"""Content-addressed artifact definitions and validation for the paper eval."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mmds import render_query

from scripts.experiments.common import (
    ExperimentDataError,
    atomic_write_json,
    load_json_object,
    load_jsonl_objects,
    sha256_file,
    sha256_json,
    sha256_text,
)
from scripts.experiments.whisper import NORMALIZATION_CONTRACT_VERSION
from scripts.lectures.download import MANIFEST_FILENAME
from scripts.lectures.materialize import (
    MATERIALIZATION_MANIFEST_FILENAME,
    MATERIALIZATION_STAGE_FILENAME,
    MATERIALIZED_INPUT_FILENAME,
    MediaProbe,
    probe_materialized_clip,
    validate_materialized_clip_stage,
)
from scripts.lectures.transcribe import RAW_CHECKPOINT_SUFFIX, TRANSCRIPT_DIRECTORY

from .catalog import (
    DEFAULT_PROFILE,
    DEFAULT_ROOT,
    EvaluationProfile,
    WHISPER_MODEL,
    catalog_payload,
    source_catalog_payload,
)
from .plans import (
    CANDIDATE_RANGE_SCHEMA,
    TRANSCRIPT_CANDIDATE_PROMPT,
    TRANSCRIPT_EVENT_SCHEMA,
    TRANSCRIPT_ONLY_PROMPT,
    VIDEO_EVENT_SCHEMA,
    VIDEO_LOCALIZATION_PROMPT,
    build_candidate_plan,
    build_naive_plan,
    build_transcript_only_plan,
    build_transcript_video_plan,
)


PREDICTION_CONFIG_FILENAME = "prediction_config.json"
EVALUATION_CONFIG_FILENAME = "evaluation_config.json"
INPUT_FILENAME = "input.jsonl"
GROUND_TRUTH_FILENAME = "ground_truth.json"
AMENDED_GROUND_TRUTH_FILENAME = "ground_truth.annotation_v2.json"
AMENDED_EVALUATION_CONFIG_FILENAME = "evaluation_config.annotation_v2.json"
ANNOTATION_AMENDMENT_FILENAME = "annotation_amendment_v1_to_v2.json"
ANNOTATION_AMENDMENT_COMPLETION_FILENAME = (
    "annotation_amendment_v1_to_v2_completion.json"
)
PREPARE_COMPLETION_FILENAME = "prepare_completion.json"
PREDICTION_PREFIX_COMPLETION_FILENAME = "prediction_prefix_completion.json"
EVALUATION_DIRECTORY = "evaluations"
COMPARISON_FILENAME = "comparison.json"
COMPARISON_COMPLETION_FILENAME = "comparison_completion.json"

NAIVE_STAGE = "naive"
TRANSCRIPT_ONLY_STAGE = "transcript_only"
CANDIDATE_STAGE = "transcript_candidates"
MATERIALIZATION_STAGE = "materialized_clips"
TRANSCRIPT_VIDEO_STAGE = "transcript_video"

PREDICTION_ARTIFACTS = ("predictions.json", "stage.json")
CANDIDATE_ARTIFACTS = ("candidates.jsonl", "audit.json", "stage.json")
MATERIALIZATION_ARTIFACTS = (
    MATERIALIZATION_MANIFEST_FILENAME,
    MATERIALIZED_INPUT_FILENAME,
    MATERIALIZATION_STAGE_FILENAME,
)
EVALUATION_ARTIFACTS = (
    "naive.json",
    "transcript_only.json",
    "transcript_video.json",
    "candidates.json",
)

FORBIDDEN_INPUT_KEYS = {
    "answer",
    "answers",
    "annotation",
    "annotations",
    "expected",
    "ground_truth",
    "label",
    "labels",
    "review_status",
    "reviewer",
    "truth",
}


def experiment_directory(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> Path:
    return root / "experiments" / profile.experiment_name


def validated_video_entries(
    root: Path,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, dict[str, Any]]:
    manifest = load_json_object(root / MANIFEST_FILENAME)
    if manifest.get("source_catalog") != source_catalog_payload(profile):
        raise ExperimentDataError("Source manifest does not match frozen paper catalog")
    entries = manifest.get("lectures")
    if not isinstance(entries, list):
        raise ExperimentDataError("Source manifest lectures must be a list")
    expected = {lecture.lecture_id: lecture for lecture in profile.lectures}
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ExperimentDataError("Source manifest entry must be an object")
        lecture_id = entry.get("lecture_id")
        if (
            not isinstance(lecture_id, str)
            or lecture_id not in expected
            or lecture_id in by_id
        ):
            raise ExperimentDataError(f"Unexpected or duplicate lecture: {lecture_id!r}")
        source = expected[lecture_id]
        if (
            entry.get("status") != "complete"
            or entry.get("download_url") != source.download_url
        ):
            raise ExperimentDataError(f"Lecture download is incomplete: {lecture_id}")
        relative = entry.get("path")
        sha = entry.get("sha256")
        media = entry.get("media")
        if (
            not isinstance(relative, str)
            or not isinstance(sha, str)
            or not isinstance(media, Mapping)
        ):
            raise ExperimentDataError(
                f"Lecture manifest metadata is incomplete: {lecture_id}"
            )
        path = root / relative
        if not path.is_file() or sha256_file(path) != sha:
            raise ExperimentDataError(f"Lecture bytes are missing or changed: {path}")
        duration = positive_finite(media.get("duration_seconds"), "video duration")
        if int(media.get("audio_stream_count", 0)) <= 0:
            raise ExperimentDataError(f"Lecture has no audio stream: {lecture_id}")
        by_id[lecture_id] = {
            **dict(entry),
            "path": path,
            "duration_seconds": duration,
        }
    if set(by_id) != set(expected):
        raise ExperimentDataError(
            "Source manifest does not contain the exact profile lectures"
        )
    return by_id


def validated_transcripts(
    root: Path,
    videos: Mapping[str, Mapping[str, Any]],
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, dict[str, Any]]:
    index = load_json_object(root / TRANSCRIPT_DIRECTORY / "index.json")
    if index.get("model") != WHISPER_MODEL:
        raise ExperimentDataError("Transcript index uses a different Whisper model")
    entries = index.get("lectures")
    if not isinstance(entries, list):
        raise ExperimentDataError("Transcript index lectures must be a list")
    indexed = {
        entry.get("lecture_id"): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    result: dict[str, dict[str, Any]] = {}
    for lecture in profile.lectures:
        lecture_id = lecture.lecture_id
        if lecture_id not in indexed:
            raise ExperimentDataError(f"Transcript index is missing {lecture_id}")
        path = root / TRANSCRIPT_DIRECTORY / f"{lecture_id}.whisper.json"
        payload = load_json_object(path)
        video = videos[lecture_id]
        expected = (
            lecture_id,
            str(video["path"].relative_to(root)),
            video["sha256"],
            WHISPER_MODEL,
            NORMALIZATION_CONTRACT_VERSION,
        )
        actual = (
            payload.get("source_id"),
            payload.get("source_path"),
            payload.get("source_sha256"),
            payload.get("model"),
            payload.get("normalization_contract_version"),
        )
        if actual != expected:
            raise ExperimentDataError(f"Transcript provenance changed: {lecture_id}")
        language = str(payload.get("language", "")).casefold()
        if language not in {"en", "eng", "english"}:
            raise ExperimentDataError(f"Transcript is not English: {lecture_id}")
        raw_path = (
            root / TRANSCRIPT_DIRECTORY / f"{lecture_id}{RAW_CHECKPOINT_SUFFIX}"
        )
        raw_sha = payload.get("raw_checkpoint_sha256")
        if (
            not isinstance(raw_sha, str)
            or not raw_path.is_file()
            or sha256_file(raw_path) != raw_sha
        ):
            raise ExperimentDataError(f"Raw Whisper checkpoint changed: {lecture_id}")
        segments = validated_segments(
            payload.get("segments"), duration=float(video["duration_seconds"])
        )
        result[lecture_id] = {**payload, "segments": segments, "path": path}
    if len(indexed) != len(result):
        raise ExperimentDataError("Transcript index contains unexpected lectures")
    return result


def validated_segments(value: Any, *, duration: float) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ExperimentDataError("Transcript segments must be a non-empty list")
    result: list[dict[str, Any]] = []
    previous_start = previous_end = -math.inf
    for index, segment in enumerate(value):
        if not isinstance(segment, Mapping) or segment.get("segment_id") != index:
            raise ExperimentDataError(
                f"Transcript segment ordering is invalid: {index}"
            )
        start = nonnegative_finite(segment.get("start_seconds"), "segment start")
        end = nonnegative_finite(segment.get("end_seconds"), "segment end")
        if (
            end <= start
            or start < previous_start
            or end < previous_end
            or end > duration
        ):
            raise ExperimentDataError(
                f"Transcript segment boundaries are invalid: {index}"
            )
        if not isinstance(segment.get("text"), str):
            raise ExperimentDataError(f"Transcript segment text is invalid: {index}")
        result.append(dict(segment))
        previous_start, previous_end = start, end
    return result


def timestamped_transcript(segments: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"[segment_id={segment['segment_id']} "
        f"{float(segment['start_seconds']):.3f}-{float(segment['end_seconds']):.3f}] "
        f"{str(segment['text']).strip()}"
        for segment in segments
    )


def rendered_plans(directory: Path) -> dict[str, str]:
    input_path = str((directory / INPUT_FILENAME).resolve())
    materialized_input = str(
        (
            directory
            / "runs"
            / MATERIALIZATION_STAGE
            / MATERIALIZED_INPUT_FILENAME
        ).resolve()
    )
    return {
        "naive": render_query(build_naive_plan(input_path)),
        "transcript_only": render_query(build_transcript_only_plan(input_path)),
        "candidates": render_query(build_candidate_plan(input_path)),
        "transcript_video": render_query(
            build_transcript_video_plan(materialized_input)
        ),
    }


def prompt_hashes() -> dict[str, str]:
    return {
        "video_localization": sha256_text(VIDEO_LOCALIZATION_PROMPT),
        "transcript_only": sha256_text(TRANSCRIPT_ONLY_PROMPT),
        "transcript_candidates": sha256_text(TRANSCRIPT_CANDIDATE_PROMPT),
    }


def schema_hashes() -> dict[str, str]:
    return {
        "video_localization": sha256_json(VIDEO_EVENT_SCHEMA),
        "transcript_only": sha256_json(TRANSCRIPT_EVENT_SCHEMA),
        "transcript_candidates": sha256_json(CANDIDATE_RANGE_SCHEMA),
    }


def prediction_prefix(
    root: Path,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> tuple[Path, dict[str, Any]]:
    validate_prediction_prefix(root, profile=profile)
    directory = experiment_directory(root, profile=profile)
    return directory, load_json_object(directory / PREDICTION_CONFIG_FILENAME)


def validate_prediction_prefix(
    root: Path,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> None:
    """Validate prediction inputs without opening labels or evaluator configuration."""
    videos = validated_video_entries(root, profile=profile)
    transcripts = validated_transcripts(root, videos, profile=profile)
    directory = experiment_directory(root, profile=profile)
    plan_sources = rendered_plans(directory)
    validate_completion(
        directory / PREDICTION_PREFIX_COMPLETION_FILENAME,
        stage_name="prediction_prefix",
        directory=directory,
        artifact_names=(
            INPUT_FILENAME,
            PREDICTION_CONFIG_FILENAME,
            *(f"plans/{name}.py" for name in sorted(plan_sources)),
        ),
        dependencies={
            "source_manifest": root / MANIFEST_FILENAME,
            **{
                f"transcript:{lecture_id}": transcript["path"]
                for lecture_id, transcript in transcripts.items()
            },
        },
    )
    config = load_json_object(directory / PREDICTION_CONFIG_FILENAME)
    if config.get("catalog") != catalog_payload(profile):
        raise ExperimentDataError("Prediction configuration catalog changed")
    if config.get("input_sha256") != sha256_file(directory / INPUT_FILENAME):
        raise ExperimentDataError("Prediction input changed after preparation")
    if (
        config.get("prompt_sha256") != prompt_hashes()
        or config.get("schema_sha256") != schema_hashes()
    ):
        raise ExperimentDataError("Frozen prompt or schema changed")
    if config.get("plan_sha256") != {
        name: sha256_text(source) for name, source in plan_sources.items()
    }:
        raise ExperimentDataError("Frozen rendered plan changed")
    validate_input_rows(
        load_jsonl_objects(directory / INPUT_FILENAME), profile=profile
    )


def validate_base_prepare(
    root: Path,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, dict[str, Any]]:
    """Validate the original preparation without resolving later label revisions."""
    videos = validated_video_entries(root, profile=profile)
    transcripts = validated_transcripts(root, videos, profile=profile)
    directory = experiment_directory(root, profile=profile)
    plan_sources = rendered_plans(directory)
    validate_completion(
        directory / PREPARE_COMPLETION_FILENAME,
        stage_name="prepare",
        directory=directory,
        artifact_names=(
            INPUT_FILENAME,
            GROUND_TRUTH_FILENAME,
            PREDICTION_CONFIG_FILENAME,
            EVALUATION_CONFIG_FILENAME,
            PREDICTION_PREFIX_COMPLETION_FILENAME,
            *(f"plans/{name}.py" for name in sorted(plan_sources)),
        ),
        dependencies={
            "source_manifest": root / MANIFEST_FILENAME,
            **{
                f"transcript:{lecture_id}": transcript["path"]
                for lecture_id, transcript in transcripts.items()
            },
        },
    )
    validate_prediction_prefix(root, profile=profile)
    _validate_evaluation_files(
        directory / GROUND_TRUTH_FILENAME,
        directory / EVALUATION_CONFIG_FILENAME,
        profile=profile,
    )
    return videos


def validate_prepare(
    root: Path,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> None:
    videos = validate_base_prepare(root, profile=profile)
    directory = experiment_directory(root, profile=profile)
    validate_evaluation_prefix(directory, videos=videos, profile=profile)


def active_evaluation_paths(directory: Path) -> tuple[Path, Path]:
    """Resolve evaluator files without exposing them to prediction validation."""
    if (directory / ANNOTATION_AMENDMENT_COMPLETION_FILENAME).is_file():
        return (
            directory / AMENDED_GROUND_TRUTH_FILENAME,
            directory / AMENDED_EVALUATION_CONFIG_FILENAME,
        )
    return (
        directory / GROUND_TRUTH_FILENAME,
        directory / EVALUATION_CONFIG_FILENAME,
    )


def validate_evaluation_prefix(
    directory: Path,
    *,
    videos: Mapping[str, Mapping[str, Any]] | None = None,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> tuple[Path, Path]:
    """Validate the active label revision and return its truth/config paths."""
    base_truth_path = directory / GROUND_TRUTH_FILENAME
    base_config_path = directory / EVALUATION_CONFIG_FILENAME
    base_truth = _validate_evaluation_files(
        base_truth_path, base_config_path, profile=profile
    )

    from .annotations import annotations_for
    from .evaluation import build_ground_truth, build_superseded_ground_truth

    annotations = annotations_for(profile)
    durations = videos or _video_durations_from_truth(base_truth)
    base_version = base_truth.get("annotation_version")
    if base_version == annotations.version:
        _reject_partial_amendment(directory)
        if base_truth != build_ground_truth(durations, profile=profile):
            raise ExperimentDataError(
                "Prepared ground truth differs from the active annotations"
            )
        return base_truth_path, base_config_path
    if (
        annotations.superseded_version is None
        or base_version != annotations.superseded_version
    ):
        raise ExperimentDataError("Prepared ground truth has an unknown annotation version")
    if base_truth != build_superseded_ground_truth(durations, profile=profile):
        raise ExperimentDataError("Amendment base differs from frozen annotation v1")

    if not (directory / ANNOTATION_AMENDMENT_COMPLETION_FILENAME).is_file():
        raise ExperimentDataError(
            "Annotation-v1 preparation requires the completed annotation-v2 amendment"
        )
    _validate_annotation_amendment(directory, profile=profile)
    truth_path, config_path = active_evaluation_paths(directory)
    truth = _validate_evaluation_files(
        truth_path, config_path, profile=profile
    )
    if truth != build_ground_truth(durations, profile=profile):
        raise ExperimentDataError("Amended ground truth differs from annotation v2")
    return truth_path, config_path


def _validate_evaluation_files(
    truth_path: Path,
    config_path: Path,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    config = load_json_object(config_path)
    if config.get("ground_truth_path") != str(truth_path.resolve()):
        raise ExperimentDataError(
            "Evaluation configuration ground-truth path changed"
        )
    if config.get("ground_truth_sha256") != sha256_file(truth_path):
        raise ExperimentDataError("Ground truth changed after preparation")
    truth = load_json_object(truth_path)
    if (
        "annotation_version" in config
        and config.get("annotation_version") != truth.get("annotation_version")
    ):
        raise ExperimentDataError("Evaluation annotation version changed")
    if (
        truth.get("experiment") != profile.experiment_name
        or config.get("experiment") != profile.experiment_name
        or truth.get("pair_count") != len(profile.pairs)
        or not isinstance(truth.get("positive_pair_count"), int)
        or isinstance(truth.get("positive_pair_count"), bool)
        or not isinstance(truth.get("negative_pair_count"), int)
        or isinstance(truth.get("negative_pair_count"), bool)
        or truth.get("positive_pair_count") + truth.get("negative_pair_count")
        != len(profile.pairs)
        or config.get("pair_count") != truth.get("pair_count")
        or config.get("positive_pair_count") != truth.get("positive_pair_count")
        or config.get("negative_pair_count") != truth.get("negative_pair_count")
        or config.get("event_count") != truth.get("event_count")
    ):
        raise ExperimentDataError("Ground-truth cardinalities changed")
    return truth


def _validate_annotation_amendment(
    directory: Path,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> None:
    if profile.profile_id != DEFAULT_PROFILE.profile_id:
        raise ExperimentDataError(
            f"Profile {profile.profile_id} does not support annotation amendments"
        )
    amendment_artifacts = (
        AMENDED_GROUND_TRUTH_FILENAME,
        AMENDED_EVALUATION_CONFIG_FILENAME,
        ANNOTATION_AMENDMENT_FILENAME,
    )
    validate_completion(
        directory / ANNOTATION_AMENDMENT_COMPLETION_FILENAME,
        stage_name="annotation_amendment_v1_to_v2",
        directory=directory,
        artifact_names=amendment_artifacts,
        dependencies={
            "base_prepare_completion": directory / PREPARE_COMPLETION_FILENAME,
            "prediction_prefix_completion": (
                directory / PREDICTION_PREFIX_COMPLETION_FILENAME
            ),
        },
    )
    amendment = load_json_object(directory / ANNOTATION_AMENDMENT_FILENAME)
    from .annotations import (
        ANNOTATION_AMENDMENT_DATE,
        ANNOTATION_AMENDMENT_ID,
        ANNOTATION_AMENDMENT_REASON,
        ANNOTATION_VERSION,
        ANNOTATOR,
        SUPERSEDED_ANNOTATION_VERSION,
    )
    base_truth = load_json_object(directory / GROUND_TRUTH_FILENAME)
    amended_truth = load_json_object(directory / AMENDED_GROUND_TRUTH_FILENAME)

    expected = {
        "amendment_id": ANNOTATION_AMENDMENT_ID,
        "amendment_date": ANNOTATION_AMENDMENT_DATE,
        "reason": ANNOTATION_AMENDMENT_REASON,
        "annotator": ANNOTATOR,
        "from_annotation_version": SUPERSEDED_ANNOTATION_VERSION,
        "to_annotation_version": ANNOTATION_VERSION,
        "base_ground_truth_path": str(
            (directory / GROUND_TRUTH_FILENAME).resolve()
        ),
        "base_ground_truth_sha256": sha256_file(
            directory / GROUND_TRUTH_FILENAME
        ),
        "base_evaluation_config_sha256": sha256_file(
            directory / EVALUATION_CONFIG_FILENAME
        ),
        "amended_ground_truth_sha256": sha256_file(
            directory / AMENDED_GROUND_TRUTH_FILENAME
        ),
        "amended_evaluation_config_sha256": sha256_file(
            directory / AMENDED_EVALUATION_CONFIG_FILENAME
        ),
        "amended_ground_truth_path": str(
            (directory / AMENDED_GROUND_TRUTH_FILENAME).resolve()
        ),
        "prediction_prefix_completion_sha256": sha256_file(
            directory / PREDICTION_PREFIX_COMPLETION_FILENAME
        ),
        "event_count_before": base_truth.get("event_count"),
        "event_count_after": amended_truth.get("event_count"),
        "positive_pair_count": amended_truth.get("positive_pair_count"),
        "negative_pair_count": amended_truth.get("negative_pair_count"),
    }
    if amendment.get("schema_version") != 1:
        raise ExperimentDataError("Annotation amendment schema changed")
    for key, value in expected.items():
        if amendment.get(key) != value:
            raise ExperimentDataError(f"Annotation amendment field changed: {key}")
    preserved = amendment.get("preserved_prediction_artifacts_sha256")
    if not isinstance(preserved, Mapping) or not preserved:
        raise ExperimentDataError("Annotation amendment has no prediction snapshot")
    required = {
        INPUT_FILENAME,
        PREDICTION_CONFIG_FILENAME,
        PREDICTION_PREFIX_COMPLETION_FILENAME,
    }
    if not required.issubset(preserved):
        raise ExperimentDataError("Annotation amendment prediction snapshot is incomplete")
    for relative, expected_sha in preserved.items():
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise ExperimentDataError("Annotation amendment prediction hash is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ExperimentDataError("Annotation amendment prediction path is unsafe")
        path = directory / relative_path
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ExperimentDataError(
                f"Prediction artifact changed across annotation amendment: {path}"
            )


def _reject_partial_amendment(directory: Path) -> None:
    paths = (
        directory / AMENDED_GROUND_TRUTH_FILENAME,
        directory / AMENDED_EVALUATION_CONFIG_FILENAME,
        directory / ANNOTATION_AMENDMENT_FILENAME,
        directory / ANNOTATION_AMENDMENT_COMPLETION_FILENAME,
    )
    if any(path.exists() for path in paths):
        raise ExperimentDataError(
            "Annotation amendment artifacts exist for an annotation-v2 preparation"
        )


def _video_durations_from_truth(
    truth: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    pairs = truth.get("pairs")
    if not isinstance(pairs, list):
        raise ExperimentDataError("Ground truth pairs must be a list")
    durations: dict[str, dict[str, float]] = {}
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ExperimentDataError("Ground truth pair must be an object")
        lecture_id = nonempty_string(pair.get("lecture_id"), "lecture_id")
        duration = positive_finite(pair.get("duration_seconds"), "video duration")
        previous = durations.get(lecture_id)
        if previous is not None and previous["duration_seconds"] != duration:
            raise ExperimentDataError("Ground-truth lecture durations disagree")
        durations[lecture_id] = {"duration_seconds": duration}
    return durations


def validate_plan(config: Mapping[str, Any], name: str, plan: Any) -> None:
    expected = config.get("plan_sha256")
    if (
        not isinstance(expected, Mapping)
        or expected.get(name) != sha256_text(render_query(plan))
    ):
        raise ExperimentDataError(f"Rendered plan changed after preparation: {name}")


def validate_input_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> None:
    _validate_profile_pair_rows(rows, profile=profile)
    for row in rows:
        reject_evaluator_keys(row)
        key = (
            nonempty_string(row.get("lecture_id"), "lecture_id"),
            nonempty_string(row.get("query_id"), "query_id"),
        )
        if not isinstance(row.get("video"), Mapping):
            raise ExperimentDataError(f"Prediction input has no video: {key}")
        if not isinstance(row.get("transcript_segments"), list):
            raise ExperimentDataError(f"Prediction input has no transcript: {key}")


def validate_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> None:
    _validate_profile_pair_rows(rows, profile=profile)
    for row in rows:
        reject_evaluator_keys(row)
        for field in (
            "candidate_windows",
            "invalid_candidate_ranges",
            "duplicate_candidate_ranges",
        ):
            if not isinstance(row.get(field), list):
                raise ExperimentDataError(f"Candidate row {field} must be a list")
        for field in (
            "raw_candidate_range_count",
            "valid_candidate_range_count",
            "invalid_candidate_range_count",
            "duplicate_candidate_range_count",
            "candidate_segment_count",
        ):
            nonnegative_integer(row.get(field), field)


def _validate_profile_pair_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: EvaluationProfile,
) -> None:
    expected = set(profile.pairs)
    actual: set[tuple[str, str]] = set()
    if len(rows) != len(profile.pairs):
        raise ExperimentDataError(
            "Row count does not match the evaluation profile"
        )
    for row in rows:
        key = (
            nonempty_string(row.get("lecture_id"), "lecture_id"),
            nonempty_string(row.get("query_id"), "query_id"),
        )
        if key in actual:
            raise ExperimentDataError(f"Duplicate evaluation pair: {key}")
        actual.add(key)
    if actual != expected:
        raise ExperimentDataError("Rows do not contain the exact profile pairs")


def validate_candidate_stage(
    directory: Path,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> None:
    stage_directory = directory / "runs" / CANDIDATE_STAGE
    validate_completion(
        stage_directory / "completion.json",
        stage_name=CANDIDATE_STAGE,
        directory=stage_directory,
        artifact_names=CANDIDATE_ARTIFACTS,
        dependencies=prediction_dependencies(directory, "candidates"),
    )
    rows = load_jsonl_objects(stage_directory / "candidates.jsonl")
    if len(rows) != len(profile.pairs):
        raise ExperimentDataError("Candidate stage must retain all profile pairs")
    validate_candidate_rows(rows, profile=profile)


def validate_materialization_stage(
    directory: Path,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
    media_probe: MediaProbe = probe_materialized_clip,
) -> None:
    validate_candidate_stage(directory, profile=profile)
    stage_directory = directory / "runs" / MATERIALIZATION_STAGE
    candidate_path = directory / "runs" / CANDIDATE_STAGE / "candidates.jsonl"
    validate_completion(
        stage_directory / "completion.json",
        stage_name=MATERIALIZATION_STAGE,
        directory=stage_directory,
        artifact_names=MATERIALIZATION_ARTIFACTS,
        dependencies={"candidates": candidate_path},
    )
    validate_materialized_clip_stage(stage_directory, media_probe=media_probe)


def validate_prediction_stage(
    directory: Path,
    stage_name: str,
    plan_name: str,
    input_path: Path,
    *,
    extra_dependencies: Mapping[str, Path] | None = None,
) -> None:
    stage_directory = directory / "runs" / stage_name
    dependencies = prediction_dependencies(directory, plan_name)
    if extra_dependencies:
        dependencies.update(extra_dependencies)
    validate_completion(
        stage_directory / "completion.json",
        stage_name=stage_name,
        directory=stage_directory,
        artifact_names=PREDICTION_ARTIFACTS,
        dependencies=dependencies,
    )
    stage = load_json_object(stage_directory / "stage.json")
    if stage.get("input_sha256") != sha256_file(input_path):
        raise ExperimentDataError(f"Prediction stage input changed: {stage_name}")


def prediction_dependencies(directory: Path, plan_name: str) -> dict[str, Path]:
    return {
        "prediction_config": directory / PREDICTION_CONFIG_FILENAME,
        "input": directory / INPUT_FILENAME,
        f"plan:{plan_name}": directory / "plans" / f"{plan_name}.py",
    }


def evaluation_dependencies(directory: Path) -> dict[str, Path]:
    ground_truth_path, evaluation_config_path = active_evaluation_paths(directory)
    return {
        "ground_truth": ground_truth_path,
        "evaluation_config": evaluation_config_path,
        "naive_predictions": directory / "runs" / NAIVE_STAGE / "predictions.json",
        "transcript_only_predictions": (
            directory / "runs" / TRANSCRIPT_ONLY_STAGE / "predictions.json"
        ),
        "transcript_video_predictions": (
            directory / "runs" / TRANSCRIPT_VIDEO_STAGE / "predictions.json"
        ),
        "candidates": directory / "runs" / CANDIDATE_STAGE / "candidates.jsonl",
    }


def write_completion(
    path: Path,
    *,
    stage_name: str,
    directory: Path,
    artifact_names: Sequence[str],
    dependencies: Mapping[str, Path],
) -> None:
    artifacts = {}
    for name in artifact_names:
        artifact = directory / name
        if not artifact.is_file():
            raise ExperimentDataError(
                f"Cannot complete stage; artifact missing: {artifact}"
            )
        artifacts[name] = sha256_file(artifact)
    dependency_hashes = {}
    for name, dependency in dependencies.items():
        if not dependency.is_file():
            raise ExperimentDataError(
                f"Cannot complete stage; dependency missing: {dependency}"
            )
        dependency_hashes[name] = sha256_file(dependency)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "stage_name": stage_name,
            "artifacts_sha256": artifacts,
            "dependencies_sha256": dependency_hashes,
        },
    )


def validate_completion(
    path: Path,
    *,
    stage_name: str,
    directory: Path,
    artifact_names: Sequence[str],
    dependencies: Mapping[str, Path],
) -> None:
    completion = load_json_object(path)
    if completion.get("stage_name") != stage_name:
        raise ExperimentDataError(f"Completion marker has wrong stage name: {path}")
    artifacts = completion.get("artifacts_sha256")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(artifact_names):
        raise ExperimentDataError(f"Completion artifact set changed: {path}")
    for name in artifact_names:
        artifact = directory / name
        if not artifact.is_file() or sha256_file(artifact) != artifacts[name]:
            raise ExperimentDataError(f"Completed artifact changed: {artifact}")
    recorded_dependencies = completion.get("dependencies_sha256")
    if (
        not isinstance(recorded_dependencies, Mapping)
        or set(recorded_dependencies) != set(dependencies)
    ):
        raise ExperimentDataError(f"Completion dependency set changed: {path}")
    for name, dependency in dependencies.items():
        if (
            not dependency.is_file()
            or sha256_file(dependency) != recorded_dependencies[name]
        ):
            raise ExperimentDataError(f"Completed dependency changed: {dependency}")


def refuse_completed(stage_directory: Path) -> None:
    if (stage_directory / "completion.json").exists():
        raise ExperimentDataError(
            f"Stage is already complete and immutable: {stage_directory}"
        )


def reject_evaluator_keys(value: Any, *, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text.casefold() in FORBIDDEN_INPUT_KEYS:
                raise ExperimentDataError(
                    f"Evaluator-only key in prediction input: {path}.{key_text}"
                )
            reject_evaluator_keys(nested, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            reject_evaluator_keys(nested, path=f"{path}[{index}]")


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentDataError(f"{label} must be a non-empty string")
    return value


def nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ExperimentDataError(f"{label} must be a non-negative integer")
    return value


def positive_finite(value: Any, label: str) -> float:
    result = nonnegative_finite(value, label)
    if result <= 0:
        raise ExperimentDataError(f"{label} must be positive")
    return result


def nonnegative_finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ExperimentDataError(f"{label} must be finite and non-negative")
    return float(value)
