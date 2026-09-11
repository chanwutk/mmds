"""Immutable artifact workflow for configured lecture paper evaluations."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mmds import execute

from scripts.experiments.common import (
    ExperimentDataError,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    load_json_object,
    load_jsonl_objects,
    sha256_file,
    sha256_json,
    sha256_text,
)
from scripts.experiments.gemini_runtime import create_experiment_executor
from scripts.experiments.whisper import NORMALIZATION_CONTRACT_VERSION
from scripts.lectures.download import MANIFEST_FILENAME, download_lectures
from scripts.lectures.materialize import (
    MATERIALIZATION_STAGE_FILENAME,
    MATERIALIZED_INPUT_FILENAME,
    ClipProcessor,
    MediaProbe,
    ffmpeg_materialize_clip,
    materialize_candidate_clips,
    probe_materialized_clip,
)
from scripts.lectures.transcribe import (
    TRANSCRIPT_DIRECTORY,
    create_whisper_model,
    transcribe_lectures,
)

from .catalog import (
    CANDIDATE_PADDING_SECONDS,
    DEFAULT_PROFILE,
    DEFAULT_ROOT,
    EvaluationProfile,
    LANGUAGE,
    MODEL,
    VIDEO_FPS,
    WHISPER_MODEL,
    catalog_payload,
    source_catalog_payload,
)
from .plans import (
    TRANSCRIPT_CANDIDATE_CONTRACT_VERSION,
    TRANSCRIPT_ONLY_CONTRACT_VERSION,
    VIDEO_LOCALIZATION_CONTRACT_VERSION,
    build_candidate_plan,
    build_naive_plan,
    build_transcript_only_plan,
    build_transcript_video_plan,
)
from .artifacts import (
    AMENDED_EVALUATION_CONFIG_FILENAME,
    AMENDED_GROUND_TRUTH_FILENAME,
    ANNOTATION_AMENDMENT_COMPLETION_FILENAME,
    ANNOTATION_AMENDMENT_FILENAME,
    CANDIDATE_ARTIFACTS,
    CANDIDATE_STAGE,
    COMPARISON_COMPLETION_FILENAME,
    COMPARISON_FILENAME,
    EVALUATION_ARTIFACTS,
    EVALUATION_CONFIG_FILENAME,
    EVALUATION_DIRECTORY,
    GROUND_TRUTH_FILENAME,
    INPUT_FILENAME,
    MATERIALIZATION_ARTIFACTS,
    MATERIALIZATION_STAGE,
    NAIVE_STAGE,
    PREDICTION_ARTIFACTS,
    PREDICTION_CONFIG_FILENAME,
    PREDICTION_PREFIX_COMPLETION_FILENAME,
    PREPARE_COMPLETION_FILENAME,
    TRANSCRIPT_ONLY_STAGE,
    TRANSCRIPT_VIDEO_STAGE,
    active_evaluation_paths as _active_evaluation_paths,
    evaluation_dependencies as _evaluation_dependencies,
    experiment_directory,
    nonempty_string as _nonempty_string,
    prediction_dependencies as _prediction_dependencies,
    prediction_prefix as _prediction_prefix,
    prompt_hashes as _prompt_hashes,
    refuse_completed as _refuse_completed,
    reject_evaluator_keys as _reject_evaluator_keys,
    rendered_plans as _rendered_plans,
    schema_hashes as _schema_hashes,
    timestamped_transcript as _timestamped_transcript,
    validate_candidate_rows as _validate_candidate_rows,
    validate_candidate_stage as _validate_candidate_stage,
    validate_completion as _validate_completion,
    validate_evaluation_prefix as _validate_evaluation_prefix,
    validate_input_rows as _validate_input_rows,
    validate_materialization_stage as _validate_materialization_stage,
    validate_plan as _validate_plan,
    validate_prediction_stage as _validate_prediction_stage,
    validate_base_prepare as _validate_base_prepare,
    validate_prepare as _validate_prepare,
    validated_transcripts as _validated_transcripts,
    validated_video_entries as _validated_video_entries,
    write_completion as _write_completion,
)
from .reporting import (
    combined_latency as _combined_latency,
    comparison_method as _comparison_method,
    reductions as _reductions,
    stage_summary as _stage_summary,
    sum_media_uploads as _sum_media_uploads,
    sum_usage as _sum_usage,
    whisper_workload as _whisper_workload,
)


def download_sources(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    return download_lectures(
        root=root,
        sources=profile.lectures,
        source_catalog=source_catalog_payload(profile),
    )


def transcribe_sources(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
    model: Any | None = None,
) -> dict[str, Any]:
    _validated_video_entries(root, profile=profile)
    whisper = model or create_whisper_model(WHISPER_MODEL)
    return transcribe_lectures(
        root=root,
        model=whisper,
        model_name=WHISPER_MODEL,
        language=LANGUAGE,
    )


def prepare(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Freeze label-isolated input, evaluator labels, plans, and their hashes."""
    directory = experiment_directory(root, profile=profile)
    completion_path = directory / PREPARE_COMPLETION_FILENAME
    if completion_path.exists():
        _validate_prepare(root, profile=profile)
        raise ExperimentDataError(
            "The paper evaluation is already prepared and immutable"
        )
    videos = _validated_video_entries(root, profile=profile)
    transcripts = _validated_transcripts(root, videos, profile=profile)
    directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    lectures = {lecture.lecture_id: lecture for lecture in profile.lectures}
    queries = {query.query_id: query for query in profile.queries}
    for lecture_id, query_id in profile.pairs:
        lecture = lectures[lecture_id]
        query = queries[query_id]
        video = videos[lecture.lecture_id]
        transcript = transcripts[lecture.lecture_id]
        row = {
            "lecture_id": lecture.lecture_id,
            "query_id": query.query_id,
            "query_text": query.text,
            "duration_seconds": video["duration_seconds"],
            "video": {
                "type": "Video",
                "path": str(video["path"].resolve()),
                "sha256": video["sha256"],
                "mime_type": "video/mp4",
                "fps": VIDEO_FPS,
            },
            "transcript_segments": transcript["segments"],
            "timestamped_transcript": _timestamped_transcript(
                transcript["segments"]
            ),
            "candidate_padding_seconds": CANDIDATE_PADDING_SECONDS,
        }
        _reject_evaluator_keys(row)
        rows.append(row)
    _validate_input_rows(rows, profile=profile)

    input_path = directory / INPUT_FILENAME
    atomic_write_jsonl(input_path, rows)
    from .evaluation import build_ground_truth

    ground_truth = build_ground_truth(
        {
            lecture_id: {
                "duration_seconds": entry["duration_seconds"],
            }
            for lecture_id, entry in videos.items()
        },
        profile=profile,
    )
    ground_truth_path = directory / GROUND_TRUTH_FILENAME
    atomic_write_json(ground_truth_path, ground_truth)

    plan_sources = _rendered_plans(directory)
    plan_directory = directory / "plans"
    plan_directory.mkdir(parents=True, exist_ok=True)
    for name, source in plan_sources.items():
        atomic_write_text(plan_directory / f"{name}.py", source)

    prediction_config = {
        "schema_version": 1,
        "experiment": profile.experiment_name,
        "catalog": catalog_payload(profile),
        "catalog_sha256": sha256_json(catalog_payload(profile)),
        "source_manifest_sha256": sha256_file(root / MANIFEST_FILENAME),
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "transcript_sha256": {
            lecture_id: sha256_file(transcript["path"])
            for lecture_id, transcript in transcripts.items()
        },
        "prompt_sha256": _prompt_hashes(),
        "schema_sha256": _schema_hashes(),
        "plan_sha256": {
            name: sha256_text(source) for name, source in plan_sources.items()
        },
        "contract_versions": {
            "video_localization": VIDEO_LOCALIZATION_CONTRACT_VERSION,
            "transcript_only": TRANSCRIPT_ONLY_CONTRACT_VERSION,
            "transcript_candidates": TRANSCRIPT_CANDIDATE_CONTRACT_VERSION,
            "whisper_normalization": NORMALIZATION_CONTRACT_VERSION,
        },
    }
    evaluation_config = _evaluation_config(
        ground_truth_path, ground_truth, profile=profile
    )
    atomic_write_json(directory / PREDICTION_CONFIG_FILENAME, prediction_config)
    atomic_write_json(directory / EVALUATION_CONFIG_FILENAME, evaluation_config)
    prediction_artifacts = (
        INPUT_FILENAME,
        PREDICTION_CONFIG_FILENAME,
        *(f"plans/{name}.py" for name in sorted(plan_sources)),
    )
    prediction_dependencies = {
        "source_manifest": root / MANIFEST_FILENAME,
        **{
            f"transcript:{lecture_id}": transcript["path"]
            for lecture_id, transcript in transcripts.items()
        },
    }
    _write_completion(
        directory / PREDICTION_PREFIX_COMPLETION_FILENAME,
        stage_name="prediction_prefix",
        directory=directory,
        artifact_names=prediction_artifacts,
        dependencies=prediction_dependencies,
    )
    prepare_artifacts = (
        INPUT_FILENAME,
        GROUND_TRUTH_FILENAME,
        PREDICTION_CONFIG_FILENAME,
        EVALUATION_CONFIG_FILENAME,
        PREDICTION_PREFIX_COMPLETION_FILENAME,
        *(f"plans/{name}.py" for name in sorted(plan_sources)),
    )
    _write_completion(
        completion_path,
        stage_name="prepare",
        directory=directory,
        artifact_names=prepare_artifacts,
        dependencies=prediction_dependencies,
    )
    return {
        "experiment": profile.experiment_name,
        "input_path": str(input_path),
        "ground_truth_path": str(ground_truth_path),
        "pair_count": ground_truth["pair_count"],
        "positive_pair_count": ground_truth["positive_pair_count"],
        "negative_pair_count": ground_truth["negative_pair_count"],
        "event_count": ground_truth["event_count"],
        "prediction_config_sha256": sha256_file(
            directory / PREDICTION_CONFIG_FILENAME
        ),
        "evaluation_config_sha256": sha256_file(
            directory / EVALUATION_CONFIG_FILENAME
        ),
    }


def amend_ground_truth(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Add annotation v2 beside an intact, prepared annotation-v1 evaluator."""
    if profile.profile_id != DEFAULT_PROFILE.profile_id:
        raise ExperimentDataError(
            f"Profile {profile.profile_id} does not support annotation amendments"
        )
    directory = experiment_directory(root, profile=profile)
    amendment_paths = (
        directory / AMENDED_GROUND_TRUTH_FILENAME,
        directory / AMENDED_EVALUATION_CONFIG_FILENAME,
        directory / ANNOTATION_AMENDMENT_FILENAME,
        directory / ANNOTATION_AMENDMENT_COMPLETION_FILENAME,
    )
    if any(path.exists() for path in amendment_paths):
        if (directory / ANNOTATION_AMENDMENT_COMPLETION_FILENAME).is_file():
            _validate_prepare(root, profile=profile)
            raise ExperimentDataError("The annotation amendment is already complete")
        raise ExperimentDataError("A partial annotation amendment already exists")
    if (directory / EVALUATION_DIRECTORY).exists() or (
        directory / COMPARISON_FILENAME
    ).exists():
        raise ExperimentDataError(
            "Ground truth cannot be amended after evaluation has started"
        )

    videos = _validate_base_prepare(root, profile=profile)
    from .annotations import (
        ANNOTATION_AMENDMENT_DATE,
        ANNOTATION_AMENDMENT_ID,
        ANNOTATION_AMENDMENT_REASON,
        ANNOTATION_VERSION,
        SUPERSEDED_ANNOTATION_VERSION,
    )
    from .evaluation import build_ground_truth, build_superseded_ground_truth

    base_truth_path = directory / GROUND_TRUTH_FILENAME
    base_truth = load_json_object(base_truth_path)
    expected_base = build_superseded_ground_truth(videos, profile=profile)
    if base_truth != expected_base:
        raise ExperimentDataError(
            "Annotation amendment requires the exact frozen annotation-v1 base"
        )

    amended_truth_path = directory / AMENDED_GROUND_TRUTH_FILENAME
    amended_truth = build_ground_truth(videos, profile=profile)
    atomic_write_json(amended_truth_path, amended_truth)
    amended_config_path = directory / AMENDED_EVALUATION_CONFIG_FILENAME
    atomic_write_json(
        amended_config_path,
        _evaluation_config(
            amended_truth_path, amended_truth, profile=profile
        ),
    )
    amendment = {
        "schema_version": 1,
        "amendment_id": ANNOTATION_AMENDMENT_ID,
        "amendment_date": ANNOTATION_AMENDMENT_DATE,
        "reason": ANNOTATION_AMENDMENT_REASON,
        "annotator": amended_truth["annotator"],
        "from_annotation_version": SUPERSEDED_ANNOTATION_VERSION,
        "to_annotation_version": ANNOTATION_VERSION,
        "base_ground_truth_path": str(base_truth_path.resolve()),
        "base_ground_truth_sha256": sha256_file(base_truth_path),
        "base_evaluation_config_sha256": sha256_file(
            directory / EVALUATION_CONFIG_FILENAME
        ),
        "amended_ground_truth_path": str(amended_truth_path.resolve()),
        "amended_ground_truth_sha256": sha256_file(amended_truth_path),
        "amended_evaluation_config_sha256": sha256_file(amended_config_path),
        "prediction_prefix_completion_sha256": sha256_file(
            directory / PREDICTION_PREFIX_COMPLETION_FILENAME
        ),
        "preserved_prediction_artifacts_sha256": _prediction_artifact_hashes(
            directory
        ),
        "event_count_before": base_truth["event_count"],
        "event_count_after": amended_truth["event_count"],
        "positive_pair_count": amended_truth["positive_pair_count"],
        "negative_pair_count": amended_truth["negative_pair_count"],
    }
    atomic_write_json(directory / ANNOTATION_AMENDMENT_FILENAME, amendment)
    _write_completion(
        directory / ANNOTATION_AMENDMENT_COMPLETION_FILENAME,
        stage_name="annotation_amendment_v1_to_v2",
        directory=directory,
        artifact_names=(
            AMENDED_GROUND_TRUTH_FILENAME,
            AMENDED_EVALUATION_CONFIG_FILENAME,
            ANNOTATION_AMENDMENT_FILENAME,
        ),
        dependencies={
            "base_prepare_completion": directory / PREPARE_COMPLETION_FILENAME,
            "prediction_prefix_completion": (
                directory / PREDICTION_PREFIX_COMPLETION_FILENAME
            ),
        },
    )
    _validate_prepare(root, profile=profile)
    return {
        "amendment_id": ANNOTATION_AMENDMENT_ID,
        "from_annotation_version": SUPERSEDED_ANNOTATION_VERSION,
        "to_annotation_version": ANNOTATION_VERSION,
        "ground_truth_path": str(amended_truth_path),
        "ground_truth_sha256": sha256_file(amended_truth_path),
        "pair_count": amended_truth["pair_count"],
        "positive_pair_count": amended_truth["positive_pair_count"],
        "negative_pair_count": amended_truth["negative_pair_count"],
        "event_count": amended_truth["event_count"],
        "prediction_artifacts_changed": False,
    }


def run_naive(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
    env_file: Path = Path(".env"),
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    directory, config = _prediction_prefix(root, profile=profile)
    plan = build_naive_plan(str((directory / INPUT_FILENAME).resolve()))
    return _run_prediction_stage(
        directory=directory,
        config=config,
        stage_name=NAIVE_STAGE,
        method=NAIVE_STAGE,
        plan=plan,
        input_path=directory / INPUT_FILENAME,
        plan_name="naive",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_transcript_only(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
    env_file: Path = Path(".env"),
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    directory, config = _prediction_prefix(root, profile=profile)
    plan = build_transcript_only_plan(str((directory / INPUT_FILENAME).resolve()))
    return _run_prediction_stage(
        directory=directory,
        config=config,
        stage_name=TRANSCRIPT_ONLY_STAGE,
        method=TRANSCRIPT_ONLY_STAGE,
        plan=plan,
        input_path=directory / INPUT_FILENAME,
        plan_name="transcript_only",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_candidates(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
    env_file: Path = Path(".env"),
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    directory, config = _prediction_prefix(root, profile=profile)
    stage_directory = directory / "runs" / CANDIDATE_STAGE
    _refuse_completed(stage_directory)
    plan = build_candidate_plan(str((directory / INPUT_FILENAME).resolve()))
    _validate_plan(config, "candidates", plan)
    executor = prompt_executor or create_experiment_executor(
        stage_directory=stage_directory, model=MODEL, env_file=env_file
    )
    started = time.perf_counter()
    rows = execute(plan, prompt_executor=executor)
    elapsed = time.perf_counter() - started
    _validate_candidate_rows(rows, profile=profile)
    candidates = [_candidate_output_row(row) for row in rows]
    candidate_path = stage_directory / "candidates.jsonl"
    atomic_write_jsonl(candidate_path, candidates)
    audit = {
        "schema_version": 1,
        "pair_count": len(candidates),
        "raw_candidate_range_count": sum(
            row["raw_candidate_range_count"] for row in candidates
        ),
        "valid_candidate_range_count": sum(
            row["valid_candidate_range_count"] for row in candidates
        ),
        "invalid_candidate_range_count": sum(
            row["invalid_candidate_range_count"] for row in candidates
        ),
        "duplicate_candidate_range_count": sum(
            row["duplicate_candidate_range_count"] for row in candidates
        ),
        "merged_candidate_window_count": sum(
            len(row["candidate_windows"]) for row in candidates
        ),
        "candidate_union_duration_seconds": sum(
            sum(window["duration_seconds"] for window in row["candidate_windows"])
            for row in candidates
        ),
    }
    atomic_write_json(stage_directory / "audit.json", audit)
    stage = _stage_summary(
        stage_directory,
        elapsed=elapsed,
        executor=executor,
        stage_name=CANDIDATE_STAGE,
        plan_name="candidates",
        input_path=directory / INPUT_FILENAME,
        config=config,
    )
    atomic_write_json(stage_directory / "stage.json", stage)
    _write_completion(
        stage_directory / "completion.json",
        stage_name=CANDIDATE_STAGE,
        directory=stage_directory,
        artifact_names=CANDIDATE_ARTIFACTS,
        dependencies=_prediction_dependencies(directory, "candidates"),
    )
    return {"stage": stage, "audit": audit, "rows": candidates}


def materialize_clips(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
    clip_processor: ClipProcessor = ffmpeg_materialize_clip,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    directory, _ = _prediction_prefix(root, profile=profile)
    _validate_candidate_stage(directory, profile=profile)
    stage_directory = directory / "runs" / MATERIALIZATION_STAGE
    _refuse_completed(stage_directory)
    candidate_path = directory / "runs" / CANDIDATE_STAGE / "candidates.jsonl"
    result = materialize_candidate_clips(
        candidate_path,
        stage_directory,
        clip_processor=clip_processor,
        media_probe=media_probe,
    )
    _write_completion(
        stage_directory / "completion.json",
        stage_name=MATERIALIZATION_STAGE,
        directory=stage_directory,
        artifact_names=MATERIALIZATION_ARTIFACTS,
        dependencies={"candidates": candidate_path},
    )
    return result


def run_transcript_video(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
    env_file: Path = Path(".env"),
    prompt_executor: Any | None = None,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    directory, config = _prediction_prefix(root, profile=profile)
    _validate_materialization_stage(
        directory, profile=profile, media_probe=media_probe
    )
    input_path = directory / "runs" / MATERIALIZATION_STAGE / MATERIALIZED_INPUT_FILENAME
    plan = build_transcript_video_plan(str(input_path.resolve()))
    return _run_prediction_stage(
        directory=directory,
        config=config,
        stage_name=TRANSCRIPT_VIDEO_STAGE,
        method=TRANSCRIPT_VIDEO_STAGE,
        plan=plan,
        input_path=input_path,
        plan_name="transcript_video",
        env_file=env_file,
        prompt_executor=prompt_executor,
        dependency_paths={
            "materialized_input": input_path,
            "materialization_completion": (
                directory / "runs" / MATERIALIZATION_STAGE / "completion.json"
            ),
        },
    )


def evaluate(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Load labels only here, after validating every prediction artifact."""
    directory, _ = _prediction_prefix(root, profile=profile)
    _validate_prediction_stage(directory, NAIVE_STAGE, "naive", directory / INPUT_FILENAME)
    _validate_prediction_stage(
        directory, TRANSCRIPT_ONLY_STAGE, "transcript_only", directory / INPUT_FILENAME
    )
    _validate_candidate_stage(directory, profile=profile)
    _validate_prediction_stage(
        directory,
        TRANSCRIPT_VIDEO_STAGE,
        "transcript_video",
        directory / "runs" / MATERIALIZATION_STAGE / MATERIALIZED_INPUT_FILENAME,
        extra_dependencies={
            "materialized_input": (
                directory / "runs" / MATERIALIZATION_STAGE / MATERIALIZED_INPUT_FILENAME
            ),
            "materialization_completion": (
                directory / "runs" / MATERIALIZATION_STAGE / "completion.json"
            ),
        },
    )
    ground_truth_path, _ = _validate_evaluation_prefix(
        directory,
        videos=_validated_video_entries(root, profile=profile),
        profile=profile,
    )
    ground_truth = load_json_object(ground_truth_path)
    from .evaluation import evaluate_candidate_windows, evaluate_predictions

    evaluation_directory = directory / EVALUATION_DIRECTORY
    _refuse_completed(evaluation_directory)
    outputs: dict[str, Any] = {}
    for method in (NAIVE_STAGE, TRANSCRIPT_ONLY_STAGE, TRANSCRIPT_VIDEO_STAGE):
        payload = load_json_object(directory / "runs" / method / "predictions.json")
        predictions = payload.get("predictions")
        if payload.get("method") != method or not isinstance(predictions, list):
            raise ExperimentDataError(f"Prediction artifact is invalid for {method}")
        output = evaluate_predictions(predictions, ground_truth)
        outputs[method] = output
        atomic_write_json(evaluation_directory / f"{method}.json", output)
    candidate_rows = load_jsonl_objects(
        directory / "runs" / CANDIDATE_STAGE / "candidates.jsonl"
    )
    candidate_evaluation = evaluate_candidate_windows(candidate_rows, ground_truth)
    outputs["candidates"] = candidate_evaluation
    atomic_write_json(evaluation_directory / "candidates.json", candidate_evaluation)
    _write_completion(
        evaluation_directory / "completion.json",
        stage_name="evaluate",
        directory=evaluation_directory,
        artifact_names=EVALUATION_ARTIFACTS,
        dependencies=_evaluation_dependencies(directory),
    )
    return outputs


def compare(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    directory, _ = _prediction_prefix(root, profile=profile)
    evaluation_directory = directory / EVALUATION_DIRECTORY
    _validate_completion(
        evaluation_directory / "completion.json",
        stage_name="evaluate",
        directory=evaluation_directory,
        artifact_names=EVALUATION_ARTIFACTS,
        dependencies=_evaluation_dependencies(directory),
    )
    output_path = directory / COMPARISON_FILENAME
    if output_path.exists():
        raise ExperimentDataError("Comparison already exists and is immutable")
    evaluations = {
        method: load_json_object(evaluation_directory / f"{method}.json")
        for method in (NAIVE_STAGE, TRANSCRIPT_ONLY_STAGE, TRANSCRIPT_VIDEO_STAGE)
    }
    stages = {
        method: load_json_object(directory / "runs" / method / "stage.json")
        for method in (NAIVE_STAGE, TRANSCRIPT_ONLY_STAGE, TRANSCRIPT_VIDEO_STAGE)
    }
    candidate_stage = load_json_object(
        directory / "runs" / CANDIDATE_STAGE / "stage.json"
    )
    materialization_stage = load_json_object(
        directory / "runs" / MATERIALIZATION_STAGE / MATERIALIZATION_STAGE_FILENAME
    )
    naive_usage = stages[NAIVE_STAGE]["api_usage"]
    transcript_usage = stages[TRANSCRIPT_ONLY_STAGE]["api_usage"]
    optimized_usage = _sum_usage(
        candidate_stage["api_usage"], stages[TRANSCRIPT_VIDEO_STAGE]["api_usage"]
    )
    optimized_media = _sum_media_uploads(
        candidate_stage["media_uploads"],
        stages[TRANSCRIPT_VIDEO_STAGE]["media_uploads"],
    )
    naive_latency = _combined_latency((NAIVE_STAGE, stages[NAIVE_STAGE]))
    transcript_latency = _combined_latency(
        (TRANSCRIPT_ONLY_STAGE, stages[TRANSCRIPT_ONLY_STAGE])
    )
    optimized_latency = _combined_latency(
        (CANDIDATE_STAGE, candidate_stage),
        (MATERIALIZATION_STAGE, materialization_stage),
        (TRANSCRIPT_VIDEO_STAGE, stages[TRANSCRIPT_VIDEO_STAGE]),
    )
    candidates = load_json_object(evaluation_directory / "candidates.json")
    comparison = {
        "schema_version": 1,
        "experiment": profile.experiment_name,
        "methods": {
            NAIVE_STAGE: _comparison_method(
                evaluations[NAIVE_STAGE],
                naive_usage,
                naive_latency,
                stages[NAIVE_STAGE]["media_uploads"],
            ),
            TRANSCRIPT_ONLY_STAGE: _comparison_method(
                evaluations[TRANSCRIPT_ONLY_STAGE],
                transcript_usage,
                transcript_latency,
                stages[TRANSCRIPT_ONLY_STAGE]["media_uploads"],
            ),
            TRANSCRIPT_VIDEO_STAGE: {
                **_comparison_method(
                    evaluations[TRANSCRIPT_VIDEO_STAGE],
                    optimized_usage,
                    optimized_latency,
                    optimized_media,
                ),
                "candidate_metrics": candidates,
                "local_materialization": materialization_stage,
                "physical_stages": [
                    CANDIDATE_STAGE,
                    MATERIALIZATION_STAGE,
                    TRANSCRIPT_VIDEO_STAGE,
                ],
            },
        },
        "relative_to_naive": {
            TRANSCRIPT_ONLY_STAGE: _reductions(
                naive_usage, transcript_usage, naive_latency, transcript_latency
            ),
            TRANSCRIPT_VIDEO_STAGE: _reductions(
                naive_usage, optimized_usage, naive_latency, optimized_latency
            ),
        },
        "one_time_whisper": _whisper_workload(root),
        "cost_scope": (
            "Query-time totals exclude one-time Whisper transcription. Transcript-video "
            "includes transcript filtering, local clip materialization, and video localization."
        ),
    }
    atomic_write_json(output_path, comparison)
    _write_completion(
        directory / COMPARISON_COMPLETION_FILENAME,
        stage_name="compare",
        directory=directory,
        artifact_names=(COMPARISON_FILENAME,),
        dependencies={
            "evaluation_completion": evaluation_directory / "completion.json"
        },
    )
    return comparison


def status(
    root: Path = DEFAULT_ROOT,
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "profile": profile.profile_id,
        "experiment": profile.experiment_name,
        "downloads": "not_started",
        "transcripts": "not_started",
        "prepared": "not_started",
        "annotation": "not_started",
        "stages": {
            NAIVE_STAGE: "not_started",
            TRANSCRIPT_ONLY_STAGE: "not_started",
            CANDIDATE_STAGE: "not_started",
            MATERIALIZATION_STAGE: "not_started",
            TRANSCRIPT_VIDEO_STAGE: "not_started",
            "evaluate": "not_started",
        },
        "comparison": "not_started",
    }
    if (root / MANIFEST_FILENAME).exists():
        try:
            videos = _validated_video_entries(root, profile=profile)
            result["downloads"] = {key: "complete" for key in videos}
        except Exception as exc:
            result["downloads"] = f"invalid: {exc}"
            return result
    else:
        return result
    if (root / TRANSCRIPT_DIRECTORY / "index.json").exists():
        try:
            transcripts = _validated_transcripts(
                root, videos, profile=profile
            )
            result["transcripts"] = {key: "complete" for key in transcripts}
        except Exception as exc:
            result["transcripts"] = f"invalid: {exc}"
            return result
    else:
        return result
    directory = experiment_directory(root, profile=profile)
    if (directory / PREPARE_COMPLETION_FILENAME).exists():
        try:
            _validate_prepare(root, profile=profile)
            result["prepared"] = "complete"
            ground_truth_path, _ = _active_evaluation_paths(directory)
            ground_truth = load_json_object(ground_truth_path)
            result["annotation"] = {
                "status": (
                    "amended"
                    if ground_truth_path.name == AMENDED_GROUND_TRUTH_FILENAME
                    else "frozen"
                ),
                "version": ground_truth["annotation_version"],
                "event_count": ground_truth["event_count"],
            }
        except Exception as exc:
            result["prepared"] = f"invalid: {exc}"
            return result
    elif directory.exists():
        result["prepared"] = "partial"
        return result
    else:
        return result
    validators = {
        NAIVE_STAGE: lambda: _validate_prediction_stage(
            directory, NAIVE_STAGE, "naive", directory / INPUT_FILENAME
        ),
        TRANSCRIPT_ONLY_STAGE: lambda: _validate_prediction_stage(
            directory,
            TRANSCRIPT_ONLY_STAGE,
            "transcript_only",
            directory / INPUT_FILENAME,
        ),
        CANDIDATE_STAGE: lambda: _validate_candidate_stage(
            directory, profile=profile
        ),
        MATERIALIZATION_STAGE: lambda: _validate_materialization_stage(
            directory, profile=profile, media_probe=media_probe
        ),
        TRANSCRIPT_VIDEO_STAGE: lambda: _validate_prediction_stage(
            directory,
            TRANSCRIPT_VIDEO_STAGE,
            "transcript_video",
            directory / "runs" / MATERIALIZATION_STAGE / MATERIALIZED_INPUT_FILENAME,
            extra_dependencies={
                "materialized_input": (
                    directory / "runs" / MATERIALIZATION_STAGE / MATERIALIZED_INPUT_FILENAME
                ),
                "materialization_completion": (
                    directory / "runs" / MATERIALIZATION_STAGE / "completion.json"
                ),
            },
        ),
    }
    for name, validator in validators.items():
        stage_directory = directory / "runs" / name
        if not stage_directory.exists():
            continue
        if not (stage_directory / "completion.json").exists():
            result["stages"][name] = "partial"
            continue
        try:
            validator()
            result["stages"][name] = "complete"
        except Exception as exc:
            result["stages"][name] = f"invalid: {exc}"
    evaluation_directory = directory / EVALUATION_DIRECTORY
    if evaluation_directory.exists():
        if not (evaluation_directory / "completion.json").exists():
            result["stages"]["evaluate"] = "partial"
        else:
            try:
                _validate_completion(
                    evaluation_directory / "completion.json",
                    stage_name="evaluate",
                    directory=evaluation_directory,
                    artifact_names=EVALUATION_ARTIFACTS,
                    dependencies=_evaluation_dependencies(directory),
                )
                result["stages"]["evaluate"] = "complete"
            except Exception as exc:
                result["stages"]["evaluate"] = f"invalid: {exc}"
    if (directory / COMPARISON_COMPLETION_FILENAME).exists():
        try:
            _validate_completion(
                directory / COMPARISON_COMPLETION_FILENAME,
                stage_name="compare",
                directory=directory,
                artifact_names=(COMPARISON_FILENAME,),
                dependencies={
                    "evaluation_completion": evaluation_directory / "completion.json"
                },
            )
            result["comparison"] = "complete"
        except Exception as exc:
            result["comparison"] = f"invalid: {exc}"
    elif (directory / COMPARISON_FILENAME).exists():
        result["comparison"] = "partial"
    return result


def _evaluation_config(
    ground_truth_path: Path,
    ground_truth: Mapping[str, Any],
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": profile.experiment_name,
        "annotation_version": ground_truth["annotation_version"],
        "ground_truth_path": str(ground_truth_path.resolve()),
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "pair_count": ground_truth["pair_count"],
        "positive_pair_count": ground_truth["positive_pair_count"],
        "negative_pair_count": ground_truth["negative_pair_count"],
        "event_count": ground_truth["event_count"],
    }


def _prediction_artifact_hashes(directory: Path) -> dict[str, str]:
    """Snapshot label-free and already-completed prediction artifacts."""
    relative_paths = {
        INPUT_FILENAME,
        PREDICTION_CONFIG_FILENAME,
        PREDICTION_PREFIX_COMPLETION_FILENAME,
        *(f"plans/{name}.py" for name in sorted(_rendered_plans(directory))),
    }
    for stage_name in (
        NAIVE_STAGE,
        TRANSCRIPT_ONLY_STAGE,
        CANDIDATE_STAGE,
        MATERIALIZATION_STAGE,
        TRANSCRIPT_VIDEO_STAGE,
    ):
        stage_directory = directory / "runs" / stage_name
        completion_path = stage_directory / "completion.json"
        if not completion_path.is_file():
            continue
        completion = load_json_object(completion_path)
        artifacts = completion.get("artifacts_sha256")
        if not isinstance(artifacts, Mapping):
            raise ExperimentDataError(
                f"Prediction completion artifact map is invalid: {completion_path}"
            )
        relative_paths.add(str(completion_path.relative_to(directory)))
        for name in artifacts:
            relative_paths.add(str((stage_directory / str(name)).relative_to(directory)))
    result: dict[str, str] = {}
    for relative in sorted(relative_paths):
        path = directory / relative
        if not path.is_file():
            raise ExperimentDataError(
                f"Prediction artifact is missing during amendment: {path}"
            )
        result[relative] = sha256_file(path)
    return result


def _run_prediction_stage(
    *,
    directory: Path,
    config: Mapping[str, Any],
    stage_name: str,
    method: str,
    plan: Any,
    input_path: Path,
    plan_name: str,
    env_file: Path,
    prompt_executor: Any | None,
    dependency_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    stage_directory = directory / "runs" / stage_name
    _refuse_completed(stage_directory)
    _validate_plan(config, plan_name, plan)
    executor = prompt_executor or create_experiment_executor(
        stage_directory=stage_directory, model=MODEL, env_file=env_file
    )
    started = time.perf_counter()
    rows = execute(plan, prompt_executor=executor)
    elapsed = time.perf_counter() - started
    predictions = [_prediction_from_row(row, method) for row in rows]
    output = {
        "schema_version": 1,
        "method": method,
        "prediction_count": len(predictions),
        "predictions": predictions,
    }
    atomic_write_json(stage_directory / "predictions.json", output)
    stage = _stage_summary(
        stage_directory,
        elapsed=elapsed,
        executor=executor,
        stage_name=stage_name,
        plan_name=plan_name,
        input_path=input_path,
        config=config,
    )
    atomic_write_json(stage_directory / "stage.json", stage)
    dependencies = _prediction_dependencies(directory, plan_name)
    if dependency_paths:
        dependencies.update(dependency_paths)
    _write_completion(
        stage_directory / "completion.json",
        stage_name=stage_name,
        directory=stage_directory,
        artifact_names=PREDICTION_ARTIFACTS,
        dependencies=dependencies,
    )
    return {"stage": stage, "predictions": predictions}


def _prediction_from_row(row: Mapping[str, Any], method: str) -> dict[str, Any]:
    event = row.get("events")
    if not isinstance(event, Mapping):
        raise ExperimentDataError("Unnested prediction row has no event object")
    interval_valid = event.get("interval_valid")
    if not isinstance(interval_valid, bool):
        raise ExperimentDataError("Prediction interval_valid must be boolean")
    prediction = {
        "lecture_id": _nonempty_string(row.get("lecture_id"), "lecture_id"),
        "query_id": _nonempty_string(row.get("query_id"), "query_id"),
        "method": method,
        **dict(event),
    }
    return prediction


def _candidate_output_row(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "lecture_id",
        "query_id",
        "query_text",
        "duration_seconds",
        "video",
        "candidate_windows",
        "invalid_candidate_ranges",
        "duplicate_candidate_ranges",
        "raw_candidate_range_count",
        "valid_candidate_range_count",
        "invalid_candidate_range_count",
        "duplicate_candidate_range_count",
        "candidate_segment_count",
    )
    return {field: row[field] for field in fields}
