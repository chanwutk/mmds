"""Gated three-lecture cross-modal pushdown experiment runner."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mmds import execute, render_query

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
from scripts.experiments.usage import aggregate_api_usage
from scripts.experiments.whisper import NORMALIZATION_CONTRACT_VERSION

from .catalog import (
    AUDIO_INPUT_USD_PER_MILLION_TOKENS,
    CANDIDATE_PADDING_SECONDS,
    DEFAULT_ROOT,
    EXPERIMENT_NAME,
    INPUT_USD_PER_MILLION_TOKENS,
    LECTURES,
    MODEL,
    OUTPUT_USD_PER_MILLION_TOKENS,
    PRICING_SOURCE,
    PRIMARY_TIOU_THRESHOLD,
    QUERIES,
    VIDEO_FPS,
    WHISPER_MODEL,
    catalog_payload,
    source_catalog_payload,
)
from .download import MANIFEST_FILENAME, download_lectures
from .evaluation import (
    evaluate_binary_decisions,
    evaluate_candidate_windows,
    evaluate_predictions,
)
from .materialize import (
    MATERIALIZATION_MANIFEST_FILENAME,
    MATERIALIZATION_STAGE_FILENAME,
    MATERIALIZED_INPUT_FILENAME,
    ClipProcessor,
    MediaProbe,
    ffmpeg_materialize_clip,
    materialize_candidate_clips,
    probe_materialized_clip,
    validate_materialized_clip_stage,
)
from .plans import (
    BINARY_VERIFICATION_CONTRACT_VERSION,
    BINARY_VERIFICATION_SCHEMA,
    BINARY_VERIFIER_PROMPT,
    CANDIDATE_RANGE_SCHEMA,
    EPISODE_LOCALIZATION_CONTRACT_VERSION,
    EPISODE_LOCALIZATION_PROMPT,
    EPISODE_SCHEMA,
    INTERVAL_CONTRACT_VERSION,
    PREDICATE_GROUNDING_CONTRACT_VERSION,
    PREDICATE_GROUNDING_PROMPT,
    PREDICATE_GROUNDING_SCHEMA,
    QUERY_CONDITION_CONTRACT_VERSION,
    QUERY_CONDITION_PROMPT,
    QUERY_CONDITION_SCHEMA,
    ROLE_AWARE_PREDICATE_GROUNDING_CONTRACT_VERSION,
    ROLE_AWARE_PREDICATE_GROUNDING_PROMPT,
    ROLE_AWARE_QUERY_CONDITION_CONTRACT_VERSION,
    ROLE_AWARE_QUERY_CONDITION_PROMPT,
    ROLE_AWARE_QUERY_CONDITION_SCHEMA,
    TRANSCRIPT_CANDIDATE_PROMPT,
    TRANSCRIPT_EPISODE_PROPOSAL_CONTRACT_VERSION,
    TRANSCRIPT_EPISODE_PROPOSAL_PROMPT,
    TRANSCRIPT_EPISODE_PROPOSAL_SCHEMA,
    TRANSCRIPT_EVENT_RANGE_SCHEMA,
    TRANSCRIPT_GROUNDED_REFINEMENT_CONTRACT_VERSION,
    TRANSCRIPT_GROUNDED_REFINEMENT_PROMPT,
    TRANSCRIPT_GROUNDED_REFINEMENT_SCHEMA,
    TRANSCRIPT_ONLY_PROMPT,
    VERIFIER_PROMPT,
    VIDEO_EVENT_SCHEMA,
    build_candidate_plan,
    build_materialized_v2_binary_plan,
    build_materialized_v2_localization_plan,
    build_materialized_verification_plan,
    build_naive_plan,
    build_naive_v2_binary_plan,
    build_naive_v2_localization_plan,
    build_transcript_only_plan,
    build_v3_refinement_plan,
    build_v3_transcript_proposal_plan,
    build_v4_predicate_grounding_plan,
    build_v4_query_condition_plan,
    build_v5_predicate_grounding_plan,
    build_v5_query_condition_plan,
    build_verification_plan,
)
from .transcribe import (
    RAW_CHECKPOINT_SUFFIX,
    TRANSCRIPT_DIRECTORY,
    create_whisper_model,
    transcribe_lectures,
)


REVIEW_FILENAME = "ground_truth_review.json"
GROUND_TRUTH_FILENAME = "ground_truth.json"
MATERIALIZED_CLIP_STAGE_NAME = "materialized_clips"
MATERIALIZED_OPTIMIZED_STAGE_NAME = "optimized_materialized"
V2_CONFIG_FILENAME = "v2_config.json"
V2_COMPARISON_FILENAME = "v2_comparison.json"
V2_RUN_DIRECTORY = Path("runs/v2")
V2_NAIVE_BINARY_STAGE_NAME = "naive_binary"
V2_OPTIMIZED_BINARY_STAGE_NAME = "optimized_binary"
V2_NAIVE_LOCALIZATION_STAGE_NAME = "naive_localization"
V2_OPTIMIZED_LOCALIZATION_STAGE_NAME = "optimized_localization"
V3_CONFIG_FILENAME = "v3_config.json"
V3_COMPARISON_FILENAME = "v3_comparison.json"
V3_RUN_DIRECTORY = Path("runs/v3")
V3_PROPOSAL_STAGE_NAME = "transcript_episode_proposals"
V3_MATERIALIZED_STAGE_NAME = "materialized_proposal_clips"
V3_REFINEMENT_STAGE_NAME = "transcript_grounded_refinement"
V4_CONFIG_FILENAME = "v4_config.json"
V4_COMPARISON_FILENAME = "v4_comparison.json"
V4_QUERY_INPUT_FILENAME = "v4_query_input.jsonl"
V4_RUN_DIRECTORY = Path("runs/v4")
V4_QUERY_CONDITION_STAGE_NAME = "query_conditions"
V4_GROUNDING_STAGE_NAME = "predicate_grounding"
V5_CONFIG_FILENAME = "v5_config.json"
V5_COMPARISON_FILENAME = "v5_comparison.json"
V5_QUERY_INPUT_FILENAME = "v5_query_input.jsonl"
V5_RUN_DIRECTORY = Path("runs/v5")
V5_QUERY_CONDITION_STAGE_NAME = "query_conditions"
V5_GROUNDING_STAGE_NAME = "role_aware_predicate_grounding"
MATERIALIZED_CLIP_ARTIFACTS = (
    MATERIALIZATION_MANIFEST_FILENAME,
    MATERIALIZED_INPUT_FILENAME,
    MATERIALIZATION_STAGE_FILENAME,
)
V2_BINARY_STAGE_ARTIFACTS = (
    "decisions.json",
    "binary_evaluation.json",
    "localization_input.jsonl",
    "stage.json",
)
V2_LOCALIZATION_STAGE_ARTIFACTS = (
    "predictions.json",
    "evaluation.json",
    "stage.json",
)
V3_PROPOSAL_STAGE_ARTIFACTS = (
    "proposals.jsonl",
    "proposal_metrics.json",
    "stage.json",
)
V3_REFINEMENT_INPUT_FILENAME = "refinement_input.jsonl"
V3_MATERIALIZED_STAGE_ARTIFACTS = (
    *MATERIALIZED_CLIP_ARTIFACTS,
    V3_REFINEMENT_INPUT_FILENAME,
)
V3_REFINEMENT_STAGE_ARTIFACTS = V2_LOCALIZATION_STAGE_ARTIFACTS
V4_QUERY_CONDITION_STAGE_ARTIFACTS = (
    "conditions.jsonl",
    "grounding_input.jsonl",
    "stage.json",
)
V4_GROUNDING_STAGE_ARTIFACTS = V2_LOCALIZATION_STAGE_ARTIFACTS
V5_QUERY_CONDITION_STAGE_ARTIFACTS = V4_QUERY_CONDITION_STAGE_ARTIFACTS
V5_GROUNDING_STAGE_ARTIFACTS = V2_LOCALIZATION_STAGE_ARTIFACTS
FORBIDDEN_INPUT_KEYS = {
    "answer",
    "answers",
    "annotation",
    "annotations",
    "events",
    "expected",
    "ground_truth",
    "label",
    "labels",
    "review_status",
    "reviewer",
    "truth",
}


def experiment_directory(root: Path) -> Path:
    return root / "experiments" / EXPERIMENT_NAME


def initialize_ground_truth_review(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Create, but never overwrite, the human annotation worksheet."""
    videos = _validated_video_entries(root)
    directory = experiment_directory(root)
    review_path = directory / REVIEW_FILENAME
    truth_path = directory / GROUND_TRUTH_FILENAME
    if review_path.exists() or truth_path.exists():
        raise ExperimentDataError(
            "Refusing to overwrite an existing review or frozen ground truth"
        )
    pairs = []
    for lecture in LECTURES:
        video = videos[lecture.lecture_id]
        for query in QUERIES:
            pairs.append(
                {
                    "lecture_id": lecture.lecture_id,
                    "lecture_title": lecture.title,
                    "query_id": query.query_id,
                    "query_text": query.text,
                    "duration_seconds": video["duration_seconds"],
                    "events": [],
                    "review_status": "pending",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
    review = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "instructions": (
            "Watch every lecture-query pair. Record every minimal event interval as "
            "start_seconds/end_seconds, including an empty events list for a reviewed "
            "negative. Then set review_status to complete and reviewer to your name."
        ),
        "source_manifest_sha256": sha256_file(root / MANIFEST_FILENAME),
        "pairs": pairs,
    }
    atomic_write_json(review_path, review)
    return review


def freeze_ground_truth(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Validate all nine human decisions and write the immutable evaluator labels."""
    directory = experiment_directory(root)
    truth_path = directory / GROUND_TRUTH_FILENAME
    if truth_path.exists():
        raise ExperimentDataError(f"Frozen ground truth already exists: {truth_path}")
    _refuse_run_artifacts(directory)
    videos = _validated_video_entries(root)
    review = load_json_object(directory / REVIEW_FILENAME)
    if review.get("experiment") != EXPERIMENT_NAME:
        raise ExperimentDataError("Ground-truth review belongs to another experiment")
    if review.get("source_manifest_sha256") != sha256_file(root / MANIFEST_FILENAME):
        raise ExperimentDataError("Source manifest changed after the review was initialized")
    expected_keys = {
        (lecture.lecture_id, query.query_id)
        for lecture in LECTURES
        for query in QUERIES
    }
    pairs = review.get("pairs")
    if not isinstance(pairs, list):
        raise ExperimentDataError("Ground-truth review pairs must be a list")
    frozen_pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ExperimentDataError("Each ground-truth review pair must be an object")
        key = (str(pair.get("lecture_id", "")), str(pair.get("query_id", "")))
        if key not in expected_keys or key in seen:
            raise ExperimentDataError(f"Unexpected or duplicate ground-truth pair: {key}")
        seen.add(key)
        if pair.get("review_status") != "complete":
            raise ExperimentDataError(f"Ground-truth pair is not complete: {key}")
        reviewer = pair.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ExperimentDataError(f"Ground-truth pair has no reviewer: {key}")
        duration = float(videos[key[0]]["duration_seconds"])
        if not math.isclose(float(pair.get("duration_seconds", -1)), duration, abs_tol=1e-6):
            raise ExperimentDataError(f"Ground-truth duration changed for {key}")
        events = _validate_ground_truth_events(pair.get("events"), duration, key)
        frozen_pairs.append(
            {
                "lecture_id": key[0],
                "query_id": key[1],
                "duration_seconds": duration,
                "events": events,
                "reviewer": reviewer.strip(),
                "review_notes": str(pair.get("review_notes", "")),
            }
        )
    if seen != expected_keys:
        raise ExperimentDataError(
            f"Ground-truth review is incomplete; missing: {sorted(expected_keys - seen)}"
        )
    order = {
        (lecture.lecture_id, query.query_id): index
        for index, (lecture, query) in enumerate(
            (lecture, query) for lecture in LECTURES for query in QUERIES
        )
    }
    frozen_pairs.sort(key=lambda pair: order[(pair["lecture_id"], pair["query_id"])])
    truth = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "label_policy": (
            "Human-verified minimal audiovisual event intervals; evaluator-only; "
            "empty events means a reviewed negative pair"
        ),
        "review_sha256": sha256_file(directory / REVIEW_FILENAME),
        "source_manifest_sha256": sha256_file(root / MANIFEST_FILENAME),
        "pairs": frozen_pairs,
    }
    atomic_write_json(truth_path, truth)
    return truth


def prepare_experiment(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Freeze label-isolated inputs, plans, prompts, schemas, and content hashes."""
    directory = experiment_directory(root)
    _refuse_run_artifacts(directory)
    config_path = directory / "config.json"
    input_path = directory / "input.jsonl"
    if config_path.exists() or input_path.exists():
        raise ExperimentDataError("Refusing to overwrite an already prepared experiment")
    videos = _validated_video_entries(root)
    transcripts = _validated_transcripts(root, videos)
    truth_path = directory / GROUND_TRUTH_FILENAME
    truth = load_json_object(truth_path)
    if truth.get("source_manifest_sha256") != sha256_file(root / MANIFEST_FILENAME):
        raise ExperimentDataError("Source manifest changed after ground truth was frozen")
    _validate_frozen_ground_truth(truth, videos)

    input_rows: list[dict[str, Any]] = []
    for lecture in LECTURES:
        video = videos[lecture.lecture_id]
        transcript = transcripts[lecture.lecture_id]
        segments = transcript["segments"]
        for query in QUERIES:
            input_rows.append(
                {
                    "lecture_id": lecture.lecture_id,
                    "query_id": query.query_id,
                    "query_text": query.text,
                    "duration_seconds": video["duration_seconds"],
                    "candidate_padding_seconds": CANDIDATE_PADDING_SECONDS,
                    "video": {
                        "type": "Video",
                        "path": str((root / video["path"]).resolve()),
                        "sha256": video["sha256"],
                        "mime_type": "video/mp4",
                        "fps": VIDEO_FPS,
                    },
                    "full_video_context": {
                        "lecture_id": lecture.lecture_id,
                        "duration_seconds": video["duration_seconds"],
                    },
                    "transcript_segments": segments,
                    "timestamped_transcript": _format_timestamped_transcript(segments),
                }
            )
    _assert_label_isolation(input_rows)
    atomic_write_jsonl(input_path, input_rows)
    _write_review_plans(directory, input_path)

    config = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "catalog": catalog_payload(),
        "pair_count": len(input_rows),
        "input_path": str(input_path.resolve()),
        "ground_truth_path": str(truth_path.resolve()),
        "hashes": {
            "catalog_sha256": sha256_json(catalog_payload()),
            "source_manifest_sha256": sha256_file(root / MANIFEST_FILENAME),
            "ground_truth_sha256": sha256_file(truth_path),
            "input_sha256": sha256_file(input_path),
            "transcript_sha256_by_lecture": {
                lecture_id: sha256_file(root / TRANSCRIPT_DIRECTORY / f"{lecture_id}.whisper.json")
                for lecture_id in transcripts
            },
            **_current_plan_contract_hashes(),
        },
        "interval_contract": {
            "version": INTERVAL_CONTRACT_VERSION,
            "model_boundaries": [
                "start_minute",
                "start_second",
                "end_minute",
                "end_second",
            ],
            "video_reference_frame": "elapsed time from start of supplied video part",
            "transcript_reference_frame": "existing inclusive Whisper segment IDs",
            "invalid_policy": "retain as false positive; never repair using labels",
        },
    }
    atomic_write_json(config_path, config)
    return config


def prepare_v2(
    root: Path = DEFAULT_ROOT,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Freeze the additive binary-filter/episode-localization v2 contract."""
    directory = _require_prepared(root)
    config_path = directory / V2_CONFIG_FILENAME
    if config_path.exists():
        raise ExperimentDataError(f"Refusing to overwrite prepared v2: {config_path}")
    v2_runs = directory / V2_RUN_DIRECTORY
    if v2_runs.exists() and any(path.is_file() for path in v2_runs.rglob("*")):
        raise ExperimentDataError(
            f"Refusing to prepare v2 after v2 run artifacts exist: {v2_runs}"
        )

    candidate_directory = directory / "runs/transcript_candidates"
    _validate_stage_completion(
        candidate_directory,
        ("candidates.jsonl", "candidate_metrics.json", "stage.json"),
    )
    materialized_directory = directory / "runs" / MATERIALIZED_CLIP_STAGE_NAME
    _validate_stage_completion(materialized_directory, MATERIALIZED_CLIP_ARTIFACTS)
    validated = validate_materialized_clip_stage(
        materialized_directory,
        media_probe=media_probe,
    )
    _assert_label_isolation(validated["rows"])

    plan_hashes = _write_v2_plans(directory)
    config = {
        "schema_version": 1,
        "method_contract_version": 2,
        "experiment": EXPERIMENT_NAME,
        "model": MODEL,
        "base_experiment_config_path": str((directory / "config.json").resolve()),
        "ground_truth_path": str(
            (directory / GROUND_TRUTH_FILENAME).resolve()
        ),
        "prefix_reuse_policy": (
            "Reuse content-addressed v1 transcript candidates and standalone clips; "
            "include their original cold-start costs in optimized v2 totals."
        ),
        "evaluation_contract": {
            "binary": "pair-level event presence; OR across candidate clips",
            "localization": (
                "one-to-one temporal IoU over complete contiguous episodes; "
                "0.3 primary"
            ),
            "ground_truth_policy": (
                "Reuse the human-reviewed intervals frozen before v1 model outputs; "
                "never alter them using v1 or v2 predictions."
            ),
        },
        "hashes": {
            "base_config_sha256": sha256_file(directory / "config.json"),
            "input_sha256": sha256_file(directory / "input.jsonl"),
            "ground_truth_sha256": sha256_file(
                directory / GROUND_TRUTH_FILENAME
            ),
            "prefix_artifacts_sha256": _v2_prefix_artifact_hashes(directory),
            "plans_sha256": plan_hashes,
            **_current_v2_contract_hashes(),
        },
        "binary_contract": {
            "version": BINARY_VERIFICATION_CONTRACT_VERSION,
            "decision_field": "verification.event_present",
            "invalid_policy": "fail stage; never coerce truthy values",
        },
        "localization_contract": {
            "version": EPISODE_LOCALIZATION_CONTRACT_VERSION,
            "conditional_on": "verification.event_present == true",
            "output_field": "episodes",
            "episode_policy": "complete contiguous physical or audible episode",
            "video_reference_frame": "elapsed time from supplied video start",
        },
    }
    atomic_write_json(config_path, config)
    return config


def prepare_v3(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Freeze transcript-owned timestamp proposals and audiovisual refinement."""
    directory = _require_prepared(root)
    config_path = directory / V3_CONFIG_FILENAME
    if config_path.exists():
        raise ExperimentDataError(f"Refusing to overwrite prepared v3: {config_path}")
    v3_runs = directory / V3_RUN_DIRECTORY
    if v3_runs.exists() and any(path.is_file() for path in v3_runs.rglob("*")):
        raise ExperimentDataError(
            f"Refusing to prepare v3 after v3 run artifacts exist: {v3_runs}"
        )

    naive_directory = directory / "runs/naive"
    naive_artifacts = (
        "predictions.json",
        "evaluation.json",
        "stage.json",
    )
    _validate_stage_completion(naive_directory, naive_artifacts)
    plan_hashes = _write_v3_plans(directory)
    config = {
        "schema_version": 1,
        "method_contract_version": 3,
        "experiment": EXPERIMENT_NAME,
        "model": MODEL,
        "evaluation_role": (
            "post_hoc_method-development diagnostic; prompts were designed after "
            "inspecting v1/v2 failures and require held-out validation before paper use"
        ),
        "base_experiment_config_path": str((directory / "config.json").resolve()),
        "ground_truth_path": str((directory / GROUND_TRUTH_FILENAME).resolve()),
        "baseline_policy": (
            "Reuse and content-bind the completed v1 naive full-lecture result; "
            "do not rerun the naive method."
        ),
        "method": {
            "timestamp_authority": "existing inclusive Whisper segment IDs",
            "proposal_policy": "one complete continuing episode per proposal",
            "video_context_padding_seconds": CANDIDATE_PADDING_SECONDS,
            "refinement_policy": (
                "return zero ranges to reject or exactly one range whose boundaries "
                "are copied from the aligned transcript excerpt"
            ),
            "free_form_timestamp_output": False,
        },
        "hashes": {
            "base_config_sha256": sha256_file(directory / "config.json"),
            "input_sha256": sha256_file(directory / "input.jsonl"),
            "ground_truth_sha256": sha256_file(
                directory / GROUND_TRUTH_FILENAME
            ),
            "naive_baseline_artifacts_sha256": {
                **{
                    name: sha256_file(naive_directory / name)
                    for name in naive_artifacts
                },
                "completion.json": sha256_file(
                    naive_directory / "completion.json"
                ),
            },
            "plans_sha256": plan_hashes,
            **_current_v3_contract_hashes(),
        },
    }
    atomic_write_json(config_path, config)
    return config


def prepare_v4(
    root: Path = DEFAULT_ROOT,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Freeze predicate compilation and evidence-grounded timestamp detection."""
    directory = _require_v3_prepared(root)
    _validate_stage_completion(
        _v3_stage_directory(directory, V3_PROPOSAL_STAGE_NAME),
        V3_PROPOSAL_STAGE_ARTIFACTS,
    )
    _validate_v3_materialized_stage(directory, media_probe=media_probe)
    config_path = directory / V4_CONFIG_FILENAME
    query_input_path = directory / V4_QUERY_INPUT_FILENAME
    if config_path.exists() or query_input_path.exists():
        raise ExperimentDataError(
            "Refusing to overwrite an existing v4 configuration or query input"
        )
    v4_runs = directory / V4_RUN_DIRECTORY
    if v4_runs.exists() and any(path.is_file() for path in v4_runs.rglob("*")):
        raise ExperimentDataError(
            f"Refusing to prepare v4 after v4 run artifacts exist: {v4_runs}"
        )

    query_rows = [
        {"query_id": query.query_id, "query_text": query.text}
        for query in QUERIES
    ]
    _assert_label_isolation(query_rows)
    atomic_write_jsonl(query_input_path, query_rows)
    plan_hashes = _write_v4_plans(directory)
    naive_directory = directory / "runs/naive"
    naive_names = (
        "predictions.json",
        "evaluation.json",
        "stage.json",
        "completion.json",
    )
    config = {
        "schema_version": 1,
        "method_contract_version": 4,
        "experiment": EXPERIMENT_NAME,
        "model": MODEL,
        "evaluation_role": (
            "post_hoc method-development diagnostic; predicate contracts were "
            "designed after inspecting v3 failures and require held-out validation"
        ),
        "method": {
            "query_compilation": (
                "one generic mandatory-condition decomposition per distinct query"
            ),
            "acceptance": (
                "return evidence for every required condition; missing any "
                "condition deterministically rejects the proposal"
            ),
            "timestamp_derivation": (
                "smallest source-time span covering all condition-evidence "
                "Whisper segment ranges"
            ),
            "free_form_timestamp_output": False,
            "v3_prefix_reuse": (
                "reuse frozen transcript proposals and standalone proposal clips; "
                "include their original cold-start costs"
            ),
        },
        "hashes": {
            "base_config_sha256": sha256_file(directory / "config.json"),
            "v3_config_sha256": sha256_file(directory / V3_CONFIG_FILENAME),
            "query_input_sha256": sha256_file(query_input_path),
            "ground_truth_sha256": sha256_file(
                directory / GROUND_TRUTH_FILENAME
            ),
            "naive_baseline_artifacts_sha256": {
                name: sha256_file(naive_directory / name) for name in naive_names
            },
            "v3_prefix_artifacts_sha256": _v4_prefix_artifact_hashes(directory),
            "plans_sha256": plan_hashes,
            **_current_v4_contract_hashes(),
        },
    }
    atomic_write_json(config_path, config)
    return config


def prepare_v5(
    root: Path = DEFAULT_ROOT,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Freeze nonredundant gate/anchor compilation and evidence grounding."""
    directory = _require_v3_prepared(root)
    _validate_stage_completion(
        _v3_stage_directory(directory, V3_PROPOSAL_STAGE_NAME),
        V3_PROPOSAL_STAGE_ARTIFACTS,
    )
    _validate_v3_materialized_stage(directory, media_probe=media_probe)
    config_path = directory / V5_CONFIG_FILENAME
    query_input_path = directory / V5_QUERY_INPUT_FILENAME
    if config_path.exists() or query_input_path.exists():
        raise ExperimentDataError(
            "Refusing to overwrite an existing v5 configuration or query input"
        )
    v5_runs = directory / V5_RUN_DIRECTORY
    if v5_runs.exists() and any(path.is_file() for path in v5_runs.rglob("*")):
        raise ExperimentDataError(
            f"Refusing to prepare v5 after v5 run artifacts exist: {v5_runs}"
        )

    query_rows = [
        {"query_id": query.query_id, "query_text": query.text}
        for query in QUERIES
    ]
    _assert_label_isolation(query_rows)
    atomic_write_jsonl(query_input_path, query_rows)
    plan_hashes = _write_v5_plans(directory)
    naive_directory = directory / "runs/naive"
    naive_names = (
        "predictions.json",
        "evaluation.json",
        "stage.json",
        "completion.json",
    )
    config = {
        "schema_version": 1,
        "method_contract_version": 5,
        "experiment": EXPERIMENT_NAME,
        "model": MODEL,
        "evaluation_role": (
            "post_hoc method-development diagnostic; gate/anchor roles were "
            "designed after inspecting v4 compilation and require held-out validation"
        ),
        "method": {
            "query_compilation": (
                "one generic nonredundant gate/anchor decomposition per query"
            ),
            "acceptance": (
                "evidence is mandatory for every gate and anchor condition"
            ),
            "timestamp_derivation": (
                "smallest source-time span covering anchor evidence only; gate "
                "evidence cannot widen the interval"
            ),
            "free_form_timestamp_output": False,
            "v3_prefix_reuse": (
                "reuse frozen transcript proposals and clips with original costs"
            ),
        },
        "hashes": {
            "base_config_sha256": sha256_file(directory / "config.json"),
            "v3_config_sha256": sha256_file(directory / V3_CONFIG_FILENAME),
            "query_input_sha256": sha256_file(query_input_path),
            "ground_truth_sha256": sha256_file(
                directory / GROUND_TRUTH_FILENAME
            ),
            "naive_baseline_artifacts_sha256": {
                name: sha256_file(naive_directory / name) for name in naive_names
            },
            "v3_prefix_artifacts_sha256": _v4_prefix_artifact_hashes(directory),
            "plans_sha256": plan_hashes,
            **_current_v5_contract_hashes(),
        },
    }
    atomic_write_json(config_path, config)
    return config


def run_naive(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    directory = _require_prepared(root)
    stage_directory = directory / "runs" / "naive"
    return _run_prediction_stage(
        directory=directory,
        stage_directory=stage_directory,
        plan=build_naive_plan(str((directory / "input.jsonl").resolve())),
        method="naive_full_audiovisual",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_transcript_only(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    directory = _require_prepared(root)
    stage_directory = directory / "runs" / "transcript_only"
    return _run_prediction_stage(
        directory=directory,
        stage_directory=stage_directory,
        plan=build_transcript_only_plan(str((directory / "input.jsonl").resolve())),
        method="transcript_only",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_candidates(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    directory = _require_prepared(root)
    stage_directory = directory / "runs" / "transcript_candidates"
    output_path = stage_directory / "candidates.jsonl"
    _refuse_completed_stage(stage_directory / "completion.json")
    executor = prompt_executor or create_experiment_executor(
        stage_directory=stage_directory, model=MODEL, env_file=env_file
    )
    started = time.perf_counter()
    rows = execute(
        build_candidate_plan(str((directory / "input.jsonl").resolve())),
        prompt_executor=executor,
    )
    elapsed = time.perf_counter() - started
    candidates = [_extract_candidate_row(row) for row in rows]
    _require_one_row_per_pair(candidates)
    truth = load_json_object(directory / GROUND_TRUTH_FILENAME)
    metrics = evaluate_candidate_windows(candidates, truth)
    stage = _stage_summary(stage_directory, elapsed, executor)
    atomic_write_json(stage_directory / "candidate_metrics.json", metrics)
    atomic_write_json(stage_directory / "stage.json", stage)
    atomic_write_jsonl(output_path, candidates)
    _write_stage_completion(
        stage_directory,
        ("candidates.jsonl", "candidate_metrics.json", "stage.json"),
    )
    return {"stage": stage, "candidate_metrics": metrics, "candidates": candidates}


def run_optimized(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    directory = _require_prepared(root)
    candidate_path = directory / "runs" / "transcript_candidates" / "candidates.jsonl"
    candidate_directory = candidate_path.parent
    if not (candidate_directory / "completion.json").is_file():
        raise ExperimentDataError(
            f"Run and validate transcript candidates first: {candidate_path}"
        )
    _validate_stage_completion(
        candidate_directory,
        ("candidates.jsonl", "candidate_metrics.json", "stage.json"),
    )
    stage_directory = directory / "runs" / "optimized"
    return _run_prediction_stage(
        directory=directory,
        stage_directory=stage_directory,
        plan=build_verification_plan(str(candidate_path.resolve())),
        method="optimized_transcript_pushdown_audiovisual",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def materialize_clips(
    root: Path = DEFAULT_ROOT,
    *,
    clip_processor: ClipProcessor = ffmpeg_materialize_clip,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Create content-addressed standalone clips for every candidate window."""
    directory = _require_prepared(root)
    candidate_directory = directory / "runs" / "transcript_candidates"
    candidate_path = candidate_directory / "candidates.jsonl"
    if not (candidate_directory / "completion.json").is_file():
        raise ExperimentDataError(
            f"Run and validate transcript candidates first: {candidate_path}"
        )
    _validate_stage_completion(
        candidate_directory,
        ("candidates.jsonl", "candidate_metrics.json", "stage.json"),
    )
    stage_directory = directory / "runs" / MATERIALIZED_CLIP_STAGE_NAME
    _refuse_completed_stage(stage_directory / "completion.json")
    finalized_paths = [stage_directory / name for name in MATERIALIZED_CLIP_ARTIFACTS]
    if all(path.is_file() for path in finalized_paths):
        validated = validate_materialized_clip_stage(
            stage_directory,
            media_probe=media_probe,
        )
        _assert_label_isolation(validated["rows"])
        _write_stage_completion(stage_directory, MATERIALIZED_CLIP_ARTIFACTS)
        checkpoint = stage_directory / "checkpoint.json"
        if checkpoint.exists():
            checkpoint.unlink()
        return {
            "manifest": validated["manifest"],
            "stage": load_json_object(stage_directory / MATERIALIZATION_STAGE_FILENAME),
            "rows": validated["rows"],
            "completion_recovered": True,
        }
    result = materialize_candidate_clips(
        candidate_path,
        stage_directory,
        clip_processor=clip_processor,
        media_probe=media_probe,
    )
    _assert_label_isolation(result["rows"])
    _write_stage_completion(stage_directory, MATERIALIZED_CLIP_ARTIFACTS)
    validate_materialized_clip_stage(stage_directory, media_probe=media_probe)
    return result


def run_optimized_materialized(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Verify standalone zero-origin candidate clips with the shared verifier."""
    directory = _require_prepared(root)
    materialized_directory = directory / "runs" / MATERIALIZED_CLIP_STAGE_NAME
    if not (materialized_directory / "completion.json").is_file():
        raise ExperimentDataError(
            "Materialize and validate standalone candidate clips first"
        )
    _validate_stage_completion(materialized_directory, MATERIALIZED_CLIP_ARTIFACTS)
    validated = validate_materialized_clip_stage(
        materialized_directory,
        media_probe=media_probe,
    )
    _assert_label_isolation(validated["rows"])
    stage_directory = directory / "runs" / MATERIALIZED_OPTIMIZED_STAGE_NAME
    materialized_input = materialized_directory / MATERIALIZED_INPUT_FILENAME
    return _run_prediction_stage(
        directory=directory,
        stage_directory=stage_directory,
        plan=build_materialized_verification_plan(str(materialized_input.resolve())),
        method="optimized_materialized_transcript_pushdown_audiovisual",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_naive_v2_binary(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    directory = _require_v2_prepared(root)
    return _run_binary_stage(
        directory=directory,
        stage_directory=_v2_stage_directory(
            directory, V2_NAIVE_BINARY_STAGE_NAME
        ),
        plan=build_naive_v2_binary_plan(
            str((directory / "input.jsonl").resolve())
        ),
        method="v2_naive_full_video_binary_verification",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_optimized_v2_binary(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    directory = _require_v2_prepared(root)
    materialized_directory = directory / "runs" / MATERIALIZED_CLIP_STAGE_NAME
    validated = validate_materialized_clip_stage(
        materialized_directory,
        media_probe=media_probe,
    )
    _assert_label_isolation(validated["rows"])
    return _run_binary_stage(
        directory=directory,
        stage_directory=_v2_stage_directory(
            directory, V2_OPTIMIZED_BINARY_STAGE_NAME
        ),
        plan=build_materialized_v2_binary_plan(
            str((materialized_directory / MATERIALIZED_INPUT_FILENAME).resolve())
        ),
        method="v2_optimized_candidate_binary_verification",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_naive_v2_localization(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    directory = _require_v2_prepared(root)
    binary_directory = _v2_stage_directory(
        directory, V2_NAIVE_BINARY_STAGE_NAME
    )
    _validate_stage_completion(binary_directory, V2_BINARY_STAGE_ARTIFACTS)
    localization_input = binary_directory / "localization_input.jsonl"
    return _run_prediction_stage(
        directory=directory,
        stage_directory=_v2_stage_directory(
            directory, V2_NAIVE_LOCALIZATION_STAGE_NAME
        ),
        plan=build_naive_v2_localization_plan(
            str(localization_input.resolve())
        ),
        method="v2_naive_full_video_complete_episode_localization",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_optimized_v2_localization(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    directory = _require_v2_prepared(root)
    materialized_directory = directory / "runs" / MATERIALIZED_CLIP_STAGE_NAME
    validate_materialized_clip_stage(
        materialized_directory,
        media_probe=media_probe,
    )
    binary_directory = _v2_stage_directory(
        directory, V2_OPTIMIZED_BINARY_STAGE_NAME
    )
    _validate_stage_completion(binary_directory, V2_BINARY_STAGE_ARTIFACTS)
    localization_input = binary_directory / "localization_input.jsonl"
    return _run_prediction_stage(
        directory=directory,
        stage_directory=_v2_stage_directory(
            directory, V2_OPTIMIZED_LOCALIZATION_STAGE_NAME
        ),
        plan=build_materialized_v2_localization_plan(
            str(localization_input.resolve())
        ),
        method="v2_optimized_candidate_complete_episode_localization",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_v3_transcript_proposals(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    """Propose complete timestamp intervals using only aligned transcripts."""
    directory = _require_v3_prepared(root)
    stage_directory = _v3_stage_directory(directory, V3_PROPOSAL_STAGE_NAME)
    _refuse_completed_stage(stage_directory / "completion.json")
    executor = prompt_executor or create_experiment_executor(
        stage_directory=stage_directory, model=MODEL, env_file=env_file
    )
    started = time.perf_counter()
    rows = execute(
        build_v3_transcript_proposal_plan(
            str((directory / "input.jsonl").resolve())
        ),
        prompt_executor=executor,
    )
    elapsed = time.perf_counter() - started
    proposals = [_extract_v3_proposal_row(row) for row in rows]
    _require_one_row_per_pair(proposals)
    _assert_label_isolation(proposals)
    truth = load_json_object(directory / GROUND_TRUTH_FILENAME)
    metrics = evaluate_candidate_windows(proposals, truth)
    stage = _stage_summary(stage_directory, elapsed, executor)
    atomic_write_jsonl(stage_directory / "proposals.jsonl", proposals)
    atomic_write_json(stage_directory / "proposal_metrics.json", metrics)
    atomic_write_json(stage_directory / "stage.json", stage)
    _write_stage_completion(stage_directory, V3_PROPOSAL_STAGE_ARTIFACTS)
    return {"stage": stage, "proposal_metrics": metrics, "proposals": proposals}


def materialize_v3_proposal_clips(
    root: Path = DEFAULT_ROOT,
    *,
    clip_processor: ClipProcessor = ffmpeg_materialize_clip,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Materialize padded proposal clips and bind aligned transcript context."""
    directory = _require_v3_prepared(root)
    proposal_directory = _v3_stage_directory(directory, V3_PROPOSAL_STAGE_NAME)
    _validate_stage_completion(proposal_directory, V3_PROPOSAL_STAGE_ARTIFACTS)
    proposal_path = proposal_directory / "proposals.jsonl"
    stage_directory = _v3_stage_directory(directory, V3_MATERIALIZED_STAGE_NAME)
    _refuse_completed_stage(stage_directory / "completion.json")

    base_finalized = [
        stage_directory / name for name in MATERIALIZED_CLIP_ARTIFACTS
    ]
    if all(path.is_file() for path in base_finalized):
        validated_clips = validate_materialized_clip_stage(
            stage_directory, media_probe=media_probe
        )
        proposal_rows = load_jsonl_objects(proposal_path)
        expected_refinement_rows = _build_v3_refinement_rows(
            proposal_rows, validated_clips["rows"]
        )
        refinement_path = stage_directory / V3_REFINEMENT_INPUT_FILENAME
        if refinement_path.is_file():
            if load_jsonl_objects(refinement_path) != expected_refinement_rows:
                raise ExperimentDataError(
                    "Refusing changed v3 refinement input during recovery"
                )
        else:
            atomic_write_jsonl(refinement_path, expected_refinement_rows)
        _assert_label_isolation(expected_refinement_rows)
        _write_stage_completion(stage_directory, V3_MATERIALIZED_STAGE_ARTIFACTS)
        validated = _validate_v3_materialized_stage(
            directory, media_probe=media_probe
        )
        checkpoint = stage_directory / "checkpoint.json"
        if checkpoint.exists():
            checkpoint.unlink()
        return {
            **validated,
            "stage": load_json_object(
                stage_directory / MATERIALIZATION_STAGE_FILENAME
            ),
            "completion_recovered": True,
        }

    result = materialize_candidate_clips(
        proposal_path,
        stage_directory,
        clip_processor=clip_processor,
        media_probe=media_probe,
    )
    proposal_rows = load_jsonl_objects(proposal_path)
    refinement_rows = _build_v3_refinement_rows(proposal_rows, result["rows"])
    _assert_label_isolation(refinement_rows)
    atomic_write_jsonl(
        stage_directory / V3_REFINEMENT_INPUT_FILENAME, refinement_rows
    )
    _write_stage_completion(stage_directory, V3_MATERIALIZED_STAGE_ARTIFACTS)
    _validate_v3_materialized_stage(directory, media_probe=media_probe)
    return {**result, "refinement_rows": refinement_rows}


def run_v3_refinement(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Refine transcript-owned boundaries using the padded audiovisual clip."""
    directory = _require_v3_prepared(root)
    _validate_v3_materialized_stage(directory, media_probe=media_probe)
    materialized_directory = _v3_stage_directory(
        directory, V3_MATERIALIZED_STAGE_NAME
    )
    return _run_prediction_stage(
        directory=directory,
        stage_directory=_v3_stage_directory(directory, V3_REFINEMENT_STAGE_NAME),
        plan=build_v3_refinement_plan(
            str(
                (
                    materialized_directory / V3_REFINEMENT_INPUT_FILENAME
                ).resolve()
            )
        ),
        method="v3_transcript_propose_audiovisual_refine",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_v4_query_conditions(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    """Compile each distinct query and attach its conditions to v3 clips."""
    directory = _require_v4_prepared(root)
    stage_directory = _v4_stage_directory(
        directory, V4_QUERY_CONDITION_STAGE_NAME
    )
    _refuse_completed_stage(stage_directory / "completion.json")
    executor = prompt_executor or create_experiment_executor(
        stage_directory=stage_directory, model=MODEL, env_file=env_file
    )
    started = time.perf_counter()
    rows = execute(
        build_v4_query_condition_plan(
            str((directory / V4_QUERY_INPUT_FILENAME).resolve())
        ),
        prompt_executor=executor,
    )
    elapsed = time.perf_counter() - started
    conditions = [_extract_v4_condition_row(row) for row in rows]
    expected_query_ids = {query.query_id for query in QUERIES}
    actual_query_ids = {row["query_id"] for row in conditions}
    if len(conditions) != len(expected_query_ids) or actual_query_ids != expected_query_ids:
        raise ExperimentDataError(
            "V4 query-condition stage did not produce one row per query"
        )
    _assert_label_isolation(conditions)
    v3_input = load_jsonl_objects(
        _v3_stage_directory(directory, V3_MATERIALIZED_STAGE_NAME)
        / V3_REFINEMENT_INPUT_FILENAME
    )
    grounding_rows = _build_v4_grounding_rows(conditions, v3_input)
    _assert_label_isolation(grounding_rows)
    stage = _stage_summary(stage_directory, elapsed, executor)
    atomic_write_jsonl(stage_directory / "conditions.jsonl", conditions)
    atomic_write_jsonl(stage_directory / "grounding_input.jsonl", grounding_rows)
    atomic_write_json(stage_directory / "stage.json", stage)
    _write_stage_completion(stage_directory, V4_QUERY_CONDITION_STAGE_ARTIFACTS)
    return {
        "stage": stage,
        "conditions": conditions,
        "grounding_row_count": len(grounding_rows),
    }


def run_v4_predicate_grounding(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Ground all mandatory conditions and derive final transcript timestamps."""
    directory = _require_v4_prepared(root)
    _validate_v3_materialized_stage(directory, media_probe=media_probe)
    condition_directory = _v4_stage_directory(
        directory, V4_QUERY_CONDITION_STAGE_NAME
    )
    _validate_stage_completion(
        condition_directory, V4_QUERY_CONDITION_STAGE_ARTIFACTS
    )
    _validate_v4_grounding_input(directory)
    return _run_prediction_stage(
        directory=directory,
        stage_directory=_v4_stage_directory(directory, V4_GROUNDING_STAGE_NAME),
        plan=build_v4_predicate_grounding_plan(
            str((condition_directory / "grounding_input.jsonl").resolve())
        ),
        method="v4_mandatory_predicate_grounding",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_v5_query_conditions(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    """Compile nonredundant gate/anchor conditions and attach them to v3 clips."""
    directory = _require_v5_prepared(root)
    stage_directory = _v5_stage_directory(
        directory, V5_QUERY_CONDITION_STAGE_NAME
    )
    _refuse_completed_stage(stage_directory / "completion.json")
    executor = prompt_executor or create_experiment_executor(
        stage_directory=stage_directory, model=MODEL, env_file=env_file
    )
    started = time.perf_counter()
    rows = execute(
        build_v5_query_condition_plan(
            str((directory / V5_QUERY_INPUT_FILENAME).resolve())
        ),
        prompt_executor=executor,
    )
    elapsed = time.perf_counter() - started
    conditions = [_extract_v4_condition_row(row) for row in rows]
    expected_query_ids = {query.query_id for query in QUERIES}
    actual_query_ids = {row["query_id"] for row in conditions}
    if len(conditions) != len(expected_query_ids) or actual_query_ids != expected_query_ids:
        raise ExperimentDataError(
            "V5 query-condition stage did not produce one row per query"
        )
    _assert_label_isolation(conditions)
    v3_input = load_jsonl_objects(
        _v3_stage_directory(directory, V3_MATERIALIZED_STAGE_NAME)
        / V3_REFINEMENT_INPUT_FILENAME
    )
    grounding_rows = _build_v4_grounding_rows(conditions, v3_input)
    _assert_label_isolation(grounding_rows)
    stage = _stage_summary(stage_directory, elapsed, executor)
    atomic_write_jsonl(stage_directory / "conditions.jsonl", conditions)
    atomic_write_jsonl(stage_directory / "grounding_input.jsonl", grounding_rows)
    atomic_write_json(stage_directory / "stage.json", stage)
    _write_stage_completion(stage_directory, V5_QUERY_CONDITION_STAGE_ARTIFACTS)
    return {
        "stage": stage,
        "conditions": conditions,
        "grounding_row_count": len(grounding_rows),
    }


def run_v5_predicate_grounding(
    root: Path = DEFAULT_ROOT,
    env_file: Path = Path(".env"),
    *,
    prompt_executor: Any | None = None,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Require all conditions and derive timestamps from anchors only."""
    directory = _require_v5_prepared(root)
    _validate_v3_materialized_stage(directory, media_probe=media_probe)
    condition_directory = _v5_stage_directory(
        directory, V5_QUERY_CONDITION_STAGE_NAME
    )
    _validate_stage_completion(
        condition_directory, V5_QUERY_CONDITION_STAGE_ARTIFACTS
    )
    _validate_v5_grounding_input(directory)
    return _run_prediction_stage(
        directory=directory,
        stage_directory=_v5_stage_directory(directory, V5_GROUNDING_STAGE_NAME),
        plan=build_v5_predicate_grounding_plan(
            str((condition_directory / "grounding_input.jsonl").resolve())
        ),
        method="v5_gate_anchor_predicate_grounding",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def compare_results(
    root: Path = DEFAULT_ROOT,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    directory = _require_prepared(root)
    required = {
        "naive": directory / "runs/naive",
        "transcript_only": directory / "runs/transcript_only",
        "transcript_candidates": directory / "runs/transcript_candidates",
        MATERIALIZED_CLIP_STAGE_NAME: directory / "runs" / MATERIALIZED_CLIP_STAGE_NAME,
        MATERIALIZED_OPTIMIZED_STAGE_NAME: directory
        / "runs"
        / MATERIALIZED_OPTIMIZED_STAGE_NAME,
    }
    artifact_names = {
        "naive": ("predictions.json", "evaluation.json", "stage.json"),
        "transcript_only": ("predictions.json", "evaluation.json", "stage.json"),
        "transcript_candidates": (
            "candidates.jsonl",
            "candidate_metrics.json",
            "stage.json",
        ),
        MATERIALIZED_CLIP_STAGE_NAME: MATERIALIZED_CLIP_ARTIFACTS,
        MATERIALIZED_OPTIMIZED_STAGE_NAME: (
            "predictions.json",
            "evaluation.json",
            "stage.json",
        ),
    }
    for name, stage_directory in required.items():
        completion = stage_directory / "completion.json"
        if not completion.is_file():
            raise ExperimentDataError(f"Stage is incomplete: {name}")
        _validate_stage_completion(stage_directory, artifact_names[name])
    validate_materialized_clip_stage(
        required[MATERIALIZED_CLIP_STAGE_NAME],
        media_probe=media_probe,
    )
    naive = _comparison_method(required["naive"])
    transcript = _comparison_method(required["transcript_only"])
    optimized_verify = _comparison_method(required[MATERIALIZED_OPTIMIZED_STAGE_NAME])
    candidate_stage = load_json_object(required["transcript_candidates"] / "stage.json")
    materialization_stage = load_json_object(
        required[MATERIALIZED_CLIP_STAGE_NAME] / MATERIALIZATION_STAGE_FILENAME
    )
    candidate_metrics = load_json_object(
        required["transcript_candidates"] / "candidate_metrics.json"
    )
    optimized = {
        "primary_accuracy": optimized_verify["primary_accuracy"],
        "api_usage": _sum_usage(candidate_stage["api_usage"], optimized_verify["api_usage"]),
        "query_time_latency": _combined_latency(
            ("transcript_candidates", candidate_stage),
            ("materialized_clips", materialization_stage),
            (
                "optimized_materialized_verification",
                load_json_object(
                    required[MATERIALIZED_OPTIMIZED_STAGE_NAME] / "stage.json"
                ),
            ),
        ),
        "candidate_metrics": candidate_metrics,
        "clip_materialization": materialization_stage,
    }
    naive["query_time_latency"] = _combined_latency(
        ("naive", load_json_object(required["naive"] / "stage.json"))
    )
    transcript["query_time_latency"] = _combined_latency(
        ("transcript_only", load_json_object(required["transcript_only"] / "stage.json"))
    )
    transcription = _transcription_cost(root)
    primary_naive = naive["primary_accuracy"]
    transcript["accuracy_delta_vs_naive"] = _accuracy_delta(primary_naive, transcript["primary_accuracy"])
    optimized["accuracy_delta_vs_naive"] = _accuracy_delta(primary_naive, optimized["primary_accuracy"])
    for method in (transcript, optimized):
        latency = method["query_time_latency"]
        method["transcript_materialization"] = transcription
        method["workload_seconds_including_one_time_transcription"] = (
            latency["end_to_end_seconds"] + transcription["total_elapsed_seconds"]
            if latency["valid"]
            else None
        )
        method["efficiency_vs_naive"] = _efficiency_vs_naive(naive, method)
    report = {
        "experiment": EXPERIMENT_NAME,
        "primary_tiou_threshold": PRIMARY_TIOU_THRESHOLD,
        "naive": naive,
        "transcript_only": transcript,
        "optimized": optimized,
    }
    offset_directory = directory / "runs" / "optimized"
    if (offset_directory / "completion.json").is_file():
        _validate_stage_completion(
            offset_directory,
            ("predictions.json", "evaluation.json", "stage.json"),
        )
        report["optimized_offset_view_diagnostic"] = {
            **_comparison_method(offset_directory),
            "excluded_from_primary_comparison": True,
            "reason": (
                "Provider outputs mixed clip-relative and source-relative timestamps "
                "for VideoMetadata offset views."
            ),
        }
    atomic_write_json(directory / "comparison.json", report)
    return report


def compare_v2_results(
    root: Path = DEFAULT_ROOT,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Compare v2 binary decisions and conditional episode localization."""
    directory = _require_v2_prepared(root)
    required = {
        V2_NAIVE_BINARY_STAGE_NAME: V2_BINARY_STAGE_ARTIFACTS,
        V2_OPTIMIZED_BINARY_STAGE_NAME: V2_BINARY_STAGE_ARTIFACTS,
        V2_NAIVE_LOCALIZATION_STAGE_NAME: V2_LOCALIZATION_STAGE_ARTIFACTS,
        V2_OPTIMIZED_LOCALIZATION_STAGE_NAME: V2_LOCALIZATION_STAGE_ARTIFACTS,
    }
    for stage_name, artifacts in required.items():
        _validate_stage_completion(
            _v2_stage_directory(directory, stage_name), artifacts
        )

    candidate_directory = directory / "runs/transcript_candidates"
    materialized_directory = directory / "runs" / MATERIALIZED_CLIP_STAGE_NAME
    validate_materialized_clip_stage(
        materialized_directory,
        media_probe=media_probe,
    )
    candidate_stage = load_json_object(candidate_directory / "stage.json")
    candidate_metrics = load_json_object(
        candidate_directory / "candidate_metrics.json"
    )
    materialization_stage = load_json_object(
        materialized_directory / MATERIALIZATION_STAGE_FILENAME
    )

    naive_binary_directory = _v2_stage_directory(
        directory, V2_NAIVE_BINARY_STAGE_NAME
    )
    optimized_binary_directory = _v2_stage_directory(
        directory, V2_OPTIMIZED_BINARY_STAGE_NAME
    )
    naive_localization_directory = _v2_stage_directory(
        directory, V2_NAIVE_LOCALIZATION_STAGE_NAME
    )
    optimized_localization_directory = _v2_stage_directory(
        directory, V2_OPTIMIZED_LOCALIZATION_STAGE_NAME
    )
    naive_binary_stage = load_json_object(naive_binary_directory / "stage.json")
    optimized_binary_stage = load_json_object(
        optimized_binary_directory / "stage.json"
    )
    naive_localization_stage = load_json_object(
        naive_localization_directory / "stage.json"
    )
    optimized_localization_stage = load_json_object(
        optimized_localization_directory / "stage.json"
    )

    naive = {
        "binary_accuracy": load_json_object(
            naive_binary_directory / "binary_evaluation.json"
        )["metrics"],
        "temporal_accuracy": load_json_object(
            naive_localization_directory / "evaluation.json"
        )["primary_metrics"],
        "api_usage_by_stage": {
            "binary_verification": naive_binary_stage["api_usage"],
            "episode_localization": naive_localization_stage["api_usage"],
        },
        "api_usage": _sum_usage(
            naive_binary_stage["api_usage"],
            naive_localization_stage["api_usage"],
        ),
        "query_time_latency": _combined_latency(
            ("naive_binary_verification", naive_binary_stage),
            ("naive_episode_localization", naive_localization_stage),
        ),
    }
    optimized_verification_usage = _sum_usage(
        optimized_binary_stage["api_usage"],
        optimized_localization_stage["api_usage"],
    )
    optimized = {
        "binary_accuracy": load_json_object(
            optimized_binary_directory / "binary_evaluation.json"
        )["metrics"],
        "temporal_accuracy": load_json_object(
            optimized_localization_directory / "evaluation.json"
        )["primary_metrics"],
        "candidate_metrics": candidate_metrics,
        "clip_materialization": materialization_stage,
        "api_usage_by_stage": {
            "transcript_candidates": candidate_stage["api_usage"],
            "binary_verification": optimized_binary_stage["api_usage"],
            "episode_localization": optimized_localization_stage["api_usage"],
        },
        "api_usage": _sum_usage(
            candidate_stage["api_usage"], optimized_verification_usage
        ),
        "query_time_latency": _combined_latency(
            ("transcript_candidates", candidate_stage),
            ("materialized_clips", materialization_stage),
            ("optimized_binary_verification", optimized_binary_stage),
            ("optimized_episode_localization", optimized_localization_stage),
        ),
    }
    optimized["binary_accuracy_delta_vs_naive"] = _binary_accuracy_delta(
        naive["binary_accuracy"], optimized["binary_accuracy"]
    )
    optimized["temporal_accuracy_delta_vs_naive"] = _accuracy_delta(
        naive["temporal_accuracy"], optimized["temporal_accuracy"]
    )
    optimized["efficiency_vs_naive"] = _efficiency_vs_naive(naive, optimized)
    transcription = _transcription_cost(root)
    optimized["transcript_materialization"] = transcription
    optimized_latency = optimized["query_time_latency"]
    optimized["workload_seconds_including_one_time_transcription"] = (
        optimized_latency["end_to_end_seconds"]
        + transcription["total_elapsed_seconds"]
        if optimized_latency["valid"]
        else None
    )
    report = {
        "schema_version": 1,
        "method_contract_version": 2,
        "experiment": EXPERIMENT_NAME,
        "binary_metric": "pair-level event presence",
        "primary_tiou_threshold": PRIMARY_TIOU_THRESHOLD,
        "naive": naive,
        "optimized": optimized,
        "prefix_reuse": {
            "transcript_candidates": str(candidate_directory.resolve()),
            "materialized_clips": str(materialized_directory.resolve()),
            "costs_included_in_optimized_totals": True,
        },
    }
    atomic_write_json(directory / V2_COMPARISON_FILENAME, report)
    return report


def compare_v3_results(
    root: Path = DEFAULT_ROOT,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Compare v3 timestamp refinement with the frozen v1 naive baseline."""
    directory = _require_v3_prepared(root)
    proposal_directory = _v3_stage_directory(directory, V3_PROPOSAL_STAGE_NAME)
    materialized_directory = _v3_stage_directory(
        directory, V3_MATERIALIZED_STAGE_NAME
    )
    refinement_directory = _v3_stage_directory(
        directory, V3_REFINEMENT_STAGE_NAME
    )
    _validate_stage_completion(proposal_directory, V3_PROPOSAL_STAGE_ARTIFACTS)
    _validate_stage_completion(
        materialized_directory, V3_MATERIALIZED_STAGE_ARTIFACTS
    )
    _validate_stage_completion(refinement_directory, V3_REFINEMENT_STAGE_ARTIFACTS)
    _validate_v3_materialized_stage(directory, media_probe=media_probe)

    naive_directory = directory / "runs/naive"
    _validate_stage_completion(
        naive_directory,
        ("predictions.json", "evaluation.json", "stage.json"),
    )
    naive = _comparison_method(naive_directory)
    naive["query_time_latency"] = _combined_latency(
        ("frozen_v1_naive", load_json_object(naive_directory / "stage.json"))
    )

    proposal_stage = load_json_object(proposal_directory / "stage.json")
    materialization_stage = load_json_object(
        materialized_directory / MATERIALIZATION_STAGE_FILENAME
    )
    refinement_stage = load_json_object(refinement_directory / "stage.json")
    optimized = {
        "primary_accuracy": load_json_object(
            refinement_directory / "evaluation.json"
        )["primary_metrics"],
        "proposal_metrics": load_json_object(
            proposal_directory / "proposal_metrics.json"
        ),
        "clip_materialization": materialization_stage,
        "api_usage_by_stage": {
            "transcript_episode_proposals": proposal_stage["api_usage"],
            "audiovisual_transcript_boundary_refinement": refinement_stage[
                "api_usage"
            ],
        },
        "api_usage": _sum_usage(
            proposal_stage["api_usage"], refinement_stage["api_usage"]
        ),
        "query_time_latency": _combined_latency(
            ("transcript_episode_proposals", proposal_stage),
            ("materialized_proposal_clips", materialization_stage),
            ("audiovisual_transcript_boundary_refinement", refinement_stage),
        ),
    }
    optimized["accuracy_delta_vs_naive"] = _accuracy_delta(
        naive["primary_accuracy"], optimized["primary_accuracy"]
    )
    optimized["efficiency_vs_naive"] = _efficiency_vs_naive(naive, optimized)
    transcription = _transcription_cost(root)
    optimized["transcript_materialization"] = transcription
    optimized_latency = optimized["query_time_latency"]
    optimized["workload_seconds_including_one_time_transcription"] = (
        optimized_latency["end_to_end_seconds"]
        + transcription["total_elapsed_seconds"]
        if optimized_latency["valid"]
        else None
    )
    report = {
        "schema_version": 1,
        "method_contract_version": 3,
        "experiment": EXPERIMENT_NAME,
        "evaluation_role": (
            "post_hoc method-development diagnostic; not held-out paper evidence"
        ),
        "metric": "one-to-one temporal IoU over complete episodes",
        "primary_tiou_threshold": PRIMARY_TIOU_THRESHOLD,
        "naive": naive,
        "optimized": optimized,
    }
    atomic_write_json(directory / V3_COMPARISON_FILENAME, report)
    return report


def compare_v4_results(
    root: Path = DEFAULT_ROOT,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Compare predicate-grounded timestamps with the frozen v1 naive result."""
    directory = _require_v4_prepared(root)
    _validate_v3_materialized_stage(directory, media_probe=media_probe)
    condition_directory = _v4_stage_directory(
        directory, V4_QUERY_CONDITION_STAGE_NAME
    )
    grounding_directory = _v4_stage_directory(directory, V4_GROUNDING_STAGE_NAME)
    _validate_stage_completion(
        condition_directory, V4_QUERY_CONDITION_STAGE_ARTIFACTS
    )
    _validate_stage_completion(grounding_directory, V4_GROUNDING_STAGE_ARTIFACTS)
    _validate_v4_grounding_input(directory)

    naive_directory = directory / "runs/naive"
    _validate_stage_completion(
        naive_directory,
        ("predictions.json", "evaluation.json", "stage.json"),
    )
    naive = _comparison_method(naive_directory)
    naive["query_time_latency"] = _combined_latency(
        ("frozen_v1_naive", load_json_object(naive_directory / "stage.json"))
    )

    proposal_directory = _v3_stage_directory(directory, V3_PROPOSAL_STAGE_NAME)
    materialized_directory = _v3_stage_directory(
        directory, V3_MATERIALIZED_STAGE_NAME
    )
    proposal_stage = load_json_object(proposal_directory / "stage.json")
    materialization_stage = load_json_object(
        materialized_directory / MATERIALIZATION_STAGE_FILENAME
    )
    condition_stage = load_json_object(condition_directory / "stage.json")
    grounding_stage = load_json_object(grounding_directory / "stage.json")
    optimized_usage = _sum_usage(
        _sum_usage(proposal_stage["api_usage"], condition_stage["api_usage"]),
        grounding_stage["api_usage"],
    )
    optimized = {
        "primary_accuracy": load_json_object(
            grounding_directory / "evaluation.json"
        )["primary_metrics"],
        "proposal_metrics": load_json_object(
            proposal_directory / "proposal_metrics.json"
        ),
        "clip_materialization": materialization_stage,
        "compiled_conditions": load_jsonl_objects(
            condition_directory / "conditions.jsonl"
        ),
        "api_usage_by_stage": {
            "transcript_episode_proposals": proposal_stage["api_usage"],
            "query_condition_compilation": condition_stage["api_usage"],
            "mandatory_predicate_grounding": grounding_stage["api_usage"],
        },
        "api_usage": optimized_usage,
        "query_time_latency": _combined_latency(
            ("transcript_episode_proposals", proposal_stage),
            ("materialized_proposal_clips", materialization_stage),
            ("query_condition_compilation", condition_stage),
            ("mandatory_predicate_grounding", grounding_stage),
        ),
    }
    optimized["accuracy_delta_vs_naive"] = _accuracy_delta(
        naive["primary_accuracy"], optimized["primary_accuracy"]
    )
    optimized["efficiency_vs_naive"] = _efficiency_vs_naive(naive, optimized)
    transcription = _transcription_cost(root)
    optimized["transcript_materialization"] = transcription
    optimized_latency = optimized["query_time_latency"]
    optimized["workload_seconds_including_one_time_transcription"] = (
        optimized_latency["end_to_end_seconds"]
        + transcription["total_elapsed_seconds"]
        if optimized_latency["valid"]
        else None
    )
    report = {
        "schema_version": 1,
        "method_contract_version": 4,
        "experiment": EXPERIMENT_NAME,
        "evaluation_role": (
            "post-hoc method-development diagnostic; not held-out paper evidence"
        ),
        "metric": "one-to-one temporal IoU over mandatory predicate evidence",
        "primary_tiou_threshold": PRIMARY_TIOU_THRESHOLD,
        "naive": naive,
        "optimized": optimized,
        "prefix_reuse": {
            "v3_transcript_proposals": str(proposal_directory.resolve()),
            "v3_materialized_proposal_clips": str(
                materialized_directory.resolve()
            ),
            "original_cold_start_costs_included": True,
            "v3_refinement_not_reused_or_counted": True,
        },
    }
    atomic_write_json(directory / V4_COMPARISON_FILENAME, report)
    return report


def compare_v5_results(
    root: Path = DEFAULT_ROOT,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Compare role-aware predicate timestamps with the frozen naive result."""
    directory = _require_v5_prepared(root)
    _validate_v3_materialized_stage(directory, media_probe=media_probe)
    condition_directory = _v5_stage_directory(
        directory, V5_QUERY_CONDITION_STAGE_NAME
    )
    grounding_directory = _v5_stage_directory(directory, V5_GROUNDING_STAGE_NAME)
    _validate_stage_completion(
        condition_directory, V5_QUERY_CONDITION_STAGE_ARTIFACTS
    )
    _validate_stage_completion(grounding_directory, V5_GROUNDING_STAGE_ARTIFACTS)
    _validate_v5_grounding_input(directory)

    naive_directory = directory / "runs/naive"
    _validate_stage_completion(
        naive_directory,
        ("predictions.json", "evaluation.json", "stage.json"),
    )
    naive = _comparison_method(naive_directory)
    naive["query_time_latency"] = _combined_latency(
        ("frozen_v1_naive", load_json_object(naive_directory / "stage.json"))
    )
    proposal_directory = _v3_stage_directory(directory, V3_PROPOSAL_STAGE_NAME)
    materialized_directory = _v3_stage_directory(
        directory, V3_MATERIALIZED_STAGE_NAME
    )
    proposal_stage = load_json_object(proposal_directory / "stage.json")
    materialization_stage = load_json_object(
        materialized_directory / MATERIALIZATION_STAGE_FILENAME
    )
    condition_stage = load_json_object(condition_directory / "stage.json")
    grounding_stage = load_json_object(grounding_directory / "stage.json")
    optimized = {
        "primary_accuracy": load_json_object(
            grounding_directory / "evaluation.json"
        )["primary_metrics"],
        "proposal_metrics": load_json_object(
            proposal_directory / "proposal_metrics.json"
        ),
        "clip_materialization": materialization_stage,
        "compiled_conditions": load_jsonl_objects(
            condition_directory / "conditions.jsonl"
        ),
        "api_usage_by_stage": {
            "transcript_episode_proposals": proposal_stage["api_usage"],
            "gate_anchor_query_compilation": condition_stage["api_usage"],
            "role_aware_predicate_grounding": grounding_stage["api_usage"],
        },
        "api_usage": _sum_usage(
            _sum_usage(
                proposal_stage["api_usage"], condition_stage["api_usage"]
            ),
            grounding_stage["api_usage"],
        ),
        "query_time_latency": _combined_latency(
            ("transcript_episode_proposals", proposal_stage),
            ("materialized_proposal_clips", materialization_stage),
            ("gate_anchor_query_compilation", condition_stage),
            ("role_aware_predicate_grounding", grounding_stage),
        ),
    }
    optimized["accuracy_delta_vs_naive"] = _accuracy_delta(
        naive["primary_accuracy"], optimized["primary_accuracy"]
    )
    optimized["efficiency_vs_naive"] = _efficiency_vs_naive(naive, optimized)
    transcription = _transcription_cost(root)
    optimized["transcript_materialization"] = transcription
    optimized_latency = optimized["query_time_latency"]
    optimized["workload_seconds_including_one_time_transcription"] = (
        optimized_latency["end_to_end_seconds"]
        + transcription["total_elapsed_seconds"]
        if optimized_latency["valid"]
        else None
    )
    report = {
        "schema_version": 1,
        "method_contract_version": 5,
        "experiment": EXPERIMENT_NAME,
        "evaluation_role": (
            "post-hoc method-development diagnostic; not held-out paper evidence"
        ),
        "metric": "one-to-one temporal IoU over anchor predicate evidence",
        "primary_tiou_threshold": PRIMARY_TIOU_THRESHOLD,
        "naive": naive,
        "optimized": optimized,
        "prefix_reuse": {
            "v3_transcript_proposals": str(proposal_directory.resolve()),
            "v3_materialized_proposal_clips": str(
                materialized_directory.resolve()
            ),
            "original_cold_start_costs_included": True,
            "v3_and_v4_grounding_not_reused_or_counted": True,
        },
    }
    atomic_write_json(directory / V5_COMPARISON_FILENAME, report)
    return report


def status(
    root: Path = DEFAULT_ROOT,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    directory = experiment_directory(root)
    manifest_path = root / MANIFEST_FILENAME
    download_status = {lecture.lecture_id: "missing" for lecture in LECTURES}
    if manifest_path.is_file():
        try:
            manifest = load_json_object(manifest_path)
            for entry in manifest.get("lectures", []):
                if isinstance(entry, Mapping) and entry.get("lecture_id") in download_status:
                    download_status[str(entry["lecture_id"])] = str(entry.get("status", "unknown"))
        except ExperimentDataError:
            download_status = {key: "invalid_manifest" for key in download_status}
    download_validation: dict[str, Any] = {
        "checked": False,
        "valid": None,
        "error": None,
    }
    validated_videos: dict[str, dict[str, Any]] | None = None
    if all(value == "complete" for value in download_status.values()):
        download_validation["checked"] = True
        try:
            validated_videos = _validated_video_entries(root)
            download_validation["valid"] = True
        except ExperimentDataError as exc:
            download_validation["valid"] = False
            download_validation["error"] = str(exc)
    transcript_status = {
        lecture.lecture_id: _transcript_status(root, lecture.lecture_id)
        for lecture in LECTURES
    }
    transcript_validation: dict[str, Any] = {
        "checked": False,
        "valid": None,
        "error": None,
    }
    if validated_videos is not None and all(
        value == "complete" for value in transcript_status.values()
    ):
        transcript_validation["checked"] = True
        try:
            _validated_transcripts(root, validated_videos)
            transcript_validation["valid"] = True
        except ExperimentDataError as exc:
            transcript_validation["valid"] = False
            transcript_validation["error"] = str(exc)
    review_summary: dict[str, Any] = {"exists": False}
    review_path = directory / REVIEW_FILENAME
    if review_path.is_file():
        review = load_json_object(review_path)
        pairs = review.get("pairs") if isinstance(review.get("pairs"), list) else []
        review_summary = {
            "exists": True,
            "complete_pairs": sum(
                isinstance(pair, Mapping) and pair.get("review_status") == "complete"
                for pair in pairs
            ),
            "total_pairs": len(pairs),
        }
    stages = {
        "naive": (
            directory / "runs/naive",
            ("predictions.json", "evaluation.json", "stage.json"),
        ),
        "transcript_only": (
            directory / "runs/transcript_only",
            ("predictions.json", "evaluation.json", "stage.json"),
        ),
        "transcript_candidates": (
            directory / "runs/transcript_candidates",
            ("candidates.jsonl", "candidate_metrics.json", "stage.json"),
        ),
        "optimized": (
            directory / "runs/optimized",
            ("predictions.json", "evaluation.json", "stage.json"),
        ),
        MATERIALIZED_CLIP_STAGE_NAME: (
            directory / "runs" / MATERIALIZED_CLIP_STAGE_NAME,
            MATERIALIZED_CLIP_ARTIFACTS,
        ),
        MATERIALIZED_OPTIMIZED_STAGE_NAME: (
            directory / "runs" / MATERIALIZED_OPTIMIZED_STAGE_NAME,
            ("predictions.json", "evaluation.json", "stage.json"),
        ),
    }
    stage_status = {
        name: _stage_status(stage_directory, artifact_names)
        for name, (stage_directory, artifact_names) in stages.items()
    }
    v2_stages = {
        V2_NAIVE_BINARY_STAGE_NAME: V2_BINARY_STAGE_ARTIFACTS,
        V2_OPTIMIZED_BINARY_STAGE_NAME: V2_BINARY_STAGE_ARTIFACTS,
        V2_NAIVE_LOCALIZATION_STAGE_NAME: V2_LOCALIZATION_STAGE_ARTIFACTS,
        V2_OPTIMIZED_LOCALIZATION_STAGE_NAME: V2_LOCALIZATION_STAGE_ARTIFACTS,
    }
    v2_stage_status = {
        name: _stage_status(_v2_stage_directory(directory, name), artifacts)
        for name, artifacts in v2_stages.items()
    }
    v2_config_status = "not_prepared"
    if (directory / V2_CONFIG_FILENAME).is_file():
        try:
            _require_v2_prepared(root)
            v2_config_status = "prepared"
        except ExperimentDataError:
            v2_config_status = "invalid_prepared_artifacts"
    v3_stages = {
        V3_PROPOSAL_STAGE_NAME: V3_PROPOSAL_STAGE_ARTIFACTS,
        V3_MATERIALIZED_STAGE_NAME: V3_MATERIALIZED_STAGE_ARTIFACTS,
        V3_REFINEMENT_STAGE_NAME: V3_REFINEMENT_STAGE_ARTIFACTS,
    }
    v3_stage_status = {
        name: _stage_status(_v3_stage_directory(directory, name), artifacts)
        for name, artifacts in v3_stages.items()
    }
    v3_config_status = "not_prepared"
    if (directory / V3_CONFIG_FILENAME).is_file():
        try:
            _require_v3_prepared(root)
            v3_config_status = "prepared"
        except ExperimentDataError:
            v3_config_status = "invalid_prepared_artifacts"
    if v3_stage_status[V3_MATERIALIZED_STAGE_NAME] == "complete":
        try:
            _validate_v3_materialized_stage(directory, media_probe=media_probe)
        except ExperimentDataError:
            v3_stage_status[V3_MATERIALIZED_STAGE_NAME] = (
                "invalid_completed_artifacts"
            )
            if v3_stage_status[V3_REFINEMENT_STAGE_NAME] == "complete":
                v3_stage_status[V3_REFINEMENT_STAGE_NAME] = (
                    "invalid_materialized_clip_dependency"
                )
    v4_stages = {
        V4_QUERY_CONDITION_STAGE_NAME: V4_QUERY_CONDITION_STAGE_ARTIFACTS,
        V4_GROUNDING_STAGE_NAME: V4_GROUNDING_STAGE_ARTIFACTS,
    }
    v4_stage_status = {
        name: _stage_status(_v4_stage_directory(directory, name), artifacts)
        for name, artifacts in v4_stages.items()
    }
    v4_config_status = "not_prepared"
    if (directory / V4_CONFIG_FILENAME).is_file():
        try:
            _require_v4_prepared(root)
            v4_config_status = "prepared"
        except ExperimentDataError:
            v4_config_status = "invalid_prepared_artifacts"
    if v4_stage_status[V4_QUERY_CONDITION_STAGE_NAME] == "complete":
        try:
            _validate_v4_grounding_input(directory)
        except ExperimentDataError:
            v4_stage_status[V4_QUERY_CONDITION_STAGE_NAME] = (
                "invalid_completed_artifacts"
            )
            if v4_stage_status[V4_GROUNDING_STAGE_NAME] == "complete":
                v4_stage_status[V4_GROUNDING_STAGE_NAME] = (
                    "invalid_condition_dependency"
                )
    v5_stages = {
        V5_QUERY_CONDITION_STAGE_NAME: V5_QUERY_CONDITION_STAGE_ARTIFACTS,
        V5_GROUNDING_STAGE_NAME: V5_GROUNDING_STAGE_ARTIFACTS,
    }
    v5_stage_status = {
        name: _stage_status(_v5_stage_directory(directory, name), artifacts)
        for name, artifacts in v5_stages.items()
    }
    v5_config_status = "not_prepared"
    if (directory / V5_CONFIG_FILENAME).is_file():
        try:
            _require_v5_prepared(root)
            v5_config_status = "prepared"
        except ExperimentDataError:
            v5_config_status = "invalid_prepared_artifacts"
    if v5_stage_status[V5_QUERY_CONDITION_STAGE_NAME] == "complete":
        try:
            _validate_v5_grounding_input(directory)
        except ExperimentDataError:
            v5_stage_status[V5_QUERY_CONDITION_STAGE_NAME] = (
                "invalid_completed_artifacts"
            )
            if v5_stage_status[V5_GROUNDING_STAGE_NAME] == "complete":
                v5_stage_status[V5_GROUNDING_STAGE_NAME] = (
                    "invalid_condition_dependency"
                )
    if stage_status[MATERIALIZED_CLIP_STAGE_NAME] == "complete":
        try:
            validate_materialized_clip_stage(
                stages[MATERIALIZED_CLIP_STAGE_NAME][0],
                media_probe=media_probe,
            )
        except ExperimentDataError:
            stage_status[MATERIALIZED_CLIP_STAGE_NAME] = (
                "invalid_completed_artifacts"
            )
            if stage_status[MATERIALIZED_OPTIMIZED_STAGE_NAME] == "complete":
                stage_status[MATERIALIZED_OPTIMIZED_STAGE_NAME] = (
                    "invalid_materialized_clip_dependency"
                )
            for name in (
                V2_OPTIMIZED_BINARY_STAGE_NAME,
                V2_OPTIMIZED_LOCALIZATION_STAGE_NAME,
            ):
                if v2_stage_status[name] == "complete":
                    v2_stage_status[name] = "invalid_materialized_clip_dependency"
    return {
        "downloads": download_status,
        "download_validation": download_validation,
        "transcripts": transcript_status,
        "transcript_validation": transcript_validation,
        "ground_truth_review": review_summary,
        "ground_truth_frozen": (directory / GROUND_TRUTH_FILENAME).is_file(),
        "prepared": (directory / "config.json").is_file(),
        "stages": stage_status,
        "comparison": (directory / "comparison.json").is_file(),
        "v2": {
            "config": v2_config_status,
            "stages": v2_stage_status,
            "comparison": (directory / V2_COMPARISON_FILENAME).is_file(),
        },
        "v3": {
            "config": v3_config_status,
            "stages": v3_stage_status,
            "comparison": (directory / V3_COMPARISON_FILENAME).is_file(),
        },
        "v4": {
            "config": v4_config_status,
            "stages": v4_stage_status,
            "comparison": (directory / V4_COMPARISON_FILENAME).is_file(),
        },
        "v5": {
            "config": v5_config_status,
            "stages": v5_stage_status,
            "comparison": (directory / V5_COMPARISON_FILENAME).is_file(),
        },
    }


def _run_prediction_stage(
    *,
    directory: Path,
    stage_directory: Path,
    plan: Any,
    method: str,
    env_file: Path,
    prompt_executor: Any | None,
) -> dict[str, Any]:
    output_path = stage_directory / "predictions.json"
    _refuse_completed_stage(stage_directory / "completion.json")
    executor = prompt_executor or create_experiment_executor(
        stage_directory=stage_directory, model=MODEL, env_file=env_file
    )
    started = time.perf_counter()
    rows = execute(plan, prompt_executor=executor)
    elapsed = time.perf_counter() - started
    predictions = [_extract_prediction(row, method) for row in rows]
    truth = load_json_object(directory / GROUND_TRUTH_FILENAME)
    evaluation = evaluate_predictions(predictions, truth)
    stage = _stage_summary(stage_directory, elapsed, executor)
    atomic_write_json(stage_directory / "evaluation.json", evaluation)
    atomic_write_json(stage_directory / "stage.json", stage)
    atomic_write_json(output_path, {"method": method, "predictions": predictions})
    _write_stage_completion(
        stage_directory,
        ("predictions.json", "evaluation.json", "stage.json"),
    )
    return {"stage": stage, "evaluation": evaluation, "predictions": predictions}


def _run_binary_stage(
    *,
    directory: Path,
    stage_directory: Path,
    plan: Any,
    method: str,
    env_file: Path,
    prompt_executor: Any | None,
) -> dict[str, Any]:
    _refuse_completed_stage(stage_directory / "completion.json")
    executor = prompt_executor or create_experiment_executor(
        stage_directory=stage_directory, model=MODEL, env_file=env_file
    )
    started = time.perf_counter()
    rows = execute(plan, prompt_executor=executor)
    elapsed = time.perf_counter() - started
    decisions = [_extract_binary_decision(row, method) for row in rows]
    localization_rows = [_extract_localization_input(row) for row in rows]
    _assert_label_isolation(localization_rows)
    truth = load_json_object(directory / GROUND_TRUTH_FILENAME)
    evaluation = evaluate_binary_decisions(decisions, truth)
    stage = _stage_summary(stage_directory, elapsed, executor)
    atomic_write_json(
        stage_directory / "decisions.json",
        {"method": method, "decisions": decisions},
    )
    atomic_write_json(
        stage_directory / "binary_evaluation.json", evaluation
    )
    atomic_write_jsonl(
        stage_directory / "localization_input.jsonl", localization_rows
    )
    atomic_write_json(stage_directory / "stage.json", stage)
    _write_stage_completion(stage_directory, V2_BINARY_STAGE_ARTIFACTS)
    return {
        "stage": stage,
        "binary_evaluation": evaluation,
        "decisions": decisions,
    }


def _extract_binary_decision(
    row: Mapping[str, Any], method: str
) -> dict[str, Any]:
    verification = row.get("verification")
    if not isinstance(verification, Mapping):
        raise ExperimentDataError("Binary row has no verification object")
    event_present = verification.get("event_present")
    confidence = verification.get("confidence")
    evidence = verification.get("evidence")
    if not isinstance(event_present, bool):
        raise ExperimentDataError("Binary event_present must be boolean")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(float(confidence))
        or not 0 <= float(confidence) <= 1
    ):
        raise ExperimentDataError("Binary confidence must satisfy 0 <= value <= 1")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ExperimentDataError("Binary evidence must be a non-empty string")
    decision = {
        "lecture_id": str(row["lecture_id"]),
        "query_id": str(row["query_id"]),
        "method": method,
        "event_present": event_present,
        "confidence": float(confidence),
        "evidence": evidence,
    }
    window = row.get("candidate_windows")
    if window is not None:
        if not isinstance(window, Mapping):
            raise ExperimentDataError("Binary candidate window must be an object")
        window_id = window.get("window_id")
        if (
            not isinstance(window_id, int)
            or isinstance(window_id, bool)
            or window_id < 0
        ):
            raise ExperimentDataError("Binary candidate window_id is invalid")
        decision["window_id"] = window_id
    return decision


def _extract_localization_input(row: Mapping[str, Any]) -> dict[str, Any]:
    verification = row.get("verification")
    if not isinstance(verification, Mapping):
        raise ExperimentDataError("Localization input has no verification object")
    base = {
        "lecture_id": str(row["lecture_id"]),
        "query_id": str(row["query_id"]),
        "query_text": str(row["query_text"]),
        "duration_seconds": float(row["duration_seconds"]),
        "verification": dict(verification),
    }
    has_full_video = "video" in row or "full_video_context" in row
    has_candidate_video = (
        "candidate_video" in row or "materialized_video_context" in row
    )
    if has_full_video == has_candidate_video:
        raise ExperimentDataError(
            "Localization input must contain exactly one supported video extent"
        )
    if has_full_video:
        video = row.get("video")
        context = row.get("full_video_context")
        if not isinstance(video, Mapping) or not isinstance(context, Mapping):
            raise ExperimentDataError("Full-video localization input is invalid")
        return {
            **base,
            "video": dict(video),
            "full_video_context": dict(context),
        }
    video = row.get("candidate_video")
    context = row.get("materialized_video_context")
    window = row.get("candidate_windows")
    if (
        not isinstance(video, Mapping)
        or not isinstance(context, Mapping)
        or not isinstance(window, Mapping)
    ):
        raise ExperimentDataError("Candidate localization input is invalid")
    return {
        **base,
        "candidate_video": dict(video),
        "materialized_video_context": dict(context),
        "candidate_windows": dict(window),
    }


def _extract_prediction(row: Mapping[str, Any], method: str) -> dict[str, Any]:
    event = row.get("events")
    if not isinstance(event, Mapping):
        raise ExperimentDataError("Unnested prediction row has no event object")
    interval_valid = event.get("interval_valid")
    if not isinstance(interval_valid, bool):
        raise ExperimentDataError("Normalized event interval_valid must be boolean")
    prediction = {
        "lecture_id": str(row["lecture_id"]),
        "query_id": str(row["query_id"]),
        "method": method,
        "start_seconds": event.get("start_seconds"),
        "end_seconds": event.get("end_seconds"),
        "interval_valid": interval_valid,
        "interval_error": event.get("interval_error"),
        "confidence": float(event.get("confidence", 0.0)),
        "evidence": str(event.get("evidence", "")),
        "timestamp_source": event.get("timestamp_source"),
    }
    for field in (
        "window_id",
        "start_segment_id",
        "end_segment_id",
        "start_offset_seconds",
        "end_offset_seconds",
        "condition_evidence",
    ):
        if field in event:
            prediction[field] = event[field]
    return prediction


def _extract_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lecture_id": str(row["lecture_id"]),
        "query_id": str(row["query_id"]),
        "query_text": str(row["query_text"]),
        "duration_seconds": float(row["duration_seconds"]),
        "video": dict(row["video"]),
        "candidate_padding_seconds": float(row["candidate_padding_seconds"]),
        "candidate_ranges": row.get("candidate_ranges", []),
        "invalid_candidate_ranges": row.get("invalid_candidate_ranges", []),
        "candidate_windows": row.get("candidate_windows", []),
    }


def _extract_v3_proposal_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lecture_id": str(row["lecture_id"]),
        "query_id": str(row["query_id"]),
        "query_text": str(row["query_text"]),
        "duration_seconds": float(row["duration_seconds"]),
        "video": dict(row["video"]),
        "candidate_padding_seconds": float(row["candidate_padding_seconds"]),
        "episode_proposals": row.get("episode_proposals", []),
        "invalid_episode_proposals": row.get(
            "invalid_episode_proposals", []
        ),
        "candidate_windows": row.get("candidate_windows", []),
        "transcript_segments": row["transcript_segments"],
        "timestamped_transcript": str(row["timestamped_transcript"]),
    }


def _extract_v4_condition_row(row: Mapping[str, Any]) -> dict[str, Any]:
    conditions = row.get("required_conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ExperimentDataError("V4 query row has no required conditions")
    return {
        "query_id": str(row["query_id"]),
        "query_text": str(row["query_text"]),
        "required_conditions": conditions,
    }


def _build_v4_grounding_rows(
    conditions: Sequence[Mapping[str, Any]],
    v3_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_query: dict[str, Mapping[str, Any]] = {}
    for row in conditions:
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in by_query:
            raise ExperimentDataError(f"Duplicate or empty v4 query_id: {query_id!r}")
        required = row.get("required_conditions")
        if not isinstance(required, list) or not required:
            raise ExperimentDataError(f"V4 query has no conditions: {query_id}")
        by_query[query_id] = row
    expected = {query.query_id for query in QUERIES}
    if set(by_query) != expected:
        raise ExperimentDataError("V4 compiled query set is incomplete")

    result: list[dict[str, Any]] = []
    for source in v3_rows:
        query_id = str(source.get("query_id", ""))
        condition_row = by_query.get(query_id)
        if condition_row is None:
            raise ExperimentDataError(f"V4 grounding row has unknown query: {query_id}")
        if str(source.get("query_text", "")) != str(condition_row["query_text"]):
            raise ExperimentDataError(f"V4 query text changed: {query_id}")
        result.append(
            {
                **dict(source),
                "required_conditions": condition_row["required_conditions"],
            }
        )
    return result


def _build_v3_refinement_rows(
    proposal_rows: Sequence[Mapping[str, Any]],
    materialized_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join clips to proposals and expose only aligned boundary segment IDs."""
    proposals_by_key: dict[tuple[str, str, int], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for row in proposal_rows:
        windows = row.get("candidate_windows")
        if not isinstance(windows, list):
            raise ExperimentDataError("V3 proposal candidate_windows must be a list")
        for window in windows:
            if not isinstance(window, Mapping):
                raise ExperimentDataError("V3 proposal window must be an object")
            key = _v3_window_key(row, window)
            if key in proposals_by_key:
                raise ExperimentDataError(f"Duplicate v3 proposal window: {key}")
            proposals_by_key[key] = (row, window)

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for materialized in materialized_rows:
        window = materialized.get("candidate_windows")
        if not isinstance(window, Mapping):
            raise ExperimentDataError("V3 materialized row has no candidate window")
        key = _v3_window_key(materialized, window)
        if key in seen or key not in proposals_by_key:
            raise ExperimentDataError(f"Unexpected v3 materialized window: {key}")
        seen.add(key)
        proposal_row, frozen_window = proposals_by_key[key]
        if dict(window) != dict(frozen_window):
            raise ExperimentDataError(f"V3 materialized window changed: {key}")
        ranges = frozen_window.get("segment_ranges")
        if not isinstance(ranges, list) or len(ranges) != 1:
            raise ExperimentDataError(
                f"V3 proposal window must contain exactly one proposal: {key}"
            )
        proposed_episode = ranges[0]
        if not isinstance(proposed_episode, Mapping):
            raise ExperimentDataError(f"V3 proposed episode is invalid: {key}")

        transcript_segments = proposal_row.get("transcript_segments")
        if not isinstance(transcript_segments, list):
            raise ExperimentDataError(f"V3 transcript segments are invalid: {key}")
        window_start = float(frozen_window["start_seconds"])
        window_end = float(frozen_window["end_seconds"])
        context_segments = [
            segment
            for segment in transcript_segments
            if isinstance(segment, Mapping)
            and float(segment.get("start_seconds", -1)) >= window_start - 1e-6
            and float(segment.get("end_seconds", math.inf)) <= window_end + 1e-6
        ]
        context_ids = [int(segment["segment_id"]) for segment in context_segments]
        proposed_ids = {
            proposed_episode.get("start_segment_id"),
            proposed_episode.get("end_segment_id"),
        }
        if not context_ids or not proposed_ids.issubset(context_ids):
            raise ExperimentDataError(
                f"V3 proposal boundaries fall outside transcript context: {key}"
            )
        candidate_video = materialized.get("candidate_video")
        materialized_context = materialized.get("materialized_video_context")
        if not isinstance(candidate_video, Mapping) or not isinstance(
            materialized_context, Mapping
        ):
            raise ExperimentDataError(f"V3 materialized media is invalid: {key}")
        result.append(
            {
                "lecture_id": key[0],
                "query_id": key[1],
                "query_text": str(proposal_row["query_text"]),
                "duration_seconds": float(proposal_row["duration_seconds"]),
                "candidate_video": dict(candidate_video),
                "candidate_windows": dict(frozen_window),
                "materialized_video_context": dict(materialized_context),
                "v3_video_context": {
                    "clip_timeline_origin_seconds": 0.0,
                    "source_window_start_seconds": window_start,
                    "source_window_end_seconds": window_end,
                    "boundary_coordinate_system": (
                        "existing Whisper segment IDs in aligned excerpt"
                    ),
                },
                "proposed_episode": dict(proposed_episode),
                "transcript_context_segment_ids": context_ids,
                "timestamped_transcript_context": _format_timestamped_transcript(
                    context_segments
                ),
                "transcript_segments": transcript_segments,
            }
        )
    if seen != set(proposals_by_key):
        raise ExperimentDataError(
            "V3 materialized rows do not cover every transcript proposal window"
        )
    result.sort(
        key=lambda row: (
            str(row["lecture_id"]),
            str(row["query_id"]),
            int(row["candidate_windows"]["window_id"]),
        )
    )
    return result


def _v3_window_key(
    row: Mapping[str, Any], window: Mapping[str, Any]
) -> tuple[str, str, int]:
    window_id = window.get("window_id")
    if (
        not isinstance(window_id, int)
        or isinstance(window_id, bool)
        or window_id < 0
    ):
        raise ExperimentDataError("V3 window_id must be a non-negative integer")
    return str(row["lecture_id"]), str(row["query_id"]), window_id


def _validated_video_entries(root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = root / MANIFEST_FILENAME
    manifest = load_json_object(manifest_path)
    if manifest.get("source_catalog") != source_catalog_payload():
        raise ExperimentDataError("Lecture source manifest does not match the frozen catalog")
    entries = manifest.get("lectures")
    if not isinstance(entries, list):
        raise ExperimentDataError("Lecture manifest lectures must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    sources = {lecture.lecture_id: lecture for lecture in LECTURES}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ExperimentDataError("Lecture manifest entry must be an object")
        lecture_id = str(entry.get("lecture_id", ""))
        source = sources.get(lecture_id)
        if source is None or lecture_id in by_id:
            raise ExperimentDataError(f"Unexpected or duplicate lecture manifest entry: {lecture_id}")
        if entry.get("status") != "complete" or entry.get("download_url") != source.download_url:
            raise ExperimentDataError(f"Lecture download is not frozen and complete: {lecture_id}")
        relative = entry.get("path")
        content_hash = entry.get("sha256")
        media = entry.get("media")
        if not isinstance(relative, str) or not isinstance(content_hash, str) or not isinstance(media, Mapping):
            raise ExperimentDataError(f"Lecture manifest metadata is incomplete: {lecture_id}")
        path = root / relative
        if not path.is_file() or sha256_file(path) != content_hash:
            raise ExperimentDataError(f"Lecture bytes are missing or changed: {path}")
        duration = _positive_finite(media.get("duration_seconds"), "duration_seconds")
        by_id[lecture_id] = {**dict(entry), "duration_seconds": duration}
    if set(by_id) != set(sources):
        raise ExperimentDataError(f"Lecture manifest is incomplete; found: {sorted(by_id)}")
    return by_id


def _validated_transcripts(
    root: Path, videos: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for lecture in LECTURES:
        path = root / TRANSCRIPT_DIRECTORY / f"{lecture.lecture_id}.whisper.json"
        payload = load_json_object(path)
        video = videos[lecture.lecture_id]
        expected = (
            lecture.lecture_id,
            video["path"],
            video["sha256"],
            WHISPER_MODEL,
        )
        actual = (
            payload.get("source_id"),
            payload.get("source_path"),
            payload.get("source_sha256"),
            payload.get("model"),
        )
        if actual != expected:
            raise ExperimentDataError(f"Transcript provenance mismatch: {lecture.lecture_id}")
        if payload.get("normalization_contract_version") != NORMALIZATION_CONTRACT_VERSION:
            raise ExperimentDataError(
                f"Transcript uses a stale normalization contract: {lecture.lecture_id}"
            )
        normalized_duration = payload.get("source_duration_seconds")
        if (
            not isinstance(normalized_duration, (int, float))
            or isinstance(normalized_duration, bool)
            or not math.isclose(
                float(normalized_duration),
                float(video["duration_seconds"]),
                abs_tol=1e-9,
            )
        ):
            raise ExperimentDataError(
                f"Transcript source duration changed: {lecture.lecture_id}"
            )
        if str(payload.get("language", "")).casefold() not in {"en", "eng", "english"}:
            raise ExperimentDataError(f"Transcript is not English: {lecture.lecture_id}")
        segments = payload.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ExperimentDataError(f"Transcript has no segments: {lecture.lecture_id}")
        _validate_segments(segments, float(video["duration_seconds"]), lecture.lecture_id)
        raw_relative = payload.get("raw_checkpoint_path")
        raw_hash = payload.get("raw_checkpoint_sha256")
        if not isinstance(raw_relative, str) or not isinstance(raw_hash, str):
            raise ExperimentDataError(
                f"Transcript has no raw-checkpoint provenance: {lecture.lecture_id}"
            )
        raw_path = root / raw_relative
        if not raw_path.is_file() or sha256_file(raw_path) != raw_hash:
            raise ExperimentDataError(
                f"Raw transcript checkpoint is missing or changed: {lecture.lecture_id}"
            )
        result[lecture.lecture_id] = payload
    return result


def _validate_segments(segments: Sequence[Any], duration: float, lecture_id: str) -> None:
    previous_start = -math.inf
    previous_end = -math.inf
    seen: set[int] = set()
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ExperimentDataError(f"Transcript segment is invalid: {lecture_id} {index}")
        segment_id = segment.get("segment_id")
        if (
            not isinstance(segment_id, int)
            or isinstance(segment_id, bool)
            or segment_id != index
            or segment_id in seen
        ):
            raise ExperimentDataError(f"Transcript segment id is invalid: {lecture_id} {index}")
        start = _nonnegative_finite(segment.get("start_seconds"), "segment start_seconds")
        end = _positive_finite(segment.get("end_seconds"), "segment end_seconds")
        if (
            start < previous_start
            or end < previous_end
            or end <= start
            or end > duration + 1e-6
        ):
            raise ExperimentDataError(f"Transcript segment ordering is invalid: {lecture_id} {segment_id}")
        if not isinstance(segment.get("text"), str):
            raise ExperimentDataError(f"Transcript segment text is invalid: {lecture_id} {segment_id}")
        seen.add(segment_id)
        previous_start, previous_end = start, end


def _validate_ground_truth_events(
    events: Any, duration: float, key: tuple[str, str]
) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        raise ExperimentDataError(f"Ground-truth events must be a list for {key}")
    normalized = []
    previous_end = 0.0
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ExperimentDataError(f"Ground-truth event must be an object for {key}")
        start = _nonnegative_finite(event.get("start_seconds"), "ground-truth start_seconds")
        end = _positive_finite(event.get("end_seconds"), "ground-truth end_seconds")
        if end <= start or end > duration or start < previous_end:
            raise ExperimentDataError(f"Ground-truth intervals overlap, reverse, or exceed duration for {key}")
        normalized.append(
            {
                "start_seconds": start,
                "end_seconds": end,
                "annotation_notes": str(event.get("annotation_notes", "")),
            }
        )
        previous_end = end
    return normalized


def _validate_frozen_ground_truth(
    truth: Mapping[str, Any], videos: Mapping[str, Mapping[str, Any]]
) -> None:
    if truth.get("experiment") != EXPERIMENT_NAME:
        raise ExperimentDataError("Ground truth belongs to a different experiment")
    if truth.get("source_manifest_sha256") is None:
        raise ExperimentDataError("Ground truth has no frozen source-manifest hash")
    pairs = truth.get("pairs")
    if not isinstance(pairs, list):
        raise ExperimentDataError("Ground truth pairs must be a list")
    expected = {
        (lecture.lecture_id, query.query_id)
        for lecture in LECTURES
        for query in QUERIES
    }
    seen = set()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ExperimentDataError("Ground truth pair must be an object")
        key = (str(pair.get("lecture_id", "")), str(pair.get("query_id", "")))
        if key not in expected or key in seen:
            raise ExperimentDataError(f"Unexpected or duplicate frozen ground-truth pair: {key}")
        duration = float(videos[key[0]]["duration_seconds"])
        if not math.isclose(float(pair.get("duration_seconds", -1)), duration, abs_tol=1e-6):
            raise ExperimentDataError(f"Frozen ground-truth duration changed for {key}")
        _validate_ground_truth_events(pair.get("events"), duration, key)
        seen.add(key)
    if seen != expected:
        raise ExperimentDataError(f"Frozen ground truth is incomplete; missing: {sorted(expected - seen)}")


def _format_timestamped_transcript(segments: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"[segment_id={int(segment['segment_id'])} "
        f"start={float(segment['start_seconds']):.3f} "
        f"end={float(segment['end_seconds']):.3f}] "
        f"{str(segment['text']).strip()}"
        for segment in segments
    )


def _assert_label_isolation(rows: Sequence[Mapping[str, Any]]) -> None:
    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).casefold()
                if normalized in FORBIDDEN_INPUT_KEYS:
                    raise ExperimentDataError(f"Evaluator-only key leaked into prediction input: {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
    for index, row in enumerate(rows):
        visit(row, f"rows[{index}]")


def _write_review_plans(directory: Path, input_path: Path) -> None:
    plans = {
        "naive.py": build_naive_plan(str(input_path.resolve())),
        "transcript_only.py": build_transcript_only_plan(str(input_path.resolve())),
        "transcript_candidates.py": build_candidate_plan(str(input_path.resolve())),
        "optimized_verification.py": build_verification_plan(
            str((directory / "runs/transcript_candidates/candidates.jsonl").resolve())
        ),
        "optimized_materialized_verification.py": build_materialized_verification_plan(
            str(
                (
                    directory
                    / "runs"
                    / MATERIALIZED_CLIP_STAGE_NAME
                    / MATERIALIZED_INPUT_FILENAME
                ).resolve()
            )
        ),
    }
    for filename, plan in plans.items():
        atomic_write_text(directory / "plans" / filename, render_query(plan))


def _write_v2_plans(directory: Path) -> dict[str, str]:
    plans = _v2_plans(directory)
    plan_directory = directory / "plans/v2"
    hashes: dict[str, str] = {}
    for filename, plan in plans.items():
        rendered = render_query(plan)
        path = plan_directory / filename
        if path.is_file() and path.read_text(encoding="utf-8") != rendered:
            raise ExperimentDataError(
                f"Refusing to overwrite a different v2 plan: {path}"
            )
        if not path.is_file():
            atomic_write_text(path, rendered)
        hashes[filename] = sha256_file(path)
    return hashes


def _write_v3_plans(directory: Path) -> dict[str, str]:
    plans = _v3_plans(directory)
    plan_directory = directory / "plans/v3"
    hashes: dict[str, str] = {}
    for filename, plan in plans.items():
        rendered = render_query(plan)
        path = plan_directory / filename
        if path.is_file() and path.read_text(encoding="utf-8") != rendered:
            raise ExperimentDataError(
                f"Refusing to overwrite a different v3 plan: {path}"
            )
        if not path.is_file():
            atomic_write_text(path, rendered)
        hashes[filename] = sha256_file(path)
    return hashes


def _write_v4_plans(directory: Path) -> dict[str, str]:
    plans = _v4_plans(directory)
    plan_directory = directory / "plans/v4"
    hashes: dict[str, str] = {}
    for filename, plan in plans.items():
        rendered = render_query(plan)
        path = plan_directory / filename
        if path.is_file() and path.read_text(encoding="utf-8") != rendered:
            raise ExperimentDataError(
                f"Refusing to overwrite a different v4 plan: {path}"
            )
        if not path.is_file():
            atomic_write_text(path, rendered)
        hashes[filename] = sha256_file(path)
    return hashes


def _v4_plans(directory: Path) -> dict[str, Any]:
    condition_directory = _v4_stage_directory(
        directory, V4_QUERY_CONDITION_STAGE_NAME
    )
    return {
        "query_conditions.py": build_v4_query_condition_plan(
            str((directory / V4_QUERY_INPUT_FILENAME).resolve())
        ),
        "predicate_grounding.py": build_v4_predicate_grounding_plan(
            str((condition_directory / "grounding_input.jsonl").resolve())
        ),
    }


def _write_v5_plans(directory: Path) -> dict[str, str]:
    plans = _v5_plans(directory)
    plan_directory = directory / "plans/v5"
    hashes: dict[str, str] = {}
    for filename, plan in plans.items():
        rendered = render_query(plan)
        path = plan_directory / filename
        if path.is_file() and path.read_text(encoding="utf-8") != rendered:
            raise ExperimentDataError(
                f"Refusing to overwrite a different v5 plan: {path}"
            )
        if not path.is_file():
            atomic_write_text(path, rendered)
        hashes[filename] = sha256_file(path)
    return hashes


def _v5_plans(directory: Path) -> dict[str, Any]:
    condition_directory = _v5_stage_directory(
        directory, V5_QUERY_CONDITION_STAGE_NAME
    )
    return {
        "query_conditions.py": build_v5_query_condition_plan(
            str((directory / V5_QUERY_INPUT_FILENAME).resolve())
        ),
        "predicate_grounding.py": build_v5_predicate_grounding_plan(
            str((condition_directory / "grounding_input.jsonl").resolve())
        ),
    }


def _v3_plans(directory: Path) -> dict[str, Any]:
    refinement_input = (
        _v3_stage_directory(directory, V3_MATERIALIZED_STAGE_NAME)
        / V3_REFINEMENT_INPUT_FILENAME
    )
    return {
        "transcript_episode_proposals.py": build_v3_transcript_proposal_plan(
            str((directory / "input.jsonl").resolve())
        ),
        "transcript_grounded_refinement.py": build_v3_refinement_plan(
            str(refinement_input.resolve())
        ),
    }


def _v2_plans(directory: Path) -> dict[str, Any]:
    materialized_input = (
        directory
        / "runs"
        / MATERIALIZED_CLIP_STAGE_NAME
        / MATERIALIZED_INPUT_FILENAME
    )
    naive_decisions = (
        _v2_stage_directory(directory, V2_NAIVE_BINARY_STAGE_NAME)
        / "localization_input.jsonl"
    )
    optimized_decisions = (
        _v2_stage_directory(directory, V2_OPTIMIZED_BINARY_STAGE_NAME)
        / "localization_input.jsonl"
    )
    return {
        "naive_binary.py": build_naive_v2_binary_plan(
            str((directory / "input.jsonl").resolve())
        ),
        "optimized_binary.py": build_materialized_v2_binary_plan(
            str(materialized_input.resolve())
        ),
        "naive_localization.py": build_naive_v2_localization_plan(
            str(naive_decisions.resolve())
        ),
        "optimized_localization.py": build_materialized_v2_localization_plan(
            str(optimized_decisions.resolve())
        ),
    }


def _v2_stage_directory(directory: Path, stage_name: str) -> Path:
    if stage_name not in {
        V2_NAIVE_BINARY_STAGE_NAME,
        V2_OPTIMIZED_BINARY_STAGE_NAME,
        V2_NAIVE_LOCALIZATION_STAGE_NAME,
        V2_OPTIMIZED_LOCALIZATION_STAGE_NAME,
    }:
        raise ExperimentDataError(f"Unknown v2 stage: {stage_name}")
    return directory / V2_RUN_DIRECTORY / stage_name


def _v3_stage_directory(directory: Path, stage_name: str) -> Path:
    if stage_name not in {
        V3_PROPOSAL_STAGE_NAME,
        V3_MATERIALIZED_STAGE_NAME,
        V3_REFINEMENT_STAGE_NAME,
    }:
        raise ExperimentDataError(f"Unknown v3 stage: {stage_name}")
    return directory / V3_RUN_DIRECTORY / stage_name


def _v4_stage_directory(directory: Path, stage_name: str) -> Path:
    if stage_name not in {
        V4_QUERY_CONDITION_STAGE_NAME,
        V4_GROUNDING_STAGE_NAME,
    }:
        raise ExperimentDataError(f"Unknown v4 stage: {stage_name}")
    return directory / V4_RUN_DIRECTORY / stage_name


def _v5_stage_directory(directory: Path, stage_name: str) -> Path:
    if stage_name not in {
        V5_QUERY_CONDITION_STAGE_NAME,
        V5_GROUNDING_STAGE_NAME,
    }:
        raise ExperimentDataError(f"Unknown v5 stage: {stage_name}")
    return directory / V5_RUN_DIRECTORY / stage_name


def _validate_v3_materialized_stage(
    directory: Path,
    *,
    media_probe: MediaProbe,
    require_completion: bool = True,
) -> dict[str, Any]:
    proposal_directory = _v3_stage_directory(directory, V3_PROPOSAL_STAGE_NAME)
    _validate_stage_completion(proposal_directory, V3_PROPOSAL_STAGE_ARTIFACTS)
    materialized_directory = _v3_stage_directory(
        directory, V3_MATERIALIZED_STAGE_NAME
    )
    if require_completion:
        _validate_stage_completion(
            materialized_directory, V3_MATERIALIZED_STAGE_ARTIFACTS
        )
    validated = validate_materialized_clip_stage(
        materialized_directory, media_probe=media_probe
    )
    proposal_rows = load_jsonl_objects(proposal_directory / "proposals.jsonl")
    expected = _build_v3_refinement_rows(proposal_rows, validated["rows"])
    actual = load_jsonl_objects(
        materialized_directory / V3_REFINEMENT_INPUT_FILENAME
    )
    if actual != expected:
        raise ExperimentDataError("V3 refinement input changed after materialization")
    _assert_label_isolation(actual)
    return {
        "manifest": validated["manifest"],
        "rows": validated["rows"],
        "refinement_rows": actual,
    }


def _validate_v4_grounding_input(directory: Path) -> list[dict[str, Any]]:
    condition_directory = _v4_stage_directory(
        directory, V4_QUERY_CONDITION_STAGE_NAME
    )
    _validate_stage_completion(
        condition_directory, V4_QUERY_CONDITION_STAGE_ARTIFACTS
    )
    conditions = load_jsonl_objects(condition_directory / "conditions.jsonl")
    v3_rows = load_jsonl_objects(
        _v3_stage_directory(directory, V3_MATERIALIZED_STAGE_NAME)
        / V3_REFINEMENT_INPUT_FILENAME
    )
    expected = _build_v4_grounding_rows(conditions, v3_rows)
    actual = load_jsonl_objects(condition_directory / "grounding_input.jsonl")
    if actual != expected:
        raise ExperimentDataError("V4 grounding input changed after compilation")
    _assert_label_isolation(actual)
    return actual


def _validate_v5_grounding_input(directory: Path) -> list[dict[str, Any]]:
    condition_directory = _v5_stage_directory(
        directory, V5_QUERY_CONDITION_STAGE_NAME
    )
    _validate_stage_completion(
        condition_directory, V5_QUERY_CONDITION_STAGE_ARTIFACTS
    )
    conditions = load_jsonl_objects(condition_directory / "conditions.jsonl")
    v3_rows = load_jsonl_objects(
        _v3_stage_directory(directory, V3_MATERIALIZED_STAGE_NAME)
        / V3_REFINEMENT_INPUT_FILENAME
    )
    expected = _build_v4_grounding_rows(conditions, v3_rows)
    actual = load_jsonl_objects(condition_directory / "grounding_input.jsonl")
    if actual != expected:
        raise ExperimentDataError("V5 grounding input changed after compilation")
    _assert_label_isolation(actual)
    return actual


def _v2_prefix_artifact_hashes(directory: Path) -> dict[str, dict[str, str]]:
    stages = {
        "transcript_candidates": (
            directory / "runs/transcript_candidates",
            (
                "candidates.jsonl",
                "candidate_metrics.json",
                "stage.json",
                "completion.json",
            ),
        ),
        "materialized_clips": (
            directory / "runs" / MATERIALIZED_CLIP_STAGE_NAME,
            (*MATERIALIZED_CLIP_ARTIFACTS, "completion.json"),
        ),
    }
    result: dict[str, dict[str, str]] = {}
    for stage_name, (stage_directory, filenames) in stages.items():
        result[stage_name] = {}
        for filename in filenames:
            path = stage_directory / filename
            if not path.is_file():
                raise ExperimentDataError(f"Missing v2 prefix artifact: {path}")
            result[stage_name][filename] = sha256_file(path)
    return result


def _v4_prefix_artifact_hashes(directory: Path) -> dict[str, dict[str, str]]:
    stages = {
        "v3_transcript_episode_proposals": (
            _v3_stage_directory(directory, V3_PROPOSAL_STAGE_NAME),
            (*V3_PROPOSAL_STAGE_ARTIFACTS, "completion.json"),
        ),
        "v3_materialized_proposal_clips": (
            _v3_stage_directory(directory, V3_MATERIALIZED_STAGE_NAME),
            (*V3_MATERIALIZED_STAGE_ARTIFACTS, "completion.json"),
        ),
    }
    result: dict[str, dict[str, str]] = {}
    for stage_name, (stage_directory, filenames) in stages.items():
        result[stage_name] = {}
        for filename in filenames:
            path = stage_directory / filename
            if not path.is_file():
                raise ExperimentDataError(f"Missing v4 prefix artifact: {path}")
            result[stage_name][filename] = sha256_file(path)
    return result


def _stage_summary(stage_directory: Path, elapsed: float, executor: Any) -> dict[str, Any]:
    usage = aggregate_api_usage(
        stage_directory / "api_calls.jsonl",
        input_usd_per_million_tokens=INPUT_USD_PER_MILLION_TOKENS,
        audio_input_usd_per_million_tokens=AUDIO_INPUT_USD_PER_MILLION_TOKENS,
        output_usd_per_million_tokens=OUTPUT_USD_PER_MILLION_TOKENS,
        pricing_source=PRICING_SOURCE,
    )
    return {
        "model": MODEL,
        "end_to_end_seconds": elapsed,
        "max_in_flight_provider_calls": 1,
        "cache_hits": int(getattr(executor, "cache_hits", 0)),
        "cache_misses": int(getattr(executor, "cache_misses", 0)),
        "api_usage": usage,
    }


def _comparison_method(stage_directory: Path) -> dict[str, Any]:
    evaluation = load_json_object(stage_directory / "evaluation.json")
    stage = load_json_object(stage_directory / "stage.json")
    return {
        "primary_accuracy": evaluation["primary_metrics"],
        "api_usage": stage["api_usage"],
    }


def _combined_latency(*stages: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
    observed = 0.0
    api_seconds = 0.0
    cache_hits = 0
    reused_materialized_clips = 0
    failed = 0
    invalid = []
    for name, stage in stages:
        usage = stage.get("api_usage")
        is_local_materialization = (
            stage.get("stage_type") == "local_candidate_clip_materialization"
        )
        if not isinstance(usage, Mapping) and not is_local_materialization:
            raise ExperimentDataError(f"Stage has invalid API usage: {name}")
        stage_seconds = _nonnegative_finite(stage.get("end_to_end_seconds"), "end_to_end_seconds")
        stage_cache_hits = int(stage.get("cache_hits", 0))
        stage_reused_clips = int(stage.get("reused_clip_count", 0))
        stage_failed = int(usage.get("failed_api_call_count", 0)) if isinstance(usage, Mapping) else 0
        observed += stage_seconds
        api_seconds += float(usage.get("api_elapsed_seconds", 0.0)) if isinstance(usage, Mapping) else 0.0
        cache_hits += stage_cache_hits
        reused_materialized_clips += stage_reused_clips
        failed += stage_failed
        reasons = []
        if stage_cache_hits:
            reasons.append("cached_responses_used")
        if stage_reused_clips:
            reasons.append("reused_materialized_clips")
        if stage_failed:
            reasons.append("failed_api_attempts_recorded")
        if reasons:
            invalid.append({"stage": name, "reasons": reasons})
    return {
        "valid": not invalid,
        "end_to_end_seconds": observed if not invalid else None,
        "observed_stage_seconds_sum": observed,
        "cumulative_api_elapsed_seconds": api_seconds,
        "cache_hits": cache_hits,
        "reused_materialized_clip_count": reused_materialized_clips,
        "failed_api_call_count": failed,
        "invalid_stages": invalid,
        "reason": (
            None
            if not invalid
            else "Cached responses, reused clips, or failed attempts invalidate cold-start latency."
        ),
    }


def _sum_usage(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    summed: dict[str, Any] = {}
    fields = (
        "api_call_count",
        "successful_api_call_count",
        "failed_api_call_count",
        "api_elapsed_seconds",
        "prompt_token_count",
        "candidate_token_count",
        "thought_token_count",
        "total_token_count",
        "estimated_input_cost_usd",
        "estimated_output_cost_usd",
        "estimated_cost_usd",
    )
    for field in fields:
        summed[field] = first.get(field, 0) + second.get(field, 0)
    modalities = set(first.get("input_tokens_by_modality", {})) | set(second.get("input_tokens_by_modality", {}))
    summed["input_tokens_by_modality"] = {
        key: first.get("input_tokens_by_modality", {}).get(key, 0)
        + second.get("input_tokens_by_modality", {}).get(key, 0)
        for key in sorted(modalities)
    }
    summed["modality_breakdown_complete"] = bool(
        first.get("modality_breakdown_complete", False)
        and second.get("modality_breakdown_complete", False)
    )
    summed["cost_assumptions"] = first.get("cost_assumptions")
    return summed


def _transcription_cost(root: Path) -> dict[str, Any]:
    index = load_json_object(root / TRANSCRIPT_DIRECTORY / "index.json")
    if index.get("model") != WHISPER_MODEL:
        raise ExperimentDataError("Transcript index uses a different Whisper model")
    lectures = index.get("lectures")
    if not isinstance(lectures, list):
        raise ExperimentDataError("Transcript index lectures must be a list")
    elapsed = []
    seen: set[str] = set()
    expected = {lecture.lecture_id for lecture in LECTURES}
    for lecture in lectures:
        if not isinstance(lecture, Mapping):
            raise ExperimentDataError("Transcript index entry must be an object")
        lecture_id = lecture.get("lecture_id")
        if not isinstance(lecture_id, str) or lecture_id not in expected or lecture_id in seen:
            raise ExperimentDataError(f"Unexpected or duplicate transcript index entry: {lecture_id!r}")
        seen.add(lecture_id)
        elapsed.append(_nonnegative_finite(lecture.get("elapsed_seconds"), "transcription elapsed_seconds"))
    if seen != expected:
        raise ExperimentDataError(f"Transcript index is incomplete; missing: {sorted(expected - seen)}")
    total = sum(elapsed)
    return {
        "model": WHISPER_MODEL,
        "lecture_count": len(elapsed),
        "total_elapsed_seconds": total,
        "amortized_seconds_per_query_pair": total / (len(LECTURES) * len(QUERIES)),
        "note": "One-time local materialization; compute cost is not converted to USD.",
    }


def _accuracy_delta(baseline: Mapping[str, Any], method: Mapping[str, Any]) -> dict[str, float]:
    return {
        field: float(method[field]) - float(baseline[field])
        for field in ("precision", "recall", "f1", "exact_pair_accuracy")
    }


def _binary_accuracy_delta(
    baseline: Mapping[str, Any], method: Mapping[str, Any]
) -> dict[str, float]:
    return {
        field: float(method[field]) - float(baseline[field])
        for field in ("precision", "recall", "f1", "accuracy")
    }


def _efficiency_vs_naive(
    naive: Mapping[str, Any], method: Mapping[str, Any]
) -> dict[str, Any]:
    naive_usage = naive["api_usage"]
    method_usage = method["api_usage"]
    naive_latency = naive["query_time_latency"]
    method_latency = method["query_time_latency"]
    return {
        "total_token_reduction_fraction": _reduction(
            float(naive_usage.get("total_token_count", 0)),
            float(method_usage.get("total_token_count", 0)),
        ),
        "estimated_cost_reduction_fraction": _reduction(
            float(naive_usage.get("estimated_cost_usd", 0)),
            float(method_usage.get("estimated_cost_usd", 0)),
        ),
        "query_time_latency_reduction_fraction": (
            _reduction(
                float(naive_latency["end_to_end_seconds"]),
                float(method_latency["end_to_end_seconds"]),
            )
            if naive_latency["valid"] and method_latency["valid"]
            else None
        ),
    }


def _reduction(baseline: float, method: float) -> float | None:
    return (baseline - method) / baseline if baseline else None


def _require_one_row_per_pair(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [(str(row.get("lecture_id")), str(row.get("query_id"))) for row in rows]
    expected = {
        (lecture.lecture_id, query.query_id)
        for lecture in LECTURES
        for query in QUERIES
    }
    if len(keys) != len(expected) or set(keys) != expected:
        raise ExperimentDataError("Candidate stage did not produce exactly one row per pair")


def _require_prepared(root: Path) -> Path:
    directory = experiment_directory(root)
    required = (
        directory / "config.json",
        directory / "input.jsonl",
        directory / GROUND_TRUTH_FILENAME,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ExperimentDataError(f"Experiment is not prepared; missing: {', '.join(missing)}")
    config = load_json_object(directory / "config.json")
    hashes = config.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ExperimentDataError("Prepared experiment has invalid hashes")
    checks = (
        (directory / "input.jsonl", hashes.get("input_sha256")),
        (directory / GROUND_TRUTH_FILENAME, hashes.get("ground_truth_sha256")),
        (root / MANIFEST_FILENAME, hashes.get("source_manifest_sha256")),
    )
    for path, expected in checks:
        if sha256_file(path) != expected:
            raise ExperimentDataError(f"Prepared artifact changed after freezing: {path}")
    if hashes.get("catalog_sha256") != sha256_json(catalog_payload()):
        raise ExperimentDataError("Frozen catalog no longer matches the implementation")
    current_contract = _current_plan_contract_hashes()
    for key, expected in current_contract.items():
        if hashes.get(key) != expected:
            raise ExperimentDataError(f"Frozen {key} no longer matches the implementation")
    # The request cache trusts the content hashes embedded in input rows, so
    # verify the underlying bytes before every model stage.
    _validated_video_entries(root)
    return directory


def _require_v2_prepared(root: Path) -> Path:
    directory = _require_prepared(root)
    config_path = directory / V2_CONFIG_FILENAME
    if not config_path.is_file():
        raise ExperimentDataError(f"V2 is not prepared; missing: {config_path}")
    config = load_json_object(config_path)
    if (
        config.get("schema_version") != 1
        or config.get("method_contract_version") != 2
        or config.get("experiment") != EXPERIMENT_NAME
        or config.get("model") != MODEL
    ):
        raise ExperimentDataError("Prepared v2 configuration is incompatible")
    hashes = config.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ExperimentDataError("Prepared v2 configuration has invalid hashes")
    checks = (
        (directory / "config.json", hashes.get("base_config_sha256")),
        (directory / "input.jsonl", hashes.get("input_sha256")),
        (
            directory / GROUND_TRUTH_FILENAME,
            hashes.get("ground_truth_sha256"),
        ),
    )
    for path, expected in checks:
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ExperimentDataError(f"Prepared v2 artifact changed: {path}")
    if hashes.get("prefix_artifacts_sha256") != _v2_prefix_artifact_hashes(
        directory
    ):
        raise ExperimentDataError("Prepared v2 prefix artifacts changed")
    for key, expected in _current_v2_contract_hashes().items():
        if hashes.get(key) != expected:
            raise ExperimentDataError(
                f"Prepared v2 {key} no longer matches the implementation"
            )
    plan_hashes = hashes.get("plans_sha256")
    if not isinstance(plan_hashes, Mapping) or not plan_hashes:
        raise ExperimentDataError("Prepared v2 plan hashes are invalid")
    current_plans = _v2_plans(directory)
    if set(plan_hashes) != set(current_plans):
        raise ExperimentDataError("Prepared v2 plan set changed")
    for filename, plan in current_plans.items():
        expected = plan_hashes[filename]
        path = directory / "plans/v2" / filename
        if not isinstance(expected, str) or not path.is_file():
            raise ExperimentDataError(f"Prepared v2 plan is missing: {path}")
        if sha256_file(path) != expected:
            raise ExperimentDataError(f"Prepared v2 plan changed: {path}")
        if sha256_text(render_query(plan)) != expected:
            raise ExperimentDataError(
                f"Prepared v2 plan no longer matches the implementation: {path}"
            )
    return directory


def _require_v3_prepared(root: Path) -> Path:
    directory = _require_prepared(root)
    config_path = directory / V3_CONFIG_FILENAME
    if not config_path.is_file():
        raise ExperimentDataError(f"V3 is not prepared; missing: {config_path}")
    config = load_json_object(config_path)
    if (
        config.get("schema_version") != 1
        or config.get("method_contract_version") != 3
        or config.get("experiment") != EXPERIMENT_NAME
        or config.get("model") != MODEL
    ):
        raise ExperimentDataError("Prepared v3 configuration is incompatible")
    hashes = config.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ExperimentDataError("Prepared v3 configuration has invalid hashes")
    for path, expected in (
        (directory / "config.json", hashes.get("base_config_sha256")),
        (directory / "input.jsonl", hashes.get("input_sha256")),
        (
            directory / GROUND_TRUTH_FILENAME,
            hashes.get("ground_truth_sha256"),
        ),
    ):
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ExperimentDataError(f"Prepared v3 artifact changed: {path}")

    naive_directory = directory / "runs/naive"
    expected_naive = hashes.get("naive_baseline_artifacts_sha256")
    naive_names = (
        "predictions.json",
        "evaluation.json",
        "stage.json",
        "completion.json",
    )
    if not isinstance(expected_naive, Mapping) or set(expected_naive) != set(
        naive_names
    ):
        raise ExperimentDataError("Prepared v3 naive baseline hashes are invalid")
    for name in naive_names:
        path = naive_directory / name
        if not path.is_file() or sha256_file(path) != expected_naive[name]:
            raise ExperimentDataError(f"Prepared v3 naive baseline changed: {path}")

    for key, expected in _current_v3_contract_hashes().items():
        if hashes.get(key) != expected:
            raise ExperimentDataError(
                f"Prepared v3 {key} no longer matches the implementation"
            )
    plan_hashes = hashes.get("plans_sha256")
    current_plans = _v3_plans(directory)
    if not isinstance(plan_hashes, Mapping) or set(plan_hashes) != set(
        current_plans
    ):
        raise ExperimentDataError("Prepared v3 plan hashes are invalid")
    for filename, plan in current_plans.items():
        expected = plan_hashes[filename]
        path = directory / "plans/v3" / filename
        if not isinstance(expected, str) or not path.is_file():
            raise ExperimentDataError(f"Prepared v3 plan is missing: {path}")
        if sha256_file(path) != expected or sha256_text(render_query(plan)) != expected:
            raise ExperimentDataError(f"Prepared v3 plan changed: {path}")
    return directory


def _require_v4_prepared(root: Path) -> Path:
    directory = _require_v3_prepared(root)
    config_path = directory / V4_CONFIG_FILENAME
    query_input_path = directory / V4_QUERY_INPUT_FILENAME
    if not config_path.is_file() or not query_input_path.is_file():
        raise ExperimentDataError(
            f"V4 is not prepared; missing: {config_path} or {query_input_path}"
        )
    config = load_json_object(config_path)
    if (
        config.get("schema_version") != 1
        or config.get("method_contract_version") != 4
        or config.get("experiment") != EXPERIMENT_NAME
        or config.get("model") != MODEL
    ):
        raise ExperimentDataError("Prepared v4 configuration is incompatible")
    hashes = config.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ExperimentDataError("Prepared v4 configuration has invalid hashes")
    for path, expected in (
        (directory / "config.json", hashes.get("base_config_sha256")),
        (directory / V3_CONFIG_FILENAME, hashes.get("v3_config_sha256")),
        (query_input_path, hashes.get("query_input_sha256")),
        (
            directory / GROUND_TRUTH_FILENAME,
            hashes.get("ground_truth_sha256"),
        ),
    ):
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ExperimentDataError(f"Prepared v4 artifact changed: {path}")
    if hashes.get("v3_prefix_artifacts_sha256") != _v4_prefix_artifact_hashes(
        directory
    ):
        raise ExperimentDataError("Prepared v4 prefix artifacts changed")

    naive_directory = directory / "runs/naive"
    naive_names = (
        "predictions.json",
        "evaluation.json",
        "stage.json",
        "completion.json",
    )
    expected_naive = hashes.get("naive_baseline_artifacts_sha256")
    if not isinstance(expected_naive, Mapping) or set(expected_naive) != set(
        naive_names
    ):
        raise ExperimentDataError("Prepared v4 naive baseline hashes are invalid")
    for name in naive_names:
        path = naive_directory / name
        if not path.is_file() or sha256_file(path) != expected_naive[name]:
            raise ExperimentDataError(f"Prepared v4 naive baseline changed: {path}")

    for key, expected in _current_v4_contract_hashes().items():
        if hashes.get(key) != expected:
            raise ExperimentDataError(
                f"Prepared v4 {key} no longer matches the implementation"
            )
    plan_hashes = hashes.get("plans_sha256")
    current_plans = _v4_plans(directory)
    if not isinstance(plan_hashes, Mapping) or set(plan_hashes) != set(
        current_plans
    ):
        raise ExperimentDataError("Prepared v4 plan hashes are invalid")
    for filename, plan in current_plans.items():
        expected = plan_hashes[filename]
        path = directory / "plans/v4" / filename
        if not isinstance(expected, str) or not path.is_file():
            raise ExperimentDataError(f"Prepared v4 plan is missing: {path}")
        if sha256_file(path) != expected or sha256_text(render_query(plan)) != expected:
            raise ExperimentDataError(f"Prepared v4 plan changed: {path}")
    return directory


def _require_v5_prepared(root: Path) -> Path:
    directory = _require_v3_prepared(root)
    config_path = directory / V5_CONFIG_FILENAME
    query_input_path = directory / V5_QUERY_INPUT_FILENAME
    if not config_path.is_file() or not query_input_path.is_file():
        raise ExperimentDataError(
            f"V5 is not prepared; missing: {config_path} or {query_input_path}"
        )
    config = load_json_object(config_path)
    if (
        config.get("schema_version") != 1
        or config.get("method_contract_version") != 5
        or config.get("experiment") != EXPERIMENT_NAME
        or config.get("model") != MODEL
    ):
        raise ExperimentDataError("Prepared v5 configuration is incompatible")
    hashes = config.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ExperimentDataError("Prepared v5 configuration has invalid hashes")
    for path, expected in (
        (directory / "config.json", hashes.get("base_config_sha256")),
        (directory / V3_CONFIG_FILENAME, hashes.get("v3_config_sha256")),
        (query_input_path, hashes.get("query_input_sha256")),
        (
            directory / GROUND_TRUTH_FILENAME,
            hashes.get("ground_truth_sha256"),
        ),
    ):
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ExperimentDataError(f"Prepared v5 artifact changed: {path}")
    if hashes.get("v3_prefix_artifacts_sha256") != _v4_prefix_artifact_hashes(
        directory
    ):
        raise ExperimentDataError("Prepared v5 prefix artifacts changed")

    naive_directory = directory / "runs/naive"
    naive_names = (
        "predictions.json",
        "evaluation.json",
        "stage.json",
        "completion.json",
    )
    expected_naive = hashes.get("naive_baseline_artifacts_sha256")
    if not isinstance(expected_naive, Mapping) or set(expected_naive) != set(
        naive_names
    ):
        raise ExperimentDataError("Prepared v5 naive baseline hashes are invalid")
    for name in naive_names:
        path = naive_directory / name
        if not path.is_file() or sha256_file(path) != expected_naive[name]:
            raise ExperimentDataError(f"Prepared v5 naive baseline changed: {path}")
    for key, expected in _current_v5_contract_hashes().items():
        if hashes.get(key) != expected:
            raise ExperimentDataError(
                f"Prepared v5 {key} no longer matches the implementation"
            )
    plan_hashes = hashes.get("plans_sha256")
    current_plans = _v5_plans(directory)
    if not isinstance(plan_hashes, Mapping) or set(plan_hashes) != set(
        current_plans
    ):
        raise ExperimentDataError("Prepared v5 plan hashes are invalid")
    for filename, plan in current_plans.items():
        expected = plan_hashes[filename]
        path = directory / "plans/v5" / filename
        if not isinstance(expected, str) or not path.is_file():
            raise ExperimentDataError(f"Prepared v5 plan is missing: {path}")
        if sha256_file(path) != expected or sha256_text(render_query(plan)) != expected:
            raise ExperimentDataError(f"Prepared v5 plan changed: {path}")
    return directory


def _refuse_run_artifacts(directory: Path) -> None:
    runs = directory / "runs"
    if runs.exists() and any(path.is_file() for path in runs.rglob("*")):
        raise ExperimentDataError(f"Refusing to change labels/config after run artifacts exist: {runs}")


def _refuse_completed_stage(path: Path) -> None:
    if path.exists():
        raise ExperimentDataError(f"Stage completion marker already exists: {path}")


def _stage_status(directory: Path, artifact_names: Sequence[str]) -> str:
    completion = directory / "completion.json"
    if completion.is_file():
        try:
            _validate_stage_completion(directory, artifact_names)
        except ExperimentDataError:
            return "invalid_completed_artifacts"
        return "complete"
    if directory.exists() and any(path.is_file() for path in directory.rglob("*")):
        return "interrupted_or_in_progress"
    return "not_started"


def _transcript_status(root: Path, lecture_id: str) -> str:
    normalized = root / TRANSCRIPT_DIRECTORY / f"{lecture_id}.whisper.json"
    raw = root / TRANSCRIPT_DIRECTORY / f"{lecture_id}{RAW_CHECKPOINT_SUFFIX}"
    if normalized.is_file():
        return "complete"
    if raw.is_file():
        return "raw_checkpoint"
    return "missing"


def _current_plan_contract_hashes() -> dict[str, Any]:
    return {
        "prompt_sha256": {
            "verification_shared_by_naive_and_optimized": sha256_text(VERIFIER_PROMPT),
            "transcript_candidates": sha256_text(TRANSCRIPT_CANDIDATE_PROMPT),
            "transcript_only": sha256_text(TRANSCRIPT_ONLY_PROMPT),
        },
        "schema_sha256": {
            "video_events_shared_by_naive_and_optimized": sha256_json(VIDEO_EVENT_SCHEMA),
            "transcript_candidates": sha256_json(CANDIDATE_RANGE_SCHEMA),
            "transcript_only": sha256_json(TRANSCRIPT_EVENT_RANGE_SCHEMA),
        },
    }


def _current_v2_contract_hashes() -> dict[str, Any]:
    return {
        "v2_prompt_sha256": {
            "binary_shared_by_naive_and_optimized": sha256_text(
                BINARY_VERIFIER_PROMPT
            ),
            "episode_localization_shared_by_naive_and_optimized": sha256_text(
                EPISODE_LOCALIZATION_PROMPT
            ),
        },
        "v2_schema_sha256": {
            "binary_shared_by_naive_and_optimized": sha256_json(
                BINARY_VERIFICATION_SCHEMA
            ),
            "episode_localization_shared_by_naive_and_optimized": sha256_json(
                EPISODE_SCHEMA
            ),
        },
        "v2_contract_versions": {
            "binary_verification": BINARY_VERIFICATION_CONTRACT_VERSION,
            "episode_localization": EPISODE_LOCALIZATION_CONTRACT_VERSION,
        },
    }


def _current_v3_contract_hashes() -> dict[str, Any]:
    return {
        "v3_prompt_sha256": {
            "transcript_complete_episode_proposal": sha256_text(
                TRANSCRIPT_EPISODE_PROPOSAL_PROMPT
            ),
            "audiovisual_transcript_boundary_refinement": sha256_text(
                TRANSCRIPT_GROUNDED_REFINEMENT_PROMPT
            ),
        },
        "v3_schema_sha256": {
            "transcript_complete_episode_proposal": sha256_json(
                TRANSCRIPT_EPISODE_PROPOSAL_SCHEMA
            ),
            "audiovisual_transcript_boundary_refinement": sha256_json(
                TRANSCRIPT_GROUNDED_REFINEMENT_SCHEMA
            ),
        },
        "v3_contract_versions": {
            "transcript_episode_proposal": (
                TRANSCRIPT_EPISODE_PROPOSAL_CONTRACT_VERSION
            ),
            "transcript_grounded_refinement": (
                TRANSCRIPT_GROUNDED_REFINEMENT_CONTRACT_VERSION
            ),
        },
    }


def _current_v4_contract_hashes() -> dict[str, Any]:
    return {
        "v4_prompt_sha256": {
            "query_condition_compilation": sha256_text(QUERY_CONDITION_PROMPT),
            "mandatory_predicate_grounding": sha256_text(
                PREDICATE_GROUNDING_PROMPT
            ),
        },
        "v4_schema_sha256": {
            "query_condition_compilation": sha256_json(QUERY_CONDITION_SCHEMA),
            "mandatory_predicate_grounding": sha256_json(
                PREDICATE_GROUNDING_SCHEMA
            ),
        },
        "v4_contract_versions": {
            "query_condition_compilation": QUERY_CONDITION_CONTRACT_VERSION,
            "mandatory_predicate_grounding": PREDICATE_GROUNDING_CONTRACT_VERSION,
        },
    }


def _current_v5_contract_hashes() -> dict[str, Any]:
    return {
        "v5_prompt_sha256": {
            "gate_anchor_query_compilation": sha256_text(
                ROLE_AWARE_QUERY_CONDITION_PROMPT
            ),
            "role_aware_predicate_grounding": sha256_text(
                ROLE_AWARE_PREDICATE_GROUNDING_PROMPT
            ),
        },
        "v5_schema_sha256": {
            "gate_anchor_query_compilation": sha256_json(
                ROLE_AWARE_QUERY_CONDITION_SCHEMA
            ),
            "role_aware_predicate_grounding": sha256_json(
                PREDICATE_GROUNDING_SCHEMA
            ),
        },
        "v5_contract_versions": {
            "gate_anchor_query_compilation": (
                ROLE_AWARE_QUERY_CONDITION_CONTRACT_VERSION
            ),
            "role_aware_predicate_grounding": (
                ROLE_AWARE_PREDICATE_GROUNDING_CONTRACT_VERSION
            ),
        },
    }


def _write_stage_completion(
    stage_directory: Path, artifact_names: Sequence[str]
) -> None:
    artifacts = {
        name: sha256_file(stage_directory / name) for name in artifact_names
    }
    atomic_write_json(
        stage_directory / "completion.json",
        {"schema_version": 1, "artifacts_sha256": artifacts},
    )


def _validate_stage_completion(
    stage_directory: Path, artifact_names: Sequence[str]
) -> None:
    completion = load_json_object(stage_directory / "completion.json")
    hashes = completion.get("artifacts_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(artifact_names):
        raise ExperimentDataError(f"Stage completion manifest is invalid: {stage_directory}")
    for name in artifact_names:
        path = stage_directory / name
        if not path.is_file() or sha256_file(path) != hashes[name]:
            raise ExperimentDataError(f"Completed stage artifact changed: {path}")


def _positive_finite(value: Any, label: str) -> float:
    result = _nonnegative_finite(value, label)
    if result <= 0:
        raise ExperimentDataError(f"{label} must be positive")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ExperimentDataError(f"{label} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentDataError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ExperimentDataError(f"{label} must be a finite non-negative number")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("catalog")
    subparsers.add_parser("status")
    subparsers.add_parser("init-review")
    subparsers.add_parser("freeze-review")
    subparsers.add_parser("prepare")
    subparsers.add_parser("prepare-v2")
    subparsers.add_parser("prepare-v3")
    subparsers.add_parser("prepare-v4")
    subparsers.add_parser("prepare-v5")
    subparsers.add_parser("compare")
    subparsers.add_parser("compare-v2")
    subparsers.add_parser("compare-v3")
    subparsers.add_parser("compare-v4")
    subparsers.add_parser("compare-v5")
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--execute", action="store_true")
    transcribe_parser = subparsers.add_parser("transcribe")
    transcribe_parser.add_argument("--execute", action="store_true")
    transcribe_parser.add_argument("--device")
    for command in (
        "run-naive",
        "run-transcript-only",
        "run-candidates",
        "run-optimized",
        "materialize-clips",
        "run-optimized-materialized",
        "run-naive-v2-binary",
        "run-optimized-v2-binary",
        "run-naive-v2-localization",
        "run-optimized-v2-localization",
        "run-v3-transcript-proposals",
        "materialize-v3-proposal-clips",
        "run-v3-refinement",
        "run-v4-query-conditions",
        "run-v4-predicate-grounding",
        "run-v5-query-conditions",
        "run-v5-predicate-grounding",
    ):
        run_parser = subparsers.add_parser(command)
        run_parser.add_argument("--execute", action="store_true")
    return parser


def _preview_message(command: str) -> dict[str, Any]:
    return {
        "preview_only": True,
        "command": command,
        "message": "No external or model work was run. Add --execute only after reviewing code, status, and frozen plan artifacts.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "catalog":
            result = catalog_payload()
        elif args.command == "status":
            result = status(args.root)
        elif args.command == "init-review":
            result = initialize_ground_truth_review(args.root)
        elif args.command == "freeze-review":
            result = freeze_ground_truth(args.root)
        elif args.command == "prepare":
            result = prepare_experiment(args.root)
        elif args.command == "prepare-v2":
            result = prepare_v2(args.root)
        elif args.command == "prepare-v3":
            result = prepare_v3(args.root)
        elif args.command == "prepare-v4":
            result = prepare_v4(args.root)
        elif args.command == "prepare-v5":
            result = prepare_v5(args.root)
        elif args.command == "compare":
            result = compare_results(args.root)
        elif args.command == "compare-v2":
            result = compare_v2_results(args.root)
        elif args.command == "compare-v3":
            result = compare_v3_results(args.root)
        elif args.command == "compare-v4":
            result = compare_v4_results(args.root)
        elif args.command == "compare-v5":
            result = compare_v5_results(args.root)
        elif not args.execute:
            result = _preview_message(args.command)
        elif args.command == "download":
            result = download_lectures(root=args.root)
        elif args.command == "transcribe":
            model = create_whisper_model(WHISPER_MODEL, device=args.device)
            result = transcribe_lectures(root=args.root, model=model)
        elif args.command == "run-naive":
            result = run_naive(args.root, args.env_file)
        elif args.command == "run-transcript-only":
            result = run_transcript_only(args.root, args.env_file)
        elif args.command == "run-candidates":
            result = run_candidates(args.root, args.env_file)
        elif args.command == "run-optimized":
            result = run_optimized(args.root, args.env_file)
        elif args.command == "materialize-clips":
            result = materialize_clips(args.root)
        elif args.command == "run-optimized-materialized":
            result = run_optimized_materialized(args.root, args.env_file)
        elif args.command == "run-naive-v2-binary":
            result = run_naive_v2_binary(args.root, args.env_file)
        elif args.command == "run-optimized-v2-binary":
            result = run_optimized_v2_binary(args.root, args.env_file)
        elif args.command == "run-naive-v2-localization":
            result = run_naive_v2_localization(args.root, args.env_file)
        elif args.command == "run-optimized-v2-localization":
            result = run_optimized_v2_localization(args.root, args.env_file)
        elif args.command == "run-v3-transcript-proposals":
            result = run_v3_transcript_proposals(args.root, args.env_file)
        elif args.command == "materialize-v3-proposal-clips":
            result = materialize_v3_proposal_clips(args.root)
        elif args.command == "run-v3-refinement":
            result = run_v3_refinement(args.root, args.env_file)
        elif args.command == "run-v4-query-conditions":
            result = run_v4_query_conditions(args.root, args.env_file)
        elif args.command == "run-v4-predicate-grounding":
            result = run_v4_predicate_grounding(args.root, args.env_file)
        elif args.command == "run-v5-query-conditions":
            result = run_v5_query_conditions(args.root, args.env_file)
        else:
            result = run_v5_predicate_grounding(args.root, args.env_file)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ExperimentDataError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
