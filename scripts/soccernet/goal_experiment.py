"""Immutable three-method SoccerNet goal experiment with materialized clips."""

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

from scripts.experiments.artifacts import (
    refuse_completed,
    validate_completion,
    write_completion,
)
from scripts.experiments.clip_materialization import (
    MATERIALIZATION_CONTRACT,
    MATERIALIZATION_CONTRACT_SHA256,
    ClipProcessor,
    MediaProbe,
    ffmpeg_materialize_clip,
    probe_materialized_clip,
)
from scripts.experiments.common import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    load_json_object as load_shared_json_object,
    load_jsonl_objects,
    sha256_file,
    sha256_json,
    sha256_text,
)
from scripts.experiments.gemini_runtime import aggregate_media_uploads

from .common import DEFAULT_ROOT, SoccerNetDataError, load_json_object
from .goal_eval import (
    DEDUPLICATION_SECONDS,
    EVALUATION_TOLERANCES_SECONDS,
    PRIMARY_TOLERANCE_SECONDS,
    aggregate_api_usage,
    evaluate_candidate_windows,
    evaluate_predictions,
)
from .goal_materialize import (
    MATERIALIZATION_MANIFEST_FILENAME,
    MATERIALIZATION_STAGE_FILENAME,
    MATERIALIZED_INPUT_FILENAME,
    materialize_goal_candidate_clips,
    validate_goal_materialized_clip_stage,
)
from .goal_plans import (
    CANDIDATE_LIST_SCHEMA,
    GOAL_LIST_SCHEMA,
    TIMESTAMP_CONTRACT_VERSION,
    TRANSCRIPT_CANDIDATE_PROMPT,
    TRANSCRIPT_GOAL_LIST_SCHEMA,
    TRANSCRIPT_ONLY_GOAL_PROMPT,
    VIDEO_GOAL_LOCALIZATION_PROMPT,
    build_llm_candidate_plan,
    build_naive_plan,
    build_transcript_only_plan,
    build_transcript_video_plan,
)
from .goal_runtime import EXPERIMENT_MODEL, create_experiment_executor


EXPERIMENT_NAME = "goal_pushdown_3_games_v4_materialized"
PREDECESSOR_EXPERIMENT_NAME = "goal_pushdown_3_games_v3"
WINDOW_RADIUS_SECONDS = 30.0
RESOLUTION = "224p"
SELECTED_GAMES = (
    "england_epl/2015-2016/2015-09-12 - 14-45 Everton 3 - 1 Chelsea",
    "england_epl/2015-2016/2015-10-17 - 17-00 Chelsea 2 - 0 Aston Villa",
    "england_epl/2015-2016/2016-01-13 - 22-45 Chelsea 2 - 2 West Brom",
)

INPUT_FILENAME = "input.jsonl"
GROUND_TRUTH_FILENAME = "ground_truth.json"
PREDICTION_CONFIG_FILENAME = "prediction_config.json"
EVALUATION_CONFIG_FILENAME = "evaluation_config.json"
PREDICTION_PREFIX_COMPLETION_FILENAME = "prediction_prefix_completion.json"
PREPARE_COMPLETION_FILENAME = "prepare_completion.json"
COMPARISON_FILENAME = "comparison.json"
COMPARISON_COMPLETION_FILENAME = "comparison_completion.json"
EVALUATION_DIRECTORY = "evaluations"

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

FORBIDDEN_PREDICTION_KEYS = {
    "annotation",
    "annotations",
    "answer",
    "expected",
    "ground_truth",
    "label",
    "labels",
    "truth",
}


def experiment_directory(root: Path) -> Path:
    return root / "experiments" / EXPERIMENT_NAME


def prepare_experiment(root: Path) -> dict[str, Any]:
    """Freeze prediction inputs separately from evaluator-only labels."""
    directory = experiment_directory(root)
    if (directory / PREPARE_COMPLETION_FILENAME).exists():
        _validate_prediction_prefix(directory)
        _validate_evaluation_prefix(directory)
        raise SoccerNetDataError("The v4 experiment is already prepared and immutable")
    if directory.exists() and any(path.is_file() for path in directory.rglob("*")):
        raise SoccerNetDataError(
            f"Refusing to prepare over partial or existing v4 artifacts: {directory}"
        )

    source_manifest_path = root / "manifest.json"
    manifest = load_json_object(source_manifest_path)
    manifest_games: dict[str, Mapping[str, Any]] = {}
    for raw_game in manifest.get("games", []):
        if not isinstance(raw_game, Mapping):
            raise SoccerNetDataError("SoccerNet manifest contains an invalid game")
        game_id = raw_game.get("game_id")
        if not isinstance(game_id, str) or game_id in manifest_games:
            raise SoccerNetDataError(f"Invalid or duplicate manifest game: {game_id!r}")
        manifest_games[game_id] = raw_game

    input_rows: list[dict[str, Any]] = []
    truth_games: list[dict[str, Any]] = []
    source_dependencies: dict[str, str] = {
        "source_manifest": str(source_manifest_path.resolve())
    }
    transcript_hashes: dict[str, str] = {}
    video_hashes: dict[str, str] = {}
    for game_id in SELECTED_GAMES:
        manifest_game = manifest_games.get(game_id)
        if not isinstance(manifest_game, Mapping):
            raise SoccerNetDataError(f"Selected game is missing from manifest: {game_id}")
        if manifest_game.get("status") != "complete":
            raise SoccerNetDataError(f"Selected game is incomplete: {game_id}")

        labels_path = root / game_id / "Labels-v2.json"
        truth_by_half = _goal_truth_by_half(labels_path, game_id)
        truth_games.append({"game_id": game_id, "halves": truth_by_half})
        videos = manifest_game.get("videos")
        if not isinstance(videos, Mapping):
            raise SoccerNetDataError(f"Selected game has no videos: {game_id}")
        resolution_videos = videos.get(RESOLUTION)
        if not isinstance(resolution_videos, Mapping):
            raise SoccerNetDataError(f"Selected game has no {RESOLUTION} videos: {game_id}")

        for half in (1, 2):
            media = resolution_videos.get(str(half))
            if not isinstance(media, Mapping) or media.get("status") != "valid":
                raise SoccerNetDataError(
                    f"Selected {RESOLUTION} video is invalid: {game_id} half {half}"
                )
            video_path_value = media.get("path")
            if not isinstance(video_path_value, str):
                raise SoccerNetDataError(f"Selected video has no path: {game_id} half {half}")
            video_path = (root / video_path_value).resolve()
            if not video_path.is_file():
                raise SoccerNetDataError(f"Selected video is missing: {video_path}")
            duration = _positive_finite(
                media.get("duration_seconds"), "video duration"
            )
            video_hash = sha256_file(video_path)
            source_key = _half_key(game_id, half)
            video_hashes[source_key] = video_hash
            source_dependencies[f"video:{source_key}"] = str(video_path)

            transcript_path = root / "transcripts" / game_id / f"{half}.whisper.json"
            transcript = load_json_object(transcript_path)
            if transcript.get("language") != "en":
                raise SoccerNetDataError(
                    f"Selected transcript is not English: {game_id} half {half}"
                )
            segments = _normalize_transcript_segments(transcript, game_id, half)
            transcript_hashes[source_key] = sha256_file(transcript_path)
            source_dependencies[f"transcript:{source_key}"] = str(
                transcript_path.resolve()
            )
            input_rows.append(
                {
                    "game_id": game_id,
                    "half": half,
                    "duration_seconds": duration,
                    "window_radius_seconds": WINDOW_RADIUS_SECONDS,
                    "video": {
                        "type": "Video",
                        "path": str(video_path),
                        "sha256": video_hash,
                        "mime_type": "video/x-matroska",
                    },
                    "transcript_segments": segments,
                    "timestamped_transcript": _format_timestamped_transcript(segments),
                }
            )

    _reject_evaluator_keys(input_rows)
    _validate_input_rows(input_rows)
    directory.mkdir(parents=True, exist_ok=False)
    input_path = directory / INPUT_FILENAME
    ground_truth_path = directory / GROUND_TRUTH_FILENAME
    atomic_write_jsonl(input_path, input_rows)
    ground_truth = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "label_policy": "visible Goals from Labels-v2.json; evaluator-only",
        "games": truth_games,
    }
    atomic_write_json(ground_truth_path, ground_truth)

    plan_sources = _rendered_plans(directory)
    for name, source in plan_sources.items():
        atomic_write_text(directory / "plans" / f"{name}.py", source)
    prediction_config = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "predecessor_experiment": PREDECESSOR_EXPERIMENT_NAME,
        "planned_methods": [NAIVE_STAGE, TRANSCRIPT_ONLY_STAGE, TRANSCRIPT_VIDEO_STAGE],
        "selected_games": list(SELECTED_GAMES),
        "resolution": RESOLUTION,
        "model": EXPERIMENT_MODEL,
        "candidate_window_radius_seconds": WINDOW_RADIUS_SECONDS,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "source_dependency_paths": source_dependencies,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "video_sha256": video_hashes,
        "transcript_sha256": transcript_hashes,
        "prompt_sha256": _prompt_hashes(),
        "schema_sha256": _schema_hashes(),
        "plan_sha256": {
            name: sha256_text(source) for name, source in plan_sources.items()
        },
        "timestamp_contract": {
            "version": TIMESTAMP_CONTRACT_VERSION,
            "model_fields": ["clip_minute", "clip_second"],
            "conversion": "offset_seconds = clip_minute * 60 + clip_second",
            "reference_frame": "elapsed time from supplied full video or standalone clip",
            "source_time_conversion": (
                "full half: time_seconds = offset_seconds; materialized clip: "
                "time_seconds = candidate_window.start_seconds + offset_seconds"
            ),
        },
        "materialization_contract": MATERIALIZATION_CONTRACT,
        "materialization_contract_sha256": MATERIALIZATION_CONTRACT_SHA256,
        "controlled_video_localizer": {
            "same_prompt": True,
            "same_schema": True,
            "only_intended_model_input_difference": (
                "complete half-video versus physically materialized transcript-selected clip"
            ),
        },
    }
    evaluation_config = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "ground_truth_path": str(ground_truth_path.resolve()),
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "evaluation_tolerances_seconds": list(EVALUATION_TOLERANCES_SECONDS),
        "primary_tolerance_seconds": PRIMARY_TOLERANCE_SECONDS,
        "deduplication_seconds": DEDUPLICATION_SECONDS,
    }
    atomic_write_json(directory / PREDICTION_CONFIG_FILENAME, prediction_config)
    atomic_write_json(directory / EVALUATION_CONFIG_FILENAME, evaluation_config)

    prediction_artifacts = _prediction_prefix_artifacts(directory)
    dependencies = _source_dependencies(prediction_config)
    write_completion(
        directory / PREDICTION_PREFIX_COMPLETION_FILENAME,
        stage_name="prediction_prefix",
        directory=directory,
        artifact_names=prediction_artifacts,
        dependencies=dependencies,
        error_type=SoccerNetDataError,
    )
    write_completion(
        directory / PREPARE_COMPLETION_FILENAME,
        stage_name="prepare",
        directory=directory,
        artifact_names=(
            *prediction_artifacts,
            GROUND_TRUTH_FILENAME,
            EVALUATION_CONFIG_FILENAME,
            PREDICTION_PREFIX_COMPLETION_FILENAME,
        ),
        dependencies=dependencies,
        error_type=SoccerNetDataError,
    )
    return {
        "experiment": EXPERIMENT_NAME,
        "input_path": str(input_path),
        "ground_truth_path": str(ground_truth_path),
        "half_count": len(input_rows),
        "goal_count": sum(
            len(goals)
            for game in truth_games
            for goals in game["halves"].values()
        ),
        "prediction_config_sha256": sha256_file(
            directory / PREDICTION_CONFIG_FILENAME
        ),
        "evaluation_config_sha256": sha256_file(
            directory / EVALUATION_CONFIG_FILENAME
        ),
        "materialization_contract_sha256": MATERIALIZATION_CONTRACT_SHA256,
    }


def run_naive(
    root: Path,
    env_file: Path,
    *,
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    directory, config = _prediction_prefix(root)
    input_path = directory / INPUT_FILENAME
    return _run_prediction_stage(
        directory=directory,
        config=config,
        stage_name=NAIVE_STAGE,
        method="naive_full_video",
        plan=build_naive_plan(str(input_path.resolve())),
        input_path=input_path,
        plan_name="naive",
        goal_field="goals",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_transcript_only(
    root: Path,
    env_file: Path,
    *,
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    directory, config = _prediction_prefix(root)
    input_path = directory / INPUT_FILENAME
    return _run_prediction_stage(
        directory=directory,
        config=config,
        stage_name=TRANSCRIPT_ONLY_STAGE,
        method=TRANSCRIPT_ONLY_STAGE,
        plan=build_transcript_only_plan(str(input_path.resolve())),
        input_path=input_path,
        plan_name="transcript_only",
        goal_field="transcript_goals",
        env_file=env_file,
        prompt_executor=prompt_executor,
    )


def run_candidates(
    root: Path,
    env_file: Path,
    *,
    prompt_executor: Any | None = None,
) -> dict[str, Any]:
    directory, config = _prediction_prefix(root)
    stage_directory = directory / "runs" / CANDIDATE_STAGE
    refuse_completed(stage_directory, error_type=SoccerNetDataError)
    if any((stage_directory / name).exists() for name in CANDIDATE_ARTIFACTS):
        raise SoccerNetDataError(
            f"Candidate stage has partial final artifacts: {stage_directory}"
        )
    input_path = directory / INPUT_FILENAME
    plan = build_llm_candidate_plan(str(input_path.resolve()))
    _validate_plan(config, "candidates", plan)
    executor = prompt_executor or create_experiment_executor(
        stage_directory=stage_directory,
        env_file=env_file,
    )
    started = time.perf_counter()
    rows = execute(plan, prompt_executor=executor)
    elapsed = time.perf_counter() - started
    candidates = [_extract_candidate_row(row) for row in rows]
    _validate_candidate_rows(candidates)
    atomic_write_jsonl(stage_directory / "candidates.jsonl", candidates)
    audit = {
        "schema_version": 1,
        "half_count": len(candidates),
        "candidate_point_count": sum(
            len(row["goal_candidates"]) for row in candidates
        ),
        "candidate_window_count": sum(
            len(row["candidate_windows"]) for row in candidates
        ),
        "candidate_union_duration_seconds": sum(
            sum(float(window["duration_seconds"]) for window in row["candidate_windows"])
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
        input_path=input_path,
        config=config,
    )
    atomic_write_json(stage_directory / "stage.json", stage)
    write_completion(
        stage_directory / "completion.json",
        stage_name=CANDIDATE_STAGE,
        directory=stage_directory,
        artifact_names=CANDIDATE_ARTIFACTS,
        dependencies=_prediction_dependencies(directory, "candidates"),
        error_type=SoccerNetDataError,
    )
    return {"stage": stage, "audit": audit, "rows": candidates}


def materialize_clips(
    root: Path,
    *,
    clip_processor: ClipProcessor = ffmpeg_materialize_clip,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    directory, _ = _prediction_prefix(root)
    _validate_candidate_stage(directory)
    stage_directory = directory / "runs" / MATERIALIZATION_STAGE
    refuse_completed(stage_directory, error_type=SoccerNetDataError)
    candidate_path = directory / "runs" / CANDIDATE_STAGE / "candidates.jsonl"
    completed_artifacts = [stage_directory / name for name in MATERIALIZATION_ARTIFACTS]
    if all(path.is_file() for path in completed_artifacts):
        result = validate_goal_materialized_clip_stage(
            stage_directory, media_probe=media_probe
        )
        write_completion(
            stage_directory / "completion.json",
            stage_name=MATERIALIZATION_STAGE,
            directory=stage_directory,
            artifact_names=MATERIALIZATION_ARTIFACTS,
            dependencies={
                "candidates": candidate_path,
                "candidate_completion": (
                    directory / "runs" / CANDIDATE_STAGE / "completion.json"
                ),
            },
            error_type=SoccerNetDataError,
        )
        return {
            **result,
            "stage": load_shared_json_object(
                stage_directory / MATERIALIZATION_STAGE_FILENAME,
                error_type=SoccerNetDataError,
            ),
            "completion_recovered": True,
        }
    if any(path.exists() for path in completed_artifacts):
        raise SoccerNetDataError(
            f"Materialization has incomplete final artifacts: {stage_directory}"
        )
    result = materialize_goal_candidate_clips(
        candidate_path,
        stage_directory,
        clip_processor=clip_processor,
        media_probe=media_probe,
    )
    write_completion(
        stage_directory / "completion.json",
        stage_name=MATERIALIZATION_STAGE,
        directory=stage_directory,
        artifact_names=MATERIALIZATION_ARTIFACTS,
        dependencies={
            "candidates": candidate_path,
            "candidate_completion": (
                directory / "runs" / CANDIDATE_STAGE / "completion.json"
            ),
        },
        error_type=SoccerNetDataError,
    )
    return result


def run_transcript_video(
    root: Path,
    env_file: Path,
    *,
    prompt_executor: Any | None = None,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    directory, config = _prediction_prefix(root)
    _validate_materialization_stage(directory, media_probe=media_probe)
    input_path = directory / "runs" / MATERIALIZATION_STAGE / MATERIALIZED_INPUT_FILENAME
    return _run_prediction_stage(
        directory=directory,
        config=config,
        stage_name=TRANSCRIPT_VIDEO_STAGE,
        method=TRANSCRIPT_VIDEO_STAGE,
        plan=build_transcript_video_plan(str(input_path.resolve())),
        input_path=input_path,
        plan_name="transcript_video",
        goal_field="transcript_video_goals",
        env_file=env_file,
        prompt_executor=prompt_executor,
        extra_dependencies={
            "materialized_input": input_path,
            "materialization_completion": (
                directory / "runs" / MATERIALIZATION_STAGE / "completion.json"
            ),
        },
    )


def evaluate_results(
    root: Path,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    """Read SoccerNet labels only here, after all prediction stages validate."""
    directory, _ = _prediction_prefix(root)
    _validate_prediction_stage(directory, NAIVE_STAGE, "naive", directory / INPUT_FILENAME)
    _validate_prediction_stage(
        directory,
        TRANSCRIPT_ONLY_STAGE,
        "transcript_only",
        directory / INPUT_FILENAME,
    )
    _validate_candidate_stage(directory)
    _validate_materialization_stage(directory, media_probe=media_probe)
    materialized_input = (
        directory / "runs" / MATERIALIZATION_STAGE / MATERIALIZED_INPUT_FILENAME
    )
    _validate_prediction_stage(
        directory,
        TRANSCRIPT_VIDEO_STAGE,
        "transcript_video",
        materialized_input,
        extra_dependencies={
            "materialized_input": materialized_input,
            "materialization_completion": (
                directory / "runs" / MATERIALIZATION_STAGE / "completion.json"
            ),
        },
    )
    ground_truth = _validate_evaluation_prefix(directory)
    evaluation_directory = directory / EVALUATION_DIRECTORY
    refuse_completed(evaluation_directory, error_type=SoccerNetDataError)
    if evaluation_directory.exists() and any(evaluation_directory.iterdir()):
        raise SoccerNetDataError(f"Evaluation stage is partial: {evaluation_directory}")

    outputs: dict[str, Any] = {}
    for method in (NAIVE_STAGE, TRANSCRIPT_ONLY_STAGE, TRANSCRIPT_VIDEO_STAGE):
        payload = load_shared_json_object(
            directory / "runs" / method / "predictions.json",
            error_type=SoccerNetDataError,
        )
        predictions = payload.get("predictions")
        if not isinstance(predictions, list):
            raise SoccerNetDataError(f"Prediction artifact is invalid: {method}")
        evaluation = evaluate_predictions(predictions, ground_truth)
        outputs[method] = evaluation
        atomic_write_json(evaluation_directory / f"{method}.json", evaluation)
    candidate_rows = load_jsonl_objects(
        directory / "runs" / CANDIDATE_STAGE / "candidates.jsonl",
        error_type=SoccerNetDataError,
    )
    candidate_evaluation = evaluate_candidate_windows(candidate_rows, ground_truth)
    outputs["candidates"] = candidate_evaluation
    atomic_write_json(evaluation_directory / "candidates.json", candidate_evaluation)
    write_completion(
        evaluation_directory / "completion.json",
        stage_name="evaluate",
        directory=evaluation_directory,
        artifact_names=EVALUATION_ARTIFACTS,
        dependencies=_evaluation_dependencies(directory),
        error_type=SoccerNetDataError,
    )
    return outputs


def compare_results(root: Path) -> dict[str, Any]:
    directory, _ = _prediction_prefix(root)
    evaluation_directory = directory / EVALUATION_DIRECTORY
    validate_completion(
        evaluation_directory / "completion.json",
        stage_name="evaluate",
        directory=evaluation_directory,
        artifact_names=EVALUATION_ARTIFACTS,
        dependencies=_evaluation_dependencies(directory),
        error_type=SoccerNetDataError,
    )
    if (directory / COMPARISON_FILENAME).exists():
        raise SoccerNetDataError("Comparison already exists and is immutable")

    evaluations = {
        method: load_shared_json_object(
            evaluation_directory / f"{method}.json", error_type=SoccerNetDataError
        )
        for method in (NAIVE_STAGE, TRANSCRIPT_ONLY_STAGE, TRANSCRIPT_VIDEO_STAGE)
    }
    stages = {
        method: load_shared_json_object(
            directory / "runs" / method / "stage.json",
            error_type=SoccerNetDataError,
        )
        for method in (NAIVE_STAGE, TRANSCRIPT_ONLY_STAGE, TRANSCRIPT_VIDEO_STAGE)
    }
    candidate_stage = load_shared_json_object(
        directory / "runs" / CANDIDATE_STAGE / "stage.json",
        error_type=SoccerNetDataError,
    )
    materialization_stage = load_shared_json_object(
        directory / "runs" / MATERIALIZATION_STAGE / MATERIALIZATION_STAGE_FILENAME,
        error_type=SoccerNetDataError,
    )
    optimized_usage = _sum_usage(
        candidate_stage["api_usage"], stages[TRANSCRIPT_VIDEO_STAGE]["api_usage"]
    )
    optimized_media = _sum_media_uploads(
        candidate_stage["media_uploads"],
        stages[TRANSCRIPT_VIDEO_STAGE]["media_uploads"],
    )
    naive_latency = _combined_latency_measurement((NAIVE_STAGE, stages[NAIVE_STAGE]))
    transcript_latency = _combined_latency_measurement(
        (TRANSCRIPT_ONLY_STAGE, stages[TRANSCRIPT_ONLY_STAGE])
    )
    optimized_latency = _combined_latency_measurement(
        (CANDIDATE_STAGE, candidate_stage),
        (MATERIALIZATION_STAGE, materialization_stage),
        (TRANSCRIPT_VIDEO_STAGE, stages[TRANSCRIPT_VIDEO_STAGE]),
    )
    primary_key = str(int(PRIMARY_TOLERANCE_SECONDS))
    naive_usage = stages[NAIVE_STAGE]["api_usage"]
    transcript_usage = stages[TRANSCRIPT_ONLY_STAGE]["api_usage"]
    report = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "primary_tolerance_seconds": PRIMARY_TOLERANCE_SECONDS,
        "methods": {
            NAIVE_STAGE: _comparison_method(
                evaluations[NAIVE_STAGE],
                stages[NAIVE_STAGE],
                naive_usage,
                stages[NAIVE_STAGE]["media_uploads"],
                naive_latency,
                primary_key,
            ),
            TRANSCRIPT_ONLY_STAGE: _comparison_method(
                evaluations[TRANSCRIPT_ONLY_STAGE],
                stages[TRANSCRIPT_ONLY_STAGE],
                transcript_usage,
                stages[TRANSCRIPT_ONLY_STAGE]["media_uploads"],
                transcript_latency,
                primary_key,
            ),
            TRANSCRIPT_VIDEO_STAGE: {
                **_comparison_method(
                    evaluations[TRANSCRIPT_VIDEO_STAGE],
                    stages[TRANSCRIPT_VIDEO_STAGE],
                    optimized_usage,
                    optimized_media,
                    optimized_latency,
                    primary_key,
                ),
                "candidate_metrics": load_shared_json_object(
                    evaluation_directory / "candidates.json",
                    error_type=SoccerNetDataError,
                ),
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
                naive_usage,
                transcript_usage,
                naive_latency,
                transcript_latency,
            ),
            TRANSCRIPT_VIDEO_STAGE: _reductions(
                naive_usage,
                optimized_usage,
                naive_latency,
                optimized_latency,
            ),
        },
        "cost_scope": (
            "Query-time transcript-video totals include transcript candidate generation, "
            "local clip materialization, clip upload, and audiovisual localization. "
            "One-time Whisper transcription is excluded."
        ),
    }
    atomic_write_json(directory / COMPARISON_FILENAME, report)
    write_completion(
        directory / COMPARISON_COMPLETION_FILENAME,
        stage_name="compare",
        directory=directory,
        artifact_names=(COMPARISON_FILENAME,),
        dependencies={
            "evaluation_completion": evaluation_directory / "completion.json"
        },
        error_type=SoccerNetDataError,
    )
    return report


def status(
    root: Path,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> dict[str, Any]:
    directory = experiment_directory(root)
    result: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "prepared": "not_started",
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
    if (directory / PREDICTION_PREFIX_COMPLETION_FILENAME).exists():
        try:
            _validate_prediction_prefix(directory)
            _validate_evaluation_prefix(directory)
            result["prepared"] = "complete"
        except (SoccerNetDataError, OSError, ValueError) as exc:
            result["prepared"] = "invalid"
            result["prepared_error"] = str(exc)
            return result
    elif directory.exists() and any(directory.rglob("*")):
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
        CANDIDATE_STAGE: lambda: _validate_candidate_stage(directory),
        MATERIALIZATION_STAGE: lambda: _validate_materialization_stage(
            directory, media_probe=media_probe
        ),
        TRANSCRIPT_VIDEO_STAGE: lambda: _validate_prediction_stage(
            directory,
            TRANSCRIPT_VIDEO_STAGE,
            "transcript_video",
            directory / "runs" / MATERIALIZATION_STAGE / MATERIALIZED_INPUT_FILENAME,
            extra_dependencies={
                "materialized_input": (
                    directory
                    / "runs"
                    / MATERIALIZATION_STAGE
                    / MATERIALIZED_INPUT_FILENAME
                ),
                "materialization_completion": (
                    directory / "runs" / MATERIALIZATION_STAGE / "completion.json"
                ),
            },
        ),
    }
    for stage_name, validator in validators.items():
        stage_directory = directory / "runs" / stage_name
        result["stages"][stage_name] = _validated_stage_status(
            stage_directory, validator
        )
    evaluation_directory = directory / EVALUATION_DIRECTORY
    result["stages"]["evaluate"] = _validated_stage_status(
        evaluation_directory,
        lambda: validate_completion(
            evaluation_directory / "completion.json",
            stage_name="evaluate",
            directory=evaluation_directory,
            artifact_names=EVALUATION_ARTIFACTS,
            dependencies=_evaluation_dependencies(directory),
            error_type=SoccerNetDataError,
        ),
    )
    comparison_completion = directory / COMPARISON_COMPLETION_FILENAME
    if comparison_completion.exists():
        try:
            validate_completion(
                comparison_completion,
                stage_name="compare",
                directory=directory,
                artifact_names=(COMPARISON_FILENAME,),
                dependencies={
                    "evaluation_completion": evaluation_directory / "completion.json"
                },
                error_type=SoccerNetDataError,
            )
            result["comparison"] = "complete"
        except (SoccerNetDataError, OSError, ValueError) as exc:
            result["comparison"] = "invalid"
            result["comparison_error"] = str(exc)
    elif (directory / COMPARISON_FILENAME).exists():
        result["comparison"] = "partial"
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
    goal_field: str,
    env_file: Path,
    prompt_executor: Any | None,
    extra_dependencies: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    stage_directory = directory / "runs" / stage_name
    refuse_completed(stage_directory, error_type=SoccerNetDataError)
    if any((stage_directory / name).exists() for name in PREDICTION_ARTIFACTS):
        raise SoccerNetDataError(
            f"Prediction stage has partial final artifacts: {stage_directory}"
        )
    _validate_plan(config, plan_name, plan)
    executor = prompt_executor or create_experiment_executor(
        stage_directory=stage_directory,
        env_file=env_file,
    )
    started = time.perf_counter()
    rows = execute(plan, prompt_executor=executor)
    elapsed = time.perf_counter() - started
    predictions = _extract_predictions(rows, goal_field=goal_field, method=method)
    artifact = {
        "schema_version": 1,
        "stage_name": stage_name,
        "method": method,
        "predictions": predictions,
    }
    atomic_write_json(stage_directory / "predictions.json", artifact)
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
    if extra_dependencies:
        dependencies.update(extra_dependencies)
    write_completion(
        stage_directory / "completion.json",
        stage_name=stage_name,
        directory=stage_directory,
        artifact_names=PREDICTION_ARTIFACTS,
        dependencies=dependencies,
        error_type=SoccerNetDataError,
    )
    return {"stage": stage, "predictions": predictions}


def _stage_summary(
    stage_directory: Path,
    *,
    elapsed: float,
    executor: Any,
    stage_name: str,
    plan_name: str,
    input_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    usage = aggregate_api_usage(stage_directory / "api_calls.jsonl")
    media = aggregate_media_uploads(stage_directory / "media_uploads.jsonl")
    cache_hits = int(getattr(executor, "cache_hits", 0))
    invalid_reasons = [
        reason
        for condition, reason in (
            (cache_hits > 0, "cached_responses_used"),
            (usage["failed_api_call_count"] > 0, "failed_api_attempts_recorded"),
            (media["failed_upload_count"] > 0, "failed_media_uploads_recorded"),
        )
        if condition
    ]
    return {
        "schema_version": 1,
        "stage_name": stage_name,
        "model": EXPERIMENT_MODEL,
        "max_in_flight_provider_calls": 1,
        "end_to_end_seconds": elapsed,
        "cache_hits": cache_hits,
        "cache_misses": int(getattr(executor, "cache_misses", 0)),
        "cold_start_latency_valid": not invalid_reasons,
        "cold_start_invalid_reasons": invalid_reasons,
        "input_sha256": sha256_file(input_path),
        "plan_sha256": config["plan_sha256"][plan_name],
        "api_usage": usage,
        "media_uploads": media,
    }


def _goal_truth_by_half(labels_path: Path, game_id: str) -> dict[str, list[dict[str, Any]]]:
    labels = load_json_object(labels_path)
    annotations = labels.get("annotations")
    if not isinstance(annotations, list):
        raise SoccerNetDataError(f"Labels have no annotations: {game_id}")
    truth: dict[str, list[dict[str, Any]]] = {"1": [], "2": []}
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        if str(annotation.get("label", "")).casefold() != "goal":
            continue
        if annotation.get("visibility") == "not shown":
            continue
        half = str(annotation.get("gameTime", "")).split("-", maxsplit=1)[0].strip()
        if half not in truth:
            raise SoccerNetDataError(f"Invalid goal half in {game_id}")
        try:
            position_seconds = int(annotation["position"]) / 1000.0
        except (KeyError, TypeError, ValueError) as exc:
            raise SoccerNetDataError(f"Invalid goal position in {game_id}") from exc
        if not math.isfinite(position_seconds) or position_seconds < 0:
            raise SoccerNetDataError(f"Invalid goal position in {game_id}")
        truth[half].append(
            {
                "time_seconds": position_seconds,
                "visibility": str(annotation.get("visibility", "visible")),
            }
        )
    for goals in truth.values():
        goals.sort(key=lambda goal: goal["time_seconds"])
    return truth


def _normalize_transcript_segments(
    transcript: Mapping[str, Any], game_id: str, half: int
) -> list[dict[str, Any]]:
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        raise SoccerNetDataError(f"Transcript has no segments: {game_id} half {half}")
    normalized: list[dict[str, Any]] = []
    previous_start = -1.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise SoccerNetDataError(
                f"Transcript segment {index} is invalid: {game_id} half {half}"
            )
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SoccerNetDataError(
                f"Transcript segment {index} has invalid timing: {game_id} half {half}"
            ) from exc
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end < start
            or start < previous_start
        ):
            raise SoccerNetDataError(
                f"Transcript segment {index} has invalid order: {game_id} half {half}"
            )
        previous_start = start
        normalized.append(
            {"start": start, "end": end, "text": str(segment.get("text", "")).strip()}
        )
    return normalized


def _format_timestamped_transcript(segments: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"[{float(segment['start']):.3f}-{float(segment['end']):.3f}] "
        f"{str(segment['text']).strip()}"
        for segment in segments
    )


def _rendered_plans(directory: Path) -> dict[str, str]:
    input_path = directory / INPUT_FILENAME
    materialized_input = (
        directory / "runs" / MATERIALIZATION_STAGE / MATERIALIZED_INPUT_FILENAME
    )
    return {
        "naive": render_query(build_naive_plan(str(input_path.resolve()))),
        "transcript_only": render_query(
            build_transcript_only_plan(str(input_path.resolve()))
        ),
        "candidates": render_query(
            build_llm_candidate_plan(str(input_path.resolve()))
        ),
        "transcript_video": render_query(
            build_transcript_video_plan(str(materialized_input.resolve()))
        ),
    }


def _prompt_hashes() -> dict[str, str]:
    video_hash = sha256_text(VIDEO_GOAL_LOCALIZATION_PROMPT)
    return {
        "video_localizer": video_hash,
        "naive": video_hash,
        "transcript_candidates": sha256_text(TRANSCRIPT_CANDIDATE_PROMPT),
        "transcript_only": sha256_text(TRANSCRIPT_ONLY_GOAL_PROMPT),
        "transcript_video": video_hash,
    }


def _schema_hashes() -> dict[str, str]:
    video_hash = sha256_json(GOAL_LIST_SCHEMA)
    return {
        "video_localizer": video_hash,
        "naive": video_hash,
        "transcript_candidates": sha256_json(CANDIDATE_LIST_SCHEMA),
        "transcript_only": sha256_json(TRANSCRIPT_GOAL_LIST_SCHEMA),
        "transcript_video": video_hash,
    }


def _prediction_prefix(root: Path) -> tuple[Path, dict[str, Any]]:
    directory = experiment_directory(root)
    return directory, _validate_prediction_prefix(directory)


def _validate_prediction_prefix(directory: Path) -> dict[str, Any]:
    config = load_shared_json_object(
        directory / PREDICTION_CONFIG_FILENAME, error_type=SoccerNetDataError
    )
    if (
        config.get("experiment") != EXPERIMENT_NAME
        or config.get("predecessor_experiment") != PREDECESSOR_EXPERIMENT_NAME
        or config.get("model") != EXPERIMENT_MODEL
        or config.get("planned_methods")
        != [NAIVE_STAGE, TRANSCRIPT_ONLY_STAGE, TRANSCRIPT_VIDEO_STAGE]
        or config.get("selected_games") != list(SELECTED_GAMES)
        or config.get("resolution") != RESOLUTION
        or config.get("candidate_window_radius_seconds") != WINDOW_RADIUS_SECONDS
        or config.get("prompt_sha256") != _prompt_hashes()
        or config.get("schema_sha256") != _schema_hashes()
        or config.get("materialization_contract") != MATERIALIZATION_CONTRACT
        or config.get("materialization_contract_sha256")
        != MATERIALIZATION_CONTRACT_SHA256
    ):
        raise SoccerNetDataError("Frozen prediction configuration changed")
    input_path = directory / INPUT_FILENAME
    if config.get("input_sha256") != sha256_file(input_path):
        raise SoccerNetDataError("Frozen prediction input changed")
    expected_plans = _rendered_plans(directory)
    expected_plan_hashes = {
        name: sha256_text(source) for name, source in expected_plans.items()
    }
    if config.get("plan_sha256") != expected_plan_hashes:
        raise SoccerNetDataError("Frozen logical or physical plans changed")
    control = config.get("controlled_video_localizer")
    if (
        not isinstance(control, Mapping)
        or control.get("same_prompt") is not True
        or control.get("same_schema") is not True
        or _prompt_hashes()["naive"] != _prompt_hashes()["transcript_video"]
        or _schema_hashes()["naive"] != _schema_hashes()["transcript_video"]
    ):
        raise SoccerNetDataError("Controlled video-localizer declaration changed")
    prediction_artifacts = _prediction_prefix_artifacts(directory)
    validate_completion(
        directory / PREDICTION_PREFIX_COMPLETION_FILENAME,
        stage_name="prediction_prefix",
        directory=directory,
        artifact_names=prediction_artifacts,
        dependencies=_source_dependencies(config),
        error_type=SoccerNetDataError,
    )
    rows = load_jsonl_objects(input_path, error_type=SoccerNetDataError)
    _reject_evaluator_keys(rows)
    _validate_input_rows(rows)
    return config


def _validate_evaluation_prefix(directory: Path) -> dict[str, Any]:
    config = load_shared_json_object(
        directory / EVALUATION_CONFIG_FILENAME, error_type=SoccerNetDataError
    )
    ground_truth_path = directory / GROUND_TRUTH_FILENAME
    if (
        config.get("experiment") != EXPERIMENT_NAME
        or config.get("ground_truth_path") != str(ground_truth_path.resolve())
        or config.get("ground_truth_sha256") != sha256_file(ground_truth_path)
        or config.get("evaluation_tolerances_seconds")
        != list(EVALUATION_TOLERANCES_SECONDS)
        or config.get("primary_tolerance_seconds") != PRIMARY_TOLERANCE_SECONDS
        or config.get("deduplication_seconds") != DEDUPLICATION_SECONDS
    ):
        raise SoccerNetDataError("Frozen evaluation configuration changed")
    ground_truth = load_shared_json_object(
        ground_truth_path, error_type=SoccerNetDataError
    )
    if ground_truth.get("experiment") != EXPERIMENT_NAME:
        raise SoccerNetDataError("Ground truth belongs to another experiment")
    prediction_config = load_shared_json_object(
        directory / PREDICTION_CONFIG_FILENAME, error_type=SoccerNetDataError
    )
    validate_completion(
        directory / PREPARE_COMPLETION_FILENAME,
        stage_name="prepare",
        directory=directory,
        artifact_names=(
            *_prediction_prefix_artifacts(directory),
            GROUND_TRUTH_FILENAME,
            EVALUATION_CONFIG_FILENAME,
            PREDICTION_PREFIX_COMPLETION_FILENAME,
        ),
        dependencies=_source_dependencies(prediction_config),
        error_type=SoccerNetDataError,
    )
    return ground_truth


def _validate_frozen_experiment(directory: Path) -> None:
    """Compatibility validator for the complete v4 prepared prefix."""
    _validate_prediction_prefix(directory)
    _validate_evaluation_prefix(directory)


def _source_dependencies(config: Mapping[str, Any]) -> dict[str, Path]:
    raw = config.get("source_dependency_paths")
    if not isinstance(raw, Mapping) or not raw:
        raise SoccerNetDataError("Prediction source dependencies are missing")
    dependencies: dict[str, Path] = {}
    for name, path_value in raw.items():
        if not isinstance(name, str) or not isinstance(path_value, str):
            raise SoccerNetDataError("Prediction source dependency is invalid")
        dependencies[name] = Path(path_value)
    return dependencies


def _prediction_prefix_artifacts(directory: Path) -> tuple[str, ...]:
    return (
        INPUT_FILENAME,
        PREDICTION_CONFIG_FILENAME,
        *(f"plans/{name}.py" for name in sorted(_rendered_plans(directory))),
    )


def _prediction_dependencies(directory: Path, plan_name: str) -> dict[str, Path]:
    return {
        "prediction_config": directory / PREDICTION_CONFIG_FILENAME,
        "input": directory / INPUT_FILENAME,
        f"plan:{plan_name}": directory / "plans" / f"{plan_name}.py",
    }


def _evaluation_dependencies(directory: Path) -> dict[str, Path]:
    return {
        "ground_truth": directory / GROUND_TRUTH_FILENAME,
        "evaluation_config": directory / EVALUATION_CONFIG_FILENAME,
        "naive_predictions": directory / "runs" / NAIVE_STAGE / "predictions.json",
        "transcript_only_predictions": (
            directory / "runs" / TRANSCRIPT_ONLY_STAGE / "predictions.json"
        ),
        "transcript_video_predictions": (
            directory / "runs" / TRANSCRIPT_VIDEO_STAGE / "predictions.json"
        ),
        "candidates": directory / "runs" / CANDIDATE_STAGE / "candidates.jsonl",
    }


def _validate_plan(config: Mapping[str, Any], plan_name: str, plan: Any) -> None:
    expected = config.get("plan_sha256")
    if not isinstance(expected, Mapping):
        raise SoccerNetDataError("Prediction plan hashes are invalid")
    if sha256_text(render_query(plan)) != expected.get(plan_name):
        raise SoccerNetDataError(f"Execution plan changed after preparation: {plan_name}")


def _validate_prediction_stage(
    directory: Path,
    stage_name: str,
    plan_name: str,
    input_path: Path,
    *,
    extra_dependencies: Mapping[str, Path] | None = None,
) -> None:
    dependencies = _prediction_dependencies(directory, plan_name)
    if extra_dependencies:
        dependencies.update(extra_dependencies)
    stage_directory = directory / "runs" / stage_name
    validate_completion(
        stage_directory / "completion.json",
        stage_name=stage_name,
        directory=stage_directory,
        artifact_names=PREDICTION_ARTIFACTS,
        dependencies=dependencies,
        error_type=SoccerNetDataError,
    )
    stage = load_shared_json_object(
        stage_directory / "stage.json", error_type=SoccerNetDataError
    )
    if stage.get("input_sha256") != sha256_file(input_path):
        raise SoccerNetDataError(f"Prediction stage input changed: {stage_name}")


def _validate_candidate_stage(directory: Path) -> None:
    stage_directory = directory / "runs" / CANDIDATE_STAGE
    validate_completion(
        stage_directory / "completion.json",
        stage_name=CANDIDATE_STAGE,
        directory=stage_directory,
        artifact_names=CANDIDATE_ARTIFACTS,
        dependencies=_prediction_dependencies(directory, "candidates"),
        error_type=SoccerNetDataError,
    )
    rows = load_jsonl_objects(
        stage_directory / "candidates.jsonl", error_type=SoccerNetDataError
    )
    _validate_candidate_rows(rows)


def _validate_materialization_stage(
    directory: Path,
    *,
    media_probe: MediaProbe = probe_materialized_clip,
) -> None:
    stage_directory = directory / "runs" / MATERIALIZATION_STAGE
    candidate_path = directory / "runs" / CANDIDATE_STAGE / "candidates.jsonl"
    validate_completion(
        stage_directory / "completion.json",
        stage_name=MATERIALIZATION_STAGE,
        directory=stage_directory,
        artifact_names=MATERIALIZATION_ARTIFACTS,
        dependencies={
            "candidates": candidate_path,
            "candidate_completion": (
                directory / "runs" / CANDIDATE_STAGE / "completion.json"
            ),
        },
        error_type=SoccerNetDataError,
    )
    result = validate_goal_materialized_clip_stage(
        stage_directory, media_probe=media_probe
    )
    _reject_evaluator_keys(result["rows"])


def _validate_input_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != len(SELECTED_GAMES) * 2:
        raise SoccerNetDataError("Prediction input must contain exactly six halves")
    seen: set[tuple[str, int]] = set()
    verified_hashes: dict[Path, str] = {}
    for row in rows:
        game_id = row.get("game_id")
        half = row.get("half")
        if (
            game_id not in SELECTED_GAMES
            or not isinstance(half, int)
            or isinstance(half, bool)
            or half not in (1, 2)
        ):
            raise SoccerNetDataError("Prediction input contains an unexpected half")
        key = (str(game_id), int(half))
        if key in seen:
            raise SoccerNetDataError(f"Prediction input contains duplicate half: {key}")
        seen.add(key)
        _positive_finite(row.get("duration_seconds"), "video duration")
        if row.get("window_radius_seconds") != WINDOW_RADIUS_SECONDS:
            raise SoccerNetDataError("Prediction input window radius changed")
        video = row.get("video")
        if not isinstance(video, Mapping) or video.get("type") != "Video":
            raise SoccerNetDataError("Prediction input video is invalid")
        path_value = video.get("path")
        expected_sha = video.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_sha, str):
            raise SoccerNetDataError("Prediction input video provenance is missing")
        path = Path(path_value)
        if not path.is_file():
            raise SoccerNetDataError(f"Prediction source video is missing: {path}")
        actual_sha = verified_hashes.get(path.resolve())
        if actual_sha is None:
            actual_sha = sha256_file(path)
            verified_hashes[path.resolve()] = actual_sha
        if actual_sha != expected_sha:
            raise SoccerNetDataError(f"Prediction source video changed: {path}")
        segments = row.get("transcript_segments")
        timestamped = row.get("timestamped_transcript")
        if not isinstance(segments, list) or not segments or not isinstance(timestamped, str):
            raise SoccerNetDataError("Prediction transcript fields are invalid")


def _validate_candidate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != len(SELECTED_GAMES) * 2:
        raise SoccerNetDataError("Candidate stage must contain exactly six halves")
    seen: set[tuple[str, int]] = set()
    for row in rows:
        game_id = row.get("game_id")
        half = row.get("half")
        if (
            game_id not in SELECTED_GAMES
            or not isinstance(half, int)
            or isinstance(half, bool)
            or half not in (1, 2)
        ):
            raise SoccerNetDataError("Candidate stage contains an unexpected half")
        key = (str(game_id), int(half))
        if key in seen:
            raise SoccerNetDataError(f"Candidate stage contains duplicate half: {key}")
        seen.add(key)
        if row.get("candidate_strategy") != "llm_high_recall":
            raise SoccerNetDataError("Candidate stage uses a non-paper strategy")
        duration = _positive_finite(row.get("duration_seconds"), "half duration")
        video = row.get("video")
        if not isinstance(video, Mapping) or not isinstance(video.get("sha256"), str):
            raise SoccerNetDataError("Candidate stage lacks video provenance")
        windows = row.get("candidate_windows")
        points = row.get("goal_candidates")
        if not isinstance(windows, list) or not isinstance(points, list):
            raise SoccerNetDataError("Candidate stage fields are invalid")
        previous_end = -1.0
        for index, window in enumerate(windows):
            if not isinstance(window, Mapping) or window.get("window_id") != index:
                raise SoccerNetDataError("Candidate window identifiers are invalid")
            start = _nonnegative_finite(window.get("start_seconds"), "window start")
            end = _positive_finite(window.get("end_seconds"), "window end")
            if not start < end <= duration or start <= previous_end:
                raise SoccerNetDataError("Candidate windows overlap or exceed the half")
            if not math.isclose(
                float(window.get("duration_seconds", -1)),
                end - start,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise SoccerNetDataError("Candidate window duration is inconsistent")
            previous_end = end
    _reject_evaluator_keys(rows)


def _extract_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "game_id": str(row["game_id"]),
        "half": int(row["half"]),
        "duration_seconds": float(row["duration_seconds"]),
        "window_radius_seconds": float(row["window_radius_seconds"]),
        "video": dict(row["video"]),
        "candidate_strategy": "llm_high_recall",
        "goal_candidates": row.get("goal_candidates", []),
        "candidate_windows": row.get("candidate_windows", []),
    }


def _extract_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    goal_field: str,
    method: str,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        goal = row.get(goal_field)
        if not isinstance(goal, Mapping):
            continue
        timestamp_valid = goal.get("timestamp_valid")
        if not isinstance(timestamp_valid, bool):
            raise SoccerNetDataError("Normalized goal lacks timestamp_valid")
        prediction = {
            "game_id": str(row["game_id"]),
            "half": int(row["half"]),
            "time_seconds": float(goal["time_seconds"]),
            "timestamp_valid": timestamp_valid,
            "timestamp_error": goal.get("timestamp_error"),
            "confidence": float(goal.get("confidence", 0.0)),
            "evidence": str(goal.get("evidence", "")),
            "method": method,
        }
        if "clip_minute" in goal:
            prediction["clip_minute"] = int(goal["clip_minute"])
            prediction["clip_second"] = float(goal["clip_second"])
        if "window_id" in goal:
            prediction["window_id"] = int(goal["window_id"])
        predictions.append(prediction)
    return predictions


def _reject_evaluator_keys(value: Any, *, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text.casefold() in FORBIDDEN_PREDICTION_KEYS:
                raise SoccerNetDataError(
                    f"Evaluator-only key in prediction artifact: {path}.{key_text}"
                )
            _reject_evaluator_keys(nested, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_evaluator_keys(nested, path=f"{path}[{index}]")


def _combined_latency_measurement(
    *stages: tuple[str, Mapping[str, Any]],
) -> dict[str, Any]:
    observed = 0.0
    cumulative_api_seconds = 0.0
    invalid: list[dict[str, Any]] = []
    cache_hits = 0
    failed_api_calls = 0
    failed_uploads = 0
    reused_clips = 0
    for stage_name, stage in stages:
        elapsed = _nonnegative_finite(stage.get("end_to_end_seconds"), "stage latency")
        observed += elapsed
        reasons: list[str] = []
        if stage.get("stage_type") == "local_candidate_clip_materialization":
            reused = int(stage.get("reused_clip_count", 0))
            failed_encodes = int(stage.get("failed_clip_attempt_count", 0))
            if min(reused, failed_encodes) < 0:
                raise SoccerNetDataError("Materialization attempt count is negative")
            reused_clips += reused
            if reused:
                reasons.append("reused_materialized_clips")
            if failed_encodes:
                reasons.append("failed_clip_encoding_attempts_recorded")
        else:
            usage = stage.get("api_usage")
            media = stage.get("media_uploads")
            if not isinstance(usage, Mapping) or not isinstance(media, Mapping):
                raise SoccerNetDataError(f"Stage accounting is invalid: {stage_name}")
            stage_cache_hits = int(stage.get("cache_hits", 0))
            stage_failed_api = int(usage.get("failed_api_call_count", 0))
            stage_failed_uploads = int(media.get("failed_upload_count", 0))
            if min(stage_cache_hits, stage_failed_api, stage_failed_uploads) < 0:
                raise SoccerNetDataError(f"Stage accounting is negative: {stage_name}")
            cache_hits += stage_cache_hits
            failed_api_calls += stage_failed_api
            failed_uploads += stage_failed_uploads
            cumulative_api_seconds += _nonnegative_finite(
                usage.get("api_elapsed_seconds", 0.0), "API latency"
            )
            reasons.extend(
                reason
                for condition, reason in (
                    (stage_cache_hits > 0, "cached_responses_used"),
                    (stage_failed_api > 0, "failed_api_attempts_recorded"),
                    (stage_failed_uploads > 0, "failed_media_uploads_recorded"),
                )
                if condition
            )
        if reasons:
            invalid.append({"stage": stage_name, "reasons": reasons})
    return {
        "valid": not invalid,
        "end_to_end_seconds": observed if not invalid else None,
        "observed_stage_seconds_sum": observed,
        "cumulative_api_elapsed_seconds": cumulative_api_seconds,
        "cache_hits": cache_hits,
        "failed_api_call_count": failed_api_calls,
        "failed_upload_count": failed_uploads,
        "reused_materialized_clip_count": reused_clips,
        "invalid_stages": invalid,
        "reason": (
            None
            if not invalid
            else "Cached responses, failed attempts, or reused clips invalidate cold-start latency."
        ),
    }


def _sum_usage(*values: Mapping[str, Any]) -> dict[str, Any]:
    count_fields = (
        "api_call_count",
        "successful_api_call_count",
        "failed_api_call_count",
        "prompt_token_count",
        "candidate_token_count",
        "thought_token_count",
        "total_token_count",
    )
    float_fields = ("api_elapsed_seconds", "estimated_cost_usd")
    result: dict[str, Any] = {
        field: sum(int(value.get(field, 0)) for value in values)
        for field in count_fields
    }
    result.update(
        {
            field: sum(float(value.get(field, 0.0)) for value in values)
            for field in float_fields
        }
    )
    result["cost_assumptions"] = values[0].get("cost_assumptions") if values else None
    return result


def _sum_media_uploads(*values: Mapping[str, Any]) -> dict[str, Any]:
    count_fields = (
        "media_reference_count",
        "upload_attempt_count",
        "successful_upload_count",
        "failed_upload_count",
        "reused_upload_count",
        "unique_uploaded_media_count",
        "uploaded_bytes",
    )
    result = {
        field: sum(int(value.get(field, 0)) for value in values)
        for field in count_fields
    }
    result["media_upload_elapsed_seconds"] = sum(
        float(value.get("media_upload_elapsed_seconds", 0.0)) for value in values
    )
    return result


def _comparison_method(
    evaluation: Mapping[str, Any],
    stage: Mapping[str, Any],
    usage: Mapping[str, Any],
    media: Mapping[str, Any],
    latency: Mapping[str, Any],
    primary_key: str,
) -> dict[str, Any]:
    return {
        "primary_accuracy": evaluation["metrics_by_tolerance_seconds"][primary_key],
        "api_usage": dict(usage),
        "media_uploads": dict(media),
        "latency": dict(latency),
        "final_model_stage_seconds": stage["end_to_end_seconds"],
    }


def _reductions(
    baseline_usage: Mapping[str, Any],
    method_usage: Mapping[str, Any],
    baseline_latency: Mapping[str, Any],
    method_latency: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "total_token_reduction_fraction": _reduction(
            baseline_usage.get("total_token_count", 0),
            method_usage.get("total_token_count", 0),
        ),
        "estimated_cost_reduction_fraction": _reduction(
            baseline_usage.get("estimated_cost_usd", 0.0),
            method_usage.get("estimated_cost_usd", 0.0),
        ),
        "cold_start_latency_reduction_fraction": None,
    }
    if baseline_latency.get("valid") and method_latency.get("valid"):
        result["cold_start_latency_reduction_fraction"] = _reduction(
            baseline_latency["end_to_end_seconds"],
            method_latency["end_to_end_seconds"],
        )
    return result


def _reduction(baseline: Any, value: Any) -> float | None:
    baseline_value = float(baseline)
    return 1.0 - float(value) / baseline_value if baseline_value > 0 else None


def _validated_stage_status(stage_directory: Path, validator: Any) -> str:
    completion = stage_directory / "completion.json"
    if completion.exists():
        try:
            validator()
            return "complete"
        except (SoccerNetDataError, OSError, ValueError):
            return "invalid"
    if stage_directory.exists() and any(stage_directory.rglob("*")):
        return "partial"
    return "not_started"


def _half_key(game_id: str, half: int) -> str:
    return f"{game_id}::half_{half}"


def _positive_finite(value: Any, label: str) -> float:
    result = _nonnegative_finite(value, label)
    if result <= 0:
        raise SoccerNetDataError(f"{label} must be positive")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise SoccerNetDataError(f"{label} must be finite and non-negative")
    return float(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "prepare",
        "status",
        "run-naive",
        "run-transcript-only",
        "run-candidates",
        "materialize-clips",
        "run-transcript-video",
        "evaluate",
        "compare",
    ):
        subparser = subparsers.add_parser(command)
        if command in {
            "run-naive",
            "run-transcript-only",
            "run-candidates",
            "materialize-clips",
            "run-transcript-video",
        }:
            subparser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = str(args.command)
    if command.startswith("run-") or command == "materialize-clips":
        if not args.execute:
            print(
                "Preview only. No model call or media encoding ran. "
                "Add --execute after reviewing the frozen plans and status."
            )
            return 0
    try:
        if command == "prepare":
            result = prepare_experiment(args.root)
        elif command == "status":
            result = status(args.root)
        elif command == "run-naive":
            result = run_naive(args.root, args.env_file)
        elif command == "run-transcript-only":
            result = run_transcript_only(args.root, args.env_file)
        elif command == "run-candidates":
            result = run_candidates(args.root, args.env_file)
        elif command == "materialize-clips":
            result = materialize_clips(args.root)
        elif command == "run-transcript-video":
            result = run_transcript_video(args.root, args.env_file)
        elif command == "evaluate":
            result = evaluate_results(args.root)
        elif command == "compare":
            result = compare_results(args.root)
        else:  # pragma: no cover - argparse enforces choices
            raise AssertionError(command)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (SoccerNetDataError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
