"""Usage, media-transfer, latency, and comparison accounting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.experiments.common import ExperimentDataError, load_json_object, sha256_file
from scripts.experiments.gemini_runtime import aggregate_media_uploads
from scripts.experiments.usage import aggregate_api_usage
from scripts.lectures.transcribe import TRANSCRIPT_DIRECTORY

from .catalog import (
    AUDIO_INPUT_USD_PER_MILLION_TOKENS,
    INPUT_USD_PER_MILLION_TOKENS,
    MODEL,
    OUTPUT_USD_PER_MILLION_TOKENS,
    PRICING_SOURCE,
    WHISPER_MODEL,
)


def stage_summary(
    stage_directory: Path,
    *,
    elapsed: float,
    executor: Any,
    stage_name: str,
    plan_name: str,
    input_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    usage = aggregate_api_usage(
        stage_directory / "api_calls.jsonl",
        input_usd_per_million_tokens=INPUT_USD_PER_MILLION_TOKENS,
        audio_input_usd_per_million_tokens=AUDIO_INPUT_USD_PER_MILLION_TOKENS,
        output_usd_per_million_tokens=OUTPUT_USD_PER_MILLION_TOKENS,
        pricing_source=PRICING_SOURCE,
    )
    media = aggregate_media_uploads(stage_directory / "media_uploads.jsonl")
    cache_hits = int(getattr(executor, "cache_hits", 0))
    cold_valid = (
        cache_hits == 0
        and usage["failed_api_call_count"] == 0
        and media["failed_upload_count"] == 0
    )
    return {
        "schema_version": 1,
        "stage_name": stage_name,
        "model": MODEL,
        "max_in_flight_provider_calls": 1,
        "end_to_end_seconds": elapsed,
        "cache_hits": cache_hits,
        "cache_misses": int(getattr(executor, "cache_misses", 0)),
        "cold_start_latency_valid": cold_valid,
        "cold_start_invalid_reasons": [
            reason
            for condition, reason in (
                (cache_hits > 0, "cached_responses_used"),
                (usage["failed_api_call_count"] > 0, "failed_api_attempts_recorded"),
                (media["failed_upload_count"] > 0, "failed_media_uploads_recorded"),
            )
            if condition
        ],
        "input_sha256": sha256_file(input_path),
        "plan_sha256": config["plan_sha256"][plan_name],
        "api_usage": usage,
        "media_uploads": media,
    }


def comparison_method(
    evaluation: Mapping[str, Any],
    usage: Mapping[str, Any],
    latency: Mapping[str, Any],
    media_uploads: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "primary_accuracy": evaluation["primary_metrics"],
        "pair_presence": evaluation["pair_presence"]["metrics"],
        "api_usage": dict(usage),
        "media_uploads": dict(media_uploads),
        "latency": dict(latency),
    }


def combined_latency(*stages: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
    observed = 0.0
    invalid: list[dict[str, Any]] = []
    for name, stage in stages:
        elapsed = nonnegative_finite(stage.get("end_to_end_seconds"), "stage latency")
        observed += elapsed
        reasons: list[str] = []
        if stage.get("stage_type") == "local_candidate_clip_materialization":
            if int(stage.get("reused_clip_count", 0)):
                reasons.append("reused_materialized_clips")
        elif not stage.get("cold_start_latency_valid", False):
            raw = stage.get("cold_start_invalid_reasons")
            reasons.extend(str(value) for value in raw if isinstance(raw, list))
        if reasons:
            invalid.append({"stage": name, "reasons": reasons})
    return {
        "valid": not invalid,
        "end_to_end_seconds": observed if not invalid else None,
        "observed_stage_seconds_sum": observed,
        "invalid_stages": invalid,
        "reason": (
            None
            if not invalid
            else "Cached responses, failed attempts, or reused clips invalidate cold-start latency."
        ),
    }


def sum_usage(*values: Mapping[str, Any]) -> dict[str, Any]:
    count_fields = (
        "api_call_count",
        "successful_api_call_count",
        "failed_api_call_count",
        "prompt_token_count",
        "candidate_token_count",
        "thought_token_count",
        "total_token_count",
    )
    float_fields = (
        "api_elapsed_seconds",
        "estimated_input_cost_usd",
        "estimated_output_cost_usd",
        "estimated_cost_usd",
    )
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
    modalities: dict[str, int] = {}
    for value in values:
        breakdown = value.get("input_tokens_by_modality")
        if isinstance(breakdown, Mapping):
            for modality, count in breakdown.items():
                key = str(modality)
                modalities[key] = modalities.get(key, 0) + int(count)
    result["input_tokens_by_modality"] = modalities
    result["modality_breakdown_complete"] = all(
        bool(value.get("modality_breakdown_complete", False)) for value in values
    )
    result["cost_assumptions"] = values[0].get("cost_assumptions") if values else None
    return result


def sum_media_uploads(*values: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "media_reference_count",
        "upload_attempt_count",
        "successful_upload_count",
        "failed_upload_count",
        "reused_upload_count",
        "unique_uploaded_media_count",
        "uploaded_bytes",
    )
    result = {
        field: sum(int(value.get(field, 0)) for value in values) for field in fields
    }
    result["media_upload_elapsed_seconds"] = sum(
        float(value.get("media_upload_elapsed_seconds", 0.0)) for value in values
    )
    return result


def reductions(
    naive_usage: Mapping[str, Any],
    method_usage: Mapping[str, Any],
    naive_latency: Mapping[str, Any],
    method_latency: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "input_token_reduction_fraction": reduction(
            float(naive_usage.get("prompt_token_count", 0)),
            float(method_usage.get("prompt_token_count", 0)),
        ),
        "total_token_reduction_fraction": reduction(
            float(naive_usage.get("total_token_count", 0)),
            float(method_usage.get("total_token_count", 0)),
        ),
        "estimated_cost_reduction_fraction": reduction(
            float(naive_usage.get("estimated_cost_usd", 0)),
            float(method_usage.get("estimated_cost_usd", 0)),
        ),
        "cold_start_latency_reduction_fraction": None,
    }
    if naive_latency.get("valid") and method_latency.get("valid"):
        result["cold_start_latency_reduction_fraction"] = reduction(
            float(naive_latency["end_to_end_seconds"]),
            float(method_latency["end_to_end_seconds"]),
        )
    return result


def reduction(baseline: float, value: float) -> float | None:
    return 1.0 - value / baseline if baseline > 0 else None


def whisper_workload(root: Path) -> dict[str, Any]:
    index = load_json_object(root / TRANSCRIPT_DIRECTORY / "index.json")
    entries = index.get("lectures")
    if not isinstance(entries, list):
        raise ExperimentDataError("Transcript index is invalid")
    return {
        "model": WHISPER_MODEL,
        "lecture_count": len(entries),
        "elapsed_seconds": sum(
            nonnegative_finite(entry.get("elapsed_seconds"), "Whisper elapsed time")
            for entry in entries
            if isinstance(entry, Mapping)
        ),
        "included_in_query_time": False,
    }


def nonnegative_finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ExperimentDataError(f"{label} must be finite and non-negative")
    return float(value)
