"""Evaluation and cost metrics for SoccerNet goal-timestamp predictions."""

from __future__ import annotations

import functools
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from .common import SoccerNetDataError


EVALUATION_TOLERANCES_SECONDS = (5.0, 10.0, 30.0, 60.0)
PRIMARY_TOLERANCE_SECONDS = 30.0
DEDUPLICATION_SECONDS = 5.0

# Standard paid-tier token prices documented for Gemini 3.1 Flash-Lite when
# this experiment was frozen. The report labels this as an estimate because
# modality-specific audio accounting may differ from the aggregate token fields.
INPUT_USD_PER_MILLION_TOKENS = 0.25
OUTPUT_USD_PER_MILLION_TOKENS = 1.50


def deduplicate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    threshold_seconds: float = DEDUPLICATION_SECONDS,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        game_id = prediction.get("game_id")
        half = prediction.get("half")
        time_seconds = prediction.get("time_seconds")
        if not isinstance(game_id, str) or not isinstance(half, int):
            continue
        if not isinstance(time_seconds, Real) or float(time_seconds) < 0:
            continue
        normalized = dict(prediction)
        normalized["time_seconds"] = float(time_seconds)
        grouped[(game_id, half)].append(normalized)

    deduplicated: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda item: (
                not _timestamp_valid(item),
                -_confidence(item),
                item["time_seconds"],
            ),
        )
        kept: list[dict[str, Any]] = []
        for prediction in ranked:
            if any(
                abs(prediction["time_seconds"] - existing["time_seconds"])
                <= threshold_seconds
                for existing in kept
            ):
                continue
            kept.append(prediction)
        deduplicated.extend(sorted(kept, key=lambda item: item["time_seconds"]))
    return deduplicated


def _confidence(prediction: Mapping[str, Any]) -> float:
    value = prediction.get("confidence")
    return float(value) if isinstance(value, Real) else 0.0


def _timestamp_valid(prediction: Mapping[str, Any]) -> bool:
    """Treat legacy predictions as valid and honor explicit v2 validation flags."""
    return prediction.get("timestamp_valid") is not False


def match_timestamps(
    predictions: Sequence[float],
    truths: Sequence[float],
    tolerance_seconds: float,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, minimum-error one-to-one timestamp matching."""
    predicted = tuple(float(value) for value in predictions)
    actual = tuple(float(value) for value in truths)
    if len(actual) > 20:
        raise ValueError("Timestamp matcher supports at most 20 truth events per half")

    @functools.lru_cache(maxsize=None)
    def solve(prediction_index: int, truth_mask: int) -> tuple[int, float, tuple]:
        if prediction_index == len(predicted):
            return (0, 0.0, ())

        best = solve(prediction_index + 1, truth_mask)
        for truth_index, truth in enumerate(actual):
            bit = 1 << truth_index
            if truth_mask & bit:
                continue
            error = abs(predicted[prediction_index] - truth)
            if error > tolerance_seconds:
                continue
            tail_matches, tail_error, tail_pairs = solve(
                prediction_index + 1,
                truth_mask | bit,
            )
            candidate = (
                tail_matches + 1,
                tail_error + error,
                ((prediction_index, truth_index, error),) + tail_pairs,
            )
            if _is_better_match(candidate, best):
                best = candidate
        return best

    return list(solve(0, 0)[2])


def _is_better_match(candidate: tuple, current: tuple) -> bool:
    return candidate[0] > current[0] or (
        candidate[0] == current[0] and candidate[1] < current[1]
    )


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, Any],
    *,
    tolerances: Sequence[float] = EVALUATION_TOLERANCES_SECONDS,
) -> dict[str, Any]:
    deduplicated = deduplicate_predictions(predictions)
    predictions_by_half: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in deduplicated:
        predictions_by_half[(prediction["game_id"], prediction["half"])].append(prediction)
    truth_by_half = _ground_truth_by_half(ground_truth)
    total_invalid_prediction_count = sum(
        not _timestamp_valid(prediction) for prediction in deduplicated
    )

    metrics: dict[str, Any] = {}
    for tolerance in tolerances:
        true_positives = false_positives = false_negatives = 0
        errors: list[float] = []
        per_half: list[dict[str, Any]] = []
        all_keys = sorted(set(truth_by_half) | set(predictions_by_half))
        for game_id, half in all_keys:
            half_predictions = predictions_by_half.get((game_id, half), [])
            valid_predictions = [
                item for item in half_predictions if _timestamp_valid(item)
            ]
            invalid_prediction_count = len(half_predictions) - len(valid_predictions)
            half_truths = truth_by_half.get((game_id, half), [])
            matches = match_timestamps(
                [item["time_seconds"] for item in valid_predictions],
                half_truths,
                tolerance,
            )
            tp = len(matches)
            fp = len(valid_predictions) - tp + invalid_prediction_count
            fn = len(half_truths) - tp
            true_positives += tp
            false_positives += fp
            false_negatives += fn
            errors.extend(pair[2] for pair in matches)
            per_half.append(
                {
                    "game_id": game_id,
                    "half": half,
                    "truth_count": len(half_truths),
                    "prediction_count": len(half_predictions),
                    "invalid_prediction_count": invalid_prediction_count,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "matched_absolute_errors_seconds": [pair[2] for pair in matches],
                }
            )
        precision = _safe_ratio(true_positives, true_positives + false_positives)
        recall = _safe_ratio(true_positives, true_positives + false_negatives)
        metrics[str(int(tolerance))] = {
            "tolerance_seconds": tolerance,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "invalid_prediction_count": total_invalid_prediction_count,
            "precision": precision,
            "recall": recall,
            "f1": _safe_ratio(2 * precision * recall, precision + recall),
            "mean_absolute_error_seconds": _mean(errors),
            "median_absolute_error_seconds": _median(errors),
            "per_half": per_half,
        }

    return {
        "raw_prediction_count": len(predictions),
        "deduplicated_prediction_count": len(deduplicated),
        "invalid_prediction_count": total_invalid_prediction_count,
        "deduplication_threshold_seconds": DEDUPLICATION_SECONDS,
        "primary_tolerance_seconds": PRIMARY_TOLERANCE_SECONDS,
        "metrics_by_tolerance_seconds": metrics,
        "deduplicated_predictions": deduplicated,
    }


def evaluate_candidate_windows(
    candidate_rows: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, Any],
) -> dict[str, Any]:
    truth_by_half = _ground_truth_by_half(ground_truth)
    covered_truths = total_truths = 0
    candidate_duration = full_duration = 0.0
    window_count = 0
    per_half: list[dict[str, Any]] = []
    rows_by_half = {
        (str(row["game_id"]), int(row["half"])): row for row in candidate_rows
    }
    for key in sorted(truth_by_half):
        row = rows_by_half.get(key)
        if row is None:
            raise SoccerNetDataError(f"Missing candidate row for {key[0]} half {key[1]}")
        windows = row.get("candidate_windows")
        duration = row.get("duration_seconds")
        if not isinstance(windows, list) or not isinstance(duration, Real):
            raise SoccerNetDataError(f"Invalid candidate row for {key[0]} half {key[1]}")
        truths = truth_by_half[key]
        covered = sum(_is_covered(truth, windows) for truth in truths)
        half_candidate_duration = sum(
            float(window["end_seconds"]) - float(window["start_seconds"])
            for window in windows
        )
        total_truths += len(truths)
        covered_truths += covered
        candidate_duration += half_candidate_duration
        full_duration += float(duration)
        window_count += len(windows)
        per_half.append(
            {
                "game_id": key[0],
                "half": key[1],
                "truth_count": len(truths),
                "covered_truth_count": covered,
                "candidate_window_count": len(windows),
                "candidate_duration_seconds": half_candidate_duration,
                "full_duration_seconds": float(duration),
                "selectivity": _safe_ratio(half_candidate_duration, float(duration)),
            }
        )
    return {
        "goal_count": total_truths,
        "covered_goal_count": covered_truths,
        "candidate_recall": _safe_ratio(covered_truths, total_truths),
        "candidate_window_count": window_count,
        "candidate_duration_seconds": candidate_duration,
        "full_video_duration_seconds": full_duration,
        "selectivity": _safe_ratio(candidate_duration, full_duration),
        "pruned_fraction": 1.0 - _safe_ratio(candidate_duration, full_duration),
        "per_half": per_half,
    }


def aggregate_api_usage(path: Path) -> dict[str, Any]:
    records = []
    if path.is_file():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SoccerNetDataError(
                    f"Invalid API usage JSON on line {line_number} of {path}"
                ) from exc
    successful = [record for record in records if record.get("status") == "ok"]
    prompt_tokens = sum(_usage_count(record, "prompt_token_count") for record in successful)
    candidate_tokens = sum(
        _usage_count(record, "candidates_token_count") for record in successful
    )
    thought_tokens = sum(_usage_count(record, "thoughts_token_count") for record in successful)
    estimated_cost = (
        prompt_tokens * INPUT_USD_PER_MILLION_TOKENS
        + (candidate_tokens + thought_tokens) * OUTPUT_USD_PER_MILLION_TOKENS
    ) / 1_000_000
    return {
        "api_call_count": len(records),
        "successful_api_call_count": len(successful),
        "failed_api_call_count": len(records) - len(successful),
        "api_elapsed_seconds": sum(float(record["elapsed_seconds"]) for record in records),
        "prompt_token_count": prompt_tokens,
        "candidate_token_count": candidate_tokens,
        "thought_token_count": thought_tokens,
        "total_token_count": sum(
            _usage_count(record, "total_token_count") for record in successful
        ),
        "estimated_cost_usd": estimated_cost,
        "cost_assumptions": {
            "input_usd_per_million_tokens": INPUT_USD_PER_MILLION_TOKENS,
            "output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION_TOKENS,
            "warning": (
                "Estimate uses aggregate token fields; modality-specific audio pricing "
                "may differ. Raw usage records are authoritative."
            ),
        },
    }


def _usage_count(record: Mapping[str, Any], field: str) -> int:
    usage = record.get("usage_metadata") or {}
    if not isinstance(usage, Mapping):
        return 0
    value = usage.get(field)
    if value is None:
        camel = field.split("_")[0] + "".join(
            word.capitalize() for word in field.split("_")[1:]
        )
        value = usage.get(camel)
    return int(value) if isinstance(value, Real) else 0


def _ground_truth_by_half(
    ground_truth: Mapping[str, Any],
) -> dict[tuple[str, int], list[float]]:
    result: dict[tuple[str, int], list[float]] = {}
    for game in ground_truth.get("games", []):
        game_id = game.get("game_id")
        halves = game.get("halves")
        if not isinstance(game_id, str) or not isinstance(halves, Mapping):
            raise SoccerNetDataError("Invalid ground-truth game record")
        for half_string in ("1", "2"):
            goals = halves.get(half_string)
            if not isinstance(goals, list):
                raise SoccerNetDataError(f"Missing goals for {game_id} half {half_string}")
            result[(game_id, int(half_string))] = [float(goal["time_seconds"]) for goal in goals]
    return result


def _is_covered(truth: float, windows: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        float(window["start_seconds"]) <= truth <= float(window["end_seconds"])
        for window in windows
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2
