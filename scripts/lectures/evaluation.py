"""Label-isolated interval and candidate-window evaluation for lectures."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from numbers import Real
from typing import Any

from scripts.experiments.common import ExperimentDataError

from .catalog import (
    DEDUPLICATION_TIOU_THRESHOLD,
    PRIMARY_TIOU_THRESHOLD,
    TIOU_THRESHOLDS,
)


def temporal_iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    """Return temporal intersection-over-union for two valid half-open intervals."""
    first_start, first_end = _valid_boundaries(first, "first interval")
    second_start, second_end = _valid_boundaries(second, "second interval")
    intersection = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / union if union > 0 else 0.0


def deduplicate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    threshold: float = DEDUPLICATION_TIOU_THRESHOLD,
) -> list[dict[str, Any]]:
    """Greedily suppress highly overlapping valid predictions.

    Invalid predictions are intentionally retained: silently dropping them would
    hide contract violations and inflate precision.
    """
    _unit_interval(threshold, "deduplication threshold")
    invalid = [dict(item) for item in predictions if not item.get("interval_valid", False)]
    valid = [dict(item) for item in predictions if item.get("interval_valid", False)]
    valid.sort(
        key=lambda item: (
            -_confidence(item),
            _boundary_for_sort(item, "start_seconds"),
            _boundary_for_sort(item, "end_seconds"),
        )
    )
    kept: list[dict[str, Any]] = []
    for prediction in valid:
        _valid_boundaries(prediction, "valid prediction")
        if any(temporal_iou(prediction, existing) >= threshold for existing in kept):
            continue
        kept.append(prediction)
    kept.sort(
        key=lambda item: (
            _boundary_for_sort(item, "start_seconds"),
            _boundary_for_sort(item, "end_seconds"),
        )
    )
    return kept + invalid


def match_intervals(
    predictions: Sequence[Mapping[str, Any]],
    truths: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Find a one-to-one matching, maximizing cardinality then total tIoU."""
    _unit_interval(threshold, "tIoU threshold")
    valid_predictions = [
        (index, item)
        for index, item in enumerate(predictions)
        if item.get("interval_valid", False)
    ]
    for _, prediction in valid_predictions:
        _valid_boundaries(prediction, "valid prediction")
    for truth in truths:
        _valid_boundaries(truth, "ground-truth interval")

    overlaps = [
        [temporal_iou(prediction, truth) for truth in truths]
        for _, prediction in valid_predictions
    ]

    @lru_cache(maxsize=None)
    def solve(prediction_index: int, used_truth_mask: int) -> tuple[int, float, tuple[tuple[int, int, float], ...]]:
        if prediction_index == len(valid_predictions):
            return 0, 0.0, ()
        best = solve(prediction_index + 1, used_truth_mask)
        original_prediction_index = valid_predictions[prediction_index][0]
        for truth_index, overlap in enumerate(overlaps[prediction_index]):
            if overlap < threshold or used_truth_mask & (1 << truth_index):
                continue
            count, total, pairs = solve(
                prediction_index + 1, used_truth_mask | (1 << truth_index)
            )
            candidate = (
                count + 1,
                total + overlap,
                ((original_prediction_index, truth_index, overlap),) + pairs,
            )
            if _matching_score(candidate) > _matching_score(best):
                best = candidate
        return best

    return [
        {"prediction_index": prediction, "truth_index": truth, "tiou": overlap}
        for prediction, truth, overlap in solve(0, 0)[2]
    ]


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, Any],
    *,
    thresholds: Sequence[float] = TIOU_THRESHOLDS,
    primary_threshold: float | None = None,
    deduplication_threshold: float = DEDUPLICATION_TIOU_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate predictions over every frozen lecture-query pair, including negatives."""
    truth_pairs = _truth_pairs(ground_truth)
    grouped = _group_predictions(predictions, truth_pairs)
    normalized_thresholds = tuple(float(value) for value in thresholds)
    for threshold in normalized_thresholds:
        _unit_interval(threshold, "tIoU threshold")
    normalized_primary = float(
        PRIMARY_TIOU_THRESHOLD if primary_threshold is None else primary_threshold
    )
    _unit_interval(normalized_primary, "primary tIoU threshold")
    if primary_threshold is not None and normalized_primary not in normalized_thresholds:
        raise ExperimentDataError(
            "Primary tIoU threshold must be present in evaluation thresholds"
        )

    metrics: dict[str, Any] = {}
    per_pair_by_threshold: dict[str, list[dict[str, Any]]] = {}
    deduplicated_by_pair = {
        key: deduplicate_predictions(values, threshold=deduplication_threshold)
        for key, values in grouped.items()
    }
    for threshold in normalized_thresholds:
        pair_results: list[dict[str, Any]] = []
        total_tp = total_fp = total_fn = exact_pairs = 0
        for key, truth_pair in truth_pairs.items():
            pair_predictions = deduplicated_by_pair[key]
            truths = truth_pair["events"]
            matches = match_intervals(pair_predictions, truths, threshold=threshold)
            tp = len(matches)
            fp = len(pair_predictions) - tp
            fn = len(truths) - tp
            exact = fp == 0 and fn == 0
            total_tp += tp
            total_fp += fp
            total_fn += fn
            exact_pairs += int(exact)
            pair_results.append(
                {
                    "lecture_id": key[0],
                    "query_id": key[1],
                    "ground_truth_count": len(truths),
                    "prediction_count_before_deduplication": len(grouped[key]),
                    "prediction_count": len(pair_predictions),
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "exact_pair": exact,
                    "matches": matches,
                }
            )
        key_name = _threshold_key(threshold)
        metrics[key_name] = _classification_metrics(
            total_tp, total_fp, total_fn, exact_pairs, len(truth_pairs)
        )
        per_pair_by_threshold[key_name] = pair_results

    primary_key = _threshold_key(normalized_primary)
    return {
        "tiou_thresholds": list(normalized_thresholds),
        "primary_tiou_threshold": normalized_primary,
        "deduplication_tiou_threshold": deduplication_threshold,
        "metrics_by_tiou": metrics,
        "per_pair_by_tiou": per_pair_by_threshold,
        "primary_metrics": metrics.get(primary_key),
        "pair_count": len(truth_pairs),
    }


def evaluate_binary_decisions(
    decisions: Sequence[Mapping[str, Any]], ground_truth: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate pair-level event presence, OR-ing multiple candidate decisions."""
    truth_pairs = _truth_pairs(ground_truth)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {
        key: [] for key in truth_pairs
    }
    seen_decisions: set[tuple[str, str, int | None]] = set()
    for decision in decisions:
        key = _pair_key(decision)
        if key not in grouped:
            raise ExperimentDataError(f"Binary decision has unknown pair: {key}")
        event_present = decision.get("event_present")
        if not isinstance(event_present, bool):
            raise ExperimentDataError(
                "Binary decision event_present must be boolean"
            )
        raw_window_id = decision.get("window_id")
        if raw_window_id is not None and (
            not isinstance(raw_window_id, int)
            or isinstance(raw_window_id, bool)
            or raw_window_id < 0
        ):
            raise ExperimentDataError(
                "Binary decision window_id must be a non-negative integer or null"
            )
        identity = (key[0], key[1], raw_window_id)
        if identity in seen_decisions:
            raise ExperimentDataError(f"Duplicate binary decision: {identity}")
        seen_decisions.add(identity)
        grouped[key].append(dict(decision))

    true_positives = false_positives = false_negatives = true_negatives = 0
    per_pair: list[dict[str, Any]] = []
    for key, truth_pair in truth_pairs.items():
        pair_decisions = grouped[key]
        predicted_present = any(
            decision["event_present"] for decision in pair_decisions
        )
        truth_present = bool(truth_pair["events"])
        if predicted_present and truth_present:
            true_positives += 1
        elif predicted_present:
            false_positives += 1
        elif truth_present:
            false_negatives += 1
        else:
            true_negatives += 1
        per_pair.append(
            {
                "lecture_id": key[0],
                "query_id": key[1],
                "truth_present": truth_present,
                "predicted_present": predicted_present,
                "correct": predicted_present == truth_present,
                "decision_count": len(pair_decisions),
                "positive_decision_count": sum(
                    decision["event_present"] for decision in pair_decisions
                ),
            }
        )
    pair_count = len(truth_pairs)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else (1.0 if false_negatives == 0 else 0.0)
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 1.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "pair_count": pair_count,
        "metrics": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (true_positives + true_negatives) / pair_count,
        },
        "per_pair": per_pair,
    }


def evaluate_candidate_windows(
    rows: Sequence[Mapping[str, Any]], ground_truth: Mapping[str, Any]
) -> dict[str, Any]:
    """Measure transcript-pushdown selectivity and truth-event coverage."""
    truth_pairs = _truth_pairs(ground_truth)
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _pair_key(row)
        if key not in truth_pairs:
            raise ExperimentDataError(f"Candidate row has unknown pair: {key}")
        if key in by_key:
            raise ExperimentDataError(f"Duplicate candidate row for pair: {key}")
        by_key[key] = row
    if set(by_key) != set(truth_pairs):
        missing = sorted(set(truth_pairs) - set(by_key))
        raise ExperimentDataError(f"Candidate rows are incomplete; missing: {missing}")

    total_duration = total_selected = 0.0
    total_truth_events = covered_half = covered_fully = 0
    per_pair: list[dict[str, Any]] = []
    for key, truth_pair in truth_pairs.items():
        duration = _positive_number(truth_pair.get("duration_seconds"), "duration_seconds")
        windows = _merged_windows(by_key[key].get("candidate_windows"), duration)
        selected = sum(end - start for start, end in windows)
        coverages = [
            _event_coverage(event, windows) for event in truth_pair.get("events", [])
        ]
        half_count = sum(value >= 0.5 for value in coverages)
        full_count = sum(math.isclose(value, 1.0, abs_tol=1e-9) for value in coverages)
        total_duration += duration
        total_selected += selected
        total_truth_events += len(coverages)
        covered_half += half_count
        covered_fully += full_count
        per_pair.append(
            {
                "lecture_id": key[0],
                "query_id": key[1],
                "video_duration_seconds": duration,
                "candidate_duration_seconds": selected,
                "candidate_selectivity": selected / duration,
                "candidate_window_count": len(windows),
                "truth_event_coverages": coverages,
                "truth_events_covered_at_least_half": half_count,
                "truth_events_fully_covered": full_count,
            }
        )
    return {
        "pair_count": len(truth_pairs),
        "total_video_duration_seconds": total_duration,
        "total_candidate_duration_seconds": total_selected,
        "candidate_selectivity": total_selected / total_duration,
        "duration_reduction_fraction": 1.0 - total_selected / total_duration,
        "ground_truth_event_count": total_truth_events,
        "candidate_recall_at_50_percent_coverage": (
            covered_half / total_truth_events if total_truth_events else 1.0
        ),
        "candidate_recall_at_full_coverage": (
            covered_fully / total_truth_events if total_truth_events else 1.0
        ),
        "per_pair": per_pair,
    }


def _truth_pairs(ground_truth: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    pairs = ground_truth.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ExperimentDataError("Ground truth must contain a non-empty pairs list")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ExperimentDataError("Ground-truth pair must be an object")
        key = _pair_key(pair)
        if key in result:
            raise ExperimentDataError(f"Duplicate ground-truth pair: {key}")
        duration = _positive_number(pair.get("duration_seconds"), "duration_seconds")
        events = pair.get("events")
        if not isinstance(events, list):
            raise ExperimentDataError(f"Ground-truth events must be a list for {key}")
        normalized_events: list[dict[str, float]] = []
        for event in events:
            start, end = _valid_boundaries(event, f"ground-truth interval for {key}")
            if end > duration:
                raise ExperimentDataError(f"Ground-truth interval exceeds duration for {key}")
            normalized_events.append({"start_seconds": start, "end_seconds": end})
        result[key] = {**dict(pair), "duration_seconds": duration, "events": normalized_events}
    return result


def _group_predictions(
    predictions: Sequence[Mapping[str, Any]],
    truth_pairs: Mapping[tuple[str, str], Any],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped = {key: [] for key in truth_pairs}
    for prediction in predictions:
        key = _pair_key(prediction)
        if key not in grouped:
            raise ExperimentDataError(f"Prediction has unknown pair: {key}")
        interval_valid = prediction.get("interval_valid")
        if not isinstance(interval_valid, bool):
            raise ExperimentDataError("Prediction interval_valid must be boolean")
        grouped[key].append(dict(prediction))
    return grouped


def _pair_key(value: Mapping[str, Any]) -> tuple[str, str]:
    lecture_id = value.get("lecture_id")
    query_id = value.get("query_id")
    if not isinstance(lecture_id, str) or not lecture_id:
        raise ExperimentDataError("lecture_id must be a non-empty string")
    if not isinstance(query_id, str) or not query_id:
        raise ExperimentDataError("query_id must be a non-empty string")
    return lecture_id, query_id


def _classification_metrics(tp: int, fp: int, fn: int, exact: int, pairs: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_pair_accuracy": exact / pairs,
        "exact_pair_count": exact,
    }


def _matching_score(value: tuple[int, float, Any]) -> tuple[int, float]:
    return value[0], round(value[1], 12)


def _merged_windows(value: Any, duration: float) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        raise ExperimentDataError("candidate_windows must be a list")
    raw: list[tuple[float, float]] = []
    for window in value:
        start, end = _valid_boundaries(window, "candidate window")
        if end > duration:
            raise ExperimentDataError("Candidate window exceeds video duration")
        raw.append((start, end))
    raw.sort()
    merged: list[tuple[float, float]] = []
    for start, end in raw:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _event_coverage(event: Mapping[str, Any], windows: Sequence[tuple[float, float]]) -> float:
    start, end = _valid_boundaries(event, "ground-truth interval")
    overlap = sum(max(0.0, min(end, right) - max(start, left)) for left, right in windows)
    return min(1.0, overlap / (end - start))


def _valid_boundaries(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, Mapping):
        raise ExperimentDataError(f"{label} must be an object")
    start = _finite_number(value.get("start_seconds"), f"{label} start_seconds")
    end = _finite_number(value.get("end_seconds"), f"{label} end_seconds")
    if start < 0 or end <= start:
        raise ExperimentDataError(f"{label} must satisfy 0 <= start_seconds < end_seconds")
    return start, end


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ExperimentDataError(f"{label} must be a finite number")
    return float(value)


def _positive_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0:
        raise ExperimentDataError(f"{label} must be positive")
    return result


def _unit_interval(value: float, label: str) -> None:
    if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ExperimentDataError(f"{label} must be between zero and one")


def _confidence(value: Mapping[str, Any]) -> float:
    confidence = value.get("confidence", 0.0)
    if not isinstance(confidence, Real) or isinstance(confidence, bool) or not math.isfinite(float(confidence)):
        return 0.0
    return float(confidence)


def _boundary_for_sort(value: Mapping[str, Any], field: str) -> float:
    boundary = value.get(field)
    return float(boundary) if isinstance(boundary, Real) and not isinstance(boundary, bool) else math.inf


def _threshold_key(value: float) -> str:
    return f"{float(value):g}"
