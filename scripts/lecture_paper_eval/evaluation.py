"""Label-owning evaluator for immutable lecture experiment profiles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from scripts.experiments.common import ExperimentDataError
from scripts.lectures.evaluation import (
    deduplicate_predictions,
    evaluate_candidate_windows as _evaluate_candidate_windows,
    evaluate_predictions as _evaluate_predictions,
)

from .annotations import annotations_for
from .catalog import (
    DEFAULT_PROFILE,
    DEDUPLICATION_TIOU_THRESHOLD,
    EvaluationProfile,
    PRIMARY_TIOU_THRESHOLD,
    SCHEMA_VERSION,
    TIOU_THRESHOLDS,
)


def build_ground_truth(
    videos: Mapping[str, Mapping[str, Any]],
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Bind approved human labels to validated physical video durations."""
    annotations = annotations_for(profile)
    return _build_ground_truth(
        videos,
        profile=profile,
        annotation_version=annotations.version,
        annotator=annotations.annotator,
        annotation_policy=annotations.policy,
        positive_intervals=annotations.positive_intervals,
    )


def build_superseded_ground_truth(
    videos: Mapping[str, Mapping[str, Any]],
    *,
    profile: EvaluationProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Reconstruct annotation v1 solely to validate the amendment base."""
    annotations = annotations_for(profile)
    if (
        annotations.superseded_version is None
        or annotations.superseded_positive_intervals is None
    ):
        raise ExperimentDataError(
            f"Profile {profile.profile_id} has no superseded annotation revision"
        )
    return _build_ground_truth(
        videos,
        profile=profile,
        annotation_version=annotations.superseded_version,
        annotator=annotations.annotator,
        annotation_policy=annotations.policy,
        positive_intervals=annotations.superseded_positive_intervals,
    )


def _build_ground_truth(
    videos: Mapping[str, Mapping[str, Any]],
    *,
    profile: EvaluationProfile,
    annotation_version: int,
    annotator: str,
    annotation_policy: str,
    positive_intervals: Mapping[
        tuple[str, str], Sequence[tuple[float, float]]
    ],
) -> dict[str, Any]:
    lecture_ids = {lecture.lecture_id for lecture in profile.lectures}
    query_ids = {query.query_id for query in profile.queries}
    if set(videos) != lecture_ids:
        raise ExperimentDataError(
            "Ground-truth video set does not match the evaluation profile"
        )
    if not set(positive_intervals).issubset(set(profile.pairs)):
        raise ExperimentDataError("Human annotations contain an unknown lecture-query pair")
    if any(not events for events in positive_intervals.values()):
        raise ExperimentDataError("Positive annotation pairs must contain an event")

    pairs: list[dict[str, Any]] = []
    event_count = 0
    positive_pair_count = 0
    for lecture_id, query_id in profile.pairs:
        if lecture_id not in lecture_ids or query_id not in query_ids:
            raise ExperimentDataError("Evaluation profile contains an unknown pair")
        duration = _positive_finite(
            videos[lecture_id].get("duration_seconds"),
            f"duration for {lecture_id}",
        )
        key = (lecture_id, query_id)
        raw_events = positive_intervals.get(key, ())
        events: list[dict[str, Any]] = []
        previous_end = -math.inf
        for index, (start, end) in enumerate(raw_events):
            start_value = _nonnegative_finite(start, "ground-truth start")
            end_value = _nonnegative_finite(end, "ground-truth end")
            if start_value >= end_value:
                raise ExperimentDataError(f"Ground-truth interval is reversed: {key}")
            if start_value < previous_end:
                raise ExperimentDataError(f"Ground-truth intervals overlap: {key}")
            if end_value > duration:
                raise ExperimentDataError(
                    f"Ground-truth interval exceeds video duration: {key}"
                )
            events.append(
                {
                    "event_id": f"{lecture_id}:{query_id}:{index}",
                    "start_seconds": start_value,
                    "end_seconds": end_value,
                }
            )
            previous_end = end_value
        positive_pair_count += int(bool(events))
        event_count += len(events)
        pairs.append(
            {
                "lecture_id": lecture_id,
                "query_id": query_id,
                "duration_seconds": duration,
                "annotation": "positive" if events else "negative",
                "events": events,
            }
        )

    expected_pair_count = len(profile.pairs)
    if (
        len(pairs) != expected_pair_count
        or len({(p["lecture_id"], p["query_id"]) for p in pairs})
        != expected_pair_count
    ):
        raise ExperimentDataError("Ground truth pair cardinality changed")
    expected_event_count = sum(len(events) for events in positive_intervals.values())
    if (
        positive_pair_count != len(positive_intervals)
        or event_count != expected_event_count
    ):
        raise ExperimentDataError("Ground-truth positive cardinalities changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": profile.experiment_name,
        "annotation_version": annotation_version,
        "annotator": annotator,
        "annotation_policy": annotation_policy,
        "pair_count": len(pairs),
        "positive_pair_count": positive_pair_count,
        "negative_pair_count": len(pairs) - positive_pair_count,
        "event_count": event_count,
        "pairs": pairs,
    }


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the shared matcher and add event boundaries and pair-presence diagnostics."""
    result = _evaluate_predictions(
        predictions,
        ground_truth,
        thresholds=TIOU_THRESHOLDS,
        primary_threshold=PRIMARY_TIOU_THRESHOLD,
        deduplication_threshold=DEDUPLICATION_TIOU_THRESHOLD,
    )
    truth_by_key = _truth_by_key(ground_truth)
    grouped = {key: [] for key in truth_by_key}
    for prediction in predictions:
        key = _pair_key(prediction)
        if key not in grouped:
            raise ExperimentDataError(f"Prediction has unknown pair: {key}")
        grouped[key].append(dict(prediction))
    deduplicated = {
        key: deduplicate_predictions(
            values, threshold=DEDUPLICATION_TIOU_THRESHOLD
        )
        for key, values in grouped.items()
    }

    presence_tp = presence_fp = presence_fn = presence_tn = 0
    presence_pairs: list[dict[str, Any]] = []
    for key, truth in truth_by_key.items():
        truth_present = bool(truth["events"])
        predicted_present = bool(deduplicated[key])
        if truth_present and predicted_present:
            presence_tp += 1
        elif predicted_present:
            presence_fp += 1
        elif truth_present:
            presence_fn += 1
        else:
            presence_tn += 1
        presence_pairs.append(
            {
                "lecture_id": key[0],
                "query_id": key[1],
                "truth_present": truth_present,
                "predicted_present": predicted_present,
                "correct": truth_present == predicted_present,
            }
        )

    for pair_results in result["per_pair_by_tiou"].values():
        for pair_result in pair_results:
            key = (pair_result["lecture_id"], pair_result["query_id"])
            predictions_for_pair = deduplicated[key]
            truths_for_pair = truth_by_key[key]["events"]
            enriched_matches = []
            for match in pair_result["matches"]:
                prediction = predictions_for_pair[match["prediction_index"]]
                truth = truths_for_pair[match["truth_index"]]
                start_error = float(prediction["start_seconds"]) - float(
                    truth["start_seconds"]
                )
                end_error = float(prediction["end_seconds"]) - float(
                    truth["end_seconds"]
                )
                enriched_matches.append(
                    {
                        **match,
                        "truth_event_id": truth.get("event_id"),
                        "prediction_start_seconds": prediction["start_seconds"],
                        "prediction_end_seconds": prediction["end_seconds"],
                        "truth_start_seconds": truth["start_seconds"],
                        "truth_end_seconds": truth["end_seconds"],
                        "start_error_seconds": start_error,
                        "end_error_seconds": end_error,
                        "absolute_start_error_seconds": abs(start_error),
                        "absolute_end_error_seconds": abs(end_error),
                        "boundary_mae_seconds": (
                            abs(start_error) + abs(end_error)
                        )
                        / 2,
                    }
                )
            pair_result["matches"] = enriched_matches
            pair_result["truth_present"] = bool(truths_for_pair)
            pair_result["predicted_present"] = bool(predictions_for_pair)
            pair_result["presence_correct"] = (
                pair_result["truth_present"] == pair_result["predicted_present"]
            )
            pair_result["invalid_prediction_count"] = sum(
                not item.get("interval_valid", False) for item in predictions_for_pair
            )

    presence_precision = _safe_precision(presence_tp, presence_fp, presence_fn)
    presence_recall = presence_tp / (presence_tp + presence_fn) if presence_tp + presence_fn else 1.0
    presence_f1 = (
        2 * presence_precision * presence_recall / (presence_precision + presence_recall)
        if presence_precision + presence_recall
        else 0.0
    )
    result["pair_presence"] = {
        "metrics": {
            "true_positives": presence_tp,
            "false_positives": presence_fp,
            "false_negatives": presence_fn,
            "true_negatives": presence_tn,
            "precision": presence_precision,
            "recall": presence_recall,
            "f1": presence_f1,
            "accuracy": (presence_tp + presence_tn) / len(truth_by_key),
        },
        "per_pair": presence_pairs,
    }
    return result


def evaluate_candidate_windows(
    rows: Sequence[Mapping[str, Any]], ground_truth: Mapping[str, Any]
) -> dict[str, Any]:
    result = _evaluate_candidate_windows(rows, ground_truth)
    coverages = [
        coverage
        for pair in result["per_pair"]
        for coverage in pair["truth_event_coverages"]
    ]
    result.update(
        {
            "candidate_recall_at_any_coverage": (
                sum(value > 0 for value in coverages) / len(coverages)
                if coverages
                else 1.0
            ),
            "mean_truth_event_coverage": (
                sum(coverages) / len(coverages) if coverages else 1.0
            ),
            "raw_candidate_range_count": sum(
                _nonnegative_count(row.get("raw_candidate_range_count"), "raw candidate count")
                for row in rows
            ),
            "valid_candidate_range_count": sum(
                _nonnegative_count(row.get("valid_candidate_range_count"), "valid candidate count")
                for row in rows
            ),
            "invalid_candidate_range_count": sum(
                _nonnegative_count(row.get("invalid_candidate_range_count"), "invalid candidate count")
                for row in rows
            ),
            "duplicate_candidate_range_count": sum(
                _nonnegative_count(row.get("duplicate_candidate_range_count"), "duplicate candidate count")
                for row in rows
            ),
        }
    )
    return result


def _truth_by_key(ground_truth: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    pairs = ground_truth.get("pairs")
    pair_count = ground_truth.get("pair_count")
    if (
        not isinstance(pair_count, int)
        or isinstance(pair_count, bool)
        or pair_count <= 0
        or not isinstance(pairs, list)
        or len(pairs) != pair_count
    ):
        raise ExperimentDataError("Ground-truth pair cardinality is invalid")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ExperimentDataError("Ground-truth pair must be an object")
        key = _pair_key(pair)
        if key in result:
            raise ExperimentDataError(f"Duplicate ground-truth pair: {key}")
        events = pair.get("events")
        if not isinstance(events, list):
            raise ExperimentDataError(f"Ground-truth events must be a list: {key}")
        result[key] = dict(pair)
    return result


def _pair_key(value: Mapping[str, Any]) -> tuple[str, str]:
    lecture_id = value.get("lecture_id")
    query_id = value.get("query_id")
    if not isinstance(lecture_id, str) or not lecture_id:
        raise ExperimentDataError("lecture_id must be a non-empty string")
    if not isinstance(query_id, str) or not query_id:
        raise ExperimentDataError("query_id must be a non-empty string")
    return lecture_id, query_id


def _safe_precision(tp: int, fp: int, fn: int) -> float:
    return tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)


def _nonnegative_count(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ExperimentDataError(f"{label} must be a non-negative integer")
    return value


def _positive_finite(value: Any, label: str) -> float:
    result = _nonnegative_finite(value, label)
    if result <= 0:
        raise ExperimentDataError(f"{label} must be positive")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ExperimentDataError(f"{label} must be finite and non-negative")
    return float(value)
