"""Soft-reference evaluation helpers for cross-camera trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

Trajectory = dict[str, Any]
TimelineSegment = dict[str, Any]

_CLASS_ALIASES = {
    "car": "sedan",
    "automobile": "sedan",
    "pickup": "truck",
    "pickup truck": "truck",
    "semi": "truck",
    "semi-truck": "truck",
    "semi truck": "truck",
}
_COLOR_FAMILY_ALIASES = {
    "blue": "cool_dark",
    "gray": "cool_dark",
    "grey": "cool_dark",
    "silver": "cool_dark",
    "black": "cool_dark",
    "red": "warm_red",
    "brown": "warm_red",
}


@dataclass(frozen=True)
class TrajectoryMatch:
    pred_index: int
    ref_index: int
    score: float
    timeline_score: float
    attribute_exact: float
    attribute_soft: float


@dataclass(frozen=True)
class EvalReport:
    n_pred: int
    n_ref: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_timeline_score: float
    mean_attribute_exact: float
    mean_attribute_soft: float
    matches: tuple[TrajectoryMatch, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matches"] = [asdict(match) for match in self.matches]
        return payload


@dataclass(frozen=True)
class CostReport:
    label: str
    wall_time_sec: float
    prompt_calls: int
    n_trajectories: int
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_trajectory_row(row: Any) -> Trajectory | None:
    if not isinstance(row, dict) or not row:
        return None
    candidate = row
    nested = row.get("vehicles")
    if isinstance(nested, dict) and "timeline" in nested:
        candidate = nested
    timeline = candidate.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        return None
    attributes = candidate.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}
    vehicle_id = candidate.get("vehicle_id")
    return {
        "vehicle_id": str(vehicle_id) if vehicle_id is not None else "",
        "attributes": {
            key: str(attributes.get(key, "") or "")
            for key in ("class", "color", "subtype")
        },
        "timeline": [dict(segment) for segment in timeline if isinstance(segment, dict)],
        "match_score": candidate.get("match_score"),
    }


def normalize_trajectory_rows(rows: list[Any]) -> list[Trajectory]:
    return [
        normalized
        for row in rows
        if (normalized := normalize_trajectory_row(row)) is not None
    ]


def normalize_label(value: Any) -> str:
    return "" if value is None else " ".join(str(value).strip().lower().split())


def canonicalize_class(value: Any) -> str:
    label = normalize_label(value)
    return _CLASS_ALIASES.get(label, label)


def canonicalize_color(value: Any) -> str:
    label = normalize_label(value)
    return _COLOR_FAMILY_ALIASES.get(label, label)


def colors_soft_match(pred_color: Any, ref_color: Any) -> bool:
    pred = canonicalize_color(pred_color)
    return bool(pred) and pred == canonicalize_color(ref_color)


def interval_iou(
    a_entered: float,
    a_exited: float,
    b_entered: float,
    b_exited: float,
) -> float:
    if a_exited < a_entered or b_exited < b_entered:
        return 0.0
    overlap = max(0.0, min(a_exited, b_exited) - max(a_entered, b_entered))
    union = max(a_exited, b_exited) - min(a_entered, b_entered)
    if union <= 0.0:
        return 1.0 if overlap > 0.0 or a_entered == b_entered else 0.0
    return overlap / union


def endpoints_within_or_tolerance(
    a_entered: float,
    a_exited: float,
    b_entered: float,
    b_exited: float,
    *,
    max_endpoint_delta_sec: float,
) -> bool:
    return (
        abs(a_entered - b_entered) <= max_endpoint_delta_sec
        or abs(a_exited - b_exited) <= max_endpoint_delta_sec
    )


def _segment_by_camera(
    timeline: list[TimelineSegment],
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for segment in timeline:
        camera_id = segment.get("camera_id")
        entered = segment.get("entered")
        exited = segment.get("exited")
        if (
            isinstance(camera_id, str)
            and camera_id
            and isinstance(entered, (int, float))
            and isinstance(exited, (int, float))
        ):
            result[camera_id] = (float(entered), float(exited))
    return result


def timeline_overlap_score(
    pred: Trajectory,
    ref: Trajectory,
    *,
    max_endpoint_delta_sec: float | None = 1.0,
) -> float:
    pred_by_camera = _segment_by_camera(list(pred.get("timeline") or []))
    ref_by_camera = _segment_by_camera(list(ref.get("timeline") or []))
    cameras = sorted(set(pred_by_camera) | set(ref_by_camera))
    if not cameras:
        return 0.0
    scores: list[float] = []
    for camera_id in cameras:
        if camera_id not in pred_by_camera or camera_id not in ref_by_camera:
            scores.append(0.0)
            continue
        a_entered, a_exited = pred_by_camera[camera_id]
        b_entered, b_exited = ref_by_camera[camera_id]
        if max_endpoint_delta_sec is not None and not endpoints_within_or_tolerance(
            a_entered,
            a_exited,
            b_entered,
            b_exited,
            max_endpoint_delta_sec=max_endpoint_delta_sec,
        ):
            scores.append(0.0)
        else:
            scores.append(interval_iou(a_entered, a_exited, b_entered, b_exited))
    return sum(scores) / len(scores)


def attribute_agreement(pred: Trajectory, ref: Trajectory) -> tuple[float, float]:
    pred_attrs = pred.get("attributes") if isinstance(pred.get("attributes"), dict) else {}
    ref_attrs = ref.get("attributes") if isinstance(ref.get("attributes"), dict) else {}
    keys = ("class", "color", "subtype")
    exact = sum(
        1
        for key in keys
        if normalize_label(pred_attrs.get(key)) == normalize_label(ref_attrs.get(key))
        and normalize_label(pred_attrs.get(key))
    ) / 3.0
    if not colors_soft_match(pred_attrs.get("color"), ref_attrs.get("color")):
        return exact, 0.0
    soft_hits = 1
    if (
        canonicalize_class(pred_attrs.get("class"))
        == canonicalize_class(ref_attrs.get("class"))
        and canonicalize_class(pred_attrs.get("class"))
    ):
        soft_hits += 1
    if (
        normalize_label(pred_attrs.get("subtype"))
        == normalize_label(ref_attrs.get("subtype"))
        and normalize_label(pred_attrs.get("subtype"))
    ):
        soft_hits += 1
    return exact, soft_hits / 3.0


def pair_score(
    pred: Trajectory,
    ref: Trajectory,
    *,
    timeline_weight: float = 0.7,
    attribute_weight: float = 0.3,
    max_endpoint_delta_sec: float | None = 1.0,
) -> tuple[float, float, float, float]:
    if timeline_weight < 0 or attribute_weight < 0:
        raise ValueError("pair_score weights must be non-negative.")
    total_weight = timeline_weight + attribute_weight
    if total_weight <= 0:
        raise ValueError("pair_score requires a positive weight sum.")
    timeline = timeline_overlap_score(
        pred, ref, max_endpoint_delta_sec=max_endpoint_delta_sec
    )
    exact, soft = attribute_agreement(pred, ref)
    combined = (timeline_weight * timeline + attribute_weight * soft) / total_weight
    return combined, timeline, exact, soft


def greedy_match_trajectories(
    predictions: list[Trajectory],
    references: list[Trajectory],
    *,
    min_score: float = 0.7,
    timeline_weight: float = 0.7,
    attribute_weight: float = 0.3,
    max_endpoint_delta_sec: float | None = 1.0,
) -> list[TrajectoryMatch]:
    candidates: list[TrajectoryMatch] = []
    for pred_index, pred in enumerate(predictions):
        for ref_index, ref in enumerate(references):
            combined, timeline, exact, soft = pair_score(
                pred,
                ref,
                timeline_weight=timeline_weight,
                attribute_weight=attribute_weight,
                max_endpoint_delta_sec=max_endpoint_delta_sec,
            )
            if combined >= float(min_score):
                candidates.append(
                    TrajectoryMatch(
                        pred_index,
                        ref_index,
                        combined,
                        timeline,
                        exact,
                        soft,
                    )
                )
    candidates.sort(key=lambda item: item.score, reverse=True)
    used_predictions: set[int] = set()
    used_references: set[int] = set()
    matches: list[TrajectoryMatch] = []
    for candidate in candidates:
        if (
            candidate.pred_index in used_predictions
            or candidate.ref_index in used_references
        ):
            continue
        used_predictions.add(candidate.pred_index)
        used_references.add(candidate.ref_index)
        matches.append(candidate)
    return matches


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def evaluate_trajectories(
    predictions: list[Any],
    references: list[Any],
    *,
    min_score: float = 0.7,
    timeline_weight: float = 0.7,
    attribute_weight: float = 0.3,
    max_endpoint_delta_sec: float | None = 1.0,
) -> EvalReport:
    pred_rows = normalize_trajectory_rows(predictions)
    ref_rows = normalize_trajectory_rows(references)
    matches = greedy_match_trajectories(
        pred_rows,
        ref_rows,
        min_score=min_score,
        timeline_weight=timeline_weight,
        attribute_weight=attribute_weight,
        max_endpoint_delta_sec=max_endpoint_delta_sec,
    )
    true_positives = len(matches)
    false_positives = len(pred_rows) - true_positives
    false_negatives = len(ref_rows) - true_positives
    precision = _safe_div(true_positives, true_positives + false_positives)
    recall = _safe_div(true_positives, true_positives + false_negatives)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    divisor = true_positives or 1
    return EvalReport(
        n_pred=len(pred_rows),
        n_ref=len(ref_rows),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_timeline_score=sum(match.timeline_score for match in matches) / divisor,
        mean_attribute_exact=sum(match.attribute_exact for match in matches) / divisor,
        mean_attribute_soft=sum(match.attribute_soft for match in matches) / divisor,
        matches=tuple(matches),
    )


__all__ = [
    "CostReport",
    "EvalReport",
    "TrajectoryMatch",
    "attribute_agreement",
    "canonicalize_class",
    "canonicalize_color",
    "colors_soft_match",
    "endpoints_within_or_tolerance",
    "evaluate_trajectories",
    "greedy_match_trajectories",
    "interval_iou",
    "normalize_label",
    "normalize_trajectory_row",
    "normalize_trajectory_rows",
    "pair_score",
    "timeline_overlap_score",
]
