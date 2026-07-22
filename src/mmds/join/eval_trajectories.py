"""Soft-reference evaluation for cross-camera vehicle trajectories.

Compares UDF join outputs to an LLM (or other) reference set. This is **not**
oracle ground truth: the reference may hallucinate or miss vehicles. Metrics
are for relative quality / cost experiments.

Assumes ``entered`` / ``exited`` share a comparable numeric time base (for the
I24V 5s clips without ``recorded_at``, both pipelines emit clip-relative
seconds).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

Trajectory = dict[str, Any]
TimelineSegment = dict[str, Any]

# Soft attribute aliases so "car"≈"sedan" style vocab drift does not dominate.
_CLASS_ALIASES: dict[str, str] = {
    "car": "sedan",
    "automobile": "sedan",
    "pickup": "truck",
    "pickup truck": "truck",
    "semi": "truck",
    "semi-truck": "truck",
    "semi truck": "truck",
}

# Soft color families: labels in the same family count as a soft color match.
# Anything else only soft-matches itself (after normalize).
_COLOR_FAMILY_ALIASES: dict[str, str] = {
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
    """One greedy assignment between a prediction and a reference trajectory."""

    pred_index: int
    ref_index: int
    score: float
    timeline_score: float
    attribute_exact: float
    attribute_soft: float


@dataclass(frozen=True)
class EvalReport:
    """Aggregate soft-reference metrics for a prediction / reference pair."""

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
    """Wall-time and coarse cost proxies for one pipeline run.

    Token counts come from ``GeminiPromptExecutor.usage_snapshot()`` and are 0
    for pure-UDF pipelines that never call an LLM.
    """

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
    """Coerce Unnest-wrapped or bare trajectory rows into a common shape."""
    if not isinstance(row, dict) or not row:
        return None
    candidate = row
    nested = row.get("vehicles")
    if isinstance(nested, dict) and "timeline" in nested:
        candidate = nested
    timeline = candidate.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        return None
    vehicle_id = candidate.get("vehicle_id")
    attributes = candidate.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}
    return {
        "vehicle_id": str(vehicle_id) if vehicle_id is not None else "",
        "attributes": {
            "class": str(attributes.get("class", "") or ""),
            "color": str(attributes.get("color", "") or ""),
            "subtype": str(attributes.get("subtype", "") or ""),
        },
        "timeline": [dict(seg) for seg in timeline if isinstance(seg, dict)],
        "match_score": candidate.get("match_score"),
    }


def normalize_trajectory_rows(rows: list[Any]) -> list[Trajectory]:
    """Normalize a list of rows, dropping empties / malformed trajectories."""
    out: list[Trajectory] = []
    for row in rows:
        normalized = normalize_trajectory_row(row)
        if normalized is not None:
            out.append(normalized)
    return out


def normalize_label(value: Any) -> str:
    """Lowercase / strip a categorical attribute for soft comparison."""
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def canonicalize_class(value: Any) -> str:
    """Map common class synonyms onto a small canonical vocabulary."""
    label = normalize_label(value)
    return _CLASS_ALIASES.get(label, label)


def canonicalize_color(value: Any) -> str:
    """Map near-confusable colors onto soft families; others stay as normalized labels.

    Families:
    - ``cool_dark``: blue ≈ gray ≈ silver ≈ black
    - ``warm_red``: red ≈ brown
    """
    label = normalize_label(value)
    return _COLOR_FAMILY_ALIASES.get(label, label)


def colors_soft_match(pred_color: Any, ref_color: Any) -> bool:
    """True when colors are identical or in the same soft family (non-empty)."""
    pred_canon = canonicalize_color(pred_color)
    ref_canon = canonicalize_color(ref_color)
    return bool(pred_canon) and pred_canon == ref_canon


def interval_iou(a_entered: float, a_exited: float, b_entered: float, b_exited: float) -> float:
    """Intersection-over-union of two closed time intervals."""
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
    """True if enter **or** exit endpoints differ by at most ``max_endpoint_delta_sec``.

    Used as a hard gate before interval IoU: a shared camera whose enter *and*
    exit both drift beyond the tolerance contributes ``0.0`` to the timeline
    score even when the intervals still overlap.
    """
    return (
        abs(a_entered - b_entered) <= max_endpoint_delta_sec
        or abs(a_exited - b_exited) <= max_endpoint_delta_sec
    )


def _segment_by_camera(timeline: list[TimelineSegment]) -> dict[str, tuple[float, float]]:
    by_camera: dict[str, tuple[float, float]] = {}
    for segment in timeline:
        camera_id = segment.get("camera_id")
        entered = segment.get("entered")
        exited = segment.get("exited")
        if not isinstance(camera_id, str) or not camera_id:
            continue
        if not isinstance(entered, (int, float)) or not isinstance(exited, (int, float)):
            continue
        by_camera[camera_id] = (float(entered), float(exited))
    return by_camera


def timeline_overlap_score(
    pred: Trajectory,
    ref: Trajectory,
    *,
    max_endpoint_delta_sec: float | None = 1.0,
) -> float:
    """Mean per-camera interval IoU over the union of camera ids.

    Cameras present on only one side contribute ``0.0``, so missing / extra
    cameras lower the score.

    When ``max_endpoint_delta_sec`` is set (default ``1.0``), a shared camera
    also contributes ``0.0`` unless at least one endpoint is within that
    tolerance: ``|Δentered| ≤ δ`` **or** ``|Δexited| ≤ δ``. Pass ``None`` to
    disable the gate and keep pure IoU.
    """
    pred_by_cam = _segment_by_camera(list(pred.get("timeline") or []))
    ref_by_cam = _segment_by_camera(list(ref.get("timeline") or []))
    cameras = sorted(set(pred_by_cam) | set(ref_by_cam))
    if not cameras:
        return 0.0
    scores: list[float] = []
    for camera_id in cameras:
        if camera_id not in pred_by_cam or camera_id not in ref_by_cam:
            scores.append(0.0)
            continue
        a_entered, a_exited = pred_by_cam[camera_id]
        b_entered, b_exited = ref_by_cam[camera_id]
        if max_endpoint_delta_sec is not None and not endpoints_within_or_tolerance(
            a_entered,
            a_exited,
            b_entered,
            b_exited,
            max_endpoint_delta_sec=max_endpoint_delta_sec,
        ):
            scores.append(0.0)
            continue
        scores.append(interval_iou(a_entered, a_exited, b_entered, b_exited))
    return sum(scores) / len(scores)


def attribute_agreement(pred: Trajectory, ref: Trajectory) -> tuple[float, float]:
    """Return ``(exact_fraction, soft_fraction)`` over class/color/subtype.

    Soft mode:
    - ``class`` uses aliases (e.g. ``car``→``sedan``)
    - ``color`` uses soft families (``blue≈gray≈silver≈black``, ``red≈brown``);
      if color does **not** soft-match, the entire soft score is ``0.0``
    - ``subtype`` is case-folded exact match
    """
    pred_attrs = pred.get("attributes") if isinstance(pred.get("attributes"), dict) else {}
    ref_attrs = ref.get("attributes") if isinstance(ref.get("attributes"), dict) else {}
    keys = ("class", "color", "subtype")
    exact_hits = 0
    for key in keys:
        pred_raw = pred_attrs.get(key, "")
        ref_raw = ref_attrs.get(key, "")
        if normalize_label(pred_raw) == normalize_label(ref_raw) and normalize_label(pred_raw):
            exact_hits += 1
    n = float(len(keys))
    exact_fraction = exact_hits / n

    pred_color = pred_attrs.get("color", "")
    ref_color = ref_attrs.get("color", "")
    if not colors_soft_match(pred_color, ref_color):
        return exact_fraction, 0.0

    soft_hits = 0
    # Color already soft-matched; count it as a hit.
    soft_hits += 1
    pred_class = pred_attrs.get("class", "")
    ref_class = ref_attrs.get("class", "")
    if canonicalize_class(pred_class) == canonicalize_class(ref_class) and canonicalize_class(
        pred_class
    ):
        soft_hits += 1
    pred_subtype = pred_attrs.get("subtype", "")
    ref_subtype = ref_attrs.get("subtype", "")
    if normalize_label(pred_subtype) == normalize_label(ref_subtype) and normalize_label(
        pred_subtype
    ):
        soft_hits += 1
    return exact_fraction, soft_hits / n


def pair_score(
    pred: Trajectory,
    ref: Trajectory,
    *,
    timeline_weight: float = 0.7,
    attribute_weight: float = 0.3,
    max_endpoint_delta_sec: float | None = 1.0,
) -> tuple[float, float, float, float]:
    """Combined match score and component scores for one pred/ref pair."""
    if timeline_weight < 0 or attribute_weight < 0:
        raise ValueError("pair_score weights must be non-negative.")
    weight_sum = timeline_weight + attribute_weight
    if weight_sum <= 0:
        raise ValueError("pair_score requires a positive weight sum.")
    timeline = timeline_overlap_score(
        pred, ref, max_endpoint_delta_sec=max_endpoint_delta_sec
    )
    exact, soft = attribute_agreement(pred, ref)
    combined = (timeline_weight * timeline + attribute_weight * soft) / weight_sum
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
    """Greedy one-to-one matching by descending ``pair_score``."""
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
            if combined < float(min_score):
                continue
            candidates.append(
                TrajectoryMatch(
                    pred_index=pred_index,
                    ref_index=ref_index,
                    score=combined,
                    timeline_score=timeline,
                    attribute_exact=exact,
                    attribute_soft=soft,
                )
            )
    candidates.sort(key=lambda item: item.score, reverse=True)
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    matches: list[TrajectoryMatch] = []
    for candidate in candidates:
        if candidate.pred_index in used_pred or candidate.ref_index in used_ref:
            continue
        used_pred.add(candidate.pred_index)
        used_ref.add(candidate.ref_index)
        matches.append(candidate)
    return matches


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def evaluate_trajectories(
    predictions: list[Any],
    references: list[Any],
    *,
    min_score: float = 0.7,
    timeline_weight: float = 0.7,
    attribute_weight: float = 0.3,
    max_endpoint_delta_sec: float | None = 1.0,
) -> EvalReport:
    """Soft-reference precision / recall / F1 plus attribute agreement."""
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
    tp = len(matches)
    fp = len(pred_rows) - tp
    fn = len(ref_rows) - tp
    precision = _safe_div(float(tp), float(tp + fp))
    recall = _safe_div(float(tp), float(tp + fn))
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    if matches:
        mean_timeline = sum(match.timeline_score for match in matches) / tp
        mean_exact = sum(match.attribute_exact for match in matches) / tp
        mean_soft = sum(match.attribute_soft for match in matches) / tp
    else:
        mean_timeline = 0.0
        mean_exact = 0.0
        mean_soft = 0.0
    return EvalReport(
        n_pred=len(pred_rows),
        n_ref=len(ref_rows),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_timeline_score=mean_timeline,
        mean_attribute_exact=mean_exact,
        mean_attribute_soft=mean_soft,
        matches=tuple(matches),
    )
