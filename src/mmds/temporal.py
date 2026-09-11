"""Canonical deterministic temporal functions used by VRA queries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from .model import (
    BuiltinSpec,
    MMDSValidationError,
    PadIntervalSpec,
    ReconcileIntervalsSpec,
    Row,
)


def execute_builtin(spec: BuiltinSpec, payload: Any) -> Mapping[str, Any]:
    """Execute one validated deterministic function specification."""
    if isinstance(spec, PadIntervalSpec):
        return _pad_interval(spec, payload)
    if isinstance(spec, ReconcileIntervalsSpec):
        return _reconcile_intervals(spec, payload)
    raise MMDSValidationError(
        f"Unsupported built-in deterministic function {type(spec).__name__}."
    )


def _pad_interval(spec: PadIntervalSpec, payload: Any) -> Mapping[str, Any]:
    row = _row(payload, "PadInterval")
    interval = row.get(spec.interval_field)
    if not isinstance(interval, Mapping):
        raise MMDSValidationError(
            f"PadInterval field {spec.interval_field!r} must contain an interval object."
        )
    start = _nonnegative_finite(
        interval.get(spec.input_start_field),
        f"PadInterval {spec.input_start_field}",
    )
    end = _nonnegative_finite(
        interval.get(spec.input_end_field),
        f"PadInterval {spec.input_end_field}",
    )
    duration = _positive_finite(
        row.get(spec.duration_field), f"PadInterval {spec.duration_field}"
    )
    if start >= end:
        raise MMDSValidationError("PadInterval requires start < end.")
    if end > duration:
        raise MMDSValidationError(
            "PadInterval input interval extends beyond the source duration."
        )
    return {
        spec.output_start_field: max(0.0, start - float(spec.padding_seconds)),
        spec.output_end_field: min(duration, end + float(spec.padding_seconds)),
    }


def _reconcile_intervals(
    spec: ReconcileIntervalsSpec, payload: Any
) -> Mapping[str, Any]:
    if not isinstance(payload, list) or not payload:
        raise MMDSValidationError(
            "ReconcileIntervals requires a non-empty Reduce group."
        )
    rows = [_row(value, "ReconcileIntervals") for value in payload]
    output: dict[str, Any] = {}
    first = rows[0]
    for field in spec.preserve_fields:
        if field not in first:
            raise MMDSValidationError(
                f"ReconcileIntervals preserve field {field!r} is missing."
            )
        value = first[field]
        if any(field not in row or row[field] != value for row in rows[1:]):
            raise MMDSValidationError(
                f"ReconcileIntervals preserve field {field!r} is not constant within the group."
            )
        output[field] = value

    translated: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        window_start = _nonnegative_finite(
            row.get(spec.window_start_field),
            f"ReconcileIntervals row {row_index} window start",
        )
        window_end = _nonnegative_finite(
            row.get(spec.window_end_field),
            f"ReconcileIntervals row {row_index} window end",
        )
        if window_start >= window_end:
            raise MMDSValidationError(
                "ReconcileIntervals requires every window to satisfy start < end."
            )
        events = row.get(spec.events_field)
        if not isinstance(events, list):
            raise MMDSValidationError(
                f"ReconcileIntervals field {spec.events_field!r} must contain a list."
            )
        clip_duration = window_end - window_start
        for event_index, event in enumerate(events):
            if not isinstance(event, Mapping):
                raise MMDSValidationError(
                    f"ReconcileIntervals event {event_index} in row {row_index} must be an object."
                )
            local_start = _nonnegative_finite(
                event.get(spec.event_start_field),
                f"ReconcileIntervals event {event_index} start",
            )
            local_end = _nonnegative_finite(
                event.get(spec.event_end_field),
                f"ReconcileIntervals event {event_index} end",
            )
            if local_start >= local_end:
                raise MMDSValidationError(
                    "ReconcileIntervals requires event start < event end."
                )
            if local_end > clip_duration:
                raise MMDSValidationError(
                    "ReconcileIntervals event extends beyond its materialized clip."
                )
            translated.append(
                {
                    **dict(event),
                    spec.event_start_field: window_start + local_start,
                    spec.event_end_field: window_start + local_end,
                    "clip_start_seconds": window_start,
                    "clip_end_seconds": window_end,
                    "timestamp_source": "materialized_clip_relative",
                }
            )

    output[spec.output_field] = _deduplicate_events(translated, spec)
    return output


def _deduplicate_events(
    events: list[dict[str, Any]], spec: ReconcileIntervalsSpec
) -> list[dict[str, Any]]:
    def priority(event: Mapping[str, Any]) -> tuple[float, float, float]:
        confidence = event.get("confidence", 0.0)
        numeric_confidence = (
            float(confidence)
            if isinstance(confidence, Real)
            and not isinstance(confidence, bool)
            and math.isfinite(float(confidence))
            else 0.0
        )
        return (
            -numeric_confidence,
            float(event[spec.event_start_field]),
            float(event[spec.event_end_field]),
        )

    retained: list[dict[str, Any]] = []
    for event in sorted(events, key=priority):
        if any(
            _temporal_iou(event, existing, spec) >= spec.deduplication_tiou_threshold
            for existing in retained
        ):
            continue
        retained.append(event)
    retained.sort(
        key=lambda event: (
            float(event[spec.event_start_field]),
            float(event[spec.event_end_field]),
        )
    )
    return retained


def _temporal_iou(
    left: Mapping[str, Any], right: Mapping[str, Any], spec: ReconcileIntervalsSpec
) -> float:
    left_start = float(left[spec.event_start_field])
    left_end = float(left[spec.event_end_field])
    right_start = float(right[spec.event_start_field])
    right_end = float(right[spec.event_end_field])
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return intersection / union if union > 0 else 0.0


def _row(value: Any, label: str) -> Row:
    if not isinstance(value, Mapping):
        raise MMDSValidationError(f"{label} requires mapping-like row values.")
    return dict(value)


def _nonnegative_finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise MMDSValidationError(f"{label} must be a finite non-negative number.")
    return float(value)


def _positive_finite(value: Any, label: str) -> float:
    number = _nonnegative_finite(value, label)
    if number <= 0:
        raise MMDSValidationError(f"{label} must be positive.")
    return number


__all__ = ["execute_builtin"]
