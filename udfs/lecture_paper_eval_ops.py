"""Strict deterministic UDF contracts for the four-lecture paper evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


def normalize_full_video_events(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize zero-origin full-video output without repairing bad intervals."""
    duration = _positive_finite(row.get("duration_seconds"), "duration_seconds")
    return {
        "events": _normalize_video_events(
            row.get("events"),
            absolute_start=0.0,
            supplied_duration=duration,
            window_id=None,
        )
    }


def normalize_candidate_video_events(row: dict[str, Any]) -> dict[str, Any]:
    """Translate zero-origin candidate-clip output to the lecture timeline."""
    window = row.get("candidate_windows")
    if not isinstance(window, Mapping):
        raise ValueError("candidate_windows must be an object")
    window_id = _nonnegative_integer(window.get("window_id"), "window_id")
    start = _nonnegative_finite(window.get("start_seconds"), "candidate start_seconds")
    end = _nonnegative_finite(window.get("end_seconds"), "candidate end_seconds")
    if start >= end:
        raise ValueError("candidate window must satisfy start_seconds < end_seconds")
    return {
        "events": _normalize_video_events(
            row.get("events"),
            absolute_start=start,
            supplied_duration=end - start,
            window_id=window_id,
        )
    }


def normalize_transcript_events(row: dict[str, Any]) -> dict[str, Any]:
    """Ground transcript-only ranges to existing segment boundaries."""
    duration = _positive_finite(row.get("duration_seconds"), "duration_seconds")
    _, by_id = _segments(row.get("transcript_segments"), duration=duration)
    values = row.get("transcript_event_ranges")
    if not isinstance(values, list):
        raise ValueError("transcript_event_ranges must be a list")
    events: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        item = _strict_segment_range(value, index=index, label="transcript event")
        error = _segment_range_error(
            item["start_segment_id"], item["end_segment_id"], by_id
        )
        if error is None:
            start = float(by_id[item["start_segment_id"]]["start_seconds"])
            end = float(by_id[item["end_segment_id"]]["end_seconds"])
        else:
            start = end = None
        events.append(
            {
                **item,
                "start_seconds": start,
                "end_seconds": end,
                "interval_valid": error is None,
                "interval_error": error,
                "timestamp_source": "whisper_segment_boundaries",
            }
        )
    return {"events": events}


def candidate_segment_ranges_to_windows(row: dict[str, Any]) -> dict[str, Any]:
    """Validate, ground, pad, deduplicate, and union-merge high-recall ranges."""
    duration = _positive_finite(row.get("duration_seconds"), "duration_seconds")
    padding = _nonnegative_finite(
        row.get("candidate_padding_seconds"), "candidate_padding_seconds"
    )
    segments, by_id = _segments(row.get("transcript_segments"), duration=duration)
    values = row.get("candidate_ranges")
    if not isinstance(values, list):
        raise ValueError("candidate_ranges must be a list")

    valid_ranges: list[dict[str, Any]] = []
    invalid_ranges: list[dict[str, Any]] = []
    duplicate_ranges: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index, value in enumerate(values):
        item = _strict_segment_range(value, index=index, label="candidate range")
        start_id = item["start_segment_id"]
        end_id = item["end_segment_id"]
        error = _segment_range_error(start_id, end_id, by_id)
        audited = {
            **item,
            "range_index": index,
            "range_valid": error is None,
            "range_error": error,
        }
        if error is not None:
            invalid_ranges.append(audited)
            continue
        key = (start_id, end_id)
        if key in seen:
            duplicate_ranges.append({**audited, "duplicate_of_segment_range": list(key)})
            continue
        seen.add(key)
        valid_ranges.append(audited)

    expanded: list[dict[str, Any]] = []
    for item in valid_ranges:
        start = float(by_id[item["start_segment_id"]]["start_seconds"])
        end = float(by_id[item["end_segment_id"]]["end_seconds"])
        expanded.append(
            {
                "start_seconds": max(0.0, start - padding),
                "end_seconds": min(duration, end + padding),
                "unpadded_start_seconds": start,
                "unpadded_end_seconds": end,
                "segment_ranges": [item],
            }
        )

    expanded.sort(key=lambda item: (item["start_seconds"], item["end_seconds"]))
    merged: list[dict[str, Any]] = []
    for window in expanded:
        if not merged or window["start_seconds"] > merged[-1]["end_seconds"]:
            merged.append(window)
            continue
        current = merged[-1]
        current["end_seconds"] = max(current["end_seconds"], window["end_seconds"])
        current["unpadded_start_seconds"] = min(
            current["unpadded_start_seconds"], window["unpadded_start_seconds"]
        )
        current["unpadded_end_seconds"] = max(
            current["unpadded_end_seconds"], window["unpadded_end_seconds"]
        )
        current["segment_ranges"].extend(window["segment_ranges"])
    for window_id, window in enumerate(merged):
        window["window_id"] = window_id
        window["duration_seconds"] = window["end_seconds"] - window["start_seconds"]

    return {
        "candidate_windows": merged,
        "invalid_candidate_ranges": invalid_ranges,
        "duplicate_candidate_ranges": duplicate_ranges,
        "raw_candidate_range_count": len(values),
        "valid_candidate_range_count": len(valid_ranges),
        "invalid_candidate_range_count": len(invalid_ranges),
        "duplicate_candidate_range_count": len(duplicate_ranges),
        "candidate_segment_count": len(segments),
    }


def _normalize_video_events(
    values: Any,
    *,
    absolute_start: float,
    supplied_duration: float,
    window_id: int | None,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError("events must be a list")
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"event {index} must be an object")
        start = _video_boundary(value, "start", index)
        end = _video_boundary(value, "end", index)
        confidence = _bounded_confidence(value.get("confidence"), f"event {index}")
        evidence = _nonempty_text(value.get("evidence"), f"event {index} evidence")
        error = None
        if start >= end:
            error = "start_not_before_end"
        elif end > supplied_duration:
            error = "outside_supplied_video"
        item = {
            "start_seconds": absolute_start + start,
            "end_seconds": absolute_start + end,
            "start_offset_seconds": start,
            "end_offset_seconds": end,
            "interval_valid": error is None,
            "interval_error": error,
            "confidence": confidence,
            "evidence": evidence,
            "timestamp_source": "supplied_video_relative",
        }
        if window_id is not None:
            item["window_id"] = window_id
        normalized.append(item)
    return normalized


def _strict_segment_range(value: Any, *, index: int, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} {index} must be an object")
    return {
        "start_segment_id": _integer(value.get("start_segment_id"), f"{label} start_segment_id"),
        "end_segment_id": _integer(value.get("end_segment_id"), f"{label} end_segment_id"),
        "confidence": _bounded_confidence(value.get("confidence"), label),
        "evidence": _nonempty_text(value.get("evidence"), f"{label} evidence"),
    }


def _segments(
    value: Any, *, duration: float
) -> tuple[list[Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    if not isinstance(value, list):
        raise ValueError("transcript_segments must be a list")
    segments: list[Mapping[str, Any]] = []
    by_id: dict[int, Mapping[str, Any]] = {}
    previous_start = -math.inf
    previous_end = -math.inf
    for index, segment in enumerate(value):
        if not isinstance(segment, Mapping):
            raise ValueError(f"transcript segment {index} must be an object")
        segment_id = _nonnegative_integer(segment.get("segment_id"), "segment_id")
        if segment_id != index:
            raise ValueError(f"transcript segment {index} must have segment_id {index}")
        start = _nonnegative_finite(segment.get("start_seconds"), "segment start")
        end = _nonnegative_finite(segment.get("end_seconds"), "segment end")
        if end <= start or start < previous_start or end < previous_end or end > duration:
            raise ValueError(f"transcript segment {segment_id} has invalid ordering")
        text = segment.get("text")
        if not isinstance(text, str):
            raise ValueError(f"transcript segment {segment_id} text must be a string")
        segments.append(segment)
        by_id[segment_id] = segment
        previous_start, previous_end = start, end
    return segments, by_id


def _segment_range_error(
    start: int, end: int, by_id: Mapping[int, Any]
) -> str | None:
    if start < 0:
        return "negative_start_segment_id"
    if end < 0:
        return "negative_end_segment_id"
    if start not in by_id:
        return "unknown_start_segment_id"
    if end not in by_id:
        return "unknown_end_segment_id"
    if start > end:
        return "start_segment_after_end_segment"
    return None


def _video_boundary(value: Mapping[str, Any], prefix: str, index: int) -> float:
    minute = _nonnegative_integer(value.get(f"{prefix}_minute"), f"event {index} {prefix}_minute")
    second = value.get(f"{prefix}_second")
    if (
        not isinstance(second, Real)
        or isinstance(second, bool)
        or not math.isfinite(float(second))
        or not 0 <= float(second) < 60
    ):
        raise ValueError(f"event {index} {prefix}_second must satisfy 0 <= value < 60")
    return minute * 60.0 + float(second)


def _bounded_confidence(value: Any, label: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise ValueError(f"{label} confidence must satisfy 0 <= value <= 1")
    return float(value)


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_finite(value: Any, label: str) -> float:
    result = _nonnegative_finite(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)

