"""Deterministic UDFs for cross-modal lecture interval retrieval."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


def candidate_segment_ranges_to_windows(row: dict[str, Any]) -> dict[str, Any]:
    """Ground model-selected segment ranges, pad, and merge their time windows."""
    duration = _positive_finite(row.get("duration_seconds"), "duration_seconds")
    padding = _nonnegative_finite(
        row.get("candidate_padding_seconds"), "candidate_padding_seconds"
    )
    segments, by_id = _segments(row.get("transcript_segments"))
    ranges = row.get("candidate_ranges")
    if not isinstance(ranges, list):
        raise ValueError("candidate_ranges must be a list")

    expanded: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index, value in enumerate(ranges):
        if not isinstance(value, Mapping):
            invalid.append(
                {"range_index": index, "range_valid": False, "range_error": "not_object"}
            )
            continue
        start_id = value.get("start_segment_id")
        end_id = value.get("end_segment_id")
        error = _segment_range_error(start_id, end_id, by_id)
        raw = {
            "range_index": index,
            "start_segment_id": start_id,
            "end_segment_id": end_id,
            "confidence": _bounded_confidence(value.get("confidence")),
            "evidence": str(value.get("evidence", "")),
        }
        if error is not None:
            invalid.append({**raw, "range_valid": False, "range_error": error})
            continue
        key = (int(start_id), int(end_id))
        if key in seen:
            continue
        seen.add(key)
        start = float(by_id[key[0]]["start_seconds"])
        end = float(by_id[key[1]]["end_seconds"])
        expanded.append(
            {
                "start_seconds": max(0.0, start - padding),
                "end_seconds": min(duration, end + padding),
                "unpadded_start_seconds": start,
                "unpadded_end_seconds": end,
                "segment_ranges": [raw],
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
        "invalid_candidate_ranges": invalid,
        "candidate_segment_count": len(segments),
    }


def transcript_episode_proposals_to_windows(row: dict[str, Any]) -> dict[str, Any]:
    """Ground complete-episode proposals without merging distinct hypotheses."""
    duration = _positive_finite(row.get("duration_seconds"), "duration_seconds")
    padding = _nonnegative_finite(
        row.get("candidate_padding_seconds"), "candidate_padding_seconds"
    )
    segments, by_id = _segments(row.get("transcript_segments"))
    proposals = row.get("episode_proposals")
    if not isinstance(proposals, list):
        raise ValueError("episode_proposals must be a list")

    windows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for proposal_index, value in enumerate(proposals):
        if not isinstance(value, Mapping):
            invalid.append(
                {
                    "proposal_index": proposal_index,
                    "range_valid": False,
                    "range_error": "not_object",
                }
            )
            continue
        start_id = value.get("start_segment_id")
        end_id = value.get("end_segment_id")
        raw = {
            "proposal_index": proposal_index,
            "start_segment_id": start_id,
            "end_segment_id": end_id,
            "confidence": _bounded_confidence(value.get("confidence")),
            "evidence": str(value.get("evidence", "")),
        }
        error = _segment_range_error(start_id, end_id, by_id)
        if error is not None:
            invalid.append({**raw, "range_valid": False, "range_error": error})
            continue
        key = (int(start_id), int(end_id))
        if key in seen:
            continue
        seen.add(key)
        start = float(by_id[key[0]]["start_seconds"])
        end = float(by_id[key[1]]["end_seconds"])
        windows.append(
            {
                "window_id": len(windows),
                "proposal_index": proposal_index,
                "start_seconds": max(0.0, start - padding),
                "end_seconds": min(duration, end + padding),
                "duration_seconds": (
                    min(duration, end + padding) - max(0.0, start - padding)
                ),
                "unpadded_start_seconds": start,
                "unpadded_end_seconds": end,
                "segment_ranges": [raw],
            }
        )
    return {
        "candidate_windows": windows,
        "invalid_episode_proposals": invalid,
        "proposal_segment_count": len(segments),
    }


def attach_candidate_video_view(row: dict[str, Any]) -> dict[str, Any]:
    video = row.get("video")
    window = row.get("candidate_windows")
    if not isinstance(video, Mapping) or not isinstance(window, Mapping):
        raise ValueError("video and candidate_windows must be objects")
    source = video.get("path") or video.get("source") or video.get("uri")
    start = window.get("start_seconds")
    end = window.get("end_seconds")
    if not isinstance(source, str) or not source:
        raise ValueError("video must contain a non-empty path, source, or uri")
    start_seconds = _nonnegative_finite(start, "candidate start_seconds")
    end_seconds = _nonnegative_finite(end, "candidate end_seconds")
    if end_seconds <= start_seconds:
        raise ValueError("candidate window must satisfy start_seconds < end_seconds")
    view = {
        "type": "VideoView",
        "source": source,
        "start": start_seconds,
        "end": end_seconds,
    }
    for field in ("fps", "sha256", "mime_type"):
        if field in video:
            view[field] = video[field]
    return {"candidate_video": view}


def normalize_naive_events(row: dict[str, Any]) -> dict[str, Any]:
    return _normalize_naive_video_field(row, "events", require_nonempty=False)


def normalize_naive_episodes(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a required complete-episode response on a full lecture."""
    return _normalize_naive_video_field(row, "episodes", require_nonempty=True)


def _normalize_naive_video_field(
    row: dict[str, Any], field: str, *, require_nonempty: bool
) -> dict[str, Any]:
    duration = _positive_finite(row.get("duration_seconds"), "duration_seconds")
    values = _required_video_values(row, field) if require_nonempty else row.get(field)
    return {"events": _normalize_video_events(values, 0.0, duration)}


def normalize_verified_events(row: dict[str, Any]) -> dict[str, Any]:
    return _normalize_verified_video_field(row, "events", require_nonempty=False)


def normalize_verified_episodes(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize required clip-relative episodes back to source-lecture time."""
    return _normalize_verified_video_field(row, "episodes", require_nonempty=True)


def _normalize_verified_video_field(
    row: dict[str, Any], field: str, *, require_nonempty: bool
) -> dict[str, Any]:
    window = row.get("candidate_windows")
    if not isinstance(window, Mapping):
        raise ValueError("candidate_windows must be an object")
    start = _nonnegative_finite(window.get("start_seconds"), "window start_seconds")
    end = _nonnegative_finite(window.get("end_seconds"), "window end_seconds")
    if end <= start:
        raise ValueError("candidate window must satisfy start_seconds < end_seconds")
    values = _required_video_values(row, field) if require_nonempty else row.get(field)
    return {
        "events": _normalize_video_events(
            values, start, end, window_id=window.get("window_id")
        )
    }


def normalize_transcript_events(row: dict[str, Any]) -> dict[str, Any]:
    duration = _positive_finite(row.get("duration_seconds"), "duration_seconds")
    _, by_id = _segments(row.get("transcript_segments"))
    ranges = row.get("transcript_event_ranges")
    if not isinstance(ranges, list):
        raise ValueError("transcript_event_ranges must be a list")
    events: list[dict[str, Any]] = []
    for index, value in enumerate(ranges):
        if not isinstance(value, Mapping):
            events.append(
                {
                    "start_seconds": None,
                    "end_seconds": None,
                    "interval_valid": False,
                    "interval_error": "not_object",
                    "range_index": index,
                }
            )
            continue
        start_id = value.get("start_segment_id")
        end_id = value.get("end_segment_id")
        error = _segment_range_error(start_id, end_id, by_id)
        item = {
            "start_segment_id": start_id,
            "end_segment_id": end_id,
            "confidence": _bounded_confidence(value.get("confidence")),
            "evidence": str(value.get("evidence", "")),
            "timestamp_source": "whisper_segment_boundaries",
        }
        if error is not None:
            events.append(
                {
                    **item,
                    "start_seconds": None,
                    "end_seconds": None,
                    "interval_valid": False,
                    "interval_error": error,
                }
            )
            continue
        start = float(by_id[int(start_id)]["start_seconds"])
        end = float(by_id[int(end_id)]["end_seconds"])
        interval_error = None
        if end > duration:
            interval_error = "outside_video_duration"
        events.append(
            {
                **item,
                "start_seconds": start,
                "end_seconds": end,
                "interval_valid": interval_error is None,
                "interval_error": interval_error,
            }
        )
    return {"events": events}


def normalize_transcript_grounded_refinement(row: dict[str, Any]) -> dict[str, Any]:
    """Ground one audiovisual refinement to allowed Whisper segment boundaries."""
    duration = _positive_finite(row.get("duration_seconds"), "duration_seconds")
    _, by_id = _segments(row.get("transcript_segments"))
    allowed = row.get("transcript_context_segment_ids")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(
            not isinstance(segment_id, int) or isinstance(segment_id, bool)
            for segment_id in allowed
        )
    ):
        raise ValueError("transcript_context_segment_ids must be a non-empty integer list")
    allowed_ids = set(allowed)
    if len(allowed_ids) != len(allowed) or not allowed_ids.issubset(by_id):
        raise ValueError("transcript_context_segment_ids are duplicate or unknown")

    window = row.get("candidate_windows")
    if not isinstance(window, Mapping):
        raise ValueError("candidate_windows must be an object")
    window_start = _nonnegative_finite(
        window.get("start_seconds"), "window start_seconds"
    )
    window_end = _nonnegative_finite(window.get("end_seconds"), "window end_seconds")
    if window_end <= window_start:
        raise ValueError("candidate window must satisfy start_seconds < end_seconds")

    values = row.get("episode_refinements")
    if not isinstance(values, list):
        raise ValueError("episode_refinements must be a list")
    if len(values) > 1:
        raise ValueError("episode_refinements must contain at most one range")
    events: list[dict[str, Any]] = []
    for refinement_index, value in enumerate(values):
        if not isinstance(value, Mapping):
            events.append(
                {
                    "start_seconds": None,
                    "end_seconds": None,
                    "interval_valid": False,
                    "interval_error": "not_object",
                    "refinement_index": refinement_index,
                }
            )
            continue
        start_id = value.get("start_segment_id")
        end_id = value.get("end_segment_id")
        error = _segment_range_error(start_id, end_id, by_id)
        if error is None and (
            int(start_id) not in allowed_ids or int(end_id) not in allowed_ids
        ):
            error = "segment_id_outside_allowed_context"
        item = {
            "start_segment_id": start_id,
            "end_segment_id": end_id,
            "confidence": _bounded_confidence(value.get("confidence")),
            "evidence": str(value.get("evidence", "")),
            "timestamp_source": "audiovisually_refined_whisper_segment_boundaries",
            "window_id": window.get("window_id"),
        }
        if error is not None:
            events.append(
                {
                    **item,
                    "start_seconds": None,
                    "end_seconds": None,
                    "interval_valid": False,
                    "interval_error": error,
                }
            )
            continue
        start = float(by_id[int(start_id)]["start_seconds"])
        end = float(by_id[int(end_id)]["end_seconds"])
        interval_error = None
        if start < window_start or end > window_end:
            interval_error = "outside_supplied_clip"
        elif end > duration:
            interval_error = "outside_video_duration"
        events.append(
            {
                **item,
                "start_seconds": start,
                "end_seconds": end,
                "interval_valid": interval_error is None,
                "interval_error": interval_error,
            }
        )
    return {"events": events}


def normalize_required_conditions(row: dict[str, Any]) -> dict[str, Any]:
    """Assign deterministic identifiers to generic query-condition descriptions."""
    descriptions = row.get("condition_descriptions")
    if not isinstance(descriptions, list) or not 1 <= len(descriptions) <= 8:
        raise ValueError("condition_descriptions must contain between 1 and 8 items")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(descriptions):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"condition description {index} must be non-empty text")
        description = " ".join(value.split())
        identity = description.casefold()
        if identity in seen:
            raise ValueError("condition_descriptions must not contain duplicates")
        seen.add(identity)
        normalized.append(description)
    return {
        "required_conditions": [
            {"condition_id": f"condition_{index}", "description": description}
            for index, description in enumerate(normalized)
        ]
    }


def normalize_role_aware_required_conditions(row: dict[str, Any]) -> dict[str, Any]:
    """Assign IDs to nonredundant gate/anchor condition specifications."""
    specs = row.get("condition_specs")
    if not isinstance(specs, list) or not 1 <= len(specs) <= 8:
        raise ValueError("condition_specs must contain between 1 and 8 items")
    required: list[dict[str, str]] = []
    seen: set[str] = set()
    anchor_count = 0
    for index, value in enumerate(specs):
        if not isinstance(value, Mapping):
            raise ValueError(f"condition spec {index} must be an object")
        description = value.get("description")
        role = value.get("role")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"condition spec {index} description is invalid")
        normalized_description = " ".join(description.split())
        identity = normalized_description.casefold()
        if identity in seen:
            raise ValueError("condition_specs must not contain duplicate descriptions")
        if role not in {"gate", "anchor"}:
            raise ValueError(f"condition spec {index} role must be gate or anchor")
        seen.add(identity)
        anchor_count += role == "anchor"
        required.append(
            {
                "condition_id": f"condition_{index}",
                "description": normalized_description,
                "role": str(role),
            }
        )
    if anchor_count == 0:
        raise ValueError("condition_specs must contain at least one anchor")
    return {"required_conditions": required}


def normalize_condition_evidence_to_event(row: dict[str, Any]) -> dict[str, Any]:
    """Require evidence for every condition and derive its minimal covering span."""
    required = row.get("required_conditions")
    if not isinstance(required, list) or not 1 <= len(required) <= 8:
        raise ValueError("required_conditions must contain between 1 and 8 items")
    required_by_id: dict[str, str] = {}
    for index, condition in enumerate(required):
        expected_id = f"condition_{index}"
        if not isinstance(condition, Mapping):
            raise ValueError(f"required condition {index} must be an object")
        condition_id = condition.get("condition_id")
        description = condition.get("description")
        if condition_id != expected_id:
            raise ValueError(f"required condition {index} must use {expected_id}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"required condition {index} description is invalid")
        required_by_id[expected_id] = description

    duration = _positive_finite(row.get("duration_seconds"), "duration_seconds")
    _, by_id = _segments(row.get("transcript_segments"))
    allowed = row.get("transcript_context_segment_ids")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(
            not isinstance(segment_id, int) or isinstance(segment_id, bool)
            for segment_id in allowed
        )
    ):
        raise ValueError("transcript_context_segment_ids must be a non-empty integer list")
    allowed_ids = set(allowed)
    if len(allowed_ids) != len(allowed) or not allowed_ids.issubset(by_id):
        raise ValueError("transcript_context_segment_ids are duplicate or unknown")

    window = row.get("candidate_windows")
    if not isinstance(window, Mapping):
        raise ValueError("candidate_windows must be an object")
    window_start = _nonnegative_finite(
        window.get("start_seconds"), "window start_seconds"
    )
    window_end = _nonnegative_finite(window.get("end_seconds"), "window end_seconds")
    if window_end <= window_start:
        raise ValueError("candidate window must satisfy start_seconds < end_seconds")

    evidence_values = row.get("condition_evidence")
    if not isinstance(evidence_values, list):
        raise ValueError("condition_evidence must be a list")
    if not evidence_values:
        return {"events": []}

    evidence_by_id: dict[str, dict[str, Any]] = {}
    contract_error: str | None = None
    for index, value in enumerate(evidence_values):
        if not isinstance(value, Mapping):
            contract_error = f"condition_evidence_{index}_not_object"
            break
        condition_id = value.get("condition_id")
        if not isinstance(condition_id, str) or condition_id not in required_by_id:
            contract_error = "unknown_condition_id"
            break
        if condition_id in evidence_by_id:
            contract_error = "duplicate_condition_id"
            break
        start_id = value.get("start_segment_id")
        end_id = value.get("end_segment_id")
        error = _segment_range_error(start_id, end_id, by_id)
        if error is None and (
            int(start_id) not in allowed_ids or int(end_id) not in allowed_ids
        ):
            error = "segment_id_outside_allowed_context"
        if error is not None:
            contract_error = error
            break
        start = float(by_id[int(start_id)]["start_seconds"])
        end = float(by_id[int(end_id)]["end_seconds"])
        if start < window_start or end > window_end:
            contract_error = "outside_supplied_clip"
            break
        evidence_by_id[condition_id] = {
            "condition_id": condition_id,
            "description": required_by_id[condition_id],
            "start_segment_id": int(start_id),
            "end_segment_id": int(end_id),
            "start_seconds": start,
            "end_seconds": end,
            "confidence": _bounded_confidence(value.get("confidence")),
            "evidence": str(value.get("evidence", "")),
        }

    if contract_error is not None:
        return {
            "events": [
                {
                    "start_seconds": None,
                    "end_seconds": None,
                    "interval_valid": False,
                    "interval_error": contract_error,
                    "confidence": 0.0,
                    "evidence": "Invalid condition-evidence contract",
                    "timestamp_source": (
                        "predicate_grounded_whisper_segment_boundaries"
                    ),
                    "window_id": window.get("window_id"),
                }
            ]
        }
    if set(evidence_by_id) != set(required_by_id):
        return {"events": []}

    ordered = [evidence_by_id[f"condition_{index}"] for index in range(len(required))]
    start_evidence = min(
        ordered, key=lambda item: (item["start_seconds"], item["start_segment_id"])
    )
    end_evidence = max(
        ordered, key=lambda item: (item["end_seconds"], item["end_segment_id"])
    )
    start = float(start_evidence["start_seconds"])
    end = float(end_evidence["end_seconds"])
    interval_error = None
    if start >= end:
        interval_error = "condition_evidence_has_no_positive_span"
    elif end > duration:
        interval_error = "outside_video_duration"
    return {
        "events": [
            {
                "start_seconds": start,
                "end_seconds": end,
                "start_segment_id": start_evidence["start_segment_id"],
                "end_segment_id": end_evidence["end_segment_id"],
                "interval_valid": interval_error is None,
                "interval_error": interval_error,
                "confidence": min(item["confidence"] for item in ordered),
                "evidence": " | ".join(
                    f"{item['condition_id']}: {item['evidence']}" for item in ordered
                ),
                "condition_evidence": ordered,
                "timestamp_source": "predicate_grounded_whisper_segment_boundaries",
                "window_id": window.get("window_id"),
            }
        ]
    }


def normalize_role_aware_condition_evidence_to_event(
    row: dict[str, Any],
) -> dict[str, Any]:
    """Require every condition but derive timestamps from anchor evidence only."""
    required = row.get("required_conditions")
    if not isinstance(required, list) or not required:
        raise ValueError("required_conditions must be a non-empty list")
    roles: dict[str, str] = {}
    for index, condition in enumerate(required):
        if not isinstance(condition, Mapping):
            raise ValueError(f"required condition {index} must be an object")
        condition_id = condition.get("condition_id")
        role = condition.get("role")
        if condition_id != f"condition_{index}" or role not in {"gate", "anchor"}:
            raise ValueError(f"required condition {index} has invalid id or role")
        roles[str(condition_id)] = str(role)
    if "anchor" not in roles.values():
        raise ValueError("required_conditions must contain at least one anchor")

    normalized = normalize_condition_evidence_to_event(row)
    events = normalized["events"]
    if not events or not events[0].get("interval_valid", False):
        return normalized
    event = events[0]
    evidence = event.get("condition_evidence")
    if not isinstance(evidence, list):
        raise ValueError("normalized condition evidence is missing")
    anchors = [
        item
        for item in evidence
        if isinstance(item, Mapping) and roles.get(str(item.get("condition_id"))) == "anchor"
    ]
    if not anchors:
        raise ValueError("normalized condition evidence has no anchors")
    start_evidence = min(
        anchors,
        key=lambda item: (item["start_seconds"], item["start_segment_id"]),
    )
    end_evidence = max(
        anchors,
        key=lambda item: (item["end_seconds"], item["end_segment_id"]),
    )
    start = float(start_evidence["start_seconds"])
    end = float(end_evidence["end_seconds"])
    if start >= end:
        return {
            "events": [
                {
                    **event,
                    "start_seconds": start,
                    "end_seconds": end,
                    "interval_valid": False,
                    "interval_error": "anchor_evidence_has_no_positive_span",
                    "timestamp_source": (
                        "role_aware_predicate_grounded_whisper_segment_boundaries"
                    ),
                }
            ]
        }
    return {
        "events": [
            {
                **event,
                "start_seconds": start,
                "end_seconds": end,
                "start_segment_id": int(start_evidence["start_segment_id"]),
                "end_segment_id": int(end_evidence["end_segment_id"]),
                "timestamp_source": (
                    "role_aware_predicate_grounded_whisper_segment_boundaries"
                ),
            }
        ]
    }


def validate_binary_verification(row: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the strict binary VLM response contract."""
    verification = _binary_verification(row)
    return {"verification": verification}


def is_event_present(row: dict[str, Any]) -> bool:
    """Return a validated binary decision without truthiness coercion."""
    return _binary_verification(row)["event_present"]


def _binary_verification(row: Mapping[str, Any]) -> dict[str, Any]:
    verification = row.get("verification")
    if not isinstance(verification, Mapping):
        raise ValueError("verification must be an object")
    event_present = verification.get("event_present")
    confidence = verification.get("confidence")
    evidence = verification.get("evidence")
    if not isinstance(event_present, bool):
        raise ValueError("verification event_present must be boolean")
    if (
        not isinstance(confidence, Real)
        or isinstance(confidence, bool)
        or not math.isfinite(float(confidence))
        or not 0 <= float(confidence) <= 1
    ):
        raise ValueError("verification confidence must satisfy 0 <= value <= 1")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("verification evidence must be a non-empty string")
    return {
        "event_present": event_present,
        "confidence": float(confidence),
        "evidence": evidence,
    }


def has_events(row: dict[str, Any]) -> bool:
    return isinstance(row.get("events"), list) and bool(row["events"])


def _required_video_values(row: Mapping[str, Any], field: str) -> list[Any]:
    values = row.get(field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be a non-empty list after positive verification")
    return values


def _normalize_video_events(
    values: Any,
    clip_start: float,
    clip_end: float,
    *,
    window_id: Any = None,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    clip_duration = clip_end - clip_start
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            continue
        start_offset = _clip_boundary(value, "start", index)
        end_offset = _clip_boundary(value, "end", index)
        error = None
        if start_offset >= end_offset:
            error = "start_not_before_end"
        elif end_offset > clip_duration:
            error = "outside_supplied_clip"
        item = {
            "start_seconds": clip_start + start_offset,
            "end_seconds": clip_start + end_offset,
            "start_offset_seconds": start_offset,
            "end_offset_seconds": end_offset,
            "interval_valid": error is None,
            "interval_error": error,
            "confidence": _bounded_confidence(value.get("confidence")),
            "evidence": str(value.get("evidence", "")),
            "timestamp_source": "supplied_clip_relative",
        }
        if window_id is not None:
            item["window_id"] = window_id
        normalized.append(item)
    return normalized


def _clip_boundary(value: Mapping[str, Any], prefix: str, index: int) -> float:
    minute = value.get(f"{prefix}_minute")
    second = value.get(f"{prefix}_second")
    if (
        not isinstance(minute, Real)
        or isinstance(minute, bool)
        or not float(minute).is_integer()
        or float(minute) < 0
    ):
        raise ValueError(
            f"event {index} {prefix}_minute must be a non-negative integer"
        )
    if (
        not isinstance(second, Real)
        or isinstance(second, bool)
        or not math.isfinite(float(second))
        or not 0 <= float(second) < 60
    ):
        raise ValueError(f"event {index} {prefix}_second must satisfy 0 <= value < 60")
    return int(minute) * 60.0 + float(second)


def _segments(value: Any) -> tuple[list[Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    if not isinstance(value, list):
        raise ValueError("transcript_segments must be a list")
    result: list[Mapping[str, Any]] = []
    by_id: dict[int, Mapping[str, Any]] = {}
    previous_start = -math.inf
    previous_end = -math.inf
    for index, segment in enumerate(value):
        if not isinstance(segment, Mapping):
            raise ValueError(f"transcript segment {index} must be an object")
        segment_id = segment.get("segment_id")
        if (
            not isinstance(segment_id, int)
            or isinstance(segment_id, bool)
            or segment_id != index
        ):
            raise ValueError(
                f"transcript segment {index} must have segment_id {index}"
            )
        if segment_id in by_id:
            raise ValueError(f"duplicate transcript segment_id {segment_id}")
        start = _nonnegative_finite(segment.get("start_seconds"), "segment start")
        end = _nonnegative_finite(segment.get("end_seconds"), "segment end")
        if (
            end <= start
            or start < previous_start
            or end < previous_end
        ):
            raise ValueError(f"transcript segment {segment_id} has invalid ordering")
        result.append(segment)
        by_id[segment_id] = segment
        previous_start, previous_end = start, end
    return result, by_id


def _segment_range_error(start: Any, end: Any, by_id: Mapping[int, Any]) -> str | None:
    if not isinstance(start, int) or isinstance(start, bool):
        return "invalid_start_segment_id"
    if not isinstance(end, int) or isinstance(end, bool):
        return "invalid_end_segment_id"
    if start not in by_id:
        return "unknown_start_segment_id"
    if end not in by_id:
        return "unknown_end_segment_id"
    if start > end:
        return "start_segment_after_end_segment"
    return None


def _bounded_confidence(value: Any) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _positive_finite(value: Any, field: str) -> float:
    result = _nonnegative_finite(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_finite(value: Any, field: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{field} must be a finite non-negative number")
    return float(value)
