"""Reusable helpers for exporting cross-camera trajectory records."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from .predicates import orient_upstream_downstream

Track = dict[str, Any]
TimelineSegment = dict[str, Any]
TrajectoryRecord = dict[str, Any]


def parse_iso_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp string to a timezone-aware datetime."""
    if not isinstance(value, str) or not value:
        return None
    timestamp = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def track_time_to_seconds(track: Track, *, field: str) -> float | None:
    """Convert a track timestamp to UTC epoch seconds."""
    parsed = parse_iso_timestamp(track.get(field))
    return parsed.timestamp() if parsed is not None else None


def timeline_segment_from_track(track: Track) -> TimelineSegment | None:
    """Build one ``{camera_id, entered, exited}`` trajectory segment."""
    camera_id = track.get("camera_id")
    if not isinstance(camera_id, str) or not camera_id:
        return None
    entered = track_time_to_seconds(track, field="start_time")
    exited = track_time_to_seconds(track, field="end_time")
    if entered is None or exited is None:
        return None
    return {"camera_id": camera_id, "entered": entered, "exited": exited}


def trajectory_attributes_from_track(track: Track) -> dict[str, str]:
    """Return the static appearance fields used by trajectory exports."""
    return {
        "class": str(track["vehicle_class"]) if track.get("vehicle_class") is not None else "",
        "color": str(track["color"]) if track.get("color") is not None else "",
        "subtype": str(track["subtype"]) if track.get("subtype") is not None else "",
    }


def vehicle_id_from_match(upstream: Track, downstream: Track) -> str:
    """Build a stable synthetic id for a matched traffic edge."""
    key = "|".join(
        [
            str(upstream.get("camera_id", "")),
            str(upstream.get("track_id", "")),
            str(downstream.get("camera_id", "")),
            str(downstream.get("track_id", "")),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"join_unique_{digest}"


def join_match_to_trajectory_record(
    left: Track,
    right: Track,
    *,
    match_score: float | None = None,
) -> TrajectoryRecord | None:
    """Turn one traffic join match into an exportable vehicle trajectory."""
    upstream, downstream = orient_upstream_downstream(left, right)
    upstream_segment = timeline_segment_from_track(upstream)
    downstream_segment = timeline_segment_from_track(downstream)
    if upstream_segment is None or downstream_segment is None:
        return None
    record: TrajectoryRecord = {
        "vehicle_id": vehicle_id_from_match(upstream, downstream),
        "attributes": trajectory_attributes_from_track(upstream),
        "timeline": [upstream_segment, downstream_segment],
    }
    if match_score is not None:
        record["match_score"] = float(match_score)
    return record


def join_match_to_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one Join output row into a vehicle trajectory record."""
    left = row.get("left")
    right = row.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return {}
    match_score = row.get("match_score")
    score = float(match_score) if isinstance(match_score, (int, float)) else None
    record = join_match_to_trajectory_record(left, right, match_score=score)
    return record if record is not None else {}


__all__ = [
    "join_match_to_trajectory",
    "join_match_to_trajectory_record",
    "parse_iso_timestamp",
    "timeline_segment_from_track",
    "track_time_to_seconds",
    "trajectory_attributes_from_track",
    "vehicle_id_from_match",
]
