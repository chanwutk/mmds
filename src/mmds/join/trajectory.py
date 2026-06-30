from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from .predicates import orient_upstream_downstream

Track: dict[str, Any]
TimelineSegment = dict[str, Any]
TrajectoryRecord = dict[str, Any]


def parse_iso_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp string to a timezone-aware datetime."""
    if not isinstance(value, str) or not value:
        return None
    timestamp = value
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def track_time_to_seconds(track: Track, *, field: str) -> float | None:
    """Convert a track's ISO ``start_time`` / ``end_time`` to UTC epoch seconds."""
    parsed = parse_iso_timestamp(track.get(field))
    if parsed is None:
        return None
    return parsed.timestamp()


def timeline_segment_from_track(track: Track) -> TimelineSegment | None:
    """Build one ``{camera_id, entered, exited}`` segment from a track summary."""
    camera_id = track.get("camera_id")
    if not isinstance(camera_id, str) or not camera_id:
        return None
    entered = track_time_to_seconds(track, field="start_time")
    exited = track_time_to_seconds(track, field="end_time")
    if entered is None or exited is None:
        return None
    return {
        "camera_id": camera_id,
        "entered": entered,
        "exited": exited,
    }


def trajectory_attributes_from_track(track: Track) -> dict[str, str]:
    """Static appearance fields (class, color, subtype) for trajectory export."""
    vehicle_class = track.get("vehicle_class")
    color = track.get("color")
    subtype = track.get("subtype")
    return {
        "class": str(vehicle_class) if vehicle_class is not None else "",
        "color": str(color) if color is not None else "",
        "subtype": str(subtype) if subtype is not None else "",
    }


def vehicle_id_from_match(upstream: Track, downstream: Track) -> str:
    """Stable synthetic id (join_unique_<8-char-hash>) for a two-camera matched edge."""
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
    """Turn one hash-join match row into an exportable vehicle trajectory."""
    # Upstream and downstream tracks are the tracks that entered and exited the corridor first and last, respectively.
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
