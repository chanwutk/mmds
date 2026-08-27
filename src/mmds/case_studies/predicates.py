"""Reusable cross-camera traffic predicates and appearance keys."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from numbers import Real
from typing import Any

VEHICLE_APPEARANCE_KEYS: tuple[str, ...] = ("vehicle_class", "color", "subtype")

_MAX_TRAVEL_SEC = 120.0
_MAX_SYNC_OVERLAP_SEC = 2.0
_MAX_SYNC_GAP_SEC = 10.0
_MAX_SPEED_RELATIVE_DIFF = 0.5
_MIN_MOVING_SPEED_PX_PER_SEC = 1.0
_MAX_DIRECTION_ANGLE_DIFF = 120.0
_COMPASS_ANGLES: dict[str, float] = {
    "E": 0.0,
    "NE": 45.0,
    "N": 90.0,
    "NW": 135.0,
    "W": 180.0,
    "SW": 225.0,
    "S": 270.0,
    "SE": 315.0,
}
_CAMERA_INDEX_RE = re.compile(r"highway(\d+)$", re.IGNORECASE)


def vehicle_appearance_hash_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    """Build the traffic example's categorical appearance key."""
    values: list[Any] = []
    for key in VEHICLE_APPEARANCE_KEYS:
        if key not in row:
            return None
        values.append(row[key])
    result = tuple(values)
    try:
        hash(result)
    except TypeError:
        return None
    return result


def _track_camera_id(track: dict[str, Any]) -> str | None:
    camera_id = track.get("camera_id")
    if isinstance(camera_id, str) and camera_id:
        return camera_id
    return None


def _parse_iso_timestamp(value: Any) -> datetime | None:
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


def _camera_index(camera_id: str) -> int | None:
    match = _CAMERA_INDEX_RE.search(camera_id)
    return int(match.group(1)) if match else None


def orient_upstream_downstream(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the traffic corridor tracks in upstream/downstream order."""
    left_camera = _track_camera_id(left)
    right_camera = _track_camera_id(right)
    if left_camera is None or right_camera is None:
        return left, right
    left_index = _camera_index(left_camera)
    right_index = _camera_index(right_camera)
    if left_index is None or right_index is None:
        return left, right
    return (right, left) if right_index < left_index else (left, right)


def _direction_angle(label: Any) -> float | None:
    return _COMPASS_ANGLES.get(label) if isinstance(label, str) else None


def _angular_difference(left_angle: float, right_angle: float) -> float:
    diff = abs(left_angle - right_angle) % 360.0
    return min(diff, 360.0 - diff)


def different_cameras(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Tracks must originate from different camera feeds."""
    left_camera = _track_camera_id(left)
    right_camera = _track_camera_id(right)
    return left_camera is not None and right_camera is not None and left_camera != right_camera


def canonical_corridor_pair(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """For highway self-joins, accept only upstream-to-downstream pairs."""
    left_camera = _track_camera_id(left)
    right_camera = _track_camera_id(right)
    if left_camera is None or right_camera is None:
        return True
    left_index = _camera_index(left_camera)
    right_index = _camera_index(right_camera)
    if left_index is None or right_index is None:
        return True
    return left_index < right_index


def travel_time_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Track times must allow a plausible cross-camera hand-off."""
    left_start = _parse_iso_timestamp(left.get("start_time"))
    left_end = _parse_iso_timestamp(left.get("end_time"))
    right_start = _parse_iso_timestamp(right.get("start_time"))
    right_end = _parse_iso_timestamp(right.get("end_time"))
    if None not in {left_start, left_end, right_start, right_end}:
        assert left_start is not None and left_end is not None
        assert right_start is not None and right_end is not None
        if left_start <= right_end and right_start <= left_end:
            return True
        upstream, downstream = orient_upstream_downstream(left, right)
        up_end = _parse_iso_timestamp(upstream.get("end_time"))
        down_start = _parse_iso_timestamp(downstream.get("start_time"))
        if up_end is None or down_start is None:
            return False
        gap_sec = (down_start - up_end).total_seconds()
        return -_MAX_SYNC_OVERLAP_SEC <= gap_sec <= _MAX_TRAVEL_SEC

    left_start_frame = left.get("first_frame_idx", left.get("start_frame_idx"))
    right_start_frame = right.get("first_frame_idx", right.get("start_frame_idx"))
    if isinstance(left_start_frame, int) and isinstance(right_start_frame, int):
        fps = left.get("fps", right.get("fps"))
        if not isinstance(fps, (int, float)) or fps <= 0:
            fps = 30.0
        return abs(left_start_frame - right_start_frame) / float(fps) <= _MAX_SYNC_GAP_SEC
    return False


def direction_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """The upstream exit direction must align with the downstream entry."""
    upstream, downstream = orient_upstream_downstream(left, right)
    exit_direction = upstream.get("exit_direction")
    entry_direction = downstream.get("entry_direction")
    if not isinstance(exit_direction, str) or not isinstance(entry_direction, str):
        return False
    if exit_direction == "stationary" or entry_direction == "stationary":
        return False
    for speed in (upstream.get("avg_speed"), downstream.get("avg_speed")):
        if isinstance(speed, (int, float)) and speed < _MIN_MOVING_SPEED_PX_PER_SEC:
            return False
    exit_angle = _direction_angle(exit_direction)
    entry_angle = _direction_angle(entry_direction)
    if exit_angle is None or entry_angle is None:
        return exit_direction == entry_direction
    return _angular_difference(exit_angle, entry_angle) <= _MAX_DIRECTION_ANGLE_DIFF


def speed_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Average image-plane speeds must be reasonably similar."""
    left_speed = left.get("avg_speed")
    right_speed = right.get("avg_speed")
    if not isinstance(left_speed, (int, float)) or not isinstance(right_speed, (int, float)):
        return False
    if left_speed < 0 or right_speed < 0:
        return False
    faster = max(float(left_speed), float(right_speed))
    slower = min(float(left_speed), float(right_speed))
    return True if faster == 0.0 else slower / faster >= 1.0 - _MAX_SPEED_RELATIVE_DIFF


def temporal_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether synchronized traffic track intervals overlap."""
    left_interval = _track_interval_seconds(left)
    right_interval = _track_interval_seconds(right)
    if left_interval is None or right_interval is None:
        return False
    left_start, left_end = left_interval
    right_start, right_end = right_interval
    return (
        left_start <= right_end + _MAX_SYNC_OVERLAP_SEC
        and right_start <= left_end + _MAX_SYNC_OVERLAP_SEC
    )


def temporal_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Return intersection-over-union for two traffic track intervals."""
    left_interval = _track_interval_seconds(left)
    right_interval = _track_interval_seconds(right)
    if left_interval is None or right_interval is None:
        return 0.0
    left_start, left_end = left_interval
    right_start, right_end = right_interval
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    if union <= 0.0:
        return 1.0 if overlap > 0.0 or left_start == right_start else 0.0
    return overlap / union


def _track_interval_seconds(track: dict[str, Any]) -> tuple[float, float] | None:
    start = _parse_iso_timestamp(track.get("start_time"))
    end = _parse_iso_timestamp(track.get("end_time"))
    if start is None or end is None or end < start:
        return None
    return start.timestamp(), end.timestamp()


def _finite_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        return 0.0
    confidence = float(value)
    return confidence if math.isfinite(confidence) else 0.0


def same_vehicle(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether a cross-camera pair passes all relational checks."""
    return (
        different_cameras(left, right)
        and canonical_corridor_pair(left, right)
        and travel_time_compatible(left, right)
        and direction_compatible(left, right)
        and speed_compatible(left, right)
    )


def vehicle_match_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Return mean track confidence for greedy one-to-one matching."""
    left_conf = _finite_confidence(left.get("confidence"))
    right_conf = _finite_confidence(right.get("confidence"))
    return (left_conf + right_conf) / 2.0


__all__ = [
    "VEHICLE_APPEARANCE_KEYS",
    "canonical_corridor_pair",
    "different_cameras",
    "direction_compatible",
    "orient_upstream_downstream",
    "same_vehicle",
    "speed_compatible",
    "temporal_iou",
    "temporal_overlap",
    "travel_time_compatible",
    "vehicle_appearance_hash_key",
    "vehicle_match_score",
]
