from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

# Maximum seconds between upstream track end and downstream track start.
_MAX_TRAVEL_SEC = 120.0
# Allow slight temporal overlap for synchronized feeds (seconds).
_MAX_SYNC_OVERLAP_SEC = 2.0
# First-seen fallback when timestamps are missing (seconds).
_MAX_SYNC_GAP_SEC = 10.0
# Maximum relative speed difference (fraction of the faster track).
_MAX_SPEED_RELATIVE_DIFF = 0.5
# Minimum speed (px/s) to treat a track as moving for direction checks.
_MIN_MOVING_SPEED_PX_PER_SEC = 1.0
# Maximum compass angle difference for compatible transfer directions.
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


def _track_camera_id(track: dict[str, Any]) -> str | None:
    camera_id = track.get("camera_id")
    if isinstance(camera_id, str) and camera_id:
        return camera_id
    return None


def _parse_iso_timestamp(value: Any) -> datetime | None:
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


def _camera_index(camera_id: str) -> int | None:
    match = _CAMERA_INDEX_RE.search(camera_id)
    if not match:
        return None
    return int(match.group(1))


def orient_upstream_downstream(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Return ``(upstream, downstream)`` when corridor order is known.
    The upstream track is the track that entered the corridor first,
    and the downstream track is the track that exited the corridor last.
    Helper function for cross-camera vehicle join and trajectory export.
    """
    left_camera = _track_camera_id(left)
    right_camera = _track_camera_id(right)
    if left_camera is None or right_camera is None:
        return left, right

    left_index = _camera_index(left_camera)
    right_index = _camera_index(right_camera)
    if left_index is None or right_index is None:
        return left, right
    if right_index < left_index:
        return right, left
    return left, right


def _direction_angle(label: Any) -> float | None:
    if not isinstance(label, str):
        return None
    return _COMPASS_ANGLES.get(label)


def _angular_difference(left_angle: float, right_angle: float) -> float:
    diff = abs(left_angle - right_angle) % 360.0
    return min(diff, 360.0 - diff)


def different_cameras(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Tracks must originate from different camera feeds."""
    left_camera = _track_camera_id(left)
    right_camera = _track_camera_id(right)
    if left_camera is None or right_camera is None:
        return False
    return left_camera != right_camera


def canonical_corridor_pair(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """For self-joins, only accept pairs with upstream corridor camera on the left.

    When both ``camera_id`` values contain a ``highway<N>`` suffix, requires
    ``N_left < N_right``. Pairs that cannot be ordered are allowed so asymmetric
    joins on non-highway ids are unchanged.

    Deduplication is handled by the join operator.
    """
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
        assert left_start is not None
        assert left_end is not None
        assert right_start is not None
        assert right_end is not None

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
    """Exit direction of the upstream track must align with downstream entry."""
    upstream, downstream = orient_upstream_downstream(left, right)
    exit_direction = upstream.get("exit_direction")
    entry_direction = downstream.get("entry_direction")
    if not isinstance(exit_direction, str) or not isinstance(entry_direction, str):
        return False
    if exit_direction == "stationary" or entry_direction == "stationary":
        return False

    up_speed = upstream.get("avg_speed")
    down_speed = downstream.get("avg_speed")
    if isinstance(up_speed, (int, float)) and up_speed < _MIN_MOVING_SPEED_PX_PER_SEC:
        return False
    if isinstance(down_speed, (int, float)) and down_speed < _MIN_MOVING_SPEED_PX_PER_SEC:
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
    if left_speed == 0.0 and right_speed == 0.0:
        return True
    faster = max(float(left_speed), float(right_speed))
    slower = min(float(left_speed), float(right_speed))
    if faster == 0.0:
        return True
    return slower / faster >= (1.0 - _MAX_SPEED_RELATIVE_DIFF)


def same_vehicle(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return ``True`` only when every relational constraint passes."""
    return (
        different_cameras(left, right)
        and canonical_corridor_pair(left, right)
        and travel_time_compatible(left, right)
        and direction_compatible(left, right)
        and speed_compatible(left, right)
    )
