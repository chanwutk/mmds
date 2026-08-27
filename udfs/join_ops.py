"""UDF wrappers for reusable cross-camera case-study predicates."""

from __future__ import annotations

from typing import Any

from mmds.case_studies import predicates as _impl

VEHICLE_APPEARANCE_KEYS = _impl.VEHICLE_APPEARANCE_KEYS


def vehicle_appearance_hash_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    return _impl.vehicle_appearance_hash_key(row)


def orient_upstream_downstream(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _impl.orient_upstream_downstream(left, right)


def different_cameras(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _impl.different_cameras(left, right)


def canonical_corridor_pair(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _impl.canonical_corridor_pair(left, right)


def travel_time_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _impl.travel_time_compatible(left, right)


def direction_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _impl.direction_compatible(left, right)


def speed_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _impl.speed_compatible(left, right)


def temporal_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _impl.temporal_overlap(left, right)


def temporal_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    return _impl.temporal_iou(left, right)


def same_vehicle(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Join/Filter UDF for cross-camera relational compatibility."""
    return _impl.same_vehicle(left, right)


def vehicle_match_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Join score UDF for greedy one-to-one matching."""
    return _impl.vehicle_match_score(left, right)
