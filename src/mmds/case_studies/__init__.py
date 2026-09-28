"""Lazily exported reusable vehicle case-study components."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "VEHICLE_APPEARANCE_KEYS": (".predicates", "VEHICLE_APPEARANCE_KEYS"),
    "canonical_corridor_pair": (".predicates", "canonical_corridor_pair"),
    "different_cameras": (".predicates", "different_cameras"),
    "direction_compatible": (".predicates", "direction_compatible"),
    "orient_upstream_downstream": (".predicates", "orient_upstream_downstream"),
    "same_vehicle": (".predicates", "same_vehicle"),
    "speed_compatible": (".predicates", "speed_compatible"),
    "temporal_iou": (".predicates", "temporal_iou"),
    "temporal_overlap": (".predicates", "temporal_overlap"),
    "travel_time_compatible": (".predicates", "travel_time_compatible"),
    "vehicle_appearance_hash_key": (".predicates", "vehicle_appearance_hash_key"),
    "vehicle_match_score": (".predicates", "vehicle_match_score"),
    "join_match_to_trajectory": (".trajectory", "join_match_to_trajectory"),
    "join_match_to_trajectory_record": (".trajectory", "join_match_to_trajectory_record"),
    "parse_iso_timestamp": (".trajectory", "parse_iso_timestamp"),
    "promote_vehicle_trajectory_row": (".trajectory", "promote_vehicle_trajectory_row"),
    "timeline_segment_from_track": (".trajectory", "timeline_segment_from_track"),
    "track_time_to_seconds": (".trajectory", "track_time_to_seconds"),
    "trajectory_attributes_from_track": (".trajectory", "trajectory_attributes_from_track"),
    "vehicle_id_from_match": (".trajectory", "vehicle_id_from_match"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, symbol_name = target
    value = getattr(import_module(module_name, __name__), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
