"""Cross-camera join helpers."""

from .hash_join import (
    VEHICLE_APPEARANCE_KEYS,
    build_hash_index,
    hash_join,
    join_hash_key,
    nested_loop_join,
    one_to_one_hash_join,
    vehicle_appearance_hash_key,
)
from .predicates import (
    canonical_corridor_pair,
    different_cameras,
    direction_compatible,
    orient_upstream_downstream,
    same_vehicle,
    speed_compatible,
    travel_time_compatible,
)
from .trajectory import (
    join_match_to_trajectory_record,
    timeline_segment_from_track,
    track_time_to_seconds,
    trajectory_attributes_from_track,
    vehicle_id_from_match,
)

__all__ = [
    "VEHICLE_APPEARANCE_KEYS",
    "build_hash_index",
    "canonical_corridor_pair",
    "different_cameras",
    "direction_compatible",
    "hash_join",
    "join_hash_key",
    "nested_loop_join",
    "one_to_one_hash_join",
    "orient_upstream_downstream",
    "join_match_to_trajectory_record",
    "same_vehicle",
    "speed_compatible",
    "timeline_segment_from_track",
    "track_time_to_seconds",
    "trajectory_attributes_from_track",
    "travel_time_compatible",
    "vehicle_appearance_hash_key",
    "vehicle_id_from_match",
]
