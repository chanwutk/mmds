"""Generic join algorithms."""

from .hash_join import (
    build_hash_index,
    hash_join,
    join_hash_key,
    nested_loop_join,
    one_to_one_hash_join,
)

__all__ = [
    "build_hash_index",
    "hash_join",
    "join_hash_key",
    "nested_loop_join",
    "one_to_one_hash_join",
]

_COMPAT_EXPORTS = {
    "VEHICLE_APPEARANCE_KEYS": ("mmds.case_studies", "VEHICLE_APPEARANCE_KEYS"),
    "CostReport": ("mmds.case_studies", "CostReport"),
    "EvalReport": ("mmds.case_studies", "EvalReport"),
    "TrajectoryMatch": ("mmds.case_studies", "TrajectoryMatch"),
    "canonical_corridor_pair": ("mmds.case_studies", "canonical_corridor_pair"),
    "different_cameras": ("mmds.case_studies", "different_cameras"),
    "direction_compatible": ("mmds.case_studies", "direction_compatible"),
    "evaluate_trajectories": ("mmds.case_studies", "evaluate_trajectories"),
    "greedy_match_trajectories": (
        "mmds.case_studies",
        "greedy_match_trajectories",
    ),
    "join_match_to_trajectory_record": (
        "mmds.case_studies",
        "join_match_to_trajectory_record",
    ),
    "normalize_trajectory_row": (
        "mmds.case_studies",
        "normalize_trajectory_row",
    ),
    "normalize_trajectory_rows": (
        "mmds.case_studies",
        "normalize_trajectory_rows",
    ),
    "orient_upstream_downstream": ("mmds.case_studies", "orient_upstream_downstream"),
    "same_vehicle": ("mmds.case_studies", "same_vehicle"),
    "speed_compatible": ("mmds.case_studies", "speed_compatible"),
    "timeline_segment_from_track": (
        "mmds.case_studies",
        "timeline_segment_from_track",
    ),
    "track_time_to_seconds": ("mmds.case_studies", "track_time_to_seconds"),
    "trajectory_attributes_from_track": (
        "mmds.case_studies",
        "trajectory_attributes_from_track",
    ),
    "travel_time_compatible": ("mmds.case_studies", "travel_time_compatible"),
    "vehicle_appearance_hash_key": ("mmds.case_studies", "vehicle_appearance_hash_key"),
    "vehicle_id_from_match": ("mmds.case_studies", "vehicle_id_from_match"),
}


def __getattr__(name: str):
    target = _COMPAT_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    import importlib
    import warnings

    module_name, symbol_name = target
    warnings.warn(
        f"mmds.join.{name} is deprecated; import {symbol_name} from "
        f"{module_name} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(module_name), symbol_name)
