"""Lazily exported reusable case-study components.

These helpers ship with :mod:`mmds` but remain separate from the generic join
execution package.
"""

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
    "timeline_segment_from_track": (".trajectory", "timeline_segment_from_track"),
    "track_time_to_seconds": (".trajectory", "track_time_to_seconds"),
    "trajectory_attributes_from_track": (".trajectory", "trajectory_attributes_from_track"),
    "vehicle_id_from_match": (".trajectory", "vehicle_id_from_match"),
    "format_gt_caption_clip": (".text_retrieval", "format_gt_caption_clip"),
    "group_gt_clips_by_video": (".text_retrieval", "group_gt_clips_by_video"),
    "tag_gallery_video": (".text_retrieval", "tag_gallery_video"),
    "CostReport": (".trajectory_evaluation", "CostReport"),
    "EvalReport": (".trajectory_evaluation", "EvalReport"),
    "TrajectoryMatch": (".trajectory_evaluation", "TrajectoryMatch"),
    "attribute_agreement": (".trajectory_evaluation", "attribute_agreement"),
    "canonicalize_class": (".trajectory_evaluation", "canonicalize_class"),
    "canonicalize_color": (".trajectory_evaluation", "canonicalize_color"),
    "colors_soft_match": (".trajectory_evaluation", "colors_soft_match"),
    "endpoints_within_or_tolerance": (
        ".trajectory_evaluation",
        "endpoints_within_or_tolerance",
    ),
    "evaluate_trajectories": (".trajectory_evaluation", "evaluate_trajectories"),
    "greedy_match_trajectories": (
        ".trajectory_evaluation",
        "greedy_match_trajectories",
    ),
    "interval_iou": (".trajectory_evaluation", "interval_iou"),
    "normalize_label": (".trajectory_evaluation", "normalize_label"),
    "normalize_trajectory_row": (
        ".trajectory_evaluation",
        "normalize_trajectory_row",
    ),
    "normalize_trajectory_rows": (
        ".trajectory_evaluation",
        "normalize_trajectory_rows",
    ),
    "pair_score": (".trajectory_evaluation", "pair_score"),
    "timeline_overlap_score": (".trajectory_evaluation", "timeline_overlap_score"),
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
