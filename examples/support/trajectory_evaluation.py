"""Compatibility re-export of packaged trajectory evaluation helpers."""

from mmds.case_studies.trajectory_evaluation import (
    CostReport,
    EvalReport,
    TrajectoryMatch,
    attribute_agreement,
    canonicalize_class,
    canonicalize_color,
    colors_soft_match,
    endpoints_within_or_tolerance,
    evaluate_trajectories,
    greedy_match_trajectories,
    interval_iou,
    normalize_label,
    normalize_trajectory_row,
    normalize_trajectory_rows,
    pair_score,
    timeline_overlap_score,
)

__all__ = [
    "CostReport",
    "EvalReport",
    "TrajectoryMatch",
    "attribute_agreement",
    "canonicalize_class",
    "canonicalize_color",
    "colors_soft_match",
    "endpoints_within_or_tolerance",
    "evaluate_trajectories",
    "greedy_match_trajectories",
    "interval_iou",
    "normalize_label",
    "normalize_trajectory_row",
    "normalize_trajectory_rows",
    "pair_score",
    "timeline_overlap_score",
]
