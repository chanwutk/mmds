"""Deprecated compatibility imports for example trajectory evaluation."""

from warnings import warn

warn(
    "mmds.join.eval_trajectories is deprecated; import evaluation helpers from "
    "mmds.case_studies.trajectory_evaluation instead.",
    DeprecationWarning,
    stacklevel=2,
)

from mmds.case_studies.trajectory_evaluation import (  # noqa: E402,F401
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
