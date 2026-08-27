"""Deprecated compatibility imports for traffic-specific join predicates."""

from warnings import warn

warn(
    "mmds.join.predicates is deprecated; import traffic predicates from "
    "mmds.case_studies.predicates instead.",
    DeprecationWarning,
    stacklevel=2,
)

from mmds.case_studies.predicates import (  # noqa: E402,F401
    canonical_corridor_pair,
    different_cameras,
    direction_compatible,
    orient_upstream_downstream,
    same_vehicle,
    speed_compatible,
    temporal_iou,
    temporal_overlap,
    travel_time_compatible,
)

__all__ = [
    "canonical_corridor_pair",
    "different_cameras",
    "direction_compatible",
    "orient_upstream_downstream",
    "same_vehicle",
    "speed_compatible",
    "temporal_iou",
    "temporal_overlap",
    "travel_time_compatible",
]
