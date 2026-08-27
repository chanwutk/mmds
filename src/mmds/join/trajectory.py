"""Deprecated compatibility imports for traffic trajectory export."""

from warnings import warn

warn(
    "mmds.join.trajectory is deprecated; import trajectory helpers from "
    "mmds.case_studies.trajectory instead.",
    DeprecationWarning,
    stacklevel=2,
)

from mmds.case_studies.trajectory import (  # noqa: E402,F401
    join_match_to_trajectory_record,
    parse_iso_timestamp,
    timeline_segment_from_track,
    track_time_to_seconds,
    trajectory_attributes_from_track,
    vehicle_id_from_match,
)

__all__ = [
    "join_match_to_trajectory_record",
    "parse_iso_timestamp",
    "timeline_segment_from_track",
    "track_time_to_seconds",
    "trajectory_attributes_from_track",
    "vehicle_id_from_match",
]
