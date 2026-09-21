"""UDF wrappers for reusable cross-camera trajectory helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mmds.case_studies import trajectory as _impl

Track = dict[str, Any]
TimelineSegment = dict[str, Any]
TrajectoryRecord = dict[str, Any]


def parse_iso_timestamp(value: Any) -> datetime | None:
    return _impl.parse_iso_timestamp(value)


def track_time_to_seconds(track: Track, *, field: str) -> float | None:
    return _impl.track_time_to_seconds(track, field=field)


def timeline_segment_from_track(track: Track) -> TimelineSegment | None:
    return _impl.timeline_segment_from_track(track)


def trajectory_attributes_from_track(track: Track) -> dict[str, str]:
    return _impl.trajectory_attributes_from_track(track)


def vehicle_id_from_match(upstream: Track, downstream: Track) -> str:
    return _impl.vehicle_id_from_match(upstream, downstream)


def join_match_to_trajectory_record(
    left: Track,
    right: Track,
    *,
    match_score: float | None = None,
) -> TrajectoryRecord | None:
    return _impl.join_match_to_trajectory_record(
        left,
        right,
        match_score=match_score,
    )


def join_match_to_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    """Map UDF: convert one Join output row into a vehicle trajectory record.

    Input rows must contain ``left`` and ``right`` track summaries (as produced
    by a one-to-one cross-camera ``Join``). Output shape::

        {
          "vehicle_id": "join_unique_<hash>",
          "attributes": {"class": "...", "color": "...", "subtype": "..."},
          "timeline": [
            {"camera_id": "...", "entered": <epoch_sec>, "exited": <epoch_sec>},
            ...
          ],
          "match_score": <float>  # when present on the join row
        }
    """
    return _impl.join_match_to_trajectory(row)
