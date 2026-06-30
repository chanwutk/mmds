from __future__ import annotations

from typing import Any

from mmds.join.trajectory import join_match_to_trajectory_record


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
    left = row.get("left")
    right = row.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return {}

    match_score = row.get("match_score")
    score = float(match_score) if isinstance(match_score, (int, float)) else None
    record = join_match_to_trajectory_record(left, right, match_score=score)
    return record if record is not None else {}
