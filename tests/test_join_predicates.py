from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds.join.predicates import (  # noqa: E402
    canonical_corridor_pair,
    different_cameras,
    direction_compatible,
    same_vehicle,
    speed_compatible,
    travel_time_compatible,
)


def _summary(
    *,
    camera_id: str,
    start_time: str,
    end_time: str,
    entry_direction: str = "E",
    exit_direction: str = "E",
    avg_speed: float = 40.0,
) -> dict:
    return {
        "track_id": f"{camera_id}-1",
        "camera_id": camera_id,
        "start_time": start_time,
        "end_time": end_time,
        "vehicle_class": "suv",
        "color": "black",
        "subtype": "suv",
        "entry_direction": entry_direction,
        "exit_direction": exit_direction,
        "avg_speed": avg_speed,
        "centroid_path": [],
        "confidence": 0.9,
    }


class JoinPredicateTests(unittest.TestCase):
    def test_same_vehicle_accepts_compatible_highway_pair(self) -> None:
        left = _summary(
            camera_id="cam-i24v-highway2",
            start_time="2023-08-10T12:00:00Z",
            end_time="2023-08-10T12:00:02Z",
            exit_direction="E",
            avg_speed=42.0,
        )
        right = _summary(
            camera_id="cam-i24v-highway3",
            start_time="2023-08-10T12:00:03Z",
            end_time="2023-08-10T12:00:05Z",
            entry_direction="E",
            avg_speed=40.0,
        )
        self.assertTrue(same_vehicle(left, right))

    def test_canonical_corridor_pair_requires_upstream_on_left(self) -> None:
        upstream = _summary(
            camera_id="cam-i24v-highway2",
            start_time="2023-08-10T12:00:00Z",
            end_time="2023-08-10T12:00:02Z",
        )
        downstream = _summary(
            camera_id="cam-i24v-highway3",
            start_time="2023-08-10T12:00:03Z",
            end_time="2023-08-10T12:00:05Z",
        )
        self.assertTrue(canonical_corridor_pair(upstream, downstream))
        self.assertFalse(canonical_corridor_pair(downstream, upstream))
        self.assertFalse(same_vehicle(downstream, upstream))

    def test_rejects_same_camera(self) -> None:
        track = _summary(
            camera_id="cam-i24v-highway2",
            start_time="2023-08-10T12:00:00Z",
            end_time="2023-08-10T12:00:01Z",
        )
        self.assertFalse(different_cameras(track, track))
        self.assertFalse(same_vehicle(track, track))

    def test_rejects_incompatible_travel_time(self) -> None:
        left = _summary(
            camera_id="cam-i24v-highway2",
            start_time="2023-08-10T12:00:00Z",
            end_time="2023-08-10T12:00:01Z",
        )
        right = _summary(
            camera_id="cam-i24v-highway3",
            start_time="2023-08-10T12:05:00Z",
            end_time="2023-08-10T12:05:02Z",
        )
        self.assertFalse(travel_time_compatible(left, right))
        self.assertFalse(same_vehicle(left, right))

    def test_rejects_incompatible_direction(self) -> None:
        left = _summary(
            camera_id="cam-i24v-highway2",
            start_time="2023-08-10T12:00:00Z",
            end_time="2023-08-10T12:00:01Z",
            exit_direction="W",
        )
        right = _summary(
            camera_id="cam-i24v-highway3",
            start_time="2023-08-10T12:00:02Z",
            end_time="2023-08-10T12:00:03Z",
            entry_direction="E",
        )
        self.assertFalse(direction_compatible(left, right))
        self.assertFalse(same_vehicle(left, right))

    def test_rejects_incompatible_speed(self) -> None:
        left = _summary(
            camera_id="cam-i24v-highway2",
            start_time="2023-08-10T12:00:00Z",
            end_time="2023-08-10T12:00:01Z",
            avg_speed=100.0,
        )
        right = _summary(
            camera_id="cam-i24v-highway3",
            start_time="2023-08-10T12:00:02Z",
            end_time="2023-08-10T12:00:03Z",
            avg_speed=20.0,
        )
        self.assertFalse(speed_compatible(left, right))
        self.assertFalse(same_vehicle(left, right))


if __name__ == "__main__":
    unittest.main()
