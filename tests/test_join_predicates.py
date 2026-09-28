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

from udfs.join_ops import (  # noqa: E402
    canonical_corridor_pair,
    different_cameras,
    direction_compatible,
    same_vehicle,
    speed_compatible,
    temporal_iou,
    temporal_overlap,
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


class PredicateEdgeCaseTests(unittest.TestCase):
    UP = "cam-i24v-highway2"
    DOWN = "cam-i24v-highway3"

    def _pair(self, *, up=("12:00:00", "12:00:02"), down=("12:00:03", "12:00:05"), **down_kwargs):
        left = _summary(
            camera_id=self.UP,
            start_time=f"2023-08-10T{up[0]}Z",
            end_time=f"2023-08-10T{up[1]}Z",
        )
        right = _summary(
            camera_id=self.DOWN,
            start_time=f"2023-08-10T{down[0]}Z",
            end_time=f"2023-08-10T{down[1]}Z",
            **down_kwargs,
        )
        return left, right

    def test_travel_time_rejects_malformed_or_missing_timestamps(self) -> None:
        for bad in ("not a time", "", None, 12345):
            with self.subTest(start_time=bad):
                left, right = self._pair()
                right["start_time"] = bad
                self.assertFalse(travel_time_compatible(left, right))
        left, right = self._pair()
        del left["end_time"]
        self.assertFalse(travel_time_compatible(left, right))

    def test_travel_time_gap_boundary(self) -> None:
        left, right = self._pair(up=("12:00:00", "12:00:02"), down=("12:02:02", "12:02:05"))
        self.assertTrue(travel_time_compatible(left, right))
        left, right = self._pair(up=("12:00:00", "12:00:02"), down=("12:02:03", "12:02:05"))
        self.assertFalse(travel_time_compatible(left, right))

    def test_travel_time_is_symmetric_in_argument_order(self) -> None:
        left, right = self._pair(up=("12:00:00", "12:00:02"), down=("12:00:30", "12:00:32"))
        self.assertTrue(travel_time_compatible(left, right))
        self.assertTrue(travel_time_compatible(right, left))

    def test_travel_time_rejects_downstream_before_upstream(self) -> None:
        left, right = self._pair(up=("12:01:00", "12:01:02"), down=("12:00:00", "12:00:02"))
        self.assertFalse(travel_time_compatible(left, right))

    def test_travel_time_overlapping_intervals_are_compatible(self) -> None:
        left, right = self._pair(up=("12:00:00", "12:00:04"), down=("12:00:03", "12:00:05"))
        self.assertTrue(travel_time_compatible(left, right))

    def test_travel_time_falls_back_to_frame_indices(self) -> None:
        left = {"camera_id": self.UP, "first_frame_idx": 0, "fps": 30.0}
        self.assertTrue(
            travel_time_compatible(left, {"camera_id": self.DOWN, "first_frame_idx": 300})
        )
        self.assertFalse(
            travel_time_compatible(left, {"camera_id": self.DOWN, "first_frame_idx": 301})
        )
        self.assertFalse(travel_time_compatible({"camera_id": self.UP}, {"camera_id": self.DOWN}))

    def test_direction_angle_boundary(self) -> None:
        cases = {"E": True, "SE": True, "N": True, "NW": False, "W": False}
        for entry, expected in cases.items():
            with self.subTest(entry=entry):
                left, right = self._pair(entry_direction=entry)
                self.assertEqual(direction_compatible(left, right), expected)

    def test_direction_rejects_stationary_slow_or_missing(self) -> None:
        left, right = self._pair(entry_direction="stationary")
        self.assertFalse(direction_compatible(left, right))
        left, right = self._pair(avg_speed=0.5)
        self.assertFalse(direction_compatible(left, right))
        left, right = self._pair()
        del right["entry_direction"]
        self.assertFalse(direction_compatible(left, right))

    def test_direction_non_compass_labels_must_match_exactly(self) -> None:
        left, right = self._pair(entry_direction="left")
        left["exit_direction"] = "left"
        self.assertTrue(direction_compatible(left, right))
        right["entry_direction"] = "right"
        self.assertFalse(direction_compatible(left, right))

    def test_speed_ratio_boundary_and_invalid_speeds(self) -> None:
        left, right = self._pair(avg_speed=20.0)
        self.assertTrue(speed_compatible(left, right))
        left, right = self._pair(avg_speed=19.9)
        self.assertFalse(speed_compatible(left, right))
        left, right = self._pair(avg_speed=0.0)
        left["avg_speed"] = 0.0
        self.assertTrue(speed_compatible(left, right))
        for bad in (-1.0, float("nan"), float("inf"), True, "40", None):
            with self.subTest(avg_speed=bad):
                left, right = self._pair()
                right["avg_speed"] = bad
                self.assertFalse(speed_compatible(left, right))

    def test_temporal_overlap_tolerance(self) -> None:
        left, right = self._pair(up=("12:00:00", "12:00:02"), down=("12:00:04", "12:00:06"))
        self.assertTrue(temporal_overlap(left, right))
        left, right = self._pair(up=("12:00:00", "12:00:02"), down=("12:00:05", "12:00:06"))
        self.assertFalse(temporal_overlap(left, right))

    def test_temporal_overlap_and_iou_reject_invalid_intervals(self) -> None:
        left, right = self._pair()
        right["start_time"] = "garbage"
        self.assertFalse(temporal_overlap(left, right))
        self.assertEqual(temporal_iou(left, right), 0.0)
        left, right = self._pair(down=("12:00:05", "12:00:03"))
        self.assertFalse(temporal_overlap(left, right))
        self.assertEqual(temporal_iou(left, right), 0.0)

    def test_temporal_iou_values(self) -> None:
        left, right = self._pair(up=("12:00:00", "12:00:02"), down=("12:00:00", "12:00:02"))
        self.assertEqual(temporal_iou(left, right), 1.0)
        left, right = self._pair(up=("12:00:00", "12:00:02"), down=("12:00:01", "12:00:03"))
        self.assertAlmostEqual(temporal_iou(left, right), 1.0 / 3.0)
        left, right = self._pair(up=("12:00:00", "12:00:02"), down=("12:00:05", "12:00:06"))
        self.assertEqual(temporal_iou(left, right), 0.0)
        left, right = self._pair(up=("12:00:00", "12:00:00"), down=("12:00:00", "12:00:00"))
        self.assertEqual(temporal_iou(left, right), 1.0)


if __name__ == "__main__":
    unittest.main()
