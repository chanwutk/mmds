from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import Input, Map, execute  # noqa: E402
from udfs.trajectory_ops import (  # noqa: E402
    join_match_to_trajectory_record,
    parse_iso_timestamp,
    timeline_segment_from_track,
    vehicle_id_from_match,
)
from udfs.trajectory_ops import join_match_to_trajectory  # noqa: E402


def _track(
    *,
    camera_id: str,
    track_id: str,
    start: str,
    end: str,
    vehicle_class: str = "suv",
    color: str = "white",
    subtype: str = "hatchback",
) -> dict:
    return {
        "camera_id": camera_id,
        "track_id": track_id,
        "start_time": start,
        "end_time": end,
        "vehicle_class": vehicle_class,
        "color": color,
        "subtype": subtype,
        "confidence": 0.9,
    }


class TrajectoryTests(unittest.TestCase):
    def test_timeline_segment_from_track(self) -> None:
        track = _track(
            camera_id="cam-i24v-highway2",
            track_id="suv-1",
            start="2024-01-01T00:01:42.400000Z",
            end="2024-01-01T00:01:48.800000Z",
        )
        segment = timeline_segment_from_track(track)
        assert segment is not None
        self.assertEqual(segment["camera_id"], "cam-i24v-highway2")
        expected_entered = parse_iso_timestamp(track["start_time"])
        expected_exited = parse_iso_timestamp(track["end_time"])
        assert expected_entered is not None and expected_exited is not None
        self.assertAlmostEqual(segment["entered"], expected_entered.timestamp(), places=6)
        self.assertAlmostEqual(segment["exited"], expected_exited.timestamp(), places=6)

    def test_vehicle_id_is_stable(self) -> None:
        up = _track(camera_id="cam-a", track_id="t1", start="...", end="...")
        down = _track(camera_id="cam-b", track_id="t2", start="...", end="...")
        self.assertEqual(
            vehicle_id_from_match(up, down),
            vehicle_id_from_match(up, down),
        )
        self.assertTrue(vehicle_id_from_match(up, down).startswith("join_unique_"))

    def test_join_match_to_trajectory_record_orders_corridor(self) -> None:
        left = _track(
            camera_id="cam-i24v-highway3",
            track_id="suv-2",
            start="2024-01-01T00:01:57.100000Z",
            end="2024-01-01T00:02:03.500000Z",
        )
        right = _track(
            camera_id="cam-i24v-highway2",
            track_id="suv-1",
            start="2024-01-01T00:01:42.400000Z",
            end="2024-01-01T00:01:48.800000Z",
        )
        record = join_match_to_trajectory_record(left, right, match_score=0.88)
        assert record is not None
        self.assertEqual(record["timeline"][0]["camera_id"], "cam-i24v-highway2")
        self.assertEqual(record["timeline"][1]["camera_id"], "cam-i24v-highway3")
        self.assertEqual(record["attributes"]["class"], "suv")
        self.assertEqual(record["attributes"]["color"], "white")
        self.assertAlmostEqual(record["match_score"], 0.88)

    def test_join_match_to_trajectory_udf(self) -> None:
        row = {
            "left": _track(
                camera_id="cam-i24v-highway2",
                track_id="suv-1",
                start="2024-01-01T00:01:42.400000Z",
                end="2024-01-01T00:01:48.800000Z",
            ),
            "right": _track(
                camera_id="cam-i24v-highway3",
                track_id="suv-2",
                start="2024-01-01T00:01:57.100000Z",
                end="2024-01-01T00:02:03.500000Z",
            ),
            "match_score": 0.91,
        }
        record = join_match_to_trajectory(row)
        self.assertIn("vehicle_id", record)
        self.assertEqual(len(record["timeline"]), 2)
        self.assertLess(record["timeline"][0]["entered"], record["timeline"][1]["entered"])

    def test_trajectory_map_replace_drops_join_fields(self) -> None:
        join_row = {
            "left": _track(
                camera_id="cam-i24v-highway2",
                track_id="suv-1",
                start="2024-01-01T00:01:42.400000Z",
                end="2024-01-01T00:01:48.800000Z",
            ),
            "right": _track(
                camera_id="cam-i24v-highway3",
                track_id="suv-2",
                start="2024-01-01T00:01:57.100000Z",
                end="2024-01-01T00:02:03.500000Z",
            ),
            "match_score": 0.91,
            "detections": [{"should": "not appear"}],
        }
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        try:
            handle.write(json.dumps(join_row))
            handle.write("\n")
            handle.close()
            plan = Map(Input(handle.name), join_match_to_trajectory, replace=True)
            result = execute(plan)
        finally:
            Path(handle.name).unlink(missing_ok=True)

        self.assertEqual(len(result), 1)
        exported = result[0]
        self.assertNotIn("left", exported)
        self.assertNotIn("right", exported)
        self.assertNotIn("detections", exported)
        self.assertIn("vehicle_id", exported)
        self.assertIn("timeline", exported)
        self.assertAlmostEqual(exported["match_score"], 0.91)


if __name__ == "__main__":
    unittest.main()
