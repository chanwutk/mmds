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
    timeline_segment_from_track,
    vehicle_id_from_match,
)
from udfs.trajectory_ops import join_match_to_trajectory  # noqa: E402
from udfs.trajectory_ops import promote_vehicle_trajectory_row  # noqa: E402


def _track(
    *,
    camera_id: str,
    track_id: str,
    first_frame_id: int,
    last_frame_id: int,
    fps: float = 30.0,
    start: str = "2024-01-01T00:00:00Z",
    end: str = "2024-01-01T00:00:01Z",
    vehicle_class: str = "suv",
    color: str = "white",
    subtype: str = "suv",
) -> dict:
    return {
        "camera_id": camera_id,
        "track_id": track_id,
        "start_time": start,
        "end_time": end,
        "first_frame_id": first_frame_id,
        "last_frame_id": last_frame_id,
        "fps": fps,
        "vehicle_class": vehicle_class,
        "color": color,
        "subtype": subtype,
        "confidence": 0.9,
    }


class TrajectoryTests(unittest.TestCase):
    def test_timeline_segment_from_track_uses_source_absolute_seconds(self) -> None:
        track = _track(
            camera_id="cam-i24v-highway2",
            track_id="suv-1",
            first_frame_id=90,
            last_frame_id=150,
            fps=30.0,
            # Wall-clock ISO that would yield huge Unix timestamps if misused.
            start="2024-01-01T12:00:03Z",
            end="2024-01-01T12:00:05Z",
        )
        segment = timeline_segment_from_track(track)
        assert segment is not None
        self.assertEqual(segment["camera_id"], "cam-i24v-highway2")
        self.assertAlmostEqual(segment["entered"], 3.0, places=6)
        self.assertAlmostEqual(segment["exited"], 5.0, places=6)

    def test_timeline_segment_requires_frame_ids_and_fps(self) -> None:
        track = _track(
            camera_id="cam-a",
            track_id="t1",
            first_frame_id=10,
            last_frame_id=20,
        )
        del track["fps"]
        self.assertIsNone(timeline_segment_from_track(track))

    def test_vehicle_id_is_stable(self) -> None:
        up = _track(camera_id="cam-a", track_id="t1", first_frame_id=0, last_frame_id=1)
        down = _track(camera_id="cam-b", track_id="t2", first_frame_id=2, last_frame_id=3)
        self.assertEqual(
            vehicle_id_from_match(up, down),
            vehicle_id_from_match(up, down),
        )
        self.assertTrue(vehicle_id_from_match(up, down).startswith("join_unique_"))

    def test_join_match_to_trajectory_record_orders_corridor(self) -> None:
        left = _track(
            camera_id="cam-i24v-highway3",
            track_id="suv-2",
            first_frame_id=300,
            last_frame_id=360,
        )
        right = _track(
            camera_id="cam-i24v-highway2",
            track_id="suv-1",
            first_frame_id=100,
            last_frame_id=160,
        )
        record = join_match_to_trajectory_record(left, right, match_score=0.88)
        assert record is not None
        self.assertEqual(record["timeline"][0]["camera_id"], "cam-i24v-highway2")
        self.assertEqual(record["timeline"][1]["camera_id"], "cam-i24v-highway3")
        self.assertEqual(record["attributes"]["class"], "suv")
        self.assertEqual(record["attributes"]["color"], "white")
        self.assertAlmostEqual(record["match_score"], 0.88)
        self.assertAlmostEqual(record["timeline"][0]["entered"], 100 / 30.0, places=6)

    def test_join_match_to_trajectory_udf(self) -> None:
        row = {
            "left": _track(
                camera_id="cam-i24v-highway2",
                track_id="suv-1",
                first_frame_id=100,
                last_frame_id=160,
            ),
            "right": _track(
                camera_id="cam-i24v-highway3",
                track_id="suv-2",
                first_frame_id=300,
                last_frame_id=360,
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
                first_frame_id=100,
                last_frame_id=160,
            ),
            "right": _track(
                camera_id="cam-i24v-highway3",
                track_id="suv-2",
                first_frame_id=300,
                last_frame_id=360,
            ),
            "match_score": 0.77,
            "extra": "drop-me",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "matches.jsonl"
            path.write_text(json.dumps(join_row) + "\n", encoding="utf-8")
            result = execute(Map(Input(str(path)), join_match_to_trajectory, replace=True))

        self.assertEqual(len(result), 1)
        exported = result[0]
        self.assertNotIn("left", exported)
        self.assertNotIn("right", exported)
        self.assertNotIn("extra", exported)
        self.assertIn("vehicle_id", exported)
        self.assertIn("timeline", exported)


class PromoteVehicleTrajectoryRowTests(unittest.TestCase):
    def _vehicle(self, **overrides) -> dict:
        record = {
            "vehicle_id": "v-1",
            "attributes": {"class": "suv", "color": "white", "subtype": "suv"},
            "timeline": [{"camera_id": "cam-a", "entered": 0.0, "exited": 2.0}],
        }
        record.update(overrides)
        return record

    def test_keeps_match_score_when_present(self) -> None:
        row = {"vehicles": self._vehicle(match_score=0.55)}
        promoted = promote_vehicle_trajectory_row(row)
        self.assertEqual(promoted["match_score"], 0.55)

    def test_keeps_records_without_match_score(self) -> None:
        row = {"vehicles": self._vehicle()}
        promoted = promote_vehicle_trajectory_row(row)
        self.assertNotIn("match_score", promoted)

    def test_matches_join_record_shape_without_score(self) -> None:
        record = {"vehicle_id": "v-2", "attributes": {}, "timeline": []}
        promoted = promote_vehicle_trajectory_row({"vehicles": record})
        self.assertEqual(set(promoted), {"vehicle_id", "attributes", "timeline"})

    def test_rejects_missing_required_fields_or_non_dict(self) -> None:
        self.assertEqual(promote_vehicle_trajectory_row({"vehicles": "x"}), {})
        for missing in ("vehicle_id", "attributes", "timeline"):
            vehicle = self._vehicle()
            del vehicle[missing]
            self.assertEqual(promote_vehicle_trajectory_row({"vehicles": vehicle}), {})


if __name__ == "__main__":
    unittest.main()
