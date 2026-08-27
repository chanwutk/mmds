from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mmds import Input, Join, Map, execute
from udfs.join_ops import same_vehicle, vehicle_match_score
from udfs.trajectory_ops import join_match_to_trajectory


def _track(
    camera_id: str,
    track_id: str,
    start_time: str,
    end_time: str,
) -> dict:
    return {
        "camera_id": camera_id,
        "track_id": track_id,
        "start_time": start_time,
        "end_time": end_time,
        "vehicle_class": "suv",
        "color": "white",
        "subtype": "hatchback",
        "confidence": 0.9,
        "entry_direction": "E",
        "exit_direction": "E",
        "avg_speed": 40.0,
    }


class JoinTrajectoryGoldenTests(unittest.TestCase):
    def test_synthetic_join_to_trajectory_golden(self) -> None:
        rows = [
            _track(
                "cam-i24v-highway2",
                "veh-1",
                "2024-01-01T00:01:42.400000Z",
                "2024-01-01T00:01:48.800000Z",
            ),
            _track(
                "cam-i24v-highway3",
                "veh-2",
                "2024-01-01T00:01:57.100000Z",
                "2024-01-01T00:02:03.500000Z",
            ),
        ]
        handle = tempfile.NamedTemporaryFile(
            "w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        )
        try:
            for row in rows:
                handle.write(json.dumps(row))
                handle.write("\n")
            handle.close()
            source = Input(handle.name)
            matches = Join(
                source,
                source,
                same_vehicle,
                one_to_one=True,
                score=vehicle_match_score,
                min_score=0.0,
                left_key=("camera_id", "track_id"),
                right_key=("camera_id", "track_id"),
            )
            result = execute(
                Map(
                    matches,
                    join_match_to_trajectory,
                    replace=True,
                )
            )
        finally:
            Path(handle.name).unlink(missing_ok=True)

        self.assertEqual(len(result), 1)
        trajectory = result[0]
        self.assertEqual(
            trajectory["attributes"],
            {"class": "suv", "color": "white", "subtype": "hatchback"},
        )
        self.assertEqual(
            [segment["camera_id"] for segment in trajectory["timeline"]],
            ["cam-i24v-highway2", "cam-i24v-highway3"],
        )
        self.assertLess(
            trajectory["timeline"][0]["entered"],
            trajectory["timeline"][1]["entered"],
        )
        self.assertEqual(trajectory["match_score"], 0.9)
        self.assertTrue(trajectory["vehicle_id"].startswith("join_unique_"))
        self.assertNotIn("left", trajectory)
        self.assertNotIn("right", trajectory)


if __name__ == "__main__":
    unittest.main()
