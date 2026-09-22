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

from examples import join_cross_camera_vehicle as example  # noqa: E402
from mmds import Input, Join, Map, execute  # noqa: E402
from mmds.model import JoinSpec  # noqa: E402
from udfs.join_ops import canonical_corridor_pair, same_vehicle  # noqa: E402
from udfs.reid_ops import appearance_match_score  # noqa: E402
from udfs.trajectory_ops import join_match_to_trajectory  # noqa: E402


def _track(
    camera_id: str,
    track_id: str,
    *,
    embedding: list[float] | None = None,
    start_time: str = "2024-01-01T00:00:01Z",
    end_time: str = "2024-01-01T00:00:02Z",
) -> dict:
    return {
        "camera_id": camera_id,
        "track_id": track_id,
        "start_time": start_time,
        "end_time": end_time,
        "vehicle_class": "sedan",
        "color": "white",
        "subtype": "compact",
        "avg_speed": 40.0,
        "entry_direction": "E",
        "exit_direction": "E",
        "confidence": 0.9,
        "embedding": embedding or [1.0, 0.0],
    }


class CrossCameraVehicleJoinTests(unittest.TestCase):
    def test_default_manifest(self) -> None:
        self.assertEqual(
            example.DEFAULT_MANIFEST,
            "data/i24v_traffic_highway2_highway3_5s.jsonl",
        )
        self.assertEqual(
            next(
                node.input_path
                for node in example.build_query().walk_postorder()
                if node.kind == "input"
            ),
            example.DEFAULT_MANIFEST,
        )

    def test_exact_named_stage_sequence_handles_shared_self_join(self) -> None:
        plan = example.build_query("tracks.jsonl")

        self.assertEqual(
            [(node.kind, node.name) for node in plan.walk_postorder()],
            [
                ("input", None),
                ("detect", "detect_vehicles"),
                ("map", "suppress_duplicate_detections"),
                ("map", "build_frame_detections"),
                ("map", "track_vehicles"),
                ("map", "embed_track_summaries"),
                ("unnest", "one_track_per_row"),
                ("map", "promote_track_summary"),
                ("map", "project_track_fields"),
                ("join", "match_cross_camera_tracks"),
                ("map", "build_vehicle_trajectories"),
            ],
        )
        join = plan.source
        self.assertIsNotNone(join)
        self.assertEqual(join.kind, "join")
        self.assertIs(join.source, join.right_source)

    def test_exact_join_spec_configuration(self) -> None:
        join = example.build_query("tracks.jsonl", min_score=0.65).source

        self.assertIsNotNone(join)
        self.assertIsInstance(join.spec, JoinSpec)
        assert isinstance(join.spec, JoinSpec)
        self.assertEqual(join.spec.keys, ())
        self.assertEqual(
            (join.spec.predicate.module, join.spec.predicate.name),
            ("udfs.join_ops", "same_vehicle"),
        )
        self.assertEqual(
            (join.spec.score.module, join.spec.score.name),
            ("udfs.reid_ops", "appearance_match_score"),
        )
        self.assertEqual(join.spec.min_score, 0.65)
        self.assertTrue(join.spec.one_to_one)
        self.assertEqual(join.spec.left_key, ("camera_id", "track_id"))
        self.assertEqual(join.spec.right_key, ("camera_id", "track_id"))

    def test_synthetic_join_to_replaced_trajectory_execution(self) -> None:
        rows = [
            _track("cam-i24v-highway2", "up-a", embedding=[1.0, 0.0]),
            _track(
                "cam-i24v-highway3",
                "down-a",
                embedding=[1.0, 0.0],
                start_time="2024-01-01T00:00:03Z",
                end_time="2024-01-01T00:00:04Z",
            ),
            _track("cam-i24v-highway2", "up-b", embedding=[0.0, 1.0]),
            _track(
                "cam-i24v-highway3",
                "down-b",
                embedding=[0.0, 1.0],
                start_time="2024-01-01T00:00:05Z",
                end_time="2024-01-01T00:00:06Z",
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tracks.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            tracks = Input(str(path))
            result = execute(
                Map(
                    Join(
                        tracks,
                        tracks,
                        same_vehicle,
                        one_to_one=True,
                        score=appearance_match_score,
                        min_score=0.4,
                        left_key=("camera_id", "track_id"),
                        right_key=("camera_id", "track_id"),
                    ),
                    join_match_to_trajectory,
                    replace=True,
                )
            )

        self.assertEqual(len(result), 2)
        timelines = {
            tuple(segment["camera_id"] for segment in row["timeline"])
            for row in result
        }
        self.assertEqual(
            timelines,
            {("cam-i24v-highway2", "cam-i24v-highway3")},
        )
        self.assertTrue(all(row["match_score"] == 1.0 for row in result))
        self.assertTrue(all(set(row) == {"vehicle_id", "attributes", "timeline", "match_score"} for row in result))

    def test_same_vehicle_rejections_and_canonical_corridor_ordering(self) -> None:
        upstream = _track("cam-i24v-highway2", "up")
        downstream = _track(
            "cam-i24v-highway3",
            "down",
            start_time="2024-01-01T00:00:03Z",
            end_time="2024-01-01T00:00:04Z",
        )
        cases = {
            "valid canonical pair": (upstream, downstream, True),
            "reverse corridor order": (downstream, upstream, False),
            "same camera": (upstream, {**downstream, "camera_id": upstream["camera_id"]}, False),
            "missing direction": (upstream, {**downstream, "entry_direction": None}, False),
            "implausible speed": (upstream, {**downstream, "avg_speed": 10.0}, False),
            "non-finite speed": (upstream, {**downstream, "avg_speed": float("nan")}, False),
            "infinite speed": (upstream, {**downstream, "avg_speed": float("inf")}, False),
            "boolean speed": (upstream, {**downstream, "avg_speed": True}, False),
            "travel gap too long": (
                upstream,
                {
                    **downstream,
                    "start_time": "2024-01-01T00:10:03Z",
                    "end_time": "2024-01-01T00:10:04Z",
                },
                False,
            ),
        }
        for label, (left, right, expected) in cases.items():
            with self.subTest(label=label):
                self.assertEqual(same_vehicle(left, right), expected)
        self.assertTrue(canonical_corridor_pair(upstream, downstream))
        self.assertFalse(canonical_corridor_pair(downstream, upstream))


if __name__ == "__main__":
    unittest.main()
