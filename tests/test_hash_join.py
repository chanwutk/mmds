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

from mmds.join.hash_join import (  # noqa: E402
    VEHICLE_APPEARANCE_KEYS,
    hash_join,
    one_to_one_hash_join,
    vehicle_appearance_hash_key,
)
from udfs.join_ops import same_vehicle, vehicle_match_score  # noqa: E402
from udfs.test_ops import join_cross_camera_test_bucket, join_test_match_score  # noqa: E402


def _track(
    *,
    camera_id: str,
    track_id: str,
    vehicle_class: str = "sedan",
    color: str = "white",
    subtype: str = "compact",
    confidence: float = 0.9,
    start_time: str = "2024-01-01T00:00:00Z",
    end_time: str = "2024-01-01T00:00:05Z",
) -> dict:
    return {
        "camera_id": camera_id,
        "track_id": track_id,
        "vehicle_class": vehicle_class,
        "color": color,
        "subtype": subtype,
        "confidence": confidence,
        "start_time": start_time,
        "end_time": end_time,
        "entry_direction": "E",
        "exit_direction": "E",
        "avg_speed": 40.0,
    }


class HashJoinTests(unittest.TestCase):
    def test_vehicle_appearance_hash_key(self) -> None:
        row = _track(camera_id="highway2", track_id="t1")
        self.assertEqual(
            vehicle_appearance_hash_key(row),
            ("sedan", "white", "compact"),
        )
        self.assertEqual(VEHICLE_APPEARANCE_KEYS, ("vehicle_class", "color", "subtype"))

    def test_hash_join_only_compares_same_bucket(self) -> None:
        left = [
            _track(camera_id="highway2", track_id="a", color="white"),
            _track(camera_id="highway2", track_id="b", color="black"),
        ]
        right = [
            _track(camera_id="highway3", track_id="c", color="white"),
            _track(camera_id="highway3", track_id="d", color="black"),
        ]
        pairs = list(
            hash_join(
                left,
                right,
                VEHICLE_APPEARANCE_KEYS,
                join_cross_camera_test_bucket,
            )
        )
        self.assertEqual(len(pairs), 2)
        colors = {
            (pair["left"]["track_id"], pair["right"]["track_id"], pair["left"]["color"])
            for pair in pairs
        }
        self.assertIn(("a", "c", "white"), colors)
        self.assertIn(("b", "d", "black"), colors)

    def test_one_to_one_hash_join_picks_highest_score(self) -> None:
        left = [
            _track(camera_id="highway2", track_id="a", confidence=0.5),
            _track(camera_id="highway2", track_id="b", confidence=0.95),
        ]
        right = [
            _track(camera_id="highway3", track_id="c", confidence=0.4),
            _track(camera_id="highway3", track_id="d", confidence=0.9),
        ]
        pairs = list(
            one_to_one_hash_join(
                left,
                right,
                keys=VEHICLE_APPEARANCE_KEYS,
                predicate=join_cross_camera_test_bucket,
                score_fn=join_test_match_score,
                left_key=("camera_id", "track_id"),
                right_key=("camera_id", "track_id"),
            )
        )
        self.assertEqual(len(pairs), 2)
        by_left = {pair["left"]["track_id"]: pair for pair in pairs}
        self.assertEqual(by_left["b"]["right"]["track_id"], "d")
        self.assertGreater(by_left["b"]["match_score"], by_left["a"]["match_score"])

    def test_one_to_one_vehicle_hash_join_with_predicate(self) -> None:
        left = [_track(camera_id="highway2", track_id="a")]
        right = [
            _track(camera_id="highway3", track_id="b"),
            _track(camera_id="highway2", track_id="c"),
        ]
        pairs = list(
            one_to_one_hash_join(
                left,
                right,
                keys=VEHICLE_APPEARANCE_KEYS,
                predicate=same_vehicle,
                score_fn=vehicle_match_score,
                left_key=("camera_id", "track_id"),
                right_key=("camera_id", "track_id"),
            )
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["right"]["track_id"], "b")

    def test_self_join_emits_canonical_orientation_once(self) -> None:
        tracks = [
            _track(
                camera_id="cam-i24v-highway2",
                track_id="a",
                start_time="2024-01-01T00:00:01Z",
                end_time="2024-01-01T00:00:02Z",
            ),
            _track(
                camera_id="cam-i24v-highway3",
                track_id="b",
                start_time="2024-01-01T00:00:03Z",
                end_time="2024-01-01T00:00:04Z",
            ),
        ]
        pairs = list(
            one_to_one_hash_join(
                tracks,
                tracks,
                keys=VEHICLE_APPEARANCE_KEYS,
                predicate=same_vehicle,
                score_fn=vehicle_match_score,
                left_key=("camera_id", "track_id"),
                right_key=("camera_id", "track_id"),
            )
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["left"]["camera_id"], "cam-i24v-highway2")
        self.assertEqual(pairs[0]["right"]["camera_id"], "cam-i24v-highway3")


if __name__ == "__main__":
    unittest.main()
