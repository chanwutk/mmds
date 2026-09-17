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

from mmds.join.hash_join import hash_join, one_to_one_hash_join  # noqa: E402
from mmds.model import MMDSValidationError  # noqa: E402
from udfs.test_ops import (  # noqa: E402
    join_cross_camera_test_bucket,
    join_test_match_score,
)


def _row(
    camera_id: str,
    track_id: str,
    confidence: float,
    *,
    bucket: object = "sedan",
) -> dict:
    return {
        "camera_id": camera_id,
        "track_id": track_id,
        "bucket": bucket,
        "confidence": confidence,
    }


class HashJoinTests(unittest.TestCase):
    def test_hash_join_skips_missing_and_unhashable_keys(self) -> None:
        pairs = list(
            hash_join(
                [
                    _row("left", "a", 0.9),
                    _row("left", "missing", 0.9) | {"bucket": ["unhashable"]},
                    {"camera_id": "left", "track_id": "absent"},
                ],
                [
                    _row("right", "b", 0.8),
                    _row("right", "bad", 0.8, bucket=["unhashable"]),
                ],
                ("bucket",),
            )
        )

        self.assertEqual(
            [(pair["left"]["track_id"], pair["right"]["track_id"]) for pair in pairs],
            [("a", "b")],
        )

    def test_greedy_one_to_one_selects_highest_unique_pairs(self) -> None:
        pairs = list(
            one_to_one_hash_join(
                [
                    _row("left", "a", 0.5),
                    _row("left", "b", 0.95),
                ],
                [
                    _row("right", "c", 0.4),
                    _row("right", "d", 0.9),
                ],
                keys=("bucket",),
                predicate=join_cross_camera_test_bucket,
                score_fn=join_test_match_score,
                left_key=("camera_id", "track_id"),
                right_key=("camera_id", "track_id"),
                min_score=0.45,
            )
        )

        self.assertEqual(len(pairs), 2)
        by_left = {pair["left"]["track_id"]: pair for pair in pairs}
        self.assertEqual(by_left["b"]["right"]["track_id"], "d")
        self.assertEqual(by_left["a"]["right"]["track_id"], "c")
        self.assertEqual(by_left["b"]["match_score"], 0.925)

    def test_one_to_one_rejects_invalid_scores(self) -> None:
        left = [_row("left", "a", 0.9)]
        right = [_row("right", "b", 0.8)]
        for invalid in (True, "0.9", None, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    MMDSValidationError,
                    "finite real number",
                ):
                    list(
                        one_to_one_hash_join(
                            left,
                            right,
                            keys=("bucket",),
                            predicate=None,
                            score_fn=lambda _left, _right, value=invalid: value,
                            left_key=("camera_id", "track_id"),
                            right_key=("camera_id", "track_id"),
                        )
                    )


    def test_one_to_one_skips_scoring_when_identity_is_missing(self) -> None:
        scored: list[tuple[str, str]] = []

        def score_fn(left: dict, right: dict) -> float:
            scored.append((left["track_id"], right["track_id"]))
            return 0.9

        pairs = list(
            one_to_one_hash_join(
                [
                    {"camera_id": "left", "track_id": "a", "bucket": "sedan"},
                    {"camera_id": "left", "bucket": "sedan"},
                ],
                [
                    {"camera_id": "right", "track_id": "b", "bucket": "sedan"},
                ],
                keys=("bucket",),
                predicate=None,
                score_fn=score_fn,
                left_key=("camera_id", "track_id"),
                right_key=("camera_id", "track_id"),
            )
        )

        self.assertEqual(pairs[0]["left"]["track_id"], "a")
        self.assertEqual(scored, [("a", "b")])


if __name__ == "__main__":
    unittest.main()
