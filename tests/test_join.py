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

from mmds import Input, Join, execute, parse_query, render_query  # noqa: E402
from mmds.model import DatasetExpr, JoinSpec, MMDSValidationError  # noqa: E402


def _write_jsonl(rows: list[dict]) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for row in rows:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    return handle.name


class JoinDslTests(unittest.TestCase):
    def test_join_requires_keys_or_predicate(self) -> None:
        left = Input("left.jsonl")
        right = Input("right.jsonl")
        with self.assertRaises(MMDSValidationError):
            Join(left, right)

    def test_join_rejects_non_callable_predicate(self) -> None:
        left = Input("left.jsonl")
        right = Input("right.jsonl")
        with self.assertRaises(TypeError):
            Join(left, right, "not-a-udf")  # type: ignore[arg-type]

    def test_join_requires_right_source(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DatasetExpr(
                kind="join",
                source=Input("a.jsonl"),
                spec=JoinSpec(keys=("incident_id",)),
            )


class JoinExecutionTests(unittest.TestCase):
    def test_join_on_incident_id_with_hash_keys(self) -> None:
        left_path = _write_jsonl(
            [
                {"incident_id": "a", "title": "video a"},
                {"incident_id": "b", "title": "video b"},
            ]
        )
        right_path = _write_jsonl(
            [
                {"incident_id": "a", "paragraph": "report a"},
                {"incident_id": "c", "paragraph": "orphan"},
            ]
        )
        try:
            rows = execute(
                Join(
                    Input(left_path),
                    Input(right_path),
                    on="incident_id",
                )
            )
        finally:
            Path(left_path).unlink(missing_ok=True)
            Path(right_path).unlink(missing_ok=True)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["left"]["incident_id"], "a")
        self.assertEqual(rows[0]["right"]["paragraph"], "report a")

    def test_join_with_predicate_and_hash_keys(self) -> None:
        from udfs.test_ops import join_rows_share_incident_id

        left_path = _write_jsonl([{"incident_id": "a", "title": "video a"}])
        right_path = _write_jsonl([{"incident_id": "a", "paragraph": "report a"}])
        try:
            rows = execute(
                Join(
                    Input(left_path),
                    Input(right_path),
                    join_rows_share_incident_id,
                    on="incident_id",
                )
            )
        finally:
            Path(left_path).unlink(missing_ok=True)
            Path(right_path).unlink(missing_ok=True)

        self.assertEqual(len(rows), 1)

    def test_one_to_one_hash_join(self) -> None:
        from udfs.test_ops import join_cross_camera_test_bucket, join_test_match_score

        left_path = _write_jsonl(
            [
                {
                    "camera_id": "cam_a",
                    "track_id": "t1",
                    "vehicle_class": "sedan",
                    "color": "white",
                    "subtype": "compact",
                    "confidence": 0.5,
                },
                {
                    "camera_id": "cam_a",
                    "track_id": "t2",
                    "vehicle_class": "sedan",
                    "color": "white",
                    "subtype": "compact",
                    "confidence": 0.95,
                },
            ]
        )
        right_path = _write_jsonl(
            [
                {
                    "camera_id": "cam_b",
                    "track_id": "t3",
                    "vehicle_class": "sedan",
                    "color": "white",
                    "subtype": "compact",
                    "confidence": 0.4,
                },
                {
                    "camera_id": "cam_b",
                    "track_id": "t4",
                    "vehicle_class": "sedan",
                    "color": "white",
                    "subtype": "compact",
                    "confidence": 0.9,
                },
            ]
        )
        try:
            rows = execute(
                Join(
                    Input(left_path),
                    Input(right_path),
                    join_cross_camera_test_bucket,
                    on=("vehicle_class", "color", "subtype"),
                    one_to_one=True,
                    score=join_test_match_score,
                    left_key=("camera_id", "track_id"),
                    right_key=("camera_id", "track_id"),
                )
            )
        finally:
            Path(left_path).unlink(missing_ok=True)
            Path(right_path).unlink(missing_ok=True)

        self.assertEqual(len(rows), 2)
        by_left = {row["left"]["track_id"]: row for row in rows}
        self.assertEqual(by_left["t2"]["right"]["track_id"], "t4")
        self.assertIn("match_score", by_left["t2"])


class JoinParseRenderTests(unittest.TestCase):
    def test_parse_and_render_join(self) -> None:
        from udfs.test_ops import join_rows_share_incident_id

        source = """
from mmds import Input, Join
from udfs.test_ops import join_rows_share_incident_id

left = Input("left.jsonl")
right = Input("right.jsonl")
output = Join(left, right, join_rows_share_incident_id, on="incident_id")
"""
        program = parse_query(source)
        rendered = render_query(program)
        self.assertIn('Join(left, right, join_rows_share_incident_id, on="incident_id")', rendered)


if __name__ == "__main__":
    unittest.main()
