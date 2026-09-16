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

from mmds import Coalesce, DatasetExpr, Input, MMDSValidationError, execute  # noqa: E402
from mmds.execution.ops.coalesce import _apply_coalesce  # noqa: E402


def _source() -> DatasetExpr:
    return DatasetExpr(kind="input", input_path="data.jsonl")


def _node() -> DatasetExpr:
    return Coalesce(_source(), "source_id", "clip")


def _row(source_id: str, start: float, end: float) -> dict:
    return {
        "source_id": source_id,
        "candidate": {"start": start, "end": end},
        "clip": {
            "type": "VideoView",
            "source": f"{source_id}.mp4",
            "start": start,
            "end": end,
        },
    }


class CoalesceDSLTests(unittest.TestCase):
    def test_constructs_coalesce_node(self) -> None:
        node = Coalesce(
            _source(),
            ["source_id", "camera_id"],
            "clip",
            name="merge_windows",
        )

        self.assertEqual(node.kind, "coalesce")
        self.assertEqual(node.source, _source())
        self.assertEqual(node.group_by, ("source_id", "camera_id"))
        self.assertEqual(node.field, "clip")
        self.assertEqual(node.name, "merge_windows")

    def test_empty_group_by_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            Coalesce(_source(), [], "clip")

    def test_empty_field_raises(self) -> None:
        with self.assertRaises(TypeError):
            Coalesce(_source(), "source_id", "")

    def test_non_dataset_source_raises(self) -> None:
        with self.assertRaises(TypeError):
            Coalesce("not-a-plan", "source_id", "clip")  # type: ignore[arg-type]


class ApplyCoalesceTests(unittest.TestCase):
    def test_merges_overlapping_and_touching_intervals(self) -> None:
        rows = [
            _row("video-1", 20, 30),
            _row("video-1", 5, 15),
            _row("video-1", 14, 22),
            _row("video-1", 30, 35),
        ]

        result = list(_apply_coalesce(_node(), rows))

        self.assertEqual(
            result,
            [
                {
                    "source_id": "video-1",
                    "clip": {
                        "type": "VideoView",
                        "source": "video-1.mp4",
                        "start": 5,
                        "end": 35,
                    },
                }
            ],
        )

    def test_keeps_disjoint_intervals_separate(self) -> None:
        rows = [_row("video-1", 5, 10), _row("video-1", 20, 30)]

        result = list(_apply_coalesce(_node(), rows))

        self.assertEqual(
            [(row["clip"]["start"], row["clip"]["end"]) for row in result],
            [(5, 10), (20, 30)],
        )

    def test_coalesces_groups_independently(self) -> None:
        rows = [
            _row("video-1", 5, 20),
            _row("video-2", 10, 15),
            _row("video-1", 15, 30),
            _row("video-2", 14, 25),
        ]

        result = list(_apply_coalesce(_node(), rows))

        self.assertEqual(
            [
                (row["source_id"], row["clip"]["start"], row["clip"]["end"])
                for row in result
            ],
            [("video-1", 5, 30), ("video-2", 10, 25)],
        )

    def test_outputs_only_grouping_fields_and_interval(self) -> None:
        result = list(_apply_coalesce(_node(), [_row("video-1", 5, 10)]))

        self.assertEqual(set(result[0]), {"source_id", "clip"})
        self.assertNotIn("candidate", result[0])

    def test_does_not_mutate_input_intervals(self) -> None:
        rows = [_row("video-1", 5, 20), _row("video-1", 15, 30)]
        original = json.loads(json.dumps(rows))

        list(_apply_coalesce(_node(), rows))

        self.assertEqual(rows, original)

    def test_invalid_interval_raises(self) -> None:
        invalid_intervals = (
            "not-an-interval",
            {"start": "5", "end": 10},
            {"start": 5, "end": None},
            {"start": 10, "end": 10},
            {"start": 10, "end": 5},
        )

        for interval in invalid_intervals:
            with self.subTest(interval=interval):
                row = {"source_id": "video-1", "clip": interval}
                with self.assertRaises(MMDSValidationError):
                    list(_apply_coalesce(_node(), [row]))

    def test_empty_input_produces_no_rows(self) -> None:
        self.assertEqual(list(_apply_coalesce(_node(), [])), [])


class CoalesceExecutionTests(unittest.TestCase):
    def test_execute_dispatches_coalesce(self) -> None:
        rows = [
            _row("video-1", 5, 20),
            _row("video-1", 15, 30),
            _row("video-1", 50, 60),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "rows.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            plan = Coalesce(Input(str(input_path)), "source_id", "clip")
            result = execute(plan)

        self.assertEqual(
            [(row["clip"]["start"], row["clip"]["end"]) for row in result],
            [(5, 30), (50, 60)],
        )


if __name__ == "__main__":
    unittest.main()
