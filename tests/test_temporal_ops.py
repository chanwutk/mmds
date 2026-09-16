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

from mmds import Input, Map, Reduce, execute  # noqa: E402
from udfs.temporal_ops import rebase_clip_events, reconcile_events  # noqa: E402


class RebaseClipEventsTests(unittest.TestCase):
    def test_adds_clip_start_to_event_times(self) -> None:
        row = {
            "clip": {"start": 120, "end": 150},
            "clip_events": [
                {"type": "goal", "start": 7, "end": 11},
                {"type": "save", "start": 20, "end": 22},
            ],
        }

        result = rebase_clip_events(row)

        self.assertEqual(
            result,
            {
                "events": [
                    {"type": "goal", "start": 127, "end": 131},
                    {"type": "save", "start": 140, "end": 142},
                ]
            },
        )

    def test_empty_event_list_stays_empty(self) -> None:
        row = {"clip": {"start": 120, "end": 150}, "clip_events": []}

        self.assertEqual(rebase_clip_events(row), {"events": []})

    def test_does_not_mutate_clip_events(self) -> None:
        row = {
            "clip": {"start": 120, "end": 150},
            "clip_events": [{"type": "goal", "start": 7, "end": 11}],
        }
        original = json.loads(json.dumps(row))

        rebase_clip_events(row)

        self.assertEqual(row, original)


class ReconcileEventsTests(unittest.TestCase):
    def test_flattens_and_sorts_events(self) -> None:
        rows = [
            {"events": [{"type": "goal", "start": 200, "end": 204}]},
            {"events": [{"type": "goal", "start": 120, "end": 125}]},
        ]

        result = reconcile_events(rows)

        self.assertEqual(
            result["events"],
            [
                {"type": "goal", "start": 120, "end": 125},
                {"type": "goal", "start": 200, "end": 204},
            ],
        )

    def test_preserves_overlapping_events(self) -> None:
        rows = [
            {"events": [{"type": "goal", "start": 120, "end": 125}]},
            {"events": [{"type": "goal", "start": 123, "end": 127}]},
        ]

        result = reconcile_events(rows)

        self.assertEqual(
            result["events"],
            [
                {"type": "goal", "start": 120, "end": 125},
                {"type": "goal", "start": 123, "end": 127},
            ],
        )

    def test_does_not_merge_different_event_types(self) -> None:
        rows = [
            {"events": [{"type": "goal", "start": 120, "end": 127}]},
            {"events": [{"type": "save", "start": 123, "end": 125}]},
        ]

        result = reconcile_events(rows)

        self.assertEqual(
            result["events"],
            [
                {"type": "goal", "start": 120, "end": 127},
                {"type": "save", "start": 123, "end": 125},
            ],
        )

    def test_empty_group_returns_empty_events(self) -> None:
        self.assertEqual(reconcile_events([]), {"events": []})

    def test_does_not_mutate_input_events(self) -> None:
        rows = [
            {"events": [{"type": "goal", "start": 120, "end": 125}]},
            {"events": [{"type": "goal", "start": 123, "end": 127}]},
        ]
        original = json.loads(json.dumps(rows))

        reconcile_events(rows)

        self.assertEqual(rows, original)


class TemporalUDFExecutionTests(unittest.TestCase):
    def test_map_then_reduce_rebases_and_reconciles_events(self) -> None:
        rows = [
            {
                "source_id": "video-1",
                "clip": {"start": 100, "end": 140},
                "clip_events": [
                    {"type": "goal", "start": 20, "end": 25}
                ],
            },
            {
                "source_id": "video-1",
                "clip": {"start": 120, "end": 150},
                "clip_events": [
                    {"type": "goal", "start": 3, "end": 7}
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "rows.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            plan = Reduce(
                Map(Input(str(input_path)), rebase_clip_events),
                "source_id",
                reconcile_events,
            )
            result = execute(plan)

        self.assertEqual(
            result,
            [
                {
                    "source_id": "video-1",
                    "events": [
                        {"type": "goal", "start": 120, "end": 125},
                        {"type": "goal", "start": 123, "end": 127},
                    ],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
