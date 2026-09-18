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

from mmds import (  # noqa: E402
    Input,
    Map,
    Record,
    StaticPromptExecutor,
    execute_measured,
)


class ExecutionMetricsTests(unittest.TestCase):
    def test_counts_rows_prompt_calls_and_video_seconds(self) -> None:
        rows = [
            {
                "id": 1,
                "clip": {
                    "type": "VideoView",
                    "source": "video.mp4",
                    "start": 10,
                    "end": 25,
                },
            },
            {
                "id": 2,
                "clip": {
                    "type": "VideoView",
                    "source": "video.mp4",
                    "start": 40,
                    "end": 45,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            plan = Map(
                Input(str(path)),
                ["Inspect ", Record["clip"]],
                schema={"answer": "string"},
            )
            result = execute_measured(
                plan,
                prompt_executor=StaticPromptExecutor(
                    {("map", plan.spec.cache_key()): {"answer": "done"}}
                ),
            )

        self.assertEqual(len(result.rows), 2)
        self.assertEqual(result.metrics.input_rows, 2)
        self.assertEqual(result.metrics.output_rows, 2)
        self.assertEqual(result.metrics.prompt_calls, 2)
        self.assertEqual(result.metrics.video_seconds, 20.0)
        self.assertGreaterEqual(result.metrics.prompt_seconds, 0.0)
        self.assertGreaterEqual(result.metrics.elapsed_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
