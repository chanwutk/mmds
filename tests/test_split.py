from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import Input, Split, execute, parse_query, render_query  # noqa: E402
from mmds.execution.ops.split import (  # noqa: E402
    build_chunk_video_view,
    resolve_clip_bounds,
    split_clip_intervals,
    split_row,
)
from mmds.model import DatasetExpr, MMDSValidationError, SplitSpec  # noqa: E402


class SplitIntervalTests(unittest.TestCase):
    def test_single_chunk_when_shorter_than_interval(self) -> None:
        self.assertEqual(
            split_clip_intervals(0.0, 5.0, chunk_sec=30.0),
            [(0.0, 5.0)],
        )

    def test_two_equal_chunks(self) -> None:
        self.assertEqual(
            split_clip_intervals(0.0, 60.0, chunk_sec=30.0),
            [(0.0, 30.0), (30.0, 60.0)],
        )

    def test_partial_tail_chunk(self) -> None:
        self.assertEqual(
            split_clip_intervals(10.0, 75.0, chunk_sec=30.0),
            [(10.0, 40.0), (40.0, 70.0), (70.0, 75.0)],
        )

    def test_empty_clip_returns_no_chunks(self) -> None:
        self.assertEqual(split_clip_intervals(5.0, 5.0, chunk_sec=30.0), [])

    def test_non_positive_chunk_sec_raises(self) -> None:
        with self.assertRaises(ValueError):
            split_clip_intervals(0.0, 10.0, chunk_sec=0.0)


class SplitBoundsTests(unittest.TestCase):
    def test_video_view_end(self) -> None:
        row = {"camera_id": "cam-1"}
        video = {"type": "VideoView", "path": "clip.mp4", "start": 2.0, "end": 62.0}
        self.assertEqual(
            resolve_clip_bounds(row, video_field="video", raw_video=video),
            (2.0, 62.0),
        )

    def test_duration_sec_fallback(self) -> None:
        row = {"camera_id": "cam-1", "duration_sec": 45.0}
        video = {"path": "clip.mp4"}
        self.assertEqual(
            resolve_clip_bounds(row, video_field="video", raw_video=video),
            (0.0, 45.0),
        )

    def test_missing_bounds_raises(self) -> None:
        row = {"camera_id": "cam-1"}
        with self.assertRaises(MMDSValidationError):
            resolve_clip_bounds(row, video_field="video", raw_video={"path": "clip.mp4"})


class SplitRowTests(unittest.TestCase):
    def test_expands_row_with_chunk_metadata(self) -> None:
        spec = SplitSpec(video_field="video", chunk_sec=30.0)
        row = {
            "camera_id": "cam-i24v-highway2",
            "video": {
                "type": "VideoView",
                "path": "data/highway2.mp4",
                "start": 0,
                "end": 60,
            },
        }
        chunks = split_row(row, spec)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["split_video_chunk_num"], 0)
        self.assertEqual(chunks[0]["split_video_chunk_start"], 0.0)
        self.assertEqual(chunks[0]["split_video_chunk_end"], 30.0)
        self.assertEqual(chunks[0]["split_video_id"], "cam-i24v-highway2")
        self.assertEqual(chunks[0]["video"]["start"], 0.0)
        self.assertEqual(chunks[0]["video"]["end"], 30.0)
        self.assertEqual(chunks[1]["split_video_chunk_num"], 1)
        self.assertEqual(chunks[1]["video"]["start"], 30.0)
        self.assertEqual(chunks[1]["video"]["end"], 60.0)

    def test_preserves_path_key_in_video_view(self) -> None:
        spec = SplitSpec(video_field="video", chunk_sec=30.0)
        row = {
            "camera_id": "cam-1",
            "video": {"path": "clip.mp4", "start": 0, "end": 5},
        }
        chunks = split_row(row, spec)
        self.assertEqual(chunks[0]["video"]["path"], "clip.mp4")

    def test_string_video_path(self) -> None:
        spec = SplitSpec(video_field="video", chunk_sec=30.0)
        row = {
            "camera_id": "cam-1",
            "duration_sec": 40.0,
            "video": "clip.mp4",
        }
        chunks = split_row(row, spec)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["video"]["source"], "clip.mp4")

    def test_missing_doc_id_raises(self) -> None:
        spec = SplitSpec(video_field="video")
        row = {"video": {"start": 0, "end": 5}}
        with self.assertRaises(MMDSValidationError):
            split_row(row, spec)


class SplitDslTests(unittest.TestCase):
    def test_split_returns_dataset_expr(self) -> None:
        source = Input("data/rows.jsonl")
        node = Split(source, "video", chunk_sec=30.0, doc_id_key="camera_id")
        self.assertEqual(node.kind, "split")
        self.assertIsInstance(node.spec, SplitSpec)
        self.assertEqual(node.spec.video_field, "video")
        self.assertEqual(node.spec.chunk_sec, 30.0)

    def test_invalid_chunk_sec_raises(self) -> None:
        with self.assertRaises(TypeError):
            Split(Input("data/rows.jsonl"), "video", chunk_sec=0)


class SplitExecutionTests(unittest.TestCase):
    def test_execute_split_over_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feeds.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "camera_id": "cam-a",
                        "video": {"path": "a.mp4", "start": 0, "end": 65},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            plan = Split(Input(str(path)), "video", chunk_sec=30.0)
            rows = execute(plan)
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                [row["split_video_chunk_num"] for row in rows],
                [0, 1, 2],
            )


class SplitParseRenderTests(unittest.TestCase):
    def test_parse_and_render_split(self) -> None:
        source = """
from mmds import Input, Split

feeds = Input("data/feeds.jsonl")
output = Split(feeds, "video", chunk_sec=30.0, doc_id_key="camera_id")
"""
        program = parse_query(source)
        rendered = render_query(program)
        self.assertIn('Split(feeds, "video")', rendered)

    def test_render_default_split_omits_default_kwargs(self) -> None:
        node = Split(Input("data/feeds.jsonl"), "video")
        rendered = render_query(node)
        self.assertIn('Split(source_feeds, "video")', rendered)


class SplitSpecValidationTests(unittest.TestCase):
    def test_invalid_split_spec_on_dataset_expr(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DatasetExpr(kind="split", source=Input("data/x.jsonl"), spec=None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
