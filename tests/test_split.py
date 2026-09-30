from __future__ import annotations

import json
import math
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
        with self.assertRaises(MMDSValidationError):
            split_clip_intervals(0.0, 10.0, chunk_sec=0.0)

    def test_non_finite_bounds_raise(self) -> None:
        with self.assertRaises(MMDSValidationError):
            split_clip_intervals(0.0, math.nan, chunk_sec=30.0)
        with self.assertRaises(MMDSValidationError):
            split_clip_intervals(0.0, 10.0, chunk_sec=math.inf)

    def test_end_before_start_raises(self) -> None:
        with self.assertRaises(MMDSValidationError):
            split_clip_intervals(10.0, 5.0, chunk_sec=30.0)


class SplitBoundsTests(unittest.TestCase):
    def test_video_view_end(self) -> None:
        row = {"id": "cam-1"}
        video = {"type": "VideoView", "path": "clip.mp4", "start": 2.0, "end": 62.0}
        self.assertEqual(
            resolve_clip_bounds(row, video_field="video", raw_video=video),
            (2.0, 62.0),
        )

    def test_duration_field_fallback(self) -> None:
        row = {"id": "cam-1", "duration_sec": 45.0}
        video = {"path": "clip.mp4"}
        self.assertEqual(
            resolve_clip_bounds(row, video_field="video", raw_video=video),
            (0.0, 45.0),
        )

    def test_custom_duration_field(self) -> None:
        row = {"id": "cam-1", "length_s": 12.0}
        video = {"path": "clip.mp4"}
        self.assertEqual(
            resolve_clip_bounds(
                row,
                video_field="video",
                raw_video=video,
                duration_field="length_s",
            ),
            (0.0, 12.0),
        )

    def test_missing_bounds_raises(self) -> None:
        row = {"id": "cam-1"}
        with self.assertRaises(MMDSValidationError):
            resolve_clip_bounds(row, video_field="video", raw_video={"path": "clip.mp4"})

    def test_malformed_end_raises(self) -> None:
        row = {"id": "cam-1"}
        with self.assertRaises(MMDSValidationError):
            resolve_clip_bounds(
                row,
                video_field="video",
                raw_video={"path": "clip.mp4", "end": "60"},
            )

    def test_bool_start_raises(self) -> None:
        row = {"id": "cam-1"}
        with self.assertRaises(MMDSValidationError):
            resolve_clip_bounds(
                row,
                video_field="video",
                raw_video={"path": "clip.mp4", "start": True, "end": 10},
            )

    def test_end_before_start_raises(self) -> None:
        row = {"id": "cam-1"}
        with self.assertRaises(MMDSValidationError):
            resolve_clip_bounds(
                row,
                video_field="video",
                raw_video={"path": "clip.mp4", "start": 10, "end": 5},
            )

    def test_zero_duration_yields_empty_split(self) -> None:
        row = {"id": "doc-1", "duration_sec": 0.0, "video": "clip.mp4"}
        chunks = split_row(row, SplitSpec(video_field="video", doc_id_key="id"))
        self.assertEqual(chunks, [])


class SplitRowTests(unittest.TestCase):
    def test_expands_row_with_chunk_metadata(self) -> None:
        spec = SplitSpec(video_field="video", chunk_sec=30.0, doc_id_key="camera_id")
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

    def test_forces_videoview_type_on_video_payload(self) -> None:
        spec = SplitSpec(video_field="video", chunk_sec=30.0, doc_id_key="id")
        row = {
            "id": "v1",
            "video": {"type": "Video", "uri": "https://example.com/a.mp4", "start": 0, "end": 5},
        }
        chunks = split_row(row, spec)
        self.assertEqual(chunks[0]["video"]["type"], "VideoView")
        self.assertEqual(chunks[0]["video"]["uri"], "https://example.com/a.mp4")

    def test_preserves_path_key_in_video_view(self) -> None:
        spec = SplitSpec(video_field="video", chunk_sec=30.0, doc_id_key="id")
        row = {
            "id": "cam-1",
            "video": {"path": "clip.mp4", "start": 0, "end": 5},
        }
        chunks = split_row(row, spec)
        self.assertEqual(chunks[0]["video"]["path"], "clip.mp4")
        self.assertEqual(chunks[0]["video"]["type"], "VideoView")

    def test_string_video_path(self) -> None:
        spec = SplitSpec(video_field="video", chunk_sec=30.0, doc_id_key="id")
        row = {
            "id": "cam-1",
            "duration_sec": 40.0,
            "video": "clip.mp4",
        }
        chunks = split_row(row, spec)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["video"]["source"], "clip.mp4")

    def test_custom_output_prefix_and_doc_id_key(self) -> None:
        spec = SplitSpec(
            video_field="video",
            chunk_sec=30.0,
            doc_id_key="video_id",
            output_prefix="clip",
        )
        row = {
            "video_id": "game-1",
            "video": {"path": "a.mp4", "start": 0, "end": 30},
        }
        chunks = split_row(row, spec)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["clip_id"], "game-1")
        self.assertEqual(chunks[0]["clip_chunk_num"], 0)
        self.assertNotIn("split_video_id", chunks[0])

    def test_missing_doc_id_raises(self) -> None:
        spec = SplitSpec(video_field="video")
        row = {"video": {"start": 0, "end": 5}}
        with self.assertRaises(MMDSValidationError):
            split_row(row, spec)

    def test_missing_video_field_raises(self) -> None:
        spec = SplitSpec(video_field="video", doc_id_key="id")
        with self.assertRaises(MMDSValidationError):
            split_row({"id": "x"}, spec)

    def test_empty_string_doc_id_raises(self) -> None:
        spec = SplitSpec(video_field="video", doc_id_key="id")
        row = {"id": "", "video": {"start": 0, "end": 5}}
        with self.assertRaises(MMDSValidationError):
            split_row(row, spec)

    def test_metadata_collision_raises(self) -> None:
        spec = SplitSpec(video_field="video", doc_id_key="id")
        row = {
            "id": "x",
            "split_video_id": "preexisting",
            "video": {"path": "a.mp4", "start": 0, "end": 5},
        }
        with self.assertRaisesRegex(MMDSValidationError, "refusing to overwrite"):
            split_row(row, spec)


class SplitDslTests(unittest.TestCase):
    def test_split_returns_dataset_expr(self) -> None:
        source = Input("data/rows.jsonl")
        node = Split(source, "video", chunk_sec=30.0, doc_id_key="id")
        self.assertEqual(node.kind, "split")
        self.assertIsInstance(node.spec, SplitSpec)
        self.assertEqual(node.spec.video_field, "video")
        self.assertEqual(node.spec.chunk_sec, 30.0)
        self.assertEqual(node.spec.doc_id_key, "id")
        self.assertEqual(node.spec.duration_field, "duration_sec")

    def test_default_doc_id_key_is_id(self) -> None:
        node = Split(Input("data/rows.jsonl"), "video")
        assert isinstance(node.spec, SplitSpec)
        self.assertEqual(node.spec.doc_id_key, "id")

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
                        "id": "cam-a",
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
            self.assertEqual(rows[0]["split_video_id"], "cam-a")

    def test_execute_custom_duration_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feeds.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "clip-1",
                        "length_s": 40.0,
                        "video": {"type": "Video", "uri": "https://example.com/v.mp4"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            plan = Split(
                Input(str(path)),
                "video",
                chunk_sec=30.0,
                duration_field="length_s",
            )
            rows = execute(plan)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["video"]["type"], "VideoView")


class SplitParseRenderTests(unittest.TestCase):
    def test_round_trip_default_split_spec(self) -> None:
        source = """
from mmds import Input, Split

feeds = Input("data/feeds.jsonl")
output = Split(feeds, "video")
"""
        program = parse_query(source)
        assert isinstance(program.assignments[-1].expr.spec, SplitSpec)
        spec = program.assignments[-1].expr.spec
        self.assertEqual(spec.doc_id_key, "id")
        self.assertEqual(spec.duration_field, "duration_sec")
        self.assertEqual(spec.chunk_sec, 30.0)

        rendered = render_query(program)
        self.assertIn('Split(feeds, "video")', rendered)
        self.assertNotIn("doc_id_key=", rendered)
        self.assertNotIn("duration_field=", rendered)

        again = parse_query(rendered)
        again_spec = again.assignments[-1].expr.spec
        assert isinstance(again_spec, SplitSpec)
        self.assertEqual(again_spec.doc_id_key, "id")
        self.assertEqual(again_spec.duration_field, "duration_sec")

    def test_round_trip_non_default_kwargs(self) -> None:
        source = """
from mmds import Input, Split

feeds = Input("data/feeds.jsonl")
output = Split(
    feeds,
    "video",
    chunk_sec=180.0,
    doc_id_key="video_id",
    output_prefix="clip",
    duration_field="length_s",
    name="chunk",
)
"""
        program = parse_query(source)
        rendered = render_query(program)
        again = parse_query(rendered)
        spec = again.assignments[-1].expr.spec
        assert isinstance(spec, SplitSpec)
        self.assertEqual(spec.chunk_sec, 180.0)
        self.assertEqual(spec.doc_id_key, "video_id")
        self.assertEqual(spec.output_prefix, "clip")
        self.assertEqual(spec.duration_field, "length_s")
        self.assertEqual(again.assignments[-1].expr.name, "chunk")

    def test_render_default_split_omits_default_kwargs(self) -> None:
        node = Split(Input("data/feeds.jsonl"), "video")
        rendered = render_query(node)
        self.assertIn('Split(source_feeds, "video")', rendered)

    def test_parse_rejects_non_positive_chunk_sec(self) -> None:
        source = """
from mmds import Input, Split

feeds = Input("data/feeds.jsonl")
output = Split(feeds, "video", chunk_sec=0)
"""
        with self.assertRaises(MMDSValidationError):
            parse_query(source)


class SplitSpecValidationTests(unittest.TestCase):
    def test_invalid_split_spec_on_dataset_expr(self) -> None:
        with self.assertRaises(MMDSValidationError):
            DatasetExpr(kind="split", source=Input("data/x.jsonl"), spec=None)  # type: ignore[arg-type]

    def test_build_chunk_video_view_from_string(self) -> None:
        view = build_chunk_video_view("clip.mp4", chunk_start=1.0, chunk_end=2.0)
        self.assertEqual(view["source"], "clip.mp4")
        self.assertEqual(view["start"], 1.0)
        self.assertEqual(view["end"], 2.0)
        self.assertEqual(view["type"], "VideoView")


if __name__ == "__main__":
    unittest.main()
