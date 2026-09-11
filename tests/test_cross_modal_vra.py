from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from examples.lecture_event_localization import (  # noqa: E402
    VIDEO_LOCALIZER_NAME,
    build_transcript_only_query,
    build_transcript_to_video_query,
    build_video_only_query,
)
from mmds import (  # noqa: E402
    ExecutionContext,
    Input,
    MMDSExecutionError,
    MMDSValidationError,
    Map,
    PadInterval,
    ReconcileIntervals,
    Resolve,
    StaticPromptExecutor,
    View,
    execute,
    load_query,
    render_query,
)
from mmds.execution.context import MaterializedClip  # noqa: E402
from mmds.execution.concurrency import bounded_map  # noqa: E402
from mmds.model import PromptSpec, RecordPath  # noqa: E402


class ParserRendererTests(unittest.TestCase):
    def test_new_operators_and_builtins_round_trip(self) -> None:
        query = '''
from mmds import Input, Map, Reduce, Unnest, Record, ForEach, Resolve, View, PadInterval, ReconcileIntervals

source = Input("rows.jsonl")
one = Unnest(source, "candidate_intervals")
padded = Map(one, PadInterval("candidate_intervals", "duration_seconds", 30.0))
resolved = Resolve(padded, ["source_id", "query_id"], "window_start_seconds", "window_end_seconds")
clips = View(resolved, "video", "window_start_seconds", "window_end_seconds", output_field="candidate_video")
output = Reduce(clips, ["source_id", "query_id"], ReconcileIntervals("events", "window_start_seconds", "window_end_seconds", preserve_fields=("query_text",)))
'''
        program = load_query(query)
        rendered = render_query(program)
        self.assertEqual(rendered, render_query(load_query(rendered)))
        self.assertIn("Resolve", rendered)
        self.assertIn("View", rendered)
        self.assertIn("PadInterval", rendered)
        self.assertIn("ReconcileIntervals", rendered)

    def test_parser_rejects_invalid_view_mode(self) -> None:
        query = '''
from mmds import Input, View
source = Input("rows.jsonl")
output = View(source, "video", "start", "end", mode="mystery")
'''
        with self.assertRaises(MMDSValidationError):
            load_query(query)


class ExecutionPolicyTests(unittest.TestCase):
    def test_bounded_map_never_exceeds_configured_concurrency(self) -> None:
        lock = threading.Lock()
        active = 0
        maximum = 0

        def work(value: int) -> int:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return value * 2

        result = list(bounded_map(work, range(8), max_workers=3))
        self.assertEqual(result, [value * 2 for value in range(8)])
        self.assertLessEqual(maximum, 3)
        self.assertGreater(maximum, 1)


class TemporalFunctionTests(unittest.TestCase):
    def test_padding_clamps_to_source_duration(self) -> None:
        path = _write_rows(
            [{"candidate": {"start_seconds": 5, "end_seconds": 95}, "duration": 100}]
        )
        result = execute(Map(Input(path), PadInterval("candidate", "duration", 30)))
        self.assertEqual(result[0]["window_start_seconds"], 0.0)
        self.assertEqual(result[0]["window_end_seconds"], 100.0)

    def test_resolve_merges_only_within_explicit_keys(self) -> None:
        path = _write_rows(
            [
                {"source_id": "a", "query_id": "q", "start": 0, "end": 10},
                {"source_id": "a", "query_id": "q", "start": 10, "end": 20},
                {"source_id": "b", "query_id": "q", "start": 5, "end": 15},
            ]
        )
        result = execute(Resolve(Input(path), ["source_id", "query_id"], "start", "end"))
        self.assertEqual(
            [(row["source_id"], row["start"], row["end"]) for row in result],
            [("a", 0.0, 20.0), ("b", 5.0, 15.0)],
        )

    def test_reconcile_translates_and_deduplicates_clip_events(self) -> None:
        path = _write_rows(
            [
                {
                    "source_id": "lecture",
                    "window_start": 100,
                    "window_end": 150,
                    "events": [{"start_seconds": 10, "end_seconds": 20, "confidence": 0.7}],
                },
                {
                    "source_id": "lecture",
                    "window_start": 105,
                    "window_end": 155,
                    "events": [{"start_seconds": 5, "end_seconds": 15, "confidence": 0.9}],
                },
            ]
        )
        result = execute(
            __import__("mmds").Reduce(
                Input(path),
                "source_id",
                ReconcileIntervals("events", "window_start", "window_end"),
            )
        )
        self.assertEqual(len(result[0]["events"]), 1)
        self.assertEqual(result[0]["events"][0]["start_seconds"], 110.0)
        self.assertEqual(result[0]["events"][0]["confidence"], 0.9)

    def test_invalid_temporal_values_fail_explicitly(self) -> None:
        path = _write_rows(
            [{"candidate": {"start_seconds": 7, "end_seconds": 4}, "duration": 10}]
        )
        with self.assertRaises(MMDSValidationError):
            execute(Map(Input(path), PadInterval("candidate", "duration", 1)))

    def test_resolve_rejects_collection_valued_group_keys(self) -> None:
        path = _write_rows([{"source_id": ["not", "scalar"], "start": 0, "end": 1}])
        with self.assertRaisesRegex(MMDSValidationError, "hashable scalar"):
            execute(Resolve(Input(path), "source_id", "start", "end"))


class FakeMaterializer:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, float, float]] = []

    def materialize(
        self, source: Path, destination: Path, start_seconds: float, end_seconds: float
    ) -> MaterializedClip:
        self.calls.append((source, destination, start_seconds, end_seconds))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-mp4")
        return MaterializedClip(
            path=destination,
            size_bytes=8,
            sha256="0" * 64,
            duration_seconds=end_seconds - start_seconds,
        )


class ViewTests(unittest.TestCase):
    def test_materialized_view_reuses_identical_clip_within_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "source.mp4"
            video.write_bytes(b"source")
            input_path = _write_rows(
                [
                    {"video": {"type": "Video", "path": str(video)}, "start": 2, "end": 4},
                    {"video": {"type": "Video", "path": str(video)}, "start": 2, "end": 4},
                ],
                directory=root,
            )
            materializer = FakeMaterializer()
            context = ExecutionContext(workspace=root / "workspace", materializer=materializer)
            result = execute(
                View(Input(input_path), "video", "start", "end", output_field="clip"),
                context=context,
            )

            self.assertEqual(len(materializer.calls), 1)
            self.assertEqual(result[0]["clip"]["path"], result[1]["clip"]["path"])
            self.assertEqual([value.reused for value in context.stats.views], [False, True])

    def test_materialized_view_requires_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "source.mp4"
            video.write_bytes(b"source")
            input_path = _write_rows(
                [{"video": str(video), "start": 0, "end": 1}], directory=root
            )
            with self.assertRaises(MMDSExecutionError):
                execute(View(Input(input_path), "video", "start", "end"))

    def test_view_fails_if_execution_local_cached_clip_is_modified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "source.mp4"
            video.write_bytes(b"source")
            input_path = _write_rows(
                [{"video": str(video), "start": 0, "end": 1}], directory=root
            )
            context = ExecutionContext(
                workspace=root / "workspace", materializer=FakeMaterializer()
            )
            plan = View(Input(input_path), "video", "start", "end", output_field="clip")
            first = execute(plan, context=context)
            Path(first[0]["clip"]["path"]).unlink()
            with self.assertRaisesRegex(MMDSExecutionError, "changed during"):
                execute(plan, context=context)

    def test_view_rejects_malformed_source_hash_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "source.mp4"
            video.write_bytes(b"source")
            input_path = _write_rows(
                [{
                    "video": {"type": "Video", "path": str(video), "sha256": "bad"},
                    "start": 0,
                    "end": 1,
                }],
                directory=root,
            )
            with self.assertRaisesRegex(MMDSValidationError, "64 hexadecimal"):
                execute(
                    View(Input(input_path), "video", "start", "end"),
                    context=ExecutionContext(
                        workspace=root / "workspace", materializer=FakeMaterializer()
                    ),
                )

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is unavailable")
    def test_ffmpeg_materialized_view_has_zero_origin_audio_and_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=10:d=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
                ],
                check=True,
            )
            input_path = _write_rows(
                [{"video": str(source), "start": 0.5, "end": 1.5}], directory=root
            )
            result = execute(
                View(Input(input_path), "video", "start", "end", output_field="clip"),
                context=ExecutionContext(workspace=root / "workspace"),
            )
            self.assertTrue(Path(result[0]["clip"]["path"]).is_file())


class LectureRewriteTests(unittest.TestCase):
    def test_video_only_baseline_executes_complete_video_localizer(self) -> None:
        input_path = _write_rows(
            [{
                "lecture_id": "lecture-3",
                "query_id": "glass",
                "video": {"type": "Video", "uri": "https://example.test/lecture"},
                "timestamped_transcript": [],
                "duration_seconds": 100,
                "query_text": "Find the glass shattering event.",
            }]
        )
        plan = build_video_only_query(input_path)
        localizer = _node_named(plan, VIDEO_LOCALIZER_NAME)
        executor = StaticPromptExecutor(
            {
                ("map", localizer.spec.cache_key()): {
                    "events": [
                        {"start_seconds": 70, "end_seconds": 72, "confidence": 0.9, "evidence": "glass breaks"}
                    ]
                }
            }
        )
        result = execute(plan, executor)
        self.assertEqual(result[0]["events"]["start_seconds"], 70)

    def test_o1_preserves_output_schema_and_replaces_modality(self) -> None:
        baseline = build_video_only_query("lectures.jsonl")
        transcript_only = build_transcript_only_query("lectures.jsonl")
        baseline_map = _node_named(baseline, VIDEO_LOCALIZER_NAME)
        transcript_map = _node_named(transcript_only, VIDEO_LOCALIZER_NAME)
        self.assertEqual(baseline_map.spec.output_schema, transcript_map.spec.output_schema)
        self.assertIn(RecordPath(("video",)), baseline_map.spec.parts)
        self.assertIn(RecordPath(("timestamped_transcript",)), transcript_map.spec.parts)

    def test_o1_executes_transcript_only_without_media_materialization(self) -> None:
        input_path = _write_rows(
            [{
                "lecture_id": "lecture-3",
                "query_id": "glass",
                "video": {"type": "Video", "uri": "https://example.test/lecture"},
                "timestamped_transcript": [
                    {"start_seconds": 10, "end_seconds": 12, "text": "glass breaks"}
                ],
                "duration_seconds": 100,
                "query_text": "Find the glass shattering event.",
            }]
        )
        plan = build_transcript_only_query(input_path)
        localizer = _node_named(plan, VIDEO_LOCALIZER_NAME)
        executor = StaticPromptExecutor(
            {
                ("map", localizer.spec.cache_key()): {
                    "events": [
                        {"start_seconds": 10, "end_seconds": 12, "confidence": 0.9, "evidence": "glass breaks"}
                    ]
                }
            }
        )
        result = execute(plan, executor)
        self.assertEqual(result[0]["events"]["start_seconds"], 10)

    def test_o2_has_paper_operator_sequence_and_same_video_localizer(self) -> None:
        baseline = build_video_only_query("lectures.jsonl")
        optimized = build_transcript_to_video_query("lectures.jsonl")
        self.assertEqual(
            [node.kind for node in optimized.walk_postorder()],
            ["input", "map", "unnest", "map", "resolve", "view", "map", "reduce", "unnest"],
        )
        baseline_spec = _node_named(baseline, VIDEO_LOCALIZER_NAME).spec
        optimized_spec = _node_named(optimized, VIDEO_LOCALIZER_NAME).spec
        self.assertIsInstance(baseline_spec, PromptSpec)
        self.assertIsInstance(optimized_spec, PromptSpec)
        self.assertEqual(baseline_spec.output_schema, optimized_spec.output_schema)
        baseline_non_media = [part for part in baseline_spec.parts if not isinstance(part, RecordPath)]
        optimized_non_media = [part for part in optimized_spec.parts if not isinstance(part, RecordPath)]
        self.assertEqual(baseline_non_media, optimized_non_media)
        self.assertIn(RecordPath(("candidate_video",)), optimized_spec.parts)
        rendered = render_query(optimized)
        self.assertEqual(rendered, render_query(load_query(rendered)))

    def test_o2_executes_end_to_end_without_provider_or_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "lecture.mp4"
            video.write_bytes(b"source")
            input_path = _write_rows(
                [{
                    "lecture_id": "lecture-3",
                    "query_id": "glass",
                    "video": {"type": "Video", "path": str(video)},
                    "timestamped_transcript": [{"start_seconds": 5, "end_seconds": 8, "text": "tone"}],
                    "duration_seconds": 100,
                    "query_text": "Find the glass shattering event.",
                }],
                directory=root,
            )
            plan = build_transcript_to_video_query(input_path)
            candidate = _node_named(plan, "o2_transcript_candidates")
            localizer = _node_named(plan, VIDEO_LOCALIZER_NAME)
            executor = StaticPromptExecutor(
                {
                    ("map", candidate.spec.cache_key()): {
                        "candidate_intervals": [
                            {"start_seconds": 50, "end_seconds": 52, "confidence": 0.8, "evidence": "tone"},
                            {"start_seconds": 55, "end_seconds": 57, "confidence": 0.7, "evidence": "glass"},
                        ]
                    },
                    ("map", localizer.spec.cache_key()): {
                        "events": [
                            {"start_seconds": 31, "end_seconds": 33, "confidence": 0.9, "evidence": "glass breaks"}
                        ]
                    },
                }
            )
            materializer = FakeMaterializer()
            context = ExecutionContext(workspace=root / "workspace", materializer=materializer)
            result = execute(plan, executor, context=context)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["events"]["start_seconds"], 51.0)
            self.assertEqual(result[0]["events"]["end_seconds"], 53.0)
            self.assertEqual(len(materializer.calls), 1)
            self.assertEqual(materializer.calls[0][2:], (20.0, 87.0))
            self.assertEqual(len(context.stats.views), 1)


def _node_named(plan, name: str):
    matches = [node for node in plan.walk_postorder() if node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"Expected one node named {name!r}, found {len(matches)}")
    return matches[0]


def _write_rows(rows: list[dict], *, directory: Path | None = None) -> str:
    if directory is None:
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        path = Path(handle.name)
        handle.close()
    else:
        path = directory / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    unittest.main()
