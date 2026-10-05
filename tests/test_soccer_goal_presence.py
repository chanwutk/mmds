from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mmds import Record  # noqa: E402
from mmds.model import MMDSValidationError, PromptSpec  # noqa: E402
from mmds.parser import parse_query  # noqa: E402
from mmds.render import program_from_plan, render_query  # noqa: E402
from scripts.query_types import soccer_goal_presence as sgp  # noqa: E402


def _goal(game_time: str, position_ms: int) -> dict:
    return {"gameTime": game_time, "label": "Goal", "position": str(position_ms)}


def _segment(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


class GoalTimesTests(unittest.TestCase):
    def test_filters_by_half_converts_ms_and_sorts(self) -> None:
        labels = {
            "annotations": [
                _goal("2 - 10:00", 600_000),
                _goal("1 - 29:32", 1_772_491),
                {"gameTime": "1 - 05:00", "label": "Corner", "position": "300000"},
                _goal("1 - 27:36", 1_656_477),
            ]
        }
        self.assertEqual(sgp.goal_times(labels, 1), [1656.477, 1772.491])
        self.assertEqual(sgp.goal_times(labels, 2), [600.0])

    def test_no_goals_returns_empty(self) -> None:
        self.assertEqual(sgp.goal_times({"annotations": []}, 1), [])
        self.assertEqual(sgp.goal_times({}, 1), [])

    def test_malformed_goal_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Malformed Goal"):
            sgp.goal_times({"annotations": [{"label": "Goal", "gameTime": "first half", "position": "1"}]}, 1)
        with self.assertRaisesRegex(ValueError, "Malformed Goal"):
            sgp.goal_times({"annotations": [{"label": "Goal", "gameTime": "1 - 00:01"}]}, 1)


class MakeChunksTests(unittest.TestCase):
    def test_exact_multiple(self) -> None:
        self.assertEqual(sgp.make_chunks(900.0, 300.0), [(0.0, 300.0), (300.0, 600.0), (600.0, 900.0)])

    def test_long_remainder_becomes_its_own_chunk(self) -> None:
        self.assertEqual(sgp.make_chunks(937.0, 300.0)[-1], (900.0, 937.0))

    def test_short_remainder_merges_into_previous_chunk(self) -> None:
        chunks = sgp.make_chunks(2700.003, 300.0)
        self.assertEqual(len(chunks), 9)
        self.assertEqual(chunks[-1], (2400.0, 2700.003))

    def test_video_shorter_than_one_chunk(self) -> None:
        self.assertEqual(sgp.make_chunks(12.0, 300.0), [(0.0, 12.0)])

    def test_chunks_cover_duration_without_gaps(self) -> None:
        chunks = sgp.make_chunks(3003.0, 300.0)
        self.assertEqual(chunks[0][0], 0.0)
        self.assertEqual(chunks[-1][1], 3003.0)
        for (_, end), (start, _) in zip(chunks, chunks[1:]):
            self.assertEqual(end, start)

    def test_invalid_inputs_raise(self) -> None:
        for duration, chunk in ((0.0, 300.0), (-1.0, 300.0), (math.nan, 300.0), (100.0, 0.0), (100.0, math.inf)):
            with self.subTest(duration=duration, chunk=chunk), self.assertRaises(ValueError):
                sgp.make_chunks(duration, chunk)


class TranscriptWindowTests(unittest.TestCase):
    def test_boundary_segment_goes_to_exactly_one_chunk_by_midpoint(self) -> None:
        segments = [_segment(290.0, 304.0, "straddles"), _segment(305.0, 310.0, "after")]
        self.assertEqual(sgp.transcript_for_window(segments, 0.0, 300.0), "straddles")
        self.assertEqual(sgp.transcript_for_window(segments, 300.0, 600.0), "after")

    def test_blank_segments_are_skipped(self) -> None:
        segments = [_segment(1.0, 2.0, "  "), _segment(3.0, 4.0, " Goal! ")]
        self.assertEqual(sgp.transcript_for_window(segments, 0.0, 10.0), "Goal!")


class ChunkRowsTests(unittest.TestCase):
    def _rows(self, goals: list[dict], segments: list[dict], duration: float = 900.0) -> list[dict]:
        return sgp.chunk_rows(
            game_id="league/season/game",
            half=1,
            video_path="/videos/1_224p.mkv",
            duration=duration,
            labels={"annotations": goals},
            segments=segments,
        )

    def test_goal_on_chunk_boundary_belongs_to_later_chunk(self) -> None:
        rows = self._rows([_goal("1 - 05:00", 300_000)], [])
        self.assertEqual([row["gt_goal_count"] for row in rows], [0, 1, 0])

    def test_goal_past_reported_duration_is_kept_in_last_chunk(self) -> None:
        rows = self._rows([_goal("1 - 15:01", 901_000)], [_segment(900.0, 904.0, "late")])
        self.assertEqual(rows[-1]["gt_goal_count"], 1)
        self.assertEqual(rows[-1]["transcript"], "late")

    def test_two_goals_in_one_chunk_are_counted(self) -> None:
        rows = self._rows([_goal("1 - 01:00", 60_000), _goal("1 - 02:00", 120_000)], [])
        self.assertEqual(rows[0]["gt_goal_count"], 2)

    def test_row_shape_and_no_speech_placeholder(self) -> None:
        rows = self._rows([], [_segment(10.0, 20.0, "kick off")])
        self.assertEqual(rows[0]["chunk_id"], "league/season/game|1|00")
        self.assertEqual(
            rows[1]["video"], {"type": "VideoView", "path": "/videos/1_224p.mkv", "start": 300.0, "end": 600.0}
        )
        self.assertEqual(rows[0]["transcript"], "kick off")
        self.assertEqual(rows[1]["transcript"], sgp.NO_SPEECH)


class SelectAndPrepareTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "soccernet"
        self.out = Path(self._tmp.name) / "out"
        games = []
        halves = []
        for name, goals in (("b-game", [_goal("2 - 05:10", 310_000)]), ("a-game", []), ("c-missing-video", [])):
            game_id = f"league/2015/{name}"
            game_dir = self.root / game_id
            game_dir.mkdir(parents=True)
            (game_dir / "Labels-v2.json").write_text(json.dumps({"annotations": goals}))
            videos = {}
            for half in (1, 2):
                path = f"{game_id}/{half}_224p.mkv"
                if name != "c-missing-video":
                    (self.root / path).write_bytes(b"")
                videos[str(half)] = {"path": path, "duration_seconds": 600.0}
                transcript_dir = self.root / "transcripts" / game_id
                transcript_dir.mkdir(parents=True, exist_ok=True)
                (transcript_dir / f"{half}.whisper.json").write_text(
                    json.dumps({"segments": [_segment(305.0, 315.0, "GOAL for the home side")]})
                )
                halves.append({"game_id": game_id, "half": half, "status": "transcribed"})
            games.append({"game_id": game_id, "labels": {"path": f"{game_id}/Labels-v2.json"}, "videos": {"224p": videos}})
        (self.root / "manifest.json").write_text(json.dumps({"games": games}))
        (self.root / "transcripts" / "index.json").write_text(json.dumps({"halves": halves}))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_select_games_sorts_and_skips_unavailable(self) -> None:
        manifest = json.loads((self.root / "manifest.json").read_text())
        index = json.loads((self.root / "transcripts" / "index.json").read_text())
        selected = sgp.select_games(manifest, index, self.root, 2)
        self.assertEqual([g["game_id"] for g in selected], ["league/2015/a-game", "league/2015/b-game"])
        with self.assertRaisesRegex(ValueError, "only 2 are available"):
            sgp.select_games(manifest, index, self.root, 3)

    def test_select_games_requires_both_halves_transcribed(self) -> None:
        manifest = json.loads((self.root / "manifest.json").read_text())
        index = {"halves": [{"game_id": "league/2015/a-game", "half": 1, "status": "transcribed"}]}
        with self.assertRaisesRegex(ValueError, "only 0 are available"):
            sgp.select_games(manifest, index, self.root, 1)

    def test_prepare_writes_one_file_per_half_with_ground_truth(self) -> None:
        written = sgp.prepare(self.root, self.out, games=2)
        self.assertEqual([p.name for p in written], ["g00_h1.jsonl", "g00_h2.jsonl", "g01_h1.jsonl", "g01_h2.jsonl"])
        rows = [json.loads(line) for line in (self.out / "chunks" / "g01_h2.jsonl").read_text().splitlines()]
        self.assertEqual([row["gt_goal_count"] for row in rows], [0, 1])
        self.assertTrue(Path(rows[0]["video"]["path"]).is_absolute())
        self.assertEqual(rows[1]["transcript"], "GOAL for the home side")


class PlanTests(unittest.TestCase):
    def test_naive_plan_round_trips_and_reads_video(self) -> None:
        plan = sgp.naive_plan("chunks.jsonl")
        self.assertEqual(plan.kind, "map")
        self.assertIn(Record["video"], plan.spec.parts)
        self.assertEqual(parse_query(render_query(program_from_plan(plan))).output_expr, plan)

    def test_transcript_plan_is_the_modality_substitution_of_naive(self) -> None:
        plan = sgp.transcript_plan("chunks.jsonl")
        self.assertIsInstance(plan.spec, PromptSpec)
        self.assertIn(Record["transcript"], plan.spec.parts)
        self.assertNotIn(Record["video"], plan.spec.parts)
        self.assertEqual(plan.spec.output_schema, sgp.naive_plan("chunks.jsonl").spec.output_schema)
        self.assertEqual(plan.spec.parts[0], sgp.TRANSCRIPT_PROMPT)
        self.assertEqual(plan.source.input_path, "chunks.jsonl")

    def test_detector_plan(self) -> None:
        plan = sgp.detector_plan("chunks.jsonl", 5)
        self.assertEqual(plan.kind, "detect")
        self.assertEqual(plan.spec.classes, ("soccer ball", "goal net"))
        self.assertEqual(plan.spec.frame_stride, 5)

    def test_detector_stride(self) -> None:
        self.assertEqual(sgp.detector_stride(25.0), 5)
        self.assertEqual(sgp.detector_stride(29.97), 6)
        self.assertEqual(sgp.detector_stride(5.0), 1)
        for fps in (0.0, -1.0, math.nan, 12.0, 3.0):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                sgp.detector_stride(fps)


class BallInNetTests(unittest.TestCase):
    NET = [100, 100, 200, 160]

    def _det(self, cls: str, frame: int, bbox: list[float]) -> dict:
        return {"type": cls, "bboxes": [{"frame_idx": frame, "bbox": bbox, "confidence": 0.9}]}

    def test_ball_centred_inside_net_in_same_frame(self) -> None:
        counts = sgp.ball_in_net_counts([self._det("goal net", 10, self.NET), self._det("soccer ball", 10, [140, 120, 150, 130])])
        self.assertEqual(counts, {"frames_with_ball": 1, "frames_with_net": 1, "frames_ball_in_net": 1})

    def test_ball_and_net_in_different_frames_do_not_count(self) -> None:
        counts = sgp.ball_in_net_counts([self._det("goal net", 10, self.NET), self._det("soccer ball", 15, [140, 120, 150, 130])])
        self.assertEqual(counts["frames_ball_in_net"], 0)

    def test_ball_outside_net_does_not_count(self) -> None:
        counts = sgp.ball_in_net_counts([self._det("goal net", 10, self.NET), self._det("soccer ball", 10, [300, 120, 310, 130])])
        self.assertEqual(counts, {"frames_with_ball": 1, "frames_with_net": 1, "frames_ball_in_net": 0})

    def test_multiple_boxes_in_a_frame_count_the_frame_once(self) -> None:
        ball = {"type": "soccer ball", "bboxes": [
            {"frame_idx": 3, "bbox": [140, 120, 150, 130], "confidence": 0.9},
            {"frame_idx": 3, "bbox": [150, 120, 160, 130], "confidence": 0.8},
        ]}
        counts = sgp.ball_in_net_counts([self._det("goal net", 3, self.NET), ball])
        self.assertEqual(counts["frames_ball_in_net"], 1)
        self.assertEqual(counts["frames_with_ball"], 1)

    def test_no_detections(self) -> None:
        self.assertEqual(sgp.ball_in_net_counts([]), {"frames_with_ball": 0, "frames_with_net": 0, "frames_ball_in_net": 0})


class ScoreTests(unittest.TestCase):
    def test_counts_and_rates(self) -> None:
        result = sgp.score([(True, 1), (True, 0), (False, 2), (False, 0), (False, 0)])
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (1, 1, 1, 2))
        self.assertEqual(result["goal_chunks"], 2)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["accuracy"], 0.6)

    def test_undefined_rates_are_none(self) -> None:
        result = sgp.score([(False, 0)])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertEqual(sgp.score([])["accuracy"], None)


class UsageTests(unittest.TestCase):
    def test_sums_tokens_and_modalities(self) -> None:
        records = [
            {"usage": {
                "prompt_token_count": 1000,
                "candidates_token_count": 5,
                "thoughts_token_count": 20,
                "prompt_tokens_details": [
                    {"modality": "VIDEO", "token_count": 900},
                    {"modality": "MediaModality.TEXT", "token_count": 100},
                ],
            }},
            {"usage": {"prompt_token_count": 50, "candidates_token_count": 3}},
            {"usage": None},
        ]
        summary = sgp.summarize_usage(records)
        self.assertEqual(summary["calls"], 3)
        self.assertEqual(summary["calls_without_usage"], 1)
        self.assertEqual(summary["input_tokens"], 1050)
        self.assertEqual(summary["output_tokens"], 8)
        self.assertEqual(summary["thinking_tokens"], 20)
        self.assertEqual(summary["input_tokens_by_modality"], {"VIDEO": 900, "TEXT": 100})


class _FakeGoalExecutor:
    """Answers from the resolved prompt: video chunks starting at 300 s, or 'GOAL' in text."""

    def __init__(self, records: list, answer=None):
        self.records = records
        self.answer = answer
        self.prompts = []

    def execute(self, op_type, prompt, resolved_prompt, payload, context):
        self.prompts.append(resolved_prompt.parts)
        self.records.append({"model": "fake", "op_type": op_type, "usage": {"prompt_token_count": 10}})
        if self.answer is not None:
            return {"goal_scored": self.answer}
        videos = [part for part in resolved_prompt.parts if isinstance(part, dict)]
        if videos:
            return {"goal_scored": videos[0]["start"] == 300.0}
        return {"goal_scored": any("GOAL" in str(part) for part in resolved_prompt.parts)}


class RunHalfTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        rows = sgp.chunk_rows(
            game_id="g",
            half=1,
            video_path="/videos/1.mkv",
            duration=900.0,
            labels={"annotations": [_goal("1 - 05:10", 310_000)]},
            segments=[_segment(305.0, 315.0, "GOAL!")],
        )
        (self.out / "chunks").mkdir()
        self.chunk_file = self.out / "chunks" / "g00_h1.jsonl"
        self.chunk_file.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_naive_half_sends_video_views_and_records_usage(self) -> None:
        records: list = []
        executor = _FakeGoalExecutor(records)
        payload = sgp.run_prompt_half("naive", self.chunk_file, executor, records, "fake")
        self.assertEqual([row["prediction"] for row in payload["rows"]], [False, True, False])
        self.assertEqual(len(payload["usage"]), 3)
        self.assertTrue(all(any(isinstance(p, dict) for p in parts) for parts in executor.prompts))

    def test_transcript_half_sends_text_only(self) -> None:
        records: list = []
        executor = _FakeGoalExecutor(records)
        payload = sgp.run_prompt_half("transcript", self.chunk_file, executor, records, "fake")
        self.assertEqual([row["prediction"] for row in payload["rows"]], [False, True, False])
        self.assertFalse(any(isinstance(p, dict) for parts in executor.prompts for p in parts))

    def test_usage_records_are_reset_per_half(self) -> None:
        records: list = [{"usage": {"prompt_token_count": 999}}]
        payload = sgp.run_prompt_half("transcript", self.chunk_file, _FakeGoalExecutor(records), records, "fake")
        self.assertEqual(len(payload["usage"]), 3)

    def test_non_boolean_answer_is_rejected(self) -> None:
        records: list = []
        with self.assertRaisesRegex(ValueError, "expected a boolean"):
            sgp.run_prompt_half("naive", self.chunk_file, _FakeGoalExecutor(records, answer="yes"), records, "fake")

    def test_unknown_prompt_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Not a prompt mode"):
            sgp.run_prompt_half("detector", self.chunk_file, _FakeGoalExecutor([]), [], "fake")

    def test_detector_half_predicts_from_ball_in_net_and_times_the_run(self) -> None:
        rows = [json.loads(line) for line in self.chunk_file.read_text().splitlines()]
        net = {"type": "goal net", "bboxes": [{"frame_idx": 7, "bbox": [0, 0, 10, 10], "confidence": 0.9}]}
        ball = {"type": "soccer ball", "bboxes": [{"frame_idx": 7, "bbox": [4, 4, 6, 6], "confidence": 0.9}]}
        detected = [{**rows[0], "detections": [net]}, {**rows[1], "detections": [net, ball]}, {**rows[2], "detections": []}]
        with mock.patch.object(sgp, "execute", return_value=detected) as fake_execute:
            payload = sgp.run_detector_half(self.chunk_file, 5, "cuda", "TITAN Xp")
        fake_execute.assert_called_once()
        self.assertEqual([row["prediction"] for row in payload["rows"]], [False, True, False])
        self.assertEqual(payload["rows"][0]["frames_with_net"], 1)
        self.assertEqual(payload["frame_stride"], 5)
        self.assertGreaterEqual(payload["detector_seconds"], 0.0)

    def test_run_requires_prepared_chunks(self) -> None:
        with self.assertRaisesRegex(SystemExit, "run 'prepare' first"):
            sgp.run("naive", self.out / "empty", model="fake")

    def test_run_skips_finished_halves_without_needing_an_api_key(self) -> None:
        done = self.out / "results" / "naive"
        done.mkdir(parents=True)
        (done / "g00_h1.json").write_text("{}")
        with mock.patch.dict("os.environ", {}, clear=True):
            sgp.run("naive", self.out, model="fake")

    def test_run_requires_an_api_key_for_prompt_modes(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(SystemExit, "GEMINI_API_KEY"):
            sgp.run("naive", self.out, model="fake")

    def test_failed_half_keeps_billed_usage(self) -> None:
        class FailingExecutor:
            """Like Gemini on a bad response: report usage, then raise."""

            def __init__(self, *, model, usage_sink):
                self.usage_sink = usage_sink

            def execute(self, *args):
                self.usage_sink({"model": "m", "op_type": "map", "usage": {"prompt_token_count": 7}})
                raise MMDSValidationError("Gemini returned invalid JSON")

        with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "x"}), mock.patch(
            "mmds.GeminiPromptExecutor", FailingExecutor
        ), self.assertRaises(MMDSValidationError):
            sgp.run("naive", self.out, model="m")
        failed = (self.out / "results" / "naive" / "_failed_usage.jsonl").read_text().splitlines()
        self.assertGreaterEqual(len(failed), 1)
        self.assertEqual(json.loads(failed[0])["usage"], {"prompt_token_count": 7})
        self.assertFalse((self.out / "results" / "naive" / "g00_h1.json").exists())


class ScoreResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, mode: str, payload: dict) -> None:
        directory = self.out / "results" / mode
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "g00_h1.json").write_text(json.dumps(payload))

    def test_scores_each_mode_and_formats_a_table(self) -> None:
        rows = [{"chunk_id": "a", "gt_goal_count": 1, "prediction": True}, {"chunk_id": "b", "gt_goal_count": 0, "prediction": True}]
        self._write("naive", {"model": "m", "rows": rows, "usage": [{"usage": {"prompt_token_count": 100, "candidates_token_count": 2}}]})
        self._write("detector", {"rows": rows, "detector_seconds": 12.5, "gpu": "TITAN Xp"})
        summaries = sgp.score_results(self.out)
        self.assertEqual([s["mode"] for s in summaries], ["naive", "detector"])
        self.assertEqual(summaries[0]["input_tokens"], 100)
        self.assertEqual(summaries[1]["detector_seconds"], 12.5)
        table = sgp.format_table(summaries)
        self.assertIn("| naive | 2 | 1 | 1 | 1 | 0 | 0 | 0.50 | 1.00 | 0.50 | 1 | 100 | 2 | 0 |", table)
        self.assertIn("| detector |", table)
        self.assertIn("12.5 |", table)

    def test_ignores_bookkeeping_files(self) -> None:
        rows = [{"chunk_id": "a", "gt_goal_count": 0, "prediction": False}]
        self._write("detector", {"rows": rows, "detector_seconds": 1.0, "gpu": "x"})
        (self.out / "results" / "detector" / "_warmup.json").write_text("{}")
        self.assertEqual(sgp.score_results(self.out)[0]["chunks"], 1)

    def test_modes_with_different_chunks_are_rejected(self) -> None:
        self._write("naive", {"model": "m", "rows": [{"chunk_id": "a", "gt_goal_count": 0, "prediction": False}], "usage": []})
        self._write("transcript", {"model": "m", "rows": [{"chunk_id": "b", "gt_goal_count": 0, "prediction": False}], "usage": []})
        with self.assertRaisesRegex(ValueError, "different chunks"):
            sgp.score_results(self.out)


if __name__ == "__main__":
    unittest.main()
