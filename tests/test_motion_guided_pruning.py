from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src", ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mmds.model import PromptSpec, ResolvedPrompt  # noqa: E402
from mmds.utilities.video import VideoView, open_video  # noqa: E402


def _load(name: str):
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MGP = _load("motion_guided_pruning")
EVAL = _load("eval_motion_guided_pruning")

WIDTH, HEIGHT, FPS = 160, 96, 10.0
CAT_A = (10, 30)   # (x, y) of a 30x30 square, present from frame 20
CAT_B = (110, 30)  # present during frames 63-89


def _cats_frame(i: int) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cats = []
    if i >= 20:
        cats.append(CAT_A)
    if 63 <= i < 90:
        cats.append(CAT_B)
    for x, y in cats:
        frame[y : y + 30, x : x + 30] = 255
    return frame


def blob_counter(frame_index: int, frame: np.ndarray) -> int:
    """Counts bright blobs; stands in for the MLLM."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    labels, _ = cv2.connectedComponents(mask)
    return labels - 1


class RecordingCounter:
    def __init__(self, counter=blob_counter) -> None:
        self.counter = counter
        self.calls: list[int] = []

    def __call__(self, frame_index: int, frame: np.ndarray) -> int:
        self.calls.append(frame_index)
        return self.counter(frame_index, frame)


class _VideoTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmpdir = Path(tmp.name)
        path = self.tmpdir / "cats.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT)
        )
        if not writer.isOpened():
            raise unittest.SkipTest("OpenCV MJPG VideoWriter is unavailable")
        for i in range(120):
            writer.write(_cats_frame(i))
        writer.release()
        self.video = open_video(str(path))
        self.view = VideoView(self.video, 1.0, 12.0)  # frames 10..119


class PlanStepTests(_VideoTestCase):
    def test_uniform_view_indices(self) -> None:
        self.assertEqual(MGP.uniform_view(self.view, 25), [10, 35, 60, 85, 110])

    def test_uniform_view_validates_sample_every(self) -> None:
        for bad, error in ((0, ValueError), (True, TypeError), (2.0, TypeError)):
            with self.subTest(bad=bad):
                with self.assertRaises(error):
                    MGP.uniform_view(self.view, bad)

    def test_pixel_change_filter_keeps_entrances_and_exit(self) -> None:
        kept = MGP.pixel_change_filter(self.view, sample_every=1, threshold=0.01)
        self.assertEqual(kept, [10, 20, 63, 90])

    def test_map_count_counts_requested_frames_in_order(self) -> None:
        counter = RecordingCounter()
        counts = MGP.map_count(self.view.path, [70, 15, 70, 25], counter)
        self.assertEqual(counts, {15: 0, 25: 1, 70: 2})
        self.assertEqual(counter.calls, [15, 25, 70])

    def test_map_count_batches_reads(self) -> None:
        with patch.object(
            MGP, "read_frames_at_indices", wraps=MGP.read_frames_at_indices
        ) as reader:
            MGP.map_count(self.view.path, range(10, 20), blob_counter, batch_size=4)
        self.assertEqual([len(c.args[1]) for c in reader.call_args_list], [4, 4, 2])

    def test_map_count_skips_undecodable_frames(self) -> None:
        counts = MGP.map_count(self.view.path, [50, 500], blob_counter)
        self.assertEqual(counts, {50: 1})

    def test_map_count_stops_at_count(self) -> None:
        counter = RecordingCounter()
        counts = MGP.map_count(
            self.view.path, range(60, 70), counter, stop_at_count=2
        )
        self.assertEqual(counter.calls, [60, 61, 62, 63])
        self.assertEqual(counts[63], 2)

    def test_map_count_records_timings(self) -> None:
        timings: dict[str, float] = {"select": 1.5}
        MGP.map_count(self.view.path, [30, 31], blob_counter, timings=timings)
        self.assertEqual(timings["select"], 1.5)
        self.assertGreater(timings["read"], 0.0)
        self.assertGreaterEqual(timings["mllm"], 0.0)

    def test_map_count_rejects_bad_counter_output(self) -> None:
        for bad in (-1, True, 1.0, "2", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    MGP.map_count(self.view.path, [30], lambda i, f, b=bad: b)

    def test_map_count_validates_arguments(self) -> None:
        for kwargs in ({"batch_size": 0}, {"batch_size": True},
                       {"stop_at_count": 0}, {"stop_at_count": 1.5}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    MGP.map_count(self.view.path, [30], blob_counter, **kwargs)

    def test_map_count_empty_input(self) -> None:
        self.assertEqual(MGP.map_count(self.view.path, [], blob_counter), {})


class ReduceAnswerTests(unittest.TestCase):
    def test_max_and_first_hit(self) -> None:
        answer = MGP.reduce_answer({10: 1, 40: 2, 30: 2, 50: 3}, fps=10.0)
        self.assertEqual(answer.max_simultaneous, 3)
        self.assertEqual(answer.first_index, 30)
        self.assertEqual(answer.first_seconds, 3.0)

    def test_no_frame_reaches_min_count(self) -> None:
        answer = MGP.reduce_answer({10: 1, 20: 0}, fps=10.0)
        self.assertEqual((answer.max_simultaneous, answer.first_index), (1, None))
        self.assertIsNone(answer.first_seconds)

    def test_empty_counts(self) -> None:
        answer = MGP.reduce_answer({}, fps=10.0)
        self.assertEqual((answer.max_simultaneous, answer.first_index), (None, None))

    def test_custom_min_count(self) -> None:
        self.assertEqual(MGP.reduce_answer({5: 1, 9: 3}, 10.0, min_count=3).first_index, 9)
        self.assertEqual(MGP.reduce_answer({5: 1, 9: 3}, 10.0, min_count=1).first_index, 5)

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            MGP.reduce_answer({1: 1}, fps=0)
        for bad in (0, True, 1.5):
            with self.subTest(min_count=bad):
                with self.assertRaises(ValueError):
                    MGP.reduce_answer({1: 1}, 10.0, min_count=bad)


class PlanTests(_VideoTestCase):
    def test_original_plan_counts_every_sample(self) -> None:
        result = MGP.run_original(self.view, blob_counter, sample_every=5)
        self.assertEqual(result.frames_sent, 22)
        self.assertEqual(result.sampled, 22)
        self.assertEqual(result.answer.max_simultaneous, 2)
        self.assertEqual(result.answer.first_index, 65)  # first sample >= 63

    def test_pruned_plan_sends_only_kept_frames(self) -> None:
        result = MGP.run_pruned(self.view, blob_counter, sample_every=5)
        self.assertEqual(result.kept_indices, [10, 20, 65, 90])
        self.assertEqual(result.frames_sent, 4)
        self.assertEqual(result.sampled, 22)
        self.assertEqual(result.answer.max_simultaneous, 2)
        self.assertEqual(result.answer.first_index, 65)
        self.assertGreaterEqual(result.seconds["select"], 0.0)

    def test_pruned_timestamp_never_earlier_than_original(self) -> None:
        for threshold in (0.005, 0.05, 0.2):
            with self.subTest(threshold=threshold):
                original = MGP.run_original(self.view, blob_counter, sample_every=5)
                pruned = MGP.run_pruned(
                    self.view, blob_counter, sample_every=5, threshold=threshold
                )
                if pruned.answer.first_index is not None:
                    self.assertGreaterEqual(
                        pruned.answer.first_index, original.answer.first_index
                    )

    def test_refinement_recovers_missed_entrance(self) -> None:
        # Pretend the filter missed cat B's entrance and kept 80 instead.
        pruned = MGP.PlanResult(
            "pruned", MGP.reduce_answer({10: 0, 20: 1, 80: 2}, FPS), 22,
            [10, 20, 80], {10: 0, 20: 1, 80: 2},
        )
        counter = RecordingCounter()
        refined = MGP.refine_first_occurrence(self.view, pruned, counter, sample_every=5)
        self.assertEqual(counter.calls, [25, 30, 35, 40, 45, 50, 55, 60, 65])
        self.assertEqual(refined.answer.first_index, 65)
        self.assertEqual(refined.answer.max_simultaneous, 2)
        self.assertEqual(refined.frames_sent, 3 + 9)
        self.assertEqual(pruned.counts, {10: 0, 20: 1, 80: 2})  # not mutated

    def test_refinement_without_earlier_hit_keeps_answer(self) -> None:
        pruned = MGP.run_pruned(self.view, blob_counter, sample_every=5)
        counter = RecordingCounter()
        refined = MGP.refine_first_occurrence(self.view, pruned, counter, sample_every=5)
        self.assertEqual(counter.calls, [25, 30, 35, 40, 45, 50, 55, 60])
        self.assertEqual(refined.answer.first_index, 65)

    def test_refinement_when_first_kept_frame_is_the_hit(self) -> None:
        pruned = MGP.PlanResult(
            "pruned", MGP.reduce_answer({70: 2}, FPS), 22, [70], {70: 2}
        )
        counter = RecordingCounter()
        refined = MGP.refine_first_occurrence(self.view, pruned, counter, sample_every=5)
        self.assertEqual(counter.calls[0], 10)  # scans from the view start
        self.assertEqual(refined.answer.first_index, 65)

    def test_refinement_skipped_when_no_hit(self) -> None:
        pruned = MGP.PlanResult(
            "pruned", MGP.reduce_answer({10: 0, 20: 1}, FPS), 22, [10, 20],
            {10: 0, 20: 1},
        )
        counter = RecordingCounter()
        refined = MGP.refine_first_occurrence(self.view, pruned, counter, sample_every=5)
        self.assertEqual(counter.calls, [])
        self.assertIsNone(refined.answer.first_index)

    def test_run_all_returns_three_plans(self) -> None:
        results = MGP.run_all(self.view, blob_counter, sample_every=5)
        self.assertEqual(
            [r.name for r in results],
            ["original (uniform)", "optimized (pixel-change)", "optimized + refinement"],
        )
        self.assertIn("max 2 cats", MGP.describe(results[0]))
        self.assertIn("6.5s", MGP.describe(results[0]))


class _FakeExecutor:
    model = "fake-model"

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[Any, ...]] = []

    def execute(self, op_type, prompt, resolved, payload, context):
        self.calls.append((op_type, prompt, resolved, payload, context))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class GeminiFrameCounterTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Path(tmp.name) / "sub" / "cache.json"
        self.frame = _cats_frame(70)
        self.sleeps: list[float] = []

    def counter(self, responses, **kwargs):
        executor = _FakeExecutor(responses)
        counter = MGP.GeminiFrameCounter(
            "clip.mp4", executor=executor, cache_path=self.cache,
            sleep=self.sleeps.append, **kwargs,
        )
        return counter, executor

    def test_sends_jpeg_with_prompt_and_schema(self) -> None:
        counter, executor = self.counter([{"cat_count": 2}])
        self.assertEqual(counter(70, self.frame), 2)
        op_type, prompt, resolved, payload, context = executor.calls[0]
        self.assertEqual(op_type, "map")
        self.assertIsInstance(prompt, PromptSpec)
        self.assertIsInstance(resolved, ResolvedPrompt)
        self.assertEqual(resolved.parts[0], MGP.COUNT_PROMPT)
        image = resolved.parts[1]
        self.assertEqual(image["mime_type"], "image/jpeg")
        decoded = cv2.imdecode(np.frombuffer(image["bytes"], np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, self.frame.shape)
        self.assertEqual(blob_counter(0, decoded), 2)
        self.assertEqual(dict(prompt.output_schema), dict(resolved.output_schema))

    def test_cache_hit_skips_api_and_persists(self) -> None:
        counter, executor = self.counter([{"cat_count": 1}])
        counter(5, self.frame)
        counter(5, self.frame)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual((counter.api_calls, counter.cache_hits), (1, 1))

        reloaded, executor2 = self.counter([])
        self.assertEqual(reloaded(5, self.frame), 1)
        self.assertEqual(executor2.calls, [])
        self.assertIn("clip.mp4|fake-model|5", json.loads(self.cache.read_text()))

    def test_cache_key_includes_model_and_video(self) -> None:
        counter, _ = self.counter([{"cat_count": 1}])
        counter(5, self.frame)
        other_model, executor = self.counter([{"cat_count": 0}], model="other")
        self.assertEqual(other_model(5, self.frame), 0)
        self.assertEqual(len(executor.calls), 1)
        other_video = MGP.GeminiFrameCounter(
            "other.mp4", executor=_FakeExecutor([{"cat_count": 3}]),
            cache_path=self.cache,
        )
        self.assertEqual(other_video(5, self.frame), 3)

    def test_retries_transient_errors_with_backoff(self) -> None:
        counter, executor = self.counter(
            [RuntimeError("503"), RuntimeError("503"), {"cat_count": 1}]
        )
        self.assertEqual(counter(5, self.frame), 1)
        self.assertEqual(len(executor.calls), 3)
        self.assertEqual(self.sleeps, [2.0, 4.0])
        self.assertEqual(counter.api_calls, 3)

    def test_raises_after_max_attempts_and_does_not_cache(self) -> None:
        counter, _ = self.counter([RuntimeError("a"), RuntimeError("b")], max_attempts=2)
        with self.assertRaises(RuntimeError):
            counter(5, self.frame)
        self.assertFalse(self.cache.exists())

    def test_rejects_invalid_responses(self) -> None:
        for response in ({}, {"cat_count": -1}, {"cat_count": True},
                         {"cat_count": 1.5}, [2], None):
            with self.subTest(response=response):
                counter, _ = self.counter([response])
                with self.assertRaises(ValueError):
                    counter(5, self.frame)

    def test_invalid_construction(self) -> None:
        with self.assertRaises(ValueError):
            MGP.GeminiFrameCounter("", executor=_FakeExecutor([]))
        with self.assertRaises(ValueError):
            MGP.GeminiFrameCounter("v", executor=_FakeExecutor([]), max_attempts=0)

    def test_works_without_cache_path(self) -> None:
        counter = MGP.GeminiFrameCounter(
            "v", executor=_FakeExecutor([{"cat_count": 2}])
        )
        self.assertEqual(counter(1, self.frame), 2)


def _labels(**query_answers: Any) -> dict[str, Any]:
    answers = {"max_simultaneous_cats": 2, "second_cat_first_visible": 6.3}
    answers.update(query_answers)
    return {
        "video": "cats.avi",
        "clip_start": 1.0,
        "clip_end": None,
        "query_answers": answers,
        "events": [{"start": 6.3, "end": 6.8, "description": "cat B enters"}],
        "static": [],
    }


class EvaluationTests(_VideoTestCase):
    def write_labels(self, data: dict[str, Any]) -> Path:
        path = self.tmpdir / "labels.json"
        path.write_text(json.dumps(data))
        return path

    def test_load_query_answers(self) -> None:
        truth = EVAL.load_query_answers(self.write_labels(_labels()))
        self.assertEqual(truth, EVAL.QueryAnswers(2, 6.3))

    def test_load_query_answers_rejects_invalid(self) -> None:
        cases = [
            {"max_simultaneous_cats": -1},
            {"max_simultaneous_cats": True},
            {"max_simultaneous_cats": 2.0},
            {"second_cat_first_visible": "6.3"},
            {"second_cat_first_visible": float("inf")},
            {"second_cat_first_visible": 0.5},
        ]
        for override in cases:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    EVAL.load_query_answers(self.write_labels(_labels(**override)))
        data = _labels()
        del data["query_answers"]
        with self.assertRaises(ValueError):
            EVAL.load_query_answers(self.write_labels(data))

    def test_repository_query_answers(self) -> None:
        truth = EVAL.load_query_answers(EVAL.wildlife.DEFAULT_LABELS)
        self.assertEqual(truth, EVAL.QueryAnswers(2, 949.6))

    def test_evaluate_scores_all_plans(self) -> None:
        report = EVAL.evaluate(
            self.view, EVAL.QueryAnswers(2, 6.3), blob_counter, sample_every=5
        )
        plans = {p["plan"]: p for p in report["plans"]}
        original = plans["original (uniform)"]
        pruned = plans["optimized (pixel-change)"]
        self.assertTrue(original["max_correct"])
        self.assertAlmostEqual(original["timestamp_error_seconds"], 0.2)
        self.assertEqual(original["reduction_vs_original"], 0.0)
        self.assertEqual(pruned["frames_sent"], 4)
        self.assertAlmostEqual(pruned["reduction_vs_original"], 1 - 4 / 22)
        self.assertTrue(pruned["matches_original_max"])
        self.assertTrue(pruned["matches_original_timestamp"])
        self.assertIn("optimized + refinement", EVAL.format_report(report))

    def test_score_plan_with_missing_answer(self) -> None:
        empty = MGP.PlanResult("p", MGP.reduce_answer({}, FPS), 0, [], {})
        score = EVAL.score_plan(empty, EVAL.QueryAnswers(2, 6.3), empty)
        self.assertFalse(score["max_correct"])
        self.assertIsNone(score["timestamp_error_seconds"])
        self.assertEqual(score["reduction_vs_original"], 0.0)
        self.assertIn("never", EVAL.format_report({
            "question": "q", "truth": {"max_simultaneous": 2, "second_cat_seconds": 6.3},
            "sample_every": 5, "threshold": 0.01, "pixel_cutoff": 25.0, "plans": [score],
        }))

    def test_main_with_fake_counter(self) -> None:
        data = _labels()
        data["video"] = str(self.video.path)
        labels = self.write_labels(data)
        output = self.tmpdir / "report.json"
        stub = RecordingCounter()
        stub.model, stub.api_calls, stub.cache_hits, stub.api_seconds = "fake", 0, 0, 0.0
        stdout = io.StringIO()
        with patch.object(EVAL, "make_counter", return_value=stub):
            with contextlib.redirect_stdout(stdout):
                exit_code = EVAL.main(
                    ["--labels", str(labels), "--output", str(output)]
                )
        self.assertEqual(exit_code, 0)
        report = json.loads(output.read_text())
        self.assertEqual(len(report["plans"]), 3)
        self.assertEqual(report["counter"]["model"], "fake")
        self.assertIn("Truth: max 2 cats", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
