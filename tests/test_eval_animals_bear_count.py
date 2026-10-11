from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_count_eval():
    path = ROOT / "examples" / "eval_animals_bear_count.py"
    spec = importlib.util.spec_from_file_location("eval_animals_bear_count", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _load_count_eval()
SOURCE = "https://www.youtube.com/watch?v=s5iU3nLOvi8"
OTHER = "https://www.youtube.com/watch?v=other"
GROUND_TRUTH = ROOT / "data" / "animals_bear_count_ground_truth.json"


def _row(count, source: str = SOURCE) -> dict:
    return {"video": {"type": "Video", "source": source}, "bear_count": count}


def _truth(count, source: str = SOURCE) -> dict:
    return {"source": source, "bear_count": count}


def _window_row(start: float, end: float, count: int) -> dict:
    return {
        "video": {"type": "VideoView", "source": SOURCE, "start": start, "end": end},
        "bear_count": count,
    }


class BearCountScoreTests(unittest.TestCase):
    def test_exact_count(self) -> None:
        report = EVAL.evaluate_bear_count([_row(21)], [_truth(21)])
        self.assertEqual(report["exact_matches"], 1)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["mae"], 0.0)
        self.assertEqual(report["relative_error"], 0.0)
        self.assertEqual(report["details"][0]["kind"], "exact")

    def test_over_count_has_positive_error(self) -> None:
        report = EVAL.evaluate_bear_count([_row(30)], [_truth(21)])
        self.assertEqual(report["over_counts"], 1)
        self.assertEqual(report["details"][0]["error"], 9)
        self.assertEqual(report["mae"], 9.0)
        self.assertAlmostEqual(report["relative_error"], 9 / 21)

    def test_under_count_has_negative_error(self) -> None:
        report = EVAL.evaluate_bear_count([_row(15)], [_truth(21)])
        self.assertEqual(report["under_counts"], 1)
        self.assertEqual(report["details"][0]["error"], -6)
        self.assertAlmostEqual(report["relative_error"], 6 / 21)

    def test_aggregates_across_several_videos(self) -> None:
        report = EVAL.evaluate_bear_count(
            [_row(20), _row(2, OTHER)], [_truth(21), _truth(4, OTHER)]
        )
        self.assertEqual(report["n_ref"], 2)
        self.assertEqual(report["under_counts"], 2)
        self.assertEqual(report["total_truth"], 25)
        self.assertEqual(report["total_abs_error"], 3)
        self.assertAlmostEqual(report["mae"], 1.5)
        self.assertAlmostEqual(report["relative_error"], 3 / 25)

    def test_missing_row_scores_as_zero(self) -> None:
        report = EVAL.evaluate_bear_count([], [_truth(21)])
        item = report["details"][0]
        self.assertEqual(item["predicted"], 0)
        self.assertEqual(item["kind"], "under")
        self.assertFalse(item["emitted"])
        self.assertEqual(report["n_pred"], 0)

    def test_zero_truth_has_no_relative_error(self) -> None:
        exact = EVAL.evaluate_bear_count([_row(0)], [_truth(0)])
        over = EVAL.evaluate_bear_count([_row(2)], [_truth(0)])
        self.assertIsNone(exact["relative_error"])
        self.assertIsNone(over["relative_error"])
        self.assertEqual(over["over_counts"], 1)

    def test_integral_float_is_accepted(self) -> None:
        report = EVAL.evaluate_bear_count([_row(21.0)], [_truth(21)])
        self.assertEqual(report["exact_matches"], 1)

    def test_invalid_predicted_counts_raise(self) -> None:
        for value in (True, False, -1, 1.5, "21", None, [21]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    EVAL.evaluate_bear_count([_row(value)], [_truth(21)])

    def test_duplicate_prediction_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            EVAL.evaluate_bear_count([_row(21), _row(20)], [_truth(21)])

    def test_non_object_prediction_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an object"):
            EVAL.evaluate_bear_count(["row"], [_truth(21)])

    def test_prediction_without_a_source_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "video.source"):
            EVAL.evaluate_bear_count([{"bear_count": 3}], [_truth(21)])

    def test_unlabeled_predictions_are_reported_not_scored(self) -> None:
        report = EVAL.evaluate_bear_count([_row(21), _row(3, OTHER)], [_truth(21)])
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(
            report["unlabeled_predictions"], [{"label": OTHER, "predicted": 3}]
        )


class CoverageTests(unittest.TestCase):
    def test_window_predictions_against_full_video_label_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "No prediction row matches"):
            EVAL.evaluate_bear_count([_window_row(0, 19, 2)], [_truth(21)])

    def test_other_video_only_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "No prediction row matches"):
            EVAL.evaluate_bear_count([_row(3, OTHER)], [_truth(21)])

    def test_presence_evaluator_has_the_same_check(self) -> None:
        bear_eval = EVAL._bear_eval
        with self.assertRaisesRegex(ValueError, "No prediction row matches"):
            bear_eval.evaluate_bear_presence(
                [{**_window_row(0, 19, 0), bear_eval.FLAG_FIELD: True}],
                [{"source": SOURCE, bear_eval.FLAG_FIELD: True}],
            )


class GroundTruthTests(unittest.TestCase):
    def test_shipped_ground_truth_is_one_full_video_with_21_bears(self) -> None:
        clips = EVAL.load_ground_truth(GROUND_TRUTH)
        self.assertEqual(clips, [{"key": ("video", SOURCE), "bear_count": 21}])

    def test_ground_truth_matches_the_query_input(self) -> None:
        clips = EVAL.load_ground_truth(GROUND_TRUTH)
        rows = [
            json.loads(line)
            for line in (ROOT / "data" / "swan_valley_full.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(
            {EVAL._bear_eval.row_key(row) for row in rows},
            {clip["key"] for clip in clips},
        )

    def test_rejects_bad_ground_truth(self) -> None:
        cases = (
            {"clips": []},
            {"clips": "clip"},
            {"clips": [_truth(-1)]},
            {"clips": [_truth(True)]},
            {"clips": [_truth(1.5)]},
            {"clips": [{"source": SOURCE}]},
            {"clips": [_truth(21), _truth(20)]},
            {"clips": ["clip"]},
            {"clips": [{"bear_count": 21}]},
        )
        for payload in cases:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "truth.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        EVAL.load_ground_truth(path)


class ReportTests(unittest.TestCase):
    def test_report_shows_signed_error_and_cost(self) -> None:
        cost = EVAL._bear_eval.CostReport(
            label="Detect count", wall_time_sec=2.0, prompt_calls=0, n_rows=1
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            EVAL.print_report([_row(24)], [_truth(21)], label="Detect count", cost=cost)
        out = buffer.getvalue()
        self.assertIn("DETECT COUNT — GROUND-TRUTH EVALUATION", out)
        self.assertIn("wall time: 2.000s", out)
        self.assertIn("exact-match accuracy : 0.000", out)
        self.assertIn("mean absolute error  : 3.000", out)
        self.assertIn("relative error       : 0.143", out)
        self.assertIn("truth=21  predicted=24  error=+3  over", out)

    def test_report_marks_absent_rows_and_zero_truth(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            EVAL.print_report([], [_truth(0)], label="Detect count")
        out = buffer.getvalue()
        self.assertIn("absent (scored 0)", out)
        self.assertIn("relative error       : n/a", out)

    def test_main_scores_saved_predictions_for_both_pipelines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            detect_path = Path(tmp) / "detect.json"
            semantic_path = Path(tmp) / "semantic.json"
            report_path = Path(tmp) / "report.json"
            detect_path.write_text(json.dumps([_row(25)]), encoding="utf-8")
            semantic_path.write_text(json.dumps([_row(18)]), encoding="utf-8")
            argv = [
                "eval_animals_bear_count.py",
                "--pred-json",
                str(detect_path),
                "--semantic-json",
                str(semantic_path),
                "--save-report",
                str(report_path),
            ]
            buffer = io.StringIO()
            cwd = os.getcwd()
            try:
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(buffer):
                    EVAL.main()
            finally:
                os.chdir(cwd)
            payload = json.loads(report_path.read_text(encoding="utf-8"))

        out = buffer.getvalue()
        self.assertIn("COMPARISON", out)
        self.assertRegex(out, r"true count\s+21\s+21")
        self.assertRegex(out, r"predicted count\s+25\s+18")
        self.assertRegex(out, r"MAE\s+4\.000\s+3\.000")
        self.assertRegex(out, r"over-counted clips\s+1\s+0")
        self.assertRegex(out, r"under-counted clips\s+0\s+1")
        self.assertEqual(set(payload), {"Detect count", "Semantic map"})
        self.assertEqual(payload["Detect count"]["accuracy"]["details"][0]["error"], 4)
        self.assertEqual(payload["Semantic map"]["accuracy"]["details"][0]["error"], -3)


class PlanDetectionTests(unittest.TestCase):
    def test_runtime_checks_follow_plan_shape(self) -> None:
        bear_eval = EVAL._bear_eval
        detect = bear_eval._load_query_output(ROOT / "examples" / "animals_bear_count.py")
        semantic = bear_eval._load_query_output(ROOT / "examples" / "animals_bear_count_map.py")
        self.assertTrue(bear_eval._plan_uses_detect(detect))
        self.assertFalse(bear_eval._plan_uses_prompt(detect))
        self.assertFalse(bear_eval._plan_uses_detect(semantic))
        self.assertTrue(bear_eval._plan_uses_prompt(semantic))

    def test_run_pipeline_checks_cv2_only_for_detect_plans(self) -> None:
        bear_eval = EVAL._bear_eval
        with mock.patch.object(bear_eval, "_require_detect_runtime") as runtime, \
                mock.patch.object(bear_eval, "execute", return_value=iter([])), \
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test"}):
            bear_eval._run_pipeline(ROOT / "examples" / "animals_bear_count.py", label="d")
            self.assertEqual(runtime.call_count, 1)
            bear_eval._run_pipeline(ROOT / "examples" / "animals_bear_count_map.py", label="s")
            self.assertEqual(runtime.call_count, 1)


if __name__ == "__main__":
    unittest.main()
