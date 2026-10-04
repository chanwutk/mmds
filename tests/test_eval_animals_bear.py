from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_eval_module():
    path = ROOT / "examples" / "eval_animals_bear.py"
    spec = importlib.util.spec_from_file_location("eval_animals_bear", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()


def _row(start: float, end: float, bear_present: bool) -> dict:
    return {
        "video": {
            "type": "VideoView",
            "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8",
            "start": start,
            "end": end,
        },
        "bear_present": bear_present,
    }


def _truth(start: float, end: float, bear_present: bool) -> dict:
    return {"start": start, "end": end, "bear_present": bear_present}


class BearPresenceScoreTests(unittest.TestCase):
    def test_dropped_positive_is_a_false_negative(self) -> None:
        ground_truth = [
            _truth(0, 19, True),
            _truth(20, 34, False),
            _truth(36, 58, True),
        ]
        predictions = [_row(0, 19, True)]
        report = EVAL.evaluate_bear_presence(predictions, ground_truth)
        self.assertEqual(report["true_positives"], 1)
        self.assertEqual(report["false_negatives"], 1)
        self.assertEqual(report["true_negatives"], 1)
        self.assertEqual(report["false_positives"], 0)
        self.assertAlmostEqual(report["recall"], 0.5)
        dropped = [item for item in report["details"] if item["label"] == "36-58s"]
        self.assertFalse(dropped[0]["emitted"])
        self.assertEqual(dropped[0]["kind"], "fn")

    def test_explicit_false_negative_and_false_positive(self) -> None:
        ground_truth = [_truth(0, 19, True), _truth(20, 34, False)]
        predictions = [_row(0, 19, False), _row(20, 34, True)]
        report = EVAL.evaluate_bear_presence(predictions, ground_truth)
        self.assertEqual(report["false_negatives"], 1)
        self.assertEqual(report["false_positives"], 1)
        self.assertEqual(report["precision"], 0.0)
        self.assertEqual(report["recall"], 0.0)

    def test_perfect_match(self) -> None:
        ground_truth = [_truth(0, 19, True), _truth(102, 108, False)]
        predictions = [_row(0, 19.0, True), _row(102, 108, False)]
        report = EVAL.evaluate_bear_presence(predictions, ground_truth)
        self.assertEqual(report["true_positives"], 1)
        self.assertEqual(report["true_negatives"], 1)
        self.assertAlmostEqual(report["precision"], 1.0)
        self.assertAlmostEqual(report["recall"], 1.0)
        self.assertAlmostEqual(report["f1"], 1.0)

    def test_full_video_matches_on_source_without_a_time_window(self) -> None:
        source = "https://www.youtube.com/watch?v=s5iU3nLOvi8"
        ground_truth = [{"source": source, "bear_present": True}]
        predictions = [
            {
                "video": {"type": "Video", "source": source},
                "bear_present": True,
            }
        ]
        report = EVAL.evaluate_bear_presence(predictions, ground_truth)
        self.assertEqual(report["true_positives"], 1)
        self.assertEqual(report["n_pred"], 1)
        self.assertEqual(report["details"][0]["label"], source)

    def test_shipped_ground_truth_is_the_full_compilation(self) -> None:
        clips = EVAL.load_ground_truth(EVAL.DEFAULT_GROUND_TRUTH)
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]["key"][0], "video")
        self.assertEqual(
            clips[0]["key"][1], "https://www.youtube.com/watch?v=s5iU3nLOvi8"
        )
        self.assertTrue(clips[0]["bear_present"])


class ReportFormattingTests(unittest.TestCase):
    def test_report_names_absent_false_negative_and_cost(self) -> None:
        ground_truth = [_truth(0, 19, True)]
        cost = EVAL.CostReport(
            label="Detect presence",
            wall_time_sec=1.5,
            prompt_calls=0,
            n_rows=0,
            total_tokens=0,
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            EVAL._print_report([], ground_truth, label="Detect presence", cost=cost)
        out = buffer.getvalue()
        self.assertIn("DETECT PRESENCE — GROUND-TRUTH EVALUATION", out)
        self.assertIn("wall time: 1.500s", out)
        self.assertIn("prompt calls: 0", out)
        self.assertIn("FALSE NEGATIVES (1)", out)
        self.assertIn("absent (scored false)", out)

    def test_comparison_table_lists_both_pipelines(self) -> None:
        report = {
            "n_pred": 2,
            "true_positives": 2,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 4,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
        presence_cost = EVAL.CostReport(
            label="Detect presence", wall_time_sec=3.0, prompt_calls=2, n_rows=2, total_tokens=400
        )
        semantic_cost = EVAL.CostReport(
            label="Semantic map",
            wall_time_sec=9.0,
            prompt_calls=6,
            n_rows=6,
            prompt_tokens=5000,
            candidates_tokens=800,
            total_tokens=5800,
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            EVAL._print_comparison(
                [
                    ("Detect presence", report, presence_cost),
                    ("Semantic map", dict(report), semantic_cost),
                ]
            )
        out = buffer.getvalue()
        self.assertIn("COMPARISON", out)
        self.assertIn("Detect presence", out)
        self.assertIn("Semantic map", out)
        self.assertRegex(out, r"prompt calls\s+2\s+6")
        self.assertRegex(out, r"total tokens\s+400\s+5800")
        json.dumps(presence_cost.to_dict())


class FiveBearsGroundTruthTests(unittest.TestCase):
    def test_shipped_label_is_one_full_video_positive(self) -> None:
        saved = EVAL.FLAG_FIELD
        EVAL.FLAG_FIELD = "at_least_five_bears"
        try:
            clips = EVAL.load_ground_truth(
                ROOT / "data" / "animals_five_bears_ground_truth.json"
            )
        finally:
            EVAL.FLAG_FIELD = saved
        self.assertEqual(len(clips), 1)
        self.assertTrue(clips[0]["at_least_five_bears"])
        self.assertEqual(clips[0]["key"][0], "video")


class PromptKeyTests(unittest.TestCase):
    def test_presence_plan_does_not_need_a_prompt_key(self) -> None:
        output = EVAL._load_query_output(EVAL.DEFAULT_PRESENCE_QUERY)
        self.assertFalse(EVAL._plan_uses_prompt(output))
        saved = {
            name: os.environ.pop(name, None)
            for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        }
        try:
            EVAL._require_prompt_key(output)
        finally:
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value

    def test_semantic_plan_requires_a_prompt_key(self) -> None:
        output = EVAL._load_query_output(EVAL.DEFAULT_SEMANTIC_QUERY)
        self.assertTrue(EVAL._plan_uses_prompt(output))
        saved = {
            name: os.environ.pop(name, None)
            for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        }
        try:
            with self.assertRaises(SystemExit):
                EVAL._require_prompt_key(output)
        finally:
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
