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


def _load(name: str):
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WRAPPER = _load("eval_vehicle_count")
EVAL = WRAPPER.configure(WRAPPER._load_count_eval())
PATH = "data/i24v_traffic/videos/highway2.mp4"


def _row(count, start: float = 0, end: float = 5) -> dict:
    return {
        "video": {"type": "VideoView", "path": PATH, "start": start, "end": end},
        "vehicle_count": count,
    }


class VehicleCountConfigTests(unittest.TestCase):
    def test_configure_points_at_the_vehicle_example(self) -> None:
        self.assertEqual(EVAL.COUNT_FIELD, "vehicle_count")
        self.assertEqual(EVAL.DEFAULT_COUNT_QUERY.name, "vehicle_count.py")
        self.assertEqual(EVAL.DEFAULT_SEMANTIC_QUERY.name, "vehicle_count_map.py")
        self.assertEqual(EVAL.DEFAULT_GROUND_TRUTH.name, "vehicle_count_ground_truth.json")
        for path in (EVAL.DEFAULT_COUNT_QUERY, EVAL.DEFAULT_SEMANTIC_QUERY, EVAL.DEFAULT_GROUND_TRUTH):
            self.assertTrue(path.exists(), path)

    def test_scoring_reads_the_configured_field(self) -> None:
        report = EVAL.evaluate_bear_count([_row(20)], [{"start": 0, "end": 5, "vehicle_count": 20}])
        self.assertEqual(report["exact_matches"], 1)
        with self.assertRaisesRegex(ValueError, "vehicle_count"):
            EVAL.evaluate_bear_count(
                [{"video": _row(0)["video"], "bear_count": 20}],
                [{"start": 0, "end": 5, "vehicle_count": 20}],
            )


class VehicleGroundTruthTests(unittest.TestCase):
    def test_shipped_label_is_twenty_vehicles_in_the_first_five_seconds(self) -> None:
        clips = EVAL.load_ground_truth(EVAL.DEFAULT_GROUND_TRUTH)
        self.assertEqual(clips, [{"key": ("window", 0.0, 5.0), "vehicle_count": 20}])

    def test_label_matches_the_query_input(self) -> None:
        clips = EVAL.load_ground_truth(EVAL.DEFAULT_GROUND_TRUTH)
        lines = (ROOT / "data" / "i24v_highway2_5s.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["video"]["path"], PATH)
        self.assertEqual(
            {EVAL._bear_eval.row_key(row) for row in rows}, {clip["key"] for clip in clips}
        )

    def test_window_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "No prediction row matches"):
            EVAL.evaluate_bear_count(
                [_row(20, 0, 10)], [{"start": 0, "end": 5, "vehicle_count": 20}]
            )


class VehicleMainTests(unittest.TestCase):
    def test_main_compares_saved_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            detect_path = Path(tmp) / "detect.json"
            semantic_path = Path(tmp) / "semantic.json"
            report_path = Path(tmp) / "report.json"
            detect_path.write_text(json.dumps([_row(20)]), encoding="utf-8")
            semantic_path.write_text(json.dumps([_row(13)]), encoding="utf-8")
            argv = [
                "eval_vehicle_count.py",
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
        self.assertIn("0-5s  truth=20  predicted=20  error=+0  exact", out)
        self.assertIn("0-5s  truth=20  predicted=13  error=-7  under", out)
        self.assertRegex(out, r"predicted count\s+20\s+13")
        self.assertRegex(out, r"relative error\s+0\.000\s+0\.350")
        self.assertEqual(payload["Detect count"]["accuracy"]["exact_matches"], 1)
        self.assertEqual(payload["Semantic map"]["accuracy"]["mae"], 7.0)

    def test_vehicle_plans_select_the_right_runtime_checks(self) -> None:
        bear_eval = EVAL._bear_eval
        detect = bear_eval._load_query_output(EVAL.DEFAULT_COUNT_QUERY)
        semantic = bear_eval._load_query_output(EVAL.DEFAULT_SEMANTIC_QUERY)
        self.assertTrue(bear_eval._plan_uses_detect(detect))
        self.assertFalse(bear_eval._plan_uses_prompt(detect))
        self.assertFalse(bear_eval._plan_uses_detect(semantic))
        self.assertTrue(bear_eval._plan_uses_prompt(semantic))


if __name__ == "__main__":
    unittest.main()
