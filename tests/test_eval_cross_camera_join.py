from __future__ import annotations

import contextlib
import importlib.util
import io
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
    path = ROOT / "examples" / "eval_cross_camera_join.py"
    spec = importlib.util.spec_from_file_location("eval_cross_camera_join", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()


def _traj(vehicle_id, cls, color, subtype, segments):
    return {
        "vehicle_id": vehicle_id,
        "attributes": {"class": cls, "color": color, "subtype": subtype},
        "timeline": [
            {"camera_id": cam, "entered": a, "exited": b} for cam, a, b in segments
        ],
    }


class FormattingTests(unittest.TestCase):
    def test_fmt_attrs(self) -> None:
        text = EVAL._fmt_attrs(_traj("x", "car", "red", "sedan", []))
        self.assertIn("class='car'", text)
        self.assertIn("color='red'", text)
        self.assertIn("subtype='sedan'", text)

    def test_fmt_timeline(self) -> None:
        text = EVAL._fmt_timeline(
            _traj("x", "car", "red", "sedan", [("cam-a", 0.0, 5.0)])
        )
        self.assertIn("cam-a[0.0..5.0]", text)

    def test_fmt_timeline_empty(self) -> None:
        self.assertEqual(EVAL._fmt_timeline({"timeline": []}), "(no timeline)")


class ReportTests(unittest.TestCase):
    def _run(self, preds, refs, min_score=0.25):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            report = EVAL._print_report(preds, refs, min_score)
        return report, buffer.getvalue()

    def test_tp_fp_fn_counts_and_diagnostics(self) -> None:
        cams = [("cam-i24v-highway2", 0.0, 5.0), ("cam-i24v-highway3", 0.0, 5.0)]
        preds = [
            _traj("p1", "car", "white", "sedan", cams),  # matches r1
            _traj(
                "p2", "suv", "gray", "suv",
                [("cam-i24v-highway2", 50.0, 51.0), ("cam-i24v-highway3", 50.0, 51.0)],
            ),  # false positive: no temporal overlap with any ref
        ]
        refs = [
            _traj("r1", "car", "white", "sedan", cams),  # matched by p1
            _traj(
                "r2", "truck", "red", "flatbed",
                [("cam-i24v-highway2", 100.0, 101.0), ("cam-i24v-highway3", 100.0, 101.0)],
            ),  # false negative
        ]
        report, out = self._run(preds, refs)

        self.assertEqual(report["true_positives"], 1)
        self.assertEqual(report["false_positives"], 1)
        self.assertEqual(report["false_negatives"], 1)
        # The FP prediction and FN reference are named in the diagnostics.
        self.assertIn("FALSE POSITIVES (1)", out)
        self.assertIn("'p2'", out)
        self.assertIn("FALSE NEGATIVES (1)", out)
        self.assertIn("'r2'", out)

    def test_single_camera_gt_flagged_as_unrecoverable(self) -> None:
        preds: list = []  # no predictions -> every ref is a false negative
        refs = [_traj("r1", "car", "white", "sedan", [("cam-i24v-highway2", 0.0, 5.0)])]
        report, out = self._run(preds, refs)
        self.assertEqual(report["false_negatives"], 1)
        self.assertIn("single-camera GT", out)

    def test_perfect_match(self) -> None:
        cams = [("cam-i24v-highway2", 0.0, 5.0), ("cam-i24v-highway3", 0.0, 5.0)]
        preds = [_traj("p1", "car", "white", "sedan", cams)]
        refs = [_traj("r1", "car", "white", "sedan", cams)]
        report, _ = self._run(preds, refs)
        self.assertEqual(report["true_positives"], 1)
        self.assertEqual(report["false_positives"], 0)
        self.assertEqual(report["false_negatives"], 0)
        self.assertAlmostEqual(report["precision"], 1.0)
        self.assertAlmostEqual(report["recall"], 1.0)

    def test_report_header_includes_label_and_cost(self) -> None:
        cams = [("cam-i24v-highway2", 0.0, 5.0), ("cam-i24v-highway3", 0.0, 5.0)]
        preds = [_traj("p1", "car", "white", "sedan", cams)]
        refs = [_traj("r1", "car", "white", "sedan", cams)]
        cost = EVAL.CostReport(
            label="Semantic join",
            wall_time_sec=1.5,
            prompt_calls=1,
            n_trajectories=1,
            total_tokens=1234,
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            EVAL._print_report(preds, refs, 0.25, label="Semantic join", cost=cost)
        out = buffer.getvalue()
        self.assertIn("SEMANTIC JOIN — GROUND-TRUTH EVALUATION", out)
        self.assertIn("wall time: 1.500s", out)
        self.assertIn("prompt calls: 1", out)
        self.assertIn("total tokens: 1234", out)


class ComparisonTableTests(unittest.TestCase):
    def _result(self, label, *, n_pred, tp, fp, fn, cost):
        report = {
            "n_pred": n_pred,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": 0.75,
            "recall": 1.0,
            "f1": 0.857,
            "mean_timeline_score": 0.5,
            "mean_attribute_exact": 0.1,
            "mean_attribute_soft": 0.2,
        }
        return (label, report, cost)

    def test_comparison_table_lists_both_pipelines_and_costs(self) -> None:
        udf_cost = EVAL.CostReport(
            label="UDF join", wall_time_sec=2.0, prompt_calls=0, n_trajectories=12
        )
        sem_cost = EVAL.CostReport(
            label="Semantic join",
            wall_time_sec=9.0,
            prompt_calls=1,
            n_trajectories=9,
            prompt_tokens=5000,
            candidates_tokens=800,
            total_tokens=5800,
        )
        results = [
            self._result("UDF join", n_pred=12, tp=9, fp=3, fn=0, cost=udf_cost),
            self._result("Semantic join", n_pred=9, tp=8, fp=1, fn=1, cost=sem_cost),
        ]
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            EVAL._print_comparison(results)
        out = buffer.getvalue()

        self.assertIn("COMPARISON", out)
        self.assertIn("UDF join", out)
        self.assertIn("Semantic join", out)
        # Cost rows reflect the per-pipeline CostReport values.
        self.assertIn("total tokens", out)
        self.assertIn("5800", out)
        self.assertIn("prompt calls", out)
        # Token-free UDF column stays at 0 tokens.
        self.assertRegex(out, r"total tokens\s+0\s+5800")


if __name__ == "__main__":
    unittest.main()
