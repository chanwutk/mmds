from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mmds import Detect, Input  # noqa: E402
from scripts.query_types import time_detect_phases as tdp  # noqa: E402


class SummarizeTests(unittest.TestCase):
    def test_splits_startup_from_detect_runs(self) -> None:
        phases = {
            "imports": 2.0,
            "device": 1.0,
            "model_load:yoloe-11s-seg.pt": 3.0,
            "text_encoder": 4.0,
            "detect_run_1": 0.5,
            "detect_run_2": 0.25,
        }
        self.assertEqual(
            tdp.summarize(phases),
            {
                "startup_seconds": 10.0,
                "detect_first_seconds": 0.5,
                "detect_warm_seconds": 0.25,
                "including_startup_seconds": 10.5,
            },
        )

    def test_single_run_is_both_first_and_warm(self) -> None:
        summary = tdp.summarize({"imports": 1.0, "detect_run_1": 2.0})
        self.assertEqual(summary["detect_first_seconds"], 2.0)
        self.assertEqual(summary["detect_warm_seconds"], 2.0)

    def test_requires_a_detect_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "No detect runs"):
            tdp.summarize({"imports": 1.0})


class PlanInspectionTests(unittest.TestCase):
    def test_reads_models_and_classes_from_detect_nodes(self) -> None:
        plan = Detect(Detect(Input("rows.jsonl"), "video", ["bear"], model="b.pt"), "video", ["dog"], model="a.pt")
        self.assertEqual(tdp.detect_models(plan), ["a.pt", "b.pt"])
        self.assertEqual(tdp.detect_classes(plan), ["bear"])

    def test_plan_without_detect_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no Detect node"):
            tdp.detect_models(Input("rows.jsonl"))
        with self.assertRaisesRegex(ValueError, "no Detect node"):
            tdp.detect_classes(Input("rows.jsonl"))


class LoadQueryTests(unittest.TestCase):
    def test_loads_output_from_a_query_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            query = Path(tmp) / "query.py"
            query.write_text(
                'from mmds import Detect, Input\noutput = Detect(Input("rows.jsonl"), "video", ["bear"])\n'
            )
            plan = tdp.load_query_output(query)
        self.assertEqual(plan.kind, "detect")

    def test_query_file_without_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            query = Path(tmp) / "query.py"
            query.write_text("x = 1\n")
            with self.assertRaisesRegex(ValueError, "does not define an 'output'"):
                tdp.load_query_output(query)


class TimedTests(unittest.TestCase):
    def test_records_elapsed_time_and_syncs(self) -> None:
        phases: dict[str, float] = {}
        synced = []
        result = tdp.timed(phases, "step", lambda: 42, lambda: synced.append(True))
        self.assertEqual(result, 42)
        self.assertEqual(synced, [True])
        self.assertGreaterEqual(phases["step"], 0.0)


if __name__ == "__main__":
    unittest.main()
