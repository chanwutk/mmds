"""The WLD result summaries kept in git must stay readable by the current code.

They are the only results in the repository (everything else under
data/query_types/ is git-ignored), so these tests need no WLD data and run
on a fresh checkout.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.query_types import wld_localization as wl  # noqa: E402
from scripts.query_types import wld_presence as wp  # noqa: E402
from scripts.query_types.wld_yoloe import VARIANTS  # noqa: E402

SUMMARIES = ROOT / "data" / "query_types" / "wld"
REPEATS = 3


def _read(experiment: str) -> dict:
    return json.loads((SUMMARIES / experiment / "summary.json").read_text(encoding="utf-8"))


class PresenceSummaryTests(unittest.TestCase):
    def test_every_plan_has_three_repeats_and_dollars(self) -> None:
        summary = _read("presence")
        self.assertEqual((summary["pairs"], summary["truth"]), (45, {"present": 16, "brief": 0, "absent": 29}))
        for plan in wp.GEMINI_PLANS:
            with self.subTest(plan=plan):
                scored = summary["gemini"][plan]
                self.assertEqual((scored["runs"], scored["incomplete"]), (REPEATS, []))
                self.assertIsNotNone(scored["dollars"]["mean"])
        self.assertEqual(set(summary["yoloe"]), set(VARIANTS))
        self.assertIn("yoloe_mllm", summary["costs"]["break_even_gpu_dollars_per_hour"])

    def test_it_renders(self) -> None:
        report = wp.format_report(_read("presence"))
        for plan in wp.GEMINI_PLANS:
            self.assertIn(f"| {plan} | {REPEATS} |", report)
        self.assertIn("GPU time is priced at", report)


class LocalizationSummaryTests(unittest.TestCase):
    def test_every_plan_has_three_repeats_and_dollars(self) -> None:
        summary = _read("localization")
        self.assertEqual((summary["pairs"], summary["runs"], summary["incomplete"]), (45, REPEATS, []))
        for plan in wl.PLANS:
            with self.subTest(plan=plan):
                self.assertIsNotNone(summary["plans"][plan]["dollars"]["mean"])
        self.assertTrue(summary["plans"]["_narration_recall"])
        self.assertIn("transcript_yoloe_mllm", summary["costs"]["break_even_gpu_dollars_per_hour"])

    def test_it_renders(self) -> None:
        report = wl.format_report(_read("localization"))
        for plan in wl.PLANS:
            self.assertIn(f"| {plan} |", report)
        self.assertIn("Share of each labeled pair's on-screen time inside the narration windows", report)


if __name__ == "__main__":
    unittest.main()
