from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mmds.parser import parse_query  # noqa: E402
from mmds.render import program_from_plan, render_query  # noqa: E402
from scripts.query_types import wld_presence as wp  # noqa: E402
from scripts.query_types.wld import Pair, load_pairs  # noqa: E402
from tests.wld_fixtures import BEAR_ON_SCREEN, FakeGemini, has_ffmpeg, make_fake_wld, write_scans  # noqa: E402


def _pair(film: str = "f", animal: str = "bear", truth: str = "present", **kwargs) -> Pair:
    values = {"film_id": film, "animal": animal, "path": f"/videos/{film}.mp4", "duration": 300.0, "fps": 25.0,
              "frames": 7500, "truth": truth, "truth_any_frame": truth != "absent",
              "first_seconds": None if truth == "absent" else 10.0}
    values.update(kwargs)
    return Pair(**values)


def bear_visible(payload: dict) -> dict:
    """Fake Gemini: yes only for the bear in film A, and only if its video (or window) shows 130-170 s."""
    video = payload["video"]
    film = video["uri"].rsplit("v=", 1)[1]
    start, end = video.get("start", 0.0), video.get("end", float("inf"))
    on_screen = start < BEAR_ON_SCREEN[1] and end > BEAR_ON_SCREEN[0]
    return {"visible": film == "A" and payload["animal"] == "bear" and on_screen}


class PlanTests(unittest.TestCase):
    def test_presence_plan_round_trips_and_reads_animal_and_video(self) -> None:
        plan = wp.presence_plan("pairs.jsonl")
        rendered = render_query(program_from_plan(plan))
        self.assertEqual(parse_query(rendered).output_expr, plan)
        self.assertIn('Record["animal"]', rendered)
        self.assertIn('Record["video"]', rendered)


class TaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.scratch = Path(self._tmp.name)
        self.records: list[dict] = []
        self.executor = FakeGemini(self.records.append, bear_visible)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_whole_film_asks_once_with_the_youtube_url(self) -> None:
        result = wp.whole_film(_pair("A"), self.executor, self.scratch)
        self.assertTrue(result["answer"])
        self.assertEqual((len(self.records), result["video_seconds_sent"]), (1, 300.0))

    def test_early_stop_stops_at_the_first_yes(self) -> None:
        result = wp.early_stop(_pair("A"), self.executor, self.scratch)
        # Windows 0-120 (no), 120-240 (yes, contains 130 s); 240-300 is never asked.
        self.assertEqual((result["answer"], result["chunks_asked"], result["chunks_total"]), (True, 2, 3))
        self.assertEqual(result["video_seconds_sent"], 240.0)

    def test_early_stop_watches_everything_for_a_no(self) -> None:
        result = wp.early_stop(_pair("B"), self.executor, self.scratch)
        self.assertEqual((result["answer"], result["chunks_asked"], result["video_seconds_sent"]), (False, 3, 300.0))

    def test_yoloe_then_gemini_asks_the_most_confident_window_first(self) -> None:
        # 25 fps, stride 5: hits at 100 s (0.3) and 150 s (0.9). The 150 s window is asked first and says yes.
        scan = {"stride": 5, "seconds": 240.0, "frame_max_conf": {"2500": 0.3, "3750": 0.9}}
        result = wp.yoloe_then_gemini({"A|bear": scan})(_pair("A"), self.executor, self.scratch)
        self.assertTrue(result["answer"])
        self.assertEqual((result["windows_asked"], result["windows_total"]), (1, 2))
        # Only Gemini's time is stored; the scan's time is added when scoring.
        self.assertGreaterEqual(result["gemini_seconds"], 0.0)
        self.assertTrue({"seconds", "yoloe_seconds"}.isdisjoint(result))

    def test_yoloe_then_gemini_without_hits_says_no_without_asking(self) -> None:
        scan = {"stride": 5, "seconds": 240.0, "frame_max_conf": {}}
        result = wp.yoloe_then_gemini({"B|bear": scan})(_pair("B"), self.executor, self.scratch)
        self.assertEqual((result["answer"], result["windows_asked"], self.records), (False, 0, []))

    def test_a_non_boolean_answer_is_an_error(self) -> None:
        executor = FakeGemini(self.records.append, lambda payload: {"visible": "yes"})
        with self.assertRaisesRegex(ValueError, "expected a boolean"):
            wp.whole_film(_pair("A"), executor, self.scratch)


class RunTests(unittest.TestCase):
    def test_bad_arguments_and_missing_scans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                wp.run("mllm_half", [_pair()], 1, out_root=Path(tmp))
            with self.assertRaises(ValueError):
                wp.run("mllm_full", [_pair()], 0, out_root=Path(tmp))
            with self.assertRaisesRegex(SystemExit, "needs 'large' scans"):
                wp.run("yoloe_mllm", [_pair()], 1, out_root=Path(tmp))


class ScoreAnswersTests(unittest.TestCase):
    PAIRS = [_pair("a", truth="present"), _pair("b", truth="brief"), _pair("c", truth="absent"), _pair("d", truth="absent")]

    def test_brief_pairs_are_left_out(self) -> None:
        result = wp.score_answers(self.PAIRS, {"a|bear": True, "b|bear": True, "c|bear": True, "d|bear": False})
        self.assertEqual((result["pairs"], result["tp"], result["fp"], result["tn"]), (3, 1, 1, 1))

    def test_strict_counts_brief_pairs_as_present(self) -> None:
        result = wp.score_answers(self.PAIRS, {"a|bear": True, "b|bear": False, "c|bear": False, "d|bear": False}, strict=True)
        self.assertEqual((result["pairs"], result["fn"]), (4, 1))
        self.assertAlmostEqual(result["recall"], 0.5)


@unittest.skipUnless(has_ffmpeg(), "needs ffmpeg")
class EndToEndTests(unittest.TestCase):
    """Two fake films and a fake Gemini, through every Gemini plan twice, then scored."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.out = self.root / "out"
        make_fake_wld(self.root)
        self.pairs = load_pairs(self.root)
        for variant in ("small", "large"):
            write_scans(self.out, variant, self.pairs)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, plan: str, repeat: int) -> None:
        wp.run(plan, self.pairs, repeat, out_root=self.out, executor_factory=FakeGemini.factory(bear_visible))

    def test_every_plan_runs_and_scores(self) -> None:
        for repeat in (1, 2):
            for plan in wp.GEMINI_PLANS:
                self._run(plan, repeat)
        summary = wp.score_results(self.pairs, self.out)

        self.assertEqual(summary["truth"], {"present": 1, "brief": 1, "absent": 16})
        gemini = summary["gemini"]
        for plan in wp.GEMINI_PLANS:
            self.assertEqual((gemini[plan]["runs"], gemini[plan]["main"]["f1"]), (2, {"mean": 1.0, "sd": 0.0}))
        # One call per pair; early stop asks film A's bear 2 windows and every other pair all 3;
        # YOLOE->Gemini asks only film A's bear (the only pair with YOLOE hits), once.
        calls = {plan: gemini[plan]["calls"]["mean"] for plan in wp.GEMINI_PLANS}
        self.assertEqual(calls, {"mllm_full": 18, "mllm_early": 2 + 17 * 3, "yoloe_mllm": 1})
        self.assertEqual(gemini["mllm_full"]["input_tokens"]["mean"], 18 * 50)
        self.assertEqual(gemini["mllm_early"]["video_seconds_sent"]["mean"], 240 + 17 * 300)
        # Both scanned YOLOE variants are scored; the unscanned ones are absent.
        self.assertEqual(set(summary["yoloe"]), {"small", "large"})
        main = summary["yoloe"]["small"]["main_rule"]
        self.assertEqual((main["tp"], main["fp"], main["fn"]), (1, 0, 0))
        report = wp.format_report(summary)
        self.assertIn("| mllm_early | 2 |", report)
        self.assertIn("yoloe_full (large)", report)

    def test_yoloe_mllm_time_comes_from_the_current_scans(self) -> None:
        self._run("yoloe_mllm", 1)
        before = wp.score_results(self.pairs, self.out)["gemini"]["yoloe_mllm"]
        self.assertEqual(before["yoloe_seconds"]["mean"], 18 * 30.0)
        self.assertAlmostEqual(before["seconds"]["mean"], 18 * 30.0 + before["gemini_seconds"]["mean"])
        # A rescan with other times changes the plan's time without running Gemini again.
        write_scans(self.out, "large", self.pairs, seconds=10.0)
        after = wp.score_results(self.pairs, self.out)["gemini"]["yoloe_mllm"]
        self.assertEqual(after["yoloe_seconds"]["mean"], 18 * 10.0)
        self.assertEqual(after["gemini_seconds"], before["gemini_seconds"])

    def test_dollars_price_tokens_and_gpu_time_and_give_break_even_rates(self) -> None:
        for plan in wp.GEMINI_PLANS:
            self._run(plan, 1)
        with mock.patch("scripts.query_types.common.GPU_DOLLARS_PER_HOUR", 0.5):
            summary = wp.score_results(self.pairs, self.out)
        token = 50 * 0.25 / 1e6  # each fake call: 50 text tokens at $0.25 per 1M
        gemini = summary["gemini"]
        full, filtered = gemini["mllm_full"], gemini["yoloe_mllm"]
        self.assertAlmostEqual(full["dollars"]["mean"], 18 * token)
        self.assertEqual(full["gpu_dollars"]["mean"], 0.0)
        # yoloe_mllm pays one Gemini call plus 18 scans of 30 s each at $0.50 per GPU-hour.
        self.assertAlmostEqual(filtered["gemini_dollars"]["mean"], token)
        self.assertAlmostEqual(filtered["gpu_dollars"]["mean"], 18 * 30 / 3600 * 0.5)
        self.assertAlmostEqual(filtered["dollars"]["mean"], token + 18 * 30 / 3600 * 0.5)
        self.assertAlmostEqual(summary["yoloe"]["small"]["main_rule"]["full_dollars"], 18 * 30 / 3600 * 0.5)
        costs = summary["costs"]
        self.assertEqual((costs["gpu_dollars_per_hour"], costs["reference_plan"]), (0.5, "mllm_full"))
        break_even = costs["break_even_gpu_dollars_per_hour"]
        self.assertAlmostEqual(break_even["yoloe_mllm"], (18 * token - token) / (18 * 30 / 3600))
        self.assertAlmostEqual(break_even["yoloe_full (large)"], 18 * token / (18 * 30 / 3600))
        report = wp.format_report(summary)
        self.assertIn("| Gemini $ | GPU $ | Total $ |", report)
        self.assertIn("- yoloe_mllm: $", report)

    def test_no_break_even_until_the_reference_plan_has_a_complete_run(self) -> None:
        self._run("yoloe_mllm", 1)
        self._run("mllm_full", 1)
        (wp.plan_dir("mllm_full", 1, self.out) / "A__bear.json").unlink()  # the reference run is unfinished
        summary = wp.score_results(self.pairs, self.out)
        self.assertEqual(summary["costs"]["break_even_gpu_dollars_per_hour"], {})
        self.assertNotIn("GPU time is priced", wp.format_report(summary))

    def test_score_command_writes_the_summary_and_prints_the_report(self) -> None:
        for plan in wp.GEMINI_PLANS:
            self._run(plan, 1)
        with mock.patch.object(wp, "OUT_ROOT", self.out), mock.patch("builtins.print") as printed:
            wp.main(["--wld-root", str(self.root), "score"])
        written = json.loads((self.out / "presence" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(written, json.loads(json.dumps(wp.score_results(self.pairs, self.out))))
        self.assertIn("| mllm_full | 1 |", printed.call_args.args[0])

    def test_scoring_yoloe_mllm_needs_every_filter_scan(self) -> None:
        self._run("yoloe_mllm", 1)
        (self.out / "yoloe" / "large" / "B__bear.json").unlink()
        with self.assertRaisesRegex(SystemExit, "needs 'large' scans for 1 pairs"):
            wp.score_results(self.pairs, self.out)

    def test_unfinished_repeat_is_listed_not_scored(self) -> None:
        self._run("mllm_full", 1)
        (wp.plan_dir("mllm_full", 1, self.out) / "A__bear.json").unlink()
        summary = wp.score_results(self.pairs, self.out)
        self.assertEqual(
            (summary["gemini"]["mllm_full"]["runs"], summary["gemini"]["mllm_full"]["incomplete"]),
            (0, [{"run": "run1", "pairs_done": 17}]),
        )
        self.assertIn("unfinished repeats", wp.format_report(summary))

    def test_partly_scanned_variant_is_reported_not_scored(self) -> None:
        (self.out / "yoloe" / "small" / "A__bear.json").unlink()
        self.assertEqual(wp.score_results(self.pairs, self.out)["yoloe"]["small"], {"incomplete": "17 of 18 pairs scanned"})


class CliTests(unittest.TestCase):
    def test_run_takes_a_plan_and_a_repeat(self) -> None:
        with mock.patch.object(wp, "load_pairs", return_value=[]), mock.patch.object(wp, "run") as run:
            wp.main(["run", "--plan", "mllm_early", "--repeat", "2"])
        self.assertEqual((run.call_args.args[0], run.call_args.args[2]), ("mllm_early", 2))
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            wp.main(["run", "--plan", "nonsense"])


if __name__ == "__main__":
    unittest.main()
