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
from scripts.query_types import wld_localization as wl  # noqa: E402
from scripts.query_types.wld import load_pairs  # noqa: E402
from tests.wld_fixtures import DURATION, FakeGemini, fake_yoloe, has_ffmpeg, make_fake_wld, write_scans  # noqa: E402


def bear_answers(payload: dict) -> dict:
    """Fake Gemini. Narration: the bear at 125-135 s. Whole film A: the bear at
    130-170 s. Any clip of film A for "bear": 5-25 s into the clip, plus one
    reversed interval that must be dropped. Everything else: nothing."""
    bear = payload["animal"] == "bear"
    if "transcript" in payload:
        mentioned = bear and "A bear walks" in payload["transcript"]
        return {"intervals": [{"start": 125.0, "end": 135.0}] if mentioned else []}
    video = payload["video"]
    if "uri" in video:  # the whole film, from YouTube
        film, whole = video["uri"].rsplit("v=", 1)[1], True
    else:  # a standalone clip, sent inline
        assert video["inline"] is True and Path(video["path"]).exists()
        film, whole = Path(video["path"]).stem.split("__")[0], False
    if not (bear and film == "A"):
        return {"intervals": []}
    if whole:
        return {"intervals": [{"start": 130.0, "end": 170.0}]}
    return {"intervals": [{"start": 5.0, "end": 25.0}, {"start": 50.0, "end": 40.0}]}


class HelperTests(unittest.TestCase):
    def test_parse_intervals_clips_and_counts_dropped(self) -> None:
        row = {"intervals": [{"start": -2, "end": 4}, {"start": 9, "end": 3}, {"start": 8, "end": 50}]}
        self.assertEqual(wl.parse_intervals(row, 10.0), ([(0.0, 4.0), (8.0, 10.0)], 1))

    def test_parse_intervals_rejects_malformed_answers(self) -> None:
        for row in ({"intervals": None}, {"intervals": [{"start": "1", "end": 2}]}, {"intervals": [[1, 2]]}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                wl.parse_intervals(row, 10.0)

    def test_long_windows_become_clips_of_at_most_about_two_minutes(self) -> None:
        self.assertEqual(wl.clip_pieces([(100.0, 130.0)]), [(100.0, 130.0)])
        self.assertEqual(wl.clip_pieces([(0.0, 300.0)]), [(0.0, 120.0), (120.0, 240.0), (240.0, 300.0)])
        # A tail under 30 s joins the last piece instead of becoming a tiny clip.
        self.assertEqual(wl.clip_pieces([(10.0, 260.0)]), [(10.0, 130.0), (130.0, 260.0)])
        self.assertEqual(wl.clip_pieces([]), [])

    def test_format_transcript(self) -> None:
        cues = [{"start": 1.25, "end": 3.0, "text": "A bear."}, {"start": 4.0, "end": 5.5, "text": "Gone."}]
        self.assertEqual(wl.format_transcript(cues), "[1.2-3.0] A bear.\n[4.0-5.5] Gone.")

    def test_plans_round_trip(self) -> None:
        for prompt in (wl.VIDEO_PROMPT, wl.TRANSCRIPT_PROMPT, wl.CANDIDATES_PROMPT):
            with self.subTest(prompt=prompt[0][:20]):
                plan = wl.localize_plan("pairs.jsonl", prompt)
                self.assertEqual(parse_query(render_query(program_from_plan(plan))).output_expr, plan)

    def test_every_plan_pays_for_known_stages(self) -> None:
        self.assertEqual(set(wl.PLAN_STAGES), set(wl.PLANS))
        for plan, stages in wl.PLAN_STAGES.items():
            for stage in stages:
                with self.subTest(plan=plan, stage=stage):
                    self.assertTrue(stage == "scan" or stage in wl.STAGES)


@unittest.skipUnless(has_ffmpeg(), "needs ffmpeg")
class EndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.out = self.root / "out"
        make_fake_wld(self.root)
        self.pairs = load_pairs(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, repeats=(1, 2)) -> None:
        kwargs = {"out_root": self.out, "wld_root": self.root, "executor_factory": FakeGemini.factory(bear_answers),
                  "allow_cpu": True}
        with mock.patch.object(wl, "execute", side_effect=fake_yoloe()), \
                mock.patch.object(wl, "detector_device", return_value=("cpu", None)), \
                mock.patch.object(wl, "load_average", return_value=[0.5, 0.4, 0.3]):
            for repeat in repeats:
                wl.run_all(self.pairs, repeat, **kwargs)

    def test_stages_need_their_inputs(self) -> None:
        kwargs = {"out_root": self.out, "wld_root": self.root, "executor_factory": FakeGemini.factory(bear_answers)}
        with self.assertRaisesRegex(SystemExit, "needs candidates"):
            wl.run_stage("transcript_mllm", self.pairs, 1, **kwargs)
        with self.assertRaisesRegex(SystemExit, "needs scan"):
            wl.run_stage("yoloe_mllm", self.pairs, 1, **kwargs)

    def test_every_plan_runs_and_scores(self) -> None:
        write_scans(self.out, wl.FILTER_VARIANT, self.pairs)
        self._run()
        with mock.patch("scripts.query_types.common.GPU_DOLLARS_PER_HOUR", 0.5):
            summary = wl.score_results(self.pairs, out_root=self.out, wld_root=self.root)
        self.assertEqual((summary["runs"], summary["incomplete"]), (2, []))
        plans = summary["plans"]

        def seconds(plan, metric):
            value = plans[plan]["seconds_scores"][metric]
            self.assertEqual(value["sd"], 0.0)
            return value["mean"]

        # The bear is on screen at 130-170 s. Each plan's prediction:
        expected = {
            "mllm_video": (1.0, 1.0),                  # 130-170
            "yoloe_video": (40 / 70, 1.0),              # 130-200, YOLOE overshoots
            "transcript_only": (0.5, 5 / 40),           # 125-135
            "transcript_windows": (0.5, 15 / 40),       # the narration window itself, 115-145
            "transcript_mllm": (0.5, 10 / 40),          # window 115-145, clip answer 120-140
            "transcript_yoloe": (1.0, 15 / 40),         # YOLOE inside 115-145: 130-145
            "yoloe_mllm": (0.75, 15 / 40),              # window 120-210, clip answer 125-145
            "transcript_yoloe_mllm": (0.75, 15 / 40),   # hits 130-145, window 120-155, answer 125-145
        }
        for plan, (precision, recall) in expected.items():
            with self.subTest(plan=plan):
                self.assertAlmostEqual(seconds(plan, "precision"), precision)
                self.assertAlmostEqual(seconds(plan, "recall"), recall)
        self.assertEqual(plans["mllm_video"]["interval_scores"]["f1"]["mean"], 1.0)

        # Narration windows: 115-145 marks 30 s of film and keeps 15 of the bear's 40 labeled seconds,
        # the only labeled pair (the brief pair's stretches are ignored time).
        total_film = len(self.pairs) * DURATION
        self.assertAlmostEqual(plans["transcript_windows"]["predicted_fraction"]["mean"], 30 / total_film)
        self.assertEqual(set(plans["_narration_recall"]), {"A|bear"})
        self.assertAlmostEqual(plans["_narration_recall"]["A|bear"]["mean"], 15 / 40)

        # Each call costs 50 input tokens; chained plans also pay for the 18 narration-candidate calls.
        tokens = {plan: plans[plan]["input_tokens"]["mean"] for plan in wl.PLANS}
        self.assertEqual(tokens, {
            "mllm_video": 18 * 50, "yoloe_video": 0, "transcript_only": 18 * 50, "transcript_windows": 18 * 50,
            "transcript_mllm": 19 * 50, "transcript_yoloe": 18 * 50, "yoloe_mllm": 50, "transcript_yoloe_mllm": 19 * 50,
        })
        # Dollars: 50 text tokens per call at $0.25 per 1M, and GPU time at $0.50 per hour.
        # yoloe_video pays only the 18 scans of 30 s; transcript_yoloe pays for YOLOE on the windows.
        token = 50 * 0.25 / 1e6
        self.assertAlmostEqual(plans["mllm_video"]["dollars"]["mean"], 18 * token)
        self.assertEqual(plans["mllm_video"]["gpu_seconds"]["mean"], 0.0)
        self.assertAlmostEqual(plans["yoloe_video"]["gpu_dollars"]["mean"], 18 * 30 / 3600 * 0.5)
        self.assertAlmostEqual(plans["yoloe_mllm"]["dollars"]["mean"], token + 18 * 30 / 3600 * 0.5)
        self.assertGreater(plans["transcript_yoloe"]["gpu_seconds"]["mean"], 0.0)
        costs = summary["costs"]
        self.assertEqual((costs["gpu_dollars_per_hour"], costs["reference_plan"]), (0.5, "mllm_video"))
        self.assertEqual(set(costs["break_even_gpu_dollars_per_hour"]),
                         {"yoloe_video", "transcript_yoloe", "yoloe_mllm", "transcript_yoloe_mllm"})
        self.assertAlmostEqual(costs["break_even_gpu_dollars_per_hour"]["yoloe_video"], 18 * token / (18 * 30 / 3600))
        sent = {plan: plans[plan]["video_fraction_sent"]["mean"] * total_film for plan in wl.PLANS}
        self.assertAlmostEqual(sent["mllm_video"], total_film)
        self.assertAlmostEqual(sent["transcript_mllm"], 30.0)
        self.assertAlmostEqual(sent["yoloe_mllm"], 90.0)
        self.assertAlmostEqual(sent["transcript_yoloe_mllm"], 35.0)
        self.assertEqual(sent["transcript_yoloe"], 0.0)

        # The reversed clip answer was dropped; the raw answer is kept in clip time.
        stage = json.loads((wl.stage_dir("transcript_mllm", 1, self.out) / "A__bear.json").read_text())
        self.assertEqual(stage["dropped"], 1)
        self.assertEqual(stage["answers"], [{"clip": [115.0, 145.0], "raw": [{"start": 5.0, "end": 25.0}, {"start": 50.0, "end": 40.0}]}])
        self.assertGreater(stage["cut_seconds"], 0)
        self.assertEqual(list(self.out.rglob("*.mp4")), [])
        # YOLOE on narration windows is timed, so it records the machine's load.
        windows = json.loads((wl.stage_dir("yoloe_windows", 1, self.out) / "A__bear.json").read_text())
        self.assertEqual(windows["load_average"], [0.5, 0.4, 0.3])
        report = wl.format_report(summary)
        for expected in ("| transcript_yoloe_mllm |", "| transcript_windows |", "| Gemini $ | GPU $ | Total $ |",
                         "- yoloe_video: $", "| A|bear | 0.375 ± 0.000 |"):
            self.assertIn(expected, report)

    def test_score_command_writes_the_summary_and_prints_the_report(self) -> None:
        write_scans(self.out, wl.FILTER_VARIANT, self.pairs)
        self._run(repeats=(1,))
        with mock.patch.object(wl, "OUT_ROOT", self.out), mock.patch("builtins.print") as printed:
            wl.main(["--wld-root", str(self.root), "score"])
        written = json.loads((self.out / "localization" / "summary.json").read_text(encoding="utf-8"))
        expected = wl.score_results(self.pairs, out_root=self.out, wld_root=self.root)
        self.assertEqual(written, json.loads(json.dumps(expected)))
        self.assertIn("| transcript_yoloe_mllm |", printed.call_args.args[0])

    def test_unfinished_repeat_is_reported_not_scored(self) -> None:
        write_scans(self.out, wl.FILTER_VARIANT, self.pairs)
        self._run(repeats=(1,))
        (wl.stage_dir("mllm_video", 1, self.out) / "A__bear.json").unlink()
        summary = wl.score_results(self.pairs, out_root=self.out, wld_root=self.root)
        self.assertEqual((summary["runs"], summary["incomplete"]), (0, ["run1"]))

    def test_scoring_needs_the_filter_scans(self) -> None:
        with self.assertRaisesRegex(SystemExit, "'large' YOLOE scan is missing"):
            wl.score_results(self.pairs, out_root=self.out, wld_root=self.root)


class CliTests(unittest.TestCase):
    def test_run_one_stage_or_all(self) -> None:
        with mock.patch.object(wl, "load_pairs", return_value=[]), mock.patch.object(wl, "run_stage") as run_stage, \
                mock.patch.object(wl, "run_all") as run_all:
            wl.main(["run", "--repeat", "3", "--stage", "candidates"])
            wl.main(["run", "--repeat", "2"])
        self.assertEqual((run_stage.call_args.args[0], run_stage.call_args.args[2]), ("candidates", 3))
        self.assertEqual(run_all.call_args.args[1], 2)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            wl.main(["run", "--stage", "nonsense"])


if __name__ == "__main__":
    unittest.main()
