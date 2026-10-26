from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.query_types import common  # noqa: E402


class MakeChunksTests(unittest.TestCase):
    def test_exact_multiple(self) -> None:
        self.assertEqual(common.make_chunks(900.0, 300.0), [(0.0, 300.0), (300.0, 600.0), (600.0, 900.0)])

    def test_long_remainder_becomes_its_own_chunk(self) -> None:
        self.assertEqual(common.make_chunks(937.0, 300.0)[-1], (900.0, 937.0))

    def test_short_remainder_merges_into_previous_chunk(self) -> None:
        chunks = common.make_chunks(2700.003, 300.0)
        self.assertEqual(len(chunks), 9)
        self.assertEqual(chunks[-1], (2400.0, 2700.003))

    def test_remainder_exactly_the_tail_limit_is_kept(self) -> None:
        self.assertEqual(common.make_chunks(330.0, 300.0), [(0.0, 300.0), (300.0, 330.0)])

    def test_video_shorter_than_one_chunk(self) -> None:
        self.assertEqual(common.make_chunks(12.0, 300.0), [(0.0, 12.0)])

    def test_chunks_cover_duration_without_gaps(self) -> None:
        chunks = common.make_chunks(3003.0, 120.0)
        self.assertEqual(chunks[0][0], 0.0)
        self.assertEqual(chunks[-1][1], 3003.0)
        for (_, end), (start, _) in zip(chunks, chunks[1:]):
            self.assertEqual(end, start)

    def test_invalid_inputs_raise(self) -> None:
        for duration, chunk in ((0.0, 300.0), (-1.0, 300.0), (math.nan, 300.0), (100.0, 0.0), (100.0, math.inf)):
            with self.subTest(duration=duration, chunk=chunk), self.assertRaises(ValueError):
                common.make_chunks(duration, chunk)


class DetectorStrideTests(unittest.TestCase):
    def test_strides_that_hit_the_target_rate(self) -> None:
        self.assertEqual(common.detector_stride(25.0), 5)
        self.assertEqual(common.detector_stride(29.97), 6)
        self.assertEqual(common.detector_stride(5.0), 1)

    def test_rates_that_cannot_be_hit_or_are_invalid_raise(self) -> None:
        for fps in (0.0, -1.0, math.nan, 12.0, 3.0):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                common.detector_stride(fps)


class BinaryScoresTests(unittest.TestCase):
    def test_counts_and_rates(self) -> None:
        result = common.binary_scores([(True, True), (True, False), (False, True), (False, False), (False, False)])
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (1, 1, 1, 2))
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["accuracy"], 0.6)

    def test_undefined_rates_are_none(self) -> None:
        only_negatives = common.binary_scores([(False, False)])
        self.assertIsNone(only_negatives["precision"])
        self.assertIsNone(only_negatives["recall"])
        self.assertEqual(only_negatives["accuracy"], 1.0)
        self.assertIsNone(common.binary_scores([])["accuracy"])


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
        records[0]["op_type"] = "map"
        records[1]["op_type"] = "filter"
        summary = common.summarize_usage(records)
        self.assertEqual(summary["calls"], 3)
        self.assertEqual(summary["calls_without_usage"], 1)
        self.assertEqual(summary["input_tokens"], 1050)
        self.assertEqual(summary["output_tokens"], 8)
        self.assertEqual(summary["thinking_tokens"], 20)
        self.assertEqual(summary["input_tokens_by_modality"], {"VIDEO": 900, "TEXT": 100})
        self.assertEqual(summary["calls_by_op"], {"map": 1, "filter": 1, "unknown": 1})
        self.assertEqual(summary["input_tokens_by_op"], {"map": 1000, "filter": 50})

    def test_collector_keeps_records_from_many_threads(self) -> None:
        records, sink = common.usage_collector()
        threads = [threading.Thread(target=lambda i=i: [sink({"i": i, "k": k}) for k in range(100)]) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(records), 800)

    def test_failed_usage_is_appended_with_its_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "_failed_usage.jsonl"
            common.append_failed_usage(path, {"film": "a"}, [{"usage": {"prompt_token_count": 7}}])
            common.append_failed_usage(path, {"film": "b"}, [{"usage": None}])
            lines = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(lines, [
            {"film": "a", "usage": {"prompt_token_count": 7}},
            {"film": "b", "usage": None},
        ])

    def test_missing_api_key_stops_the_run(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(SystemExit, "GEMINI_API_KEY"):
            common.require_gemini_key()
        with mock.patch.dict("os.environ", {"GOOGLE_API_KEY": "x"}, clear=True):
            common.require_gemini_key()


class DetectorDeviceTests(unittest.TestCase):
    def test_cpu_is_refused_unless_allowed(self) -> None:
        with mock.patch("mmds.execution.ops.detect._get_device", return_value="cpu"):
            with self.assertRaisesRegex(SystemExit, "refusing to record CPU time"):
                common.detector_device(allow_cpu=False)
            self.assertEqual(common.detector_device(allow_cpu=True), ("cpu", None))

    def test_mps_is_accepted_without_a_gpu_name(self) -> None:
        with mock.patch("mmds.execution.ops.detect._get_device", return_value="mps"):
            self.assertEqual(common.detector_device(allow_cpu=False), ("mps", None))


class WarmUpTests(unittest.TestCase):
    ROW = {"film": "a", "video": {"type": "VideoView", "path": "a.mp4", "start": 30.0, "end": 90.0}}

    def test_runs_one_second_and_records_the_time(self) -> None:
        seen: list[dict] = []

        def run_plan(path: str) -> None:
            seen.extend(json.loads(line) for line in Path(path).read_text().splitlines())

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            common.warm_up_detector(run_plan, self.ROW, out, "cuda", "TITAN Xp")
            record = json.loads((out / "_warmup.json").read_text())
            leftovers = sorted(p.name for p in out.iterdir())
        self.assertEqual(seen[0]["video"], {"type": "VideoView", "path": "a.mp4", "start": 0.0, "end": 1.0})
        self.assertEqual((record["device"], record["gpu"]), ("cuda", "TITAN Xp"))
        self.assertGreaterEqual(record["seconds_including_model_load"], 0.0)
        self.assertEqual(leftovers, ["_warmup.json"])

    def test_failed_warm_up_leaves_no_input_file(self) -> None:
        def run_plan(path: str) -> None:
            raise RuntimeError("model failed to load")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaises(RuntimeError):
                common.warm_up_detector(run_plan, self.ROW, out, "cpu", None)
            self.assertEqual(list(out.iterdir()), [])


class IntervalTests(unittest.TestCase):
    def test_merge_overlapping_touching_and_nested(self) -> None:
        self.assertEqual(common.merge_intervals([(5, 6), (0, 2), (1, 3), (3, 4), (0.5, 1)]), [(0, 4), (5, 6)])

    def test_merge_gap_is_inclusive(self) -> None:
        self.assertEqual(common.merge_intervals([(0, 1), (2, 3)], gap=1.0), [(0, 3)])
        self.assertEqual(common.merge_intervals([(0, 1), (2.001, 3)], gap=1.0), [(0, 1), (2.001, 3)])

    def test_merge_empty_and_invalid(self) -> None:
        self.assertEqual(common.merge_intervals([]), [])
        for bad in ([(2, 1)], [(math.nan, 1)], [(0, math.inf)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                common.merge_intervals(bad)
        with self.assertRaises(ValueError):
            common.merge_intervals([(0, 1)], gap=-1)

    def test_total_length_counts_overlap_once(self) -> None:
        self.assertAlmostEqual(common.total_length([(0, 2), (1, 3), (10, 11)]), 4.0)
        self.assertEqual(common.total_length([]), 0)

    def test_intersect_length(self) -> None:
        self.assertEqual(common.intersect_length([(0, 1)], [(2, 3)]), 0)
        self.assertAlmostEqual(common.intersect_length([(0, 2)], [(1, 3)]), 1.0)
        self.assertAlmostEqual(common.intersect_length([(0, 10)], [(1, 2), (4, 6), (9, 12)]), 4.0)
        self.assertAlmostEqual(common.intersect_length([(1, 2), (4, 6)], [(0, 10)]), 3.0)
        self.assertEqual(common.intersect_length([], [(0, 1)]), 0)

    def test_subtract(self) -> None:
        self.assertEqual(common.subtract_intervals([(0, 10)], [(2, 3), (5, 6)]), [(0, 2), (3, 5), (6, 10)])
        self.assertEqual(common.subtract_intervals([(0, 10)], [(-1, 4)]), [(4, 10)])
        self.assertEqual(common.subtract_intervals([(0, 10)], [(8, 12)]), [(0, 8)])
        self.assertEqual(common.subtract_intervals([(0, 10)], [(0, 10)]), [])
        self.assertEqual(common.subtract_intervals([(0, 1), (5, 6)], [(1, 5)]), [(0, 1), (5, 6)])
        self.assertEqual(common.subtract_intervals([(0, 1)], []), [(0, 1)])

    def test_clamp_drops_reversed_empty_outside_and_nan(self) -> None:
        kept, dropped = common.clamp_intervals(
            [(-3, 2), (5, 99), (4, 3), (7, 7), (50, 60), (math.nan, 1), (8, 9)], duration=10
        )
        self.assertEqual(kept, [(0.0, 2), (5, 10), (8, 9)])
        self.assertEqual(dropped, 4)

    def test_tiou(self) -> None:
        self.assertEqual(common.tiou((0, 2), (0, 2)), 1.0)
        self.assertEqual(common.tiou((0, 1), (2, 3)), 0.0)
        self.assertAlmostEqual(common.tiou((0, 2), (1, 3)), 1 / 3)
        self.assertEqual(common.tiou((1, 1), (1, 1)), 0.0)

    def test_f1(self) -> None:
        self.assertAlmostEqual(common.f1_score(0.5, 1.0), 2 / 3)
        self.assertEqual(common.f1_score(None, 0.0), 0.0)
        self.assertEqual(common.f1_score(0.0, None), 0.0)
        self.assertIsNone(common.f1_score(None, None))
        self.assertIsNone(common.f1_score(None, 0.5))


class SecondsScoresTests(unittest.TestCase):
    def test_hand_computed_example_with_ignored_time(self) -> None:
        # Film 1: labeled 0-10, prediction 5-15, 8-9 ignored on both sides.
        # Film 2: no bear and no prediction, so it adds nothing.
        result = common.seconds_scores([
            ([(5, 15)], [(0, 10)], [(8, 9)]),
            ([], [], []),
        ])
        self.assertAlmostEqual(result["overlap_seconds"], 4.0)
        self.assertAlmostEqual(result["predicted_seconds"], 9.0)
        self.assertAlmostEqual(result["labeled_seconds"], 9.0)
        self.assertAlmostEqual(result["precision"], 4 / 9)
        self.assertAlmostEqual(result["recall"], 4 / 9)

    def test_false_positive_on_a_film_without_the_animal_counts(self) -> None:
        result = common.seconds_scores([([(0, 5)], [(0, 5)], []), ([(0, 5)], [], [])])
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 1.0)

    def test_no_predictions_scores_zero_f1(self) -> None:
        result = common.seconds_scores([([], [(0, 5)], [])])
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)

    def test_nothing_to_score(self) -> None:
        result = common.seconds_scores([([], [], [])])
        self.assertIsNone(result["f1"])

    def test_prediction_only_in_ignored_time_is_not_a_false_positive(self) -> None:
        result = common.seconds_scores([([(8, 9)], [(0, 5)], [(8, 9)])])
        self.assertEqual(result["predicted_seconds"], 0)


class IntervalScoresTests(unittest.TestCase):
    def test_one_prediction_spanning_two_labels_matches_one(self) -> None:
        result = common.interval_scores([([(0, 10)], [(0, 4), (6, 10)], [])], threshold=0.3)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 1))

    def test_greedy_matching_prefers_the_highest_tiou(self) -> None:
        # p0 = (1, 8) clears the threshold with both labels (0.375 with g0,
        # 0.556 with g1); p1 = (0, 3) only with g0 (0.75). Taking pairs by
        # descending tIoU gives p1-g0 then p0-g1: two matches. Giving p0 its
        # first candidate g0 would leave p1 unmatched.
        result = common.interval_scores([([(1, 8), (0, 3)], [(0, 4), (3, 10)], [])], threshold=0.3)
        self.assertEqual(result["tp"], 2)

    def test_threshold_is_inclusive_and_zero_overlap_never_matches(self) -> None:
        # tIoU of (0, 1) and (0, 2) is exactly 0.5.
        self.assertEqual(common.interval_scores([([(0, 1)], [(0, 2)], [])], threshold=0.5)["tp"], 1)
        self.assertEqual(common.interval_scores([([(0, 1)], [(0, 2)], [])], threshold=0.51)["tp"], 0)
        self.assertEqual(common.interval_scores([([(0, 1)], [(5, 6)], [])], threshold=0.0)["tp"], 0)

    def test_predictions_inside_ignored_time_are_dropped(self) -> None:
        result = common.interval_scores([([(8, 9), (0, 5)], [(0, 5)], [(8, 9)])], threshold=0.3)
        self.assertEqual((result["tp"], result["fp"]), (1, 0))

    def test_counts_are_summed_over_films(self) -> None:
        result = common.interval_scores(
            [([(0, 5)], [(0, 5)], []), ([(0, 5)], [], []), ([], [(0, 5)], [])], threshold=0.3
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertAlmostEqual(result["f1"], 0.5)


class EncodeCommandTests(unittest.TestCase):
    def test_full_re_encode_keeps_every_frame_and_matches_the_paper_encoding(self) -> None:
        command = common.encode_command(Path("in.mp4"), Path("out.mp4"), height=360)
        self.assertNotIn("-ss", command)
        self.assertEqual(command[command.index("-vf") + 1], "scale=-2:360")
        self.assertEqual(command[command.index("-fps_mode") + 1], "passthrough")
        self.assertEqual(command[command.index("-crf") + 1], "18")
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertEqual(command[-1], "out.mp4")

    def test_cut_seeks_before_the_input_and_sets_a_duration(self) -> None:
        command = common.encode_command(Path("in.mp4"), Path("out.mp4"), start=60.0, end=180.5)
        self.assertLess(command.index("-ss"), command.index("-i"))
        self.assertEqual(command[command.index("-ss") + 1], "60.000")
        self.assertEqual(command[command.index("-t") + 1], "120.500")
        self.assertNotIn("-vf", command)

    def test_invalid_arguments_raise(self) -> None:
        for kwargs in ({"start": 1.0}, {"start": 5.0, "end": 5.0}, {"start": -1.0, "end": 2.0}, {"height": 361}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                common.encode_command(Path("in.mp4"), Path("out.mp4"), **kwargs)

    def test_parse_probe(self) -> None:
        probe = {
            "streams": [{"avg_frame_rate": "30000/1001", "nb_read_packets": "900", "width": 640, "height": 360}],
            "format": {"duration": "30.03"},
        }
        self.assertEqual(
            common.parse_probe(probe),
            {"frames": 900, "fps": 30000 / 1001, "width": 640, "height": 360, "duration": 30.03},
        )
        with self.assertRaises(ValueError):
            common.parse_probe({"streams": [], "format": {}})
        with self.assertRaises(ValueError):
            common.parse_probe({"streams": [{**probe["streams"][0], "avg_frame_rate": "0/0"}], "format": probe["format"]})


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class EncodeVideoTests(unittest.TestCase):
    def test_re_encode_keeps_frames_and_a_cut_has_the_requested_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.mp4"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=96x64:rate=25:duration=2",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-shortest", "-pix_fmt", "yuv420p", str(source)],
                check=True,
            )
            full = Path(tmp) / "full.mp4"
            common.encode_video(source, full, height=32)
            cut = Path(tmp) / "cut.mp4"
            common.encode_video(source, cut, start=0.4, end=1.2)
            full_probe = common.probe_video(full)
            cut_probe = common.probe_video(cut)
            leftovers = sorted(p.name for p in Path(tmp).iterdir())
        self.assertEqual(full_probe["frames"], 50)
        self.assertEqual(full_probe["height"], 32)
        self.assertEqual(cut_probe["frames"], 20)
        self.assertEqual(leftovers, ["cut.mp4", "full.mp4", "source.mp4"])


class SummaryTests(unittest.TestCase):
    def test_mean_sd(self) -> None:
        self.assertEqual(common.mean_sd([1.0, 3.0]), {"mean": 2.0, "sd": math.sqrt(2.0)})
        self.assertEqual(common.mean_sd([5.0]), {"mean": 5.0, "sd": 0.0})
        self.assertEqual(common.mean_sd([None, 4.0]), {"mean": 4.0, "sd": 0.0})
        self.assertEqual(common.mean_sd([None]), {"mean": None, "sd": None})

    def test_aggregate_walks_nested_results(self) -> None:
        runs = [{"f1": 0.5, "scores": {"p": 1.0, "r": None}}, {"f1": 0.7, "scores": {"p": 0.0, "r": None}}]
        result = common.aggregate(runs)
        self.assertAlmostEqual(result["f1"]["mean"], 0.6)
        self.assertEqual(result["scores"]["p"]["mean"], 0.5)
        self.assertEqual(result["scores"]["r"], {"mean": None, "sd": None})
        with self.assertRaises(ValueError):
            common.aggregate([])

    def test_fmt(self) -> None:
        self.assertEqual(common.fmt({"mean": 0.5, "sd": 0.25}), "0.500 ± 0.250")
        self.assertEqual(common.fmt({"mean": None, "sd": None}), "n/a")
        self.assertEqual(common.fmt(12.345, 1), "12.3")
        self.assertEqual(common.fmt(None), "n/a")


class RunInfoTests(unittest.TestCase):
    def test_records_versions_weights_and_one_line_per_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "model.pt"
            weights.write_bytes(b"weights")
            folder = Path(tmp) / "results"
            common.record_run_info(folder, weights=[weights, Path(tmp) / "missing.pt"])
            common.record_run_info(folder)
            lines = [json.loads(line) for line in (folder / "_run_info.jsonl").read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        info = lines[0]
        self.assertEqual(set(info["weights_sha256"]), {"model.pt"})
        self.assertEqual(len(info["weights_sha256"]["model.pt"]), 64)
        for key in ("time", "git_commit", "git_dirty", "python", "packages", "ffmpeg", "machine"):
            self.assertIn(key, info)

    def test_unknown_git_state_is_not_reported_as_clean(self) -> None:
        with mock.patch.object(common, "_command_output", return_value=None):
            info = common.run_info()
        self.assertEqual((info["git_commit"], info["git_dirty"]), (None, None))


class CostTests(unittest.TestCase):
    MODEL = "gemini-3.1-flash-lite"

    def _record(self, **usage):
        return {"model": self.MODEL, "usage": usage}

    def test_input_is_priced_per_modality_and_output_includes_thinking(self) -> None:
        record = self._record(
            prompt_token_count=3_000_000,
            prompt_tokens_details=[{"modality": "VIDEO", "token_count": 2_000_000},
                                   {"modality": "MediaModality.AUDIO", "token_count": 1_000_000}],
            candidates_token_count=600_000, thoughts_token_count=400_000,
        )
        # 2M video at $0.25 + 1M audio at $0.50 + 1M output (thinking included) at $1.50.
        self.assertAlmostEqual(common.gemini_dollars(record), 0.50 + 0.50 + 1.50)

    def test_cached_input_is_priced_at_the_cache_rate(self) -> None:
        record = self._record(
            prompt_token_count=1_000_000, prompt_tokens_details=[{"modality": "VIDEO", "token_count": 1_000_000}],
            cached_content_token_count=400_000, cache_tokens_details=[{"modality": "VIDEO", "token_count": 400_000}],
        )
        self.assertAlmostEqual(common.gemini_dollars(record), 0.6 * 0.25 + 0.4 * 0.025)

    def test_unpriceable_calls_raise(self) -> None:
        details = [{"modality": "TEXT", "token_count": 10}]
        bad = {
            "No price for Gemini model": {"model": "gemini-9", "usage": {"prompt_token_count": 0}},
            "no usage": {"model": self.MODEL, "usage": None},
            "do not add up": self._record(prompt_token_count=11, prompt_tokens_details=details),
            "No gemini-3.1-flash-lite price for SMELL": self._record(
                prompt_token_count=10, prompt_tokens_details=[{"modality": "SMELL", "token_count": 10}]),
            "cached TEXT tokens, but only 10": self._record(
                prompt_token_count=10, prompt_tokens_details=details,
                cached_content_token_count=20, cache_tokens_details=[{"modality": "TEXT", "token_count": 20}]),
        }
        for message, record in bad.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                common.gemini_dollars(record)

    def test_usage_dollars_is_unknown_if_any_call_lacks_usage(self) -> None:
        priced = self._record(prompt_token_count=4_000_000,
                              prompt_tokens_details=[{"modality": "TEXT", "token_count": 4_000_000}])
        self.assertAlmostEqual(common.usage_dollars([priced, priced]), 2.0)
        self.assertEqual(common.usage_dollars([]), 0)
        self.assertIsNone(common.usage_dollars([priced, {"model": self.MODEL, "usage": None}]))

    def test_gpu_time_plan_totals_and_break_even(self) -> None:
        with mock.patch.object(common, "GPU_DOLLARS_PER_HOUR", 0.5):
            self.assertAlmostEqual(common.gpu_dollars(7200), 1.0)
            self.assertEqual(common.plan_dollars(0.25, 7200), {"gemini_dollars": 0.25, "gpu_dollars": 1.0, "dollars": 1.25})
            self.assertIsNone(common.plan_dollars(None, 7200)["dollars"])
        # $1.40 reference, $0.40 of Gemini and 2 GPU-hours: equal cost at $0.50 per hour.
        self.assertAlmostEqual(common.break_even_gpu_rate(1.40, 0.40, 7200), 0.5)
        self.assertIsNone(common.break_even_gpu_rate(1.40, 0.40, 0))
        self.assertLess(common.break_even_gpu_rate(1.0, 2.0, 3600), 0)

    def test_cost_comparison_needs_a_known_reference_and_known_plan_costs(self) -> None:
        plans = {"filter": (0.40, 7200.0)}
        no_reference = common.cost_comparison("full", None, plans)
        self.assertEqual(no_reference["break_even_gpu_dollars_per_hour"], {})
        self.assertEqual(common.format_cost_comparison(no_reference), [])
        compared = common.cost_comparison("full", 1.40, {**plans, "unknown": (None, 3600.0)})
        self.assertEqual(set(compared["break_even_gpu_dollars_per_hour"]), {"filter"})
        lines = common.format_cost_comparison(compared)
        self.assertIn("- filter: $0.50/h", lines)
        self.assertIn("costs as much as full", "\n".join(lines))


class LoadAverageTests(unittest.TestCase):
    def test_three_rounded_averages(self) -> None:
        with mock.patch.object(common.os, "getloadavg", return_value=(0.123, 1.0, 12.3456)):
            self.assertEqual(common.load_average(), [0.12, 1.0, 12.35])

    def test_none_where_the_os_has_no_load_average(self) -> None:
        with mock.patch.object(common.os, "getloadavg", side_effect=OSError):
            self.assertIsNone(common.load_average())
        with mock.patch.object(common, "os") as fake_os:
            del fake_os.getloadavg  # e.g. Windows
            self.assertIsNone(common.load_average())


class WriteJsonlTests(unittest.TestCase):
    def test_one_line_per_row_and_the_path_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.jsonl"
            self.assertEqual(common.write_jsonl(path, [{"a": 1}, {"b": [2]}]), path)
            self.assertEqual(path.read_text(encoding="utf-8"), '{"a": 1}\n{"b": [2]}\n')
            common.write_jsonl(path, [])
            self.assertEqual(path.read_text(encoding="utf-8"), "")


class AtomicWriteTests(unittest.TestCase):
    def test_round_trip_and_no_temporary_file_left(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            common.write_json_atomic(path, {"a": [1, 2]})
            common.write_json_atomic(path, {"a": [3]})
            self.assertEqual(common.read_json(path), {"a": [3]})
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["out.json"])


if __name__ == "__main__":
    unittest.main()
