from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.query_types import wld_yoloe as wy  # noqa: E402
from scripts.query_types.wld import DETECTOR_VOCABULARY, QUERY_CLASSES, Pair, load_pairs  # noqa: E402
from tests.wld_fixtures import DURATION, FPS, fake_yoloe, has_ffmpeg, make_fake_wld, write_scans  # noqa: E402


def _pair(film: str = "f", animal: str = "bear", truth: str = "present", **kwargs) -> Pair:
    values = {"film_id": film, "animal": animal, "path": f"data/wld/videos_360p/{film}.mp4", "duration": 300.0,
              "fps": 25.0, "frames": 7500, "truth": truth, "truth_any_frame": truth != "absent",
              "first_seconds": None if truth == "absent" else 10.0}
    values.update(kwargs)
    return Pair(**values)


class VariantTests(unittest.TestCase):
    def test_the_four_variants(self) -> None:
        self.assertEqual(list(wy.VARIANTS), ["small", "small_vocab", "large", "large_vocab"])
        large = wy.VARIANTS["large"]
        self.assertEqual((large.model, large.native, large.vocabulary), ("yoloe-11l-seg.pt", True, False))
        self.assertEqual((wy.MAIN_VARIANT, wy.FILTER_VARIANT), ("small", "large"))

    def test_native_variants_read_the_original_download(self) -> None:
        pair = _pair("A")
        self.assertEqual(wy.video_path(pair, wy.VARIANTS["small"]), "data/wld/videos_360p/A.mp4")
        self.assertEqual(wy.video_path(pair, wy.VARIANTS["large"]), "data/wld/videos/A.mp4")

    def test_plans_prompt_one_name_or_the_whole_vocabulary(self) -> None:
        one = wy.detect_plan("p.jsonl", wy.VARIANTS["large"], "bear", 5)
        self.assertEqual((one.spec.classes, one.spec.model, one.spec.conf, one.spec.frame_stride), (("bear",), "yoloe-11l-seg.pt", 0.1, 5))
        vocab = wy.detect_plan("p.jsonl", wy.VARIANTS["small_vocab"], "bear", 5)
        self.assertEqual(vocab.spec.classes, DETECTOR_VOCABULARY)
        early = wy.detect_plan("p.jsonl", wy.VARIANTS["small"], "bear", 5, conf=0.25, stop_after_n=1)
        self.assertEqual((early.spec.conf, early.spec.stop_after_n), (0.25, 1))

    def test_vocabulary_is_coco_animals_plus_query_animals(self) -> None:
        self.assertEqual(len(DETECTOR_VOCABULARY), len(set(DETECTOR_VOCABULARY)))
        self.assertTrue(set(QUERY_CLASSES) <= set(DETECTOR_VOCABULARY))

    def test_scan_units_are_pairs_or_films(self) -> None:
        pairs = [_pair("A", "bear"), _pair("A", "dog"), _pair("B", "bear")]
        self.assertEqual([name for name, _ in wy.scan_units(wy.VARIANTS["small"], pairs)], ["A__bear", "A__dog", "B__bear"])
        films = wy.scan_units(wy.VARIANTS["large_vocab"], pairs)
        self.assertEqual([(name, [p.animal for p in covered]) for name, covered in films], [("A", ["bear", "dog"]), ("B", ["bear"])])


class RuleTests(unittest.TestCase):
    def test_one_frame_rule_fires_at_the_first_hit(self) -> None:
        self.assertEqual(wy.rule_fires({50: 0.3, 10: 0.9}, stride=5, conf=0.25, persistence=1), 2)

    def test_confidence_threshold_is_inclusive(self) -> None:
        self.assertEqual(wy.rule_fires({5: 0.25}, stride=5, conf=0.25, persistence=1), 1)
        self.assertIsNone(wy.rule_fires({5: 0.2499}, stride=5, conf=0.25, persistence=1))

    def test_persistence_needs_consecutive_sampled_frames(self) -> None:
        max_conf = {0: 0.9, 10: 0.9, 15: 0.9, 20: 0.9}  # sampled indices 0, 2, 3, 4
        self.assertEqual(wy.rule_fires(max_conf, stride=5, conf=0.5, persistence=2), 3)
        self.assertEqual(wy.rule_fires(max_conf, stride=5, conf=0.5, persistence=3), 4)
        self.assertIsNone(wy.rule_fires(max_conf, stride=5, conf=0.5, persistence=4))

    def test_a_low_frame_breaks_a_run(self) -> None:
        self.assertEqual(wy.rule_fires({0: 0.9, 5: 0.1, 10: 0.9, 15: 0.9}, stride=5, conf=0.5, persistence=2), 3)

    def test_no_boxes_never_fires_and_bad_input_raises(self) -> None:
        self.assertIsNone(wy.rule_fires({}, stride=5, conf=0.1, persistence=1))
        with self.assertRaisesRegex(ValueError, "not a sampled frame"):
            wy.rule_fires({7: 0.9}, stride=5, conf=0.1, persistence=1)
        with self.assertRaises(ValueError):
            wy.rule_fires({}, stride=5, conf=0.1, persistence=0)

    def test_frame_max_conf_keeps_the_best_box_per_frame_and_can_keep_one_animal(self) -> None:
        detections = [
            {"type": "bear", "bboxes": [{"frame_idx": 5, "confidence": 0.2}, {"frame_idx": 5, "confidence": 0.7}]},
            {"type": "dog", "bboxes": [{"frame_idx": 5, "confidence": 0.9}, {"frame_idx": 10, "confidence": 0.3}]},
        ]
        self.assertEqual(wy.frame_max_conf(detections), {5: 0.9, 10: 0.3})
        self.assertEqual(wy.frame_max_conf(detections, "bear"), {5: 0.7})

    def test_sampled_count(self) -> None:
        self.assertEqual((wy.sampled_count(10, 5), wy.sampled_count(11, 5), wy.sampled_count(1, 5)), (2, 3, 1))


class OutcomeTests(unittest.TestCase):
    SCAN = {"stride": 5, "sampled_frames": 100, "seconds": 50.0, "frame_max_conf": {"95": 0.4, "300": 0.9}}

    def test_early_stop_time_is_proportional_to_frames_scanned(self) -> None:
        outcome = wy.yoloe_outcomes(self.SCAN, conf=0.25, persistence=1)
        self.assertTrue(outcome["yoloe_full"]["answer"])
        self.assertEqual(outcome["yoloe_full"]["seconds"], 50.0)
        # Fires at sampled index 19 (frame 95): 20 of 100 frames scanned.
        self.assertAlmostEqual(outcome["yoloe_early"]["seconds"], 10.0)
        self.assertAlmostEqual(outcome["yoloe_early"]["fraction_scanned"], 0.2)

    def test_full_and_early_always_agree_and_no_fire_is_a_full_scan(self) -> None:
        for conf in (0.1, 0.5, 0.95):
            with self.subTest(conf=conf):
                outcome = wy.yoloe_outcomes(self.SCAN, conf=conf, persistence=1)
                self.assertEqual(outcome["yoloe_full"]["answer"], outcome["yoloe_early"]["answer"])
        silent = wy.yoloe_outcomes(self.SCAN, conf=0.95, persistence=1)["yoloe_early"]
        self.assertEqual((silent["answer"], silent["seconds"]), (False, 50.0))


class WindowTests(unittest.TestCase):
    def test_hits_cover_their_stride_and_merge_short_gaps(self) -> None:
        # Hits at frames 0 and 5 (stride 5, 5 fps) cover 0-2 s; frame 20 is 2 s later, past the 1 s gap.
        self.assertEqual(wy.hits_to_intervals([0, 5, 20], fps=5.0, stride=5), [(0.0, 2.0), (4.0, 5.0)])

    def test_scan_hits_use_the_main_confidence(self) -> None:
        scan = {"stride": 5, "frame_max_conf": {"0": 0.3, "5": 0.2, "50": 0.9}}
        self.assertEqual(wy.scan_hits(scan, fps=5.0), [(0.0, 1.0), (10.0, 11.0)])

    def test_pad_windows_pads_clips_and_merges(self) -> None:
        self.assertEqual(wy.pad_windows([(5, 8), (25, 30), (300, 310)], duration=305), [(0.0, 40), (290, 305)])
        self.assertEqual(wy.pad_windows([], duration=100), [])

    def test_ranked_windows_put_the_most_confident_first(self) -> None:
        # 25 fps, stride 5: hits at 100 s (0.3), 500 s (0.9) and 1000 s (0.5); frame 5000 is below 0.25.
        scan = {"stride": 5, "frame_max_conf": {"2500": 0.3, "5000": 0.2, "12500": 0.9, "25000": 0.5}}
        ranked = wy.ranked_windows(scan, _pair("A", duration=1200.0))
        self.assertEqual([conf for _, conf in ranked], [0.9, 0.5, 0.3])
        self.assertEqual(ranked[0][0], (490.0, 510.2))


class ValidationPickTests(unittest.TestCase):
    def test_picks_early_mid_late_and_the_two_longest_silent_films(self) -> None:
        pairs = [_pair(f"p{i}") for i in range(11)] + [
            _pair("short", truth="absent", duration=100.0),
            _pair("long", truth="absent", duration=900.0),
            _pair("mid", truth="absent", duration=500.0),
        ]
        scans = {
            p.pair_id: {"stride": 5, "sampled_frames": 100, "seconds": 1.0,
                        "frame_max_conf": {str(5 * (10 * i + 1)): 0.9} if p.truth == "present" else {}}
            for i, p in enumerate(pairs)
        }
        self.assertEqual([p.film_id for p in wy.pick_validation_pairs(pairs, scans)], ["p1", "p5", "p9", "long", "mid"])

    def test_too_few_pairs_raise(self) -> None:
        pairs = [_pair("a"), _pair("b", truth="absent")]
        scans = {p.pair_id: {"stride": 5, "sampled_frames": 10, "seconds": 1.0, "frame_max_conf": {}} for p in pairs}
        with self.assertRaisesRegex(ValueError, "Need 3 detected"):
            wy.pick_validation_pairs(pairs, scans)


@unittest.skipUnless(has_ffmpeg(), "needs ffmpeg")
class ScanTests(unittest.TestCase):
    """Scans on two fake films with a fake YOLOE that sees only film A's bear."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.out = self.root / "out"
        make_fake_wld(self.root)
        self.pairs = load_pairs(self.root)
        self.passes: list[tuple[str, tuple[str, ...]]] = []
        detect = fake_yoloe()

        def execute(plan, executor=None):
            row = json.loads(Path(plan.source.input_path).read_text().splitlines()[0])
            self.passes.append((row["video"]["path"], plan.spec.classes))
            return detect(plan, executor)

        self._patches = [mock.patch.object(wy, "execute", side_effect=execute),
                         mock.patch.object(wy, "detector_device", return_value=("cuda", "TITAN Xp")),
                         mock.patch.object(wy, "load_average", return_value=[0.5, 0.4, 0.3])]
        for patch in self._patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()
        self._tmp.cleanup()

    def test_one_name_scan_covers_every_pair(self) -> None:
        wy.run_scan("small", self.pairs, allow_cpu=False, out_root=self.out)
        files = sorted(p.stem for p in wy.scan_dir("small", self.out).glob("*.json"))
        self.assertEqual(files, sorted(p.file_stem for p in self.pairs))
        scans = wy.load_scans("small", self.pairs, self.out)
        self.assertEqual(len(scans), len(self.pairs))
        self.assertTrue(wy.yoloe_outcomes(scans["A|bear"], *wy.MAIN_RULE)["yoloe_full"]["answer"])
        self.assertFalse(wy.yoloe_outcomes(scans["A|dog"], *wy.MAIN_RULE)["yoloe_full"]["answer"])
        self.assertEqual(scans["A|bear"]["gpu"], "TITAN Xp")
        self.assertEqual(scans["A|bear"]["load_average"], [0.5, 0.4, 0.3])
        self.assertTrue((wy.scan_dir("small", self.out) / "_run_info.jsonl").exists())
        # One warm-up, kept in scratch, not next to the results.
        self.assertEqual(len(list((self.out / "scratch").rglob("_warmup.json"))), 1)
        self.assertEqual(len(self.passes), len(self.pairs) + 1)

    def test_vocabulary_scan_runs_once_per_film_on_the_original_video(self) -> None:
        wy.run_scan("large_vocab", self.pairs, allow_cpu=False, out_root=self.out)
        scan_passes = self.passes[1:]  # the first pass is the warm-up
        self.assertEqual(len(scan_passes), 2)
        self.assertTrue(all("/videos/" in path and classes == DETECTOR_VOCABULARY for path, classes in scan_passes))
        scans = wy.load_scans("large_vocab", self.pairs, self.out)
        self.assertEqual(len(scans), len(self.pairs))
        self.assertTrue(scans["A|bear"]["frame_max_conf"])
        self.assertEqual(scans["A|dog"]["frame_max_conf"], {})

    def test_finished_units_are_not_scanned_again(self) -> None:
        wy.run_scan("small", self.pairs, allow_cpu=False, out_root=self.out)
        count = len(self.passes)
        wy.run_scan("small", self.pairs, allow_cpu=False, out_root=self.out)
        self.assertEqual(len(self.passes), count)

    def test_loader_skips_pairs_not_yet_scanned(self) -> None:
        wy.run_scan("small", self.pairs[:2], allow_cpu=False, out_root=self.out)
        self.assertEqual(set(wy.load_scans("small", self.pairs, self.out)), {p.pair_id for p in self.pairs[:2]})

    def test_validation_compares_real_early_stop_with_the_estimate(self) -> None:
        # Enough pairs for the pick: the bear in A is present; give two more present animals.
        def boxes(film, animal, start, end, stride):
            seen = {("A", "bear"): (130, 200), ("A", "deer"): (10, 20), ("B", "tiger"): (250, 260)}.get((film, animal))
            frames = range(int(round(start * FPS)), int(round(end * FPS)), stride)
            return [{"frame_idx": f, "bbox": [0, 0, 1, 1], "confidence": 0.8} for f in frames
                    if seen and seen[0] <= f / FPS < seen[1]]

        pairs = [p if (p.film_id, p.animal) not in {("A", "deer"), ("B", "tiger")}
                 else Pair(**{**p.__dict__, "truth": "present", "truth_any_frame": True}) for p in self.pairs]
        write_scans(self.out, "small", pairs, stride=1, boxes=lambda film, animal: boxes(film, animal, 0, DURATION, 1))
        with mock.patch.object(wy, "execute", side_effect=fake_yoloe(boxes)), \
                mock.patch.object(wy, "detector_stride", return_value=1):
            report = wy.run_validation(pairs, allow_cpu=False, out_root=self.out)
        self.assertEqual(len(report["pairs"]), 5)
        self.assertTrue(all(check["answers_match"] for check in report["pairs"]))
        self.assertTrue(all(check["load_average"] == [0.5, 0.4, 0.3] for check in report["pairs"]))
        self.assertEqual(json.loads((self.out / "yoloe" / "validation.json").read_text())["variant"], "small")


class CliTests(unittest.TestCase):
    def test_scan_needs_a_known_variant(self) -> None:
        with mock.patch.object(wy, "load_pairs", return_value=[]), mock.patch.object(wy, "run_scan") as run_scan:
            wy.main(["scan", "--variant", "large"])
        self.assertEqual(run_scan.call_args.args[0], "large")
        self.assertEqual(run_scan.call_args.kwargs, {"allow_cpu": False})
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            wy.main(["scan", "--variant", "huge"])

    def test_scans_cannot_be_split_across_processes(self) -> None:
        # Side-by-side scans slow each other down, so there is no way to shard them.
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            wy.main(["scan", "--variant", "large", "--shard", "0/6"])


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class GpuScriptTests(unittest.TestCase):
    """run_yoloe_gpu.sh with a fake interpreter that records each call."""

    SCRIPT = ROOT / "scripts" / "query_types" / "run_yoloe_gpu.sh"

    def _run(self, *args: str, fail_on: str | None = None) -> tuple[subprocess.CompletedProcess, list[str]]:
        with tempfile.TemporaryDirectory() as tmp:
            calls = Path(tmp) / "calls.txt"
            fake = Path(tmp) / "fake_python"
            fail = f'case "$*" in *"{fail_on}"*) exit 3;; esac\n' if fail_on else ""
            fake.write_text(f'#!/bin/sh\necho "$CUDA_VISIBLE_DEVICES $*" >> "{calls}"\n{fail}')
            fake.chmod(0o755)
            result = subprocess.run(["bash", str(self.SCRIPT), *args], cwd=tmp, capture_output=True, text=True,
                                    env={**os.environ, "PYTHON": str(fake)})
            return result, calls.read_text().splitlines() if calls.exists() else []

    def test_runs_every_step_in_order_on_the_named_gpu(self) -> None:
        result, calls = self._run("3", "1", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [
            "3 -m scripts.query_types.wld_yoloe scan --variant small",
            "3 -m scripts.query_types.wld_yoloe scan --variant large",
            "3 -m scripts.query_types.wld_yoloe scan --variant small_vocab",
            "3 -m scripts.query_types.wld_yoloe scan --variant large_vocab",
            "3 -m scripts.query_types.wld_yoloe validate",
            "3 -m scripts.query_types.wld_localization run --repeat 1 --stage yoloe_windows",
            "3 -m scripts.query_types.wld_localization run --repeat 2 --stage yoloe_windows",
        ])
        self.assertIn("YOLOE work done.", result.stdout)

    def test_a_failed_step_stops_the_rest(self) -> None:
        result, calls = self._run("0", fail_on="--variant large")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(calls), 2)  # small, then the failing large scan
        self.assertIn("Failed; see logs/yoloe_scan_large.log", result.stderr)
        self.assertNotIn("YOLOE work done.", result.stdout)

    def test_needs_a_gpu(self) -> None:
        result, calls = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage: run_yoloe_gpu.sh GPU", result.stderr)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
