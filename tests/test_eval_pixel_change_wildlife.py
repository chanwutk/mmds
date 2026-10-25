from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mmds.utilities.video import VideoView, open_video  # noqa: E402


def _load_eval_module():
    path = ROOT / "examples" / "eval_pixel_change_wildlife.py"
    spec = importlib.util.spec_from_file_location("eval_pixel_change_wildlife", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()

WIDTH, HEIGHT, FPS = 160, 96, 10.0


def _labels_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "video": "clip.avi",
        "clip_start": 1.0,
        "clip_end": 10.0,
        "events": [
            {"start": 3.0, "end": 4.0, "description": "bear enters"},
            {"start": 6.0, "end": 7.5, "description": "bear leaves"},
        ],
        "static": [
            {"start": 1.0, "end": 3.0, "description": "empty"},
            {"start": 4.0, "end": 6.0, "description": "bear sits"},
        ],
    }
    data.update(overrides)
    return data


def _write_labels(directory: Path, data: Any) -> Path:
    path = directory / "labels.json"
    path.write_text(json.dumps(data))
    return path


def _square_frame(present: bool) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    if present:
        frame[20:60, 40:80] = 255
    return frame


def _write_video(directory: Path, frames: list[np.ndarray]) -> Path:
    path = directory / "clip.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV MJPG VideoWriter is unavailable")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


class _TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmpdir = Path(tmp.name)


class LoadLabelsTests(_TempDirTestCase):
    def load(self, data: Any):
        return EVAL.load_labels(_write_labels(self.tmpdir, data))

    def test_valid_labels(self) -> None:
        labels = self.load(_labels_dict())
        self.assertEqual(labels.video, "clip.avi")
        self.assertEqual(labels.clip_start, 1.0)
        self.assertEqual(labels.clip_end, 10.0)
        self.assertEqual(
            [(e.start, e.end, e.description) for e in labels.events],
            [(3.0, 4.0, "bear enters"), (6.0, 7.5, "bear leaves")],
        )
        self.assertEqual(len(labels.static), 2)

    def test_clip_end_and_static_are_optional(self) -> None:
        data = _labels_dict(clip_end=None)
        del data["static"]
        labels = self.load(data)
        self.assertIsNone(labels.clip_end)
        self.assertEqual(labels.static, ())

    def test_interval_contains_is_half_open(self) -> None:
        interval = EVAL.Interval(3.0, 4.0)
        self.assertTrue(interval.contains(3.0))
        self.assertTrue(interval.contains(3.99))
        self.assertFalse(interval.contains(4.0))
        self.assertFalse(interval.contains(2.99))

    def test_rejects_invalid_files(self) -> None:
        cases = {
            "not an object": [1, 2],
            "missing video": _labels_dict(video=""),
            "bool clip_start": _labels_dict(clip_start=True),
            "negative clip_start": _labels_dict(clip_start=-1),
            "clip_end before start": _labels_dict(clip_end=1.0),
            "no events": _labels_dict(events=[]),
            "events not list": _labels_dict(events={"start": 3, "end": 4}),
            "event not object": _labels_dict(events=[[3, 4]]),
            "empty interval": _labels_dict(events=[{"start": 3.0, "end": 3.0}]),
            "reversed interval": _labels_dict(events=[{"start": 5.0, "end": 3.0}]),
            "string bound": _labels_dict(events=[{"start": "3", "end": 4.0}]),
            "missing bound": _labels_dict(events=[{"start": 3.0}]),
            "nan bound": _labels_dict(events=[{"start": float("nan"), "end": 4.0}]),
            "bad description": _labels_dict(
                events=[{"start": 3.0, "end": 4.0, "description": 7}]
            ),
            "unsorted": _labels_dict(
                events=[{"start": 6.0, "end": 7.0}, {"start": 3.0, "end": 4.0}],
                static=[],
            ),
            "overlapping events": _labels_dict(
                events=[{"start": 3.0, "end": 5.0}, {"start": 4.0, "end": 6.0}],
                static=[],
            ),
            "event overlaps static": _labels_dict(
                static=[{"start": 2.0, "end": 3.5}]
            ),
            "outside clip end": _labels_dict(
                events=[{"start": 9.0, "end": 11.0}], static=[]
            ),
            "before clip start": _labels_dict(
                events=[{"start": 0.5, "end": 2.0}], static=[]
            ),
        }
        for name, data in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    self.load(data)

    def test_adjacent_intervals_are_allowed(self) -> None:
        labels = self.load(
            _labels_dict(
                events=[{"start": 3.0, "end": 4.0}, {"start": 4.0, "end": 5.0}],
                static=[{"start": 5.0, "end": 6.0}],
            )
        )
        self.assertEqual(len(labels.events), 2)

    def test_validate_against_actual_clip(self) -> None:
        labels = self.load(_labels_dict(clip_end=None))
        EVAL.validate_labels_against_clip(labels, 1.0, 10.0)
        with self.assertRaises(ValueError):
            EVAL.validate_labels_against_clip(labels, 1.0, 7.0)
        with self.assertRaises(ValueError):
            EVAL.validate_labels_against_clip(labels, 3.5, 10.0)

    def test_repository_labels_file_is_valid(self) -> None:
        labels = EVAL.load_labels(EVAL.DEFAULT_LABELS)
        self.assertEqual(labels.video, EVAL.VIDEO_URL)
        self.assertEqual(labels.clip_start, EVAL.CLIP_START)
        self.assertGreater(len(labels.events), 0)


class ScoreSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = EVAL.Labels(
            video="v",
            clip_start=0.0,
            clip_end=None,
            events=(
                EVAL.Interval(3.0, 4.0, "enter"),
                EVAL.Interval(6.0, 7.0, "leave"),
            ),
            static=(EVAL.Interval(0.0, 3.0), EVAL.Interval(4.0, 6.0)),
        )

    def score(self, selected, sampled_count=100, fps=10.0):
        return EVAL.score_selection(
            selected, sampled_count=sampled_count, fps=fps, labels=self.labels
        )

    def test_all_events_hit(self) -> None:
        result = self.score([0, 32, 35, 61])
        self.assertEqual(result["event_recall"], 1.0)
        self.assertEqual(result["missed_events"], [])
        self.assertEqual(result["latency_seconds"], [0.2, 0.1])
        self.assertEqual(result["selected"], 4)
        self.assertAlmostEqual(result["kept_fraction"], 0.04)

    def test_missed_event_reported_by_description(self) -> None:
        result = self.score([0, 30])
        self.assertEqual(result["event_recall"], 0.5)
        self.assertEqual(result["missed_events"], ["leave"])
        self.assertEqual(result["latency_seconds"], [0.0, None])

    def test_event_end_is_exclusive(self) -> None:
        result = self.score([0, 40, 70])
        self.assertEqual(result["event_recall"], 0.0)

    def test_missed_event_without_description_uses_bounds(self) -> None:
        labels = EVAL.Labels("v", 0.0, None, (EVAL.Interval(3.0, 4.0),), ())
        result = EVAL.score_selection([0], sampled_count=1, fps=10.0, labels=labels)
        self.assertEqual(result["missed_events"], ["3.0-4.0"])

    def test_static_selections_exclude_forced_first(self) -> None:
        result = self.score([0, 5, 10, 45, 50])
        # 0 is the forced first selection; 5, 10 (static 0-3s) and 45, 50
        # (static 4-6s) count.
        self.assertEqual(result["static_selections"], 4)
        self.assertAlmostEqual(result["static_selections_per_minute"], 4 / (5 / 60))

    def test_empty_selection(self) -> None:
        result = self.score([], sampled_count=0)
        self.assertEqual(result["event_recall"], 0.0)
        self.assertEqual(result["kept_fraction"], 0.0)
        self.assertEqual(result["static_selections"], 0)

    def test_no_static_intervals_gives_zero_rate(self) -> None:
        labels = EVAL.Labels("v", 0.0, None, self.labels.events, ())
        result = EVAL.score_selection([0, 5], sampled_count=2, fps=10.0, labels=labels)
        self.assertEqual(result["static_selections_per_minute"], 0.0)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.score([0], fps=0.0)
        with self.assertRaises(ValueError):
            self.score([0], sampled_count=-1)


class BackgroundSubtractionTests(unittest.TestCase):
    @staticmethod
    def grays(values_or_frames):
        return list(enumerate(values_or_frames))

    def test_empty_input(self) -> None:
        self.assertEqual(EVAL.select_by_background_subtraction([], 0.01), [])

    def test_static_scene_selects_first_only(self) -> None:
        frames = [np.full((90, 160), 40, dtype=np.uint8)] * 30
        self.assertEqual(
            EVAL.select_by_background_subtraction(self.grays(frames), 0.01), [0]
        )

    def test_new_object_after_warmup_is_selected(self) -> None:
        background = np.full((90, 160), 40, dtype=np.uint8)
        with_object = background.copy()
        with_object[20:60, 40:100] = 230
        frames = [background] * 30 + [with_object] * 3
        selected = EVAL.select_by_background_subtraction(self.grays(frames), 0.05)
        self.assertEqual(selected[0], 0)
        self.assertIn(30, selected)
        self.assertTrue(all(index == 0 or index >= 30 for index in selected))

    def test_preserves_given_indices(self) -> None:
        background = np.full((90, 160), 40, dtype=np.uint8)
        with_object = background.copy()
        with_object[20:60, 40:100] = 230
        indexed = [(500 + 5 * i, background) for i in range(30)]
        indexed.append((650, with_object))
        selected = EVAL.select_by_background_subtraction(indexed, 0.05)
        self.assertEqual(selected[0], 500)
        self.assertIn(650, selected)

    def test_tiny_noise_removed_by_opening(self) -> None:
        background = np.full((90, 160), 40, dtype=np.uint8)
        speck = background.copy()
        speck[10, 10] = 255
        frames = [background] * 30 + [speck]
        self.assertEqual(
            EVAL.select_by_background_subtraction(self.grays(frames), 0.00001), [0]
        )

    def test_invalid_fraction(self) -> None:
        for fraction in (-0.1, 1.5, True):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    EVAL.select_by_background_subtraction([], fraction)


class _FakeDetect:
    def __init__(self, detections: list[dict[str, Any]]) -> None:
        self.detections = detections
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((args, kwargs))
        return self.detections


def _box(frame_idx: int) -> dict[str, Any]:
    return {"frame_idx": frame_idx, "bbox": [0, 0, 1, 1], "confidence": 0.9}


class SelectByYoloeTests(unittest.TestCase):
    sampled = [100, 105, 110, 115, 120, 125]

    def run_yoloe(self, detections, sampled=None):
        fake = _FakeDetect(detections)
        result = EVAL.select_by_yoloe(
            "video",
            self.sampled if sampled is None else sampled,
            ["bear", "deer"],
            sample_every=5,
            model="m.pt",
            conf=0.3,
            detect_fn=fake,
        )
        return result, fake

    def test_passes_stride_classes_model_and_conf(self) -> None:
        _, fake = self.run_yoloe([])
        self.assertEqual(
            fake.calls,
            [(("video", ["bear", "deer"], "m.pt"), {"frame_stride": 5, "conf": 0.3})],
        )

    def test_no_detections_selects_first_only(self) -> None:
        result, _ = self.run_yoloe([])
        self.assertEqual(result, {"presence": [100], "count_change": [100]})

    def test_presence_and_count_change(self) -> None:
        detections = [
            {"type": "bear", "bboxes": [_box(110), _box(115), _box(120)]},
            {"type": "deer", "bboxes": [_box(120)]},
        ]
        result, _ = self.run_yoloe(detections)
        self.assertEqual(result["presence"], [100, 110, 115, 120])
        # 110: 0->1 bear; 115: unchanged; 120: deer added; 125: back to none.
        self.assertEqual(result["count_change"], [100, 110, 120, 125])

    def test_two_boxes_same_class_count_as_two(self) -> None:
        detections = [{"type": "bear", "bboxes": [_box(105), _box(110), _box(110)]}]
        result, _ = self.run_yoloe(detections)
        self.assertEqual(result["count_change"], [100, 105, 110, 115])

    def test_detection_at_unsampled_frame_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.run_yoloe([{"type": "bear", "bboxes": [_box(101)]}])

    def test_empty_sample_skips_detection(self) -> None:
        result, fake = self.run_yoloe([], sampled=[])
        self.assertEqual(result, {"presence": [], "count_change": []})
        self.assertEqual(fake.calls, [])


class MedianTimeTests(unittest.TestCase):
    def test_calls_fn_repeats_times_and_returns_last_result(self) -> None:
        calls = []
        seconds, result = EVAL.median_time(lambda: calls.append(1) or len(calls), 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result, 3)
        self.assertGreaterEqual(seconds, 0.0)

    def test_invalid_repeats(self) -> None:
        for repeats in (0, -1, True, 1.5):
            with self.subTest(repeats=repeats):
                with self.assertRaises(ValueError):
                    EVAL.median_time(lambda: None, repeats)


class VideoHarnessTests(_TempDirTestCase):
    """End-to-end harness checks on a synthetic clip with known changes."""

    def setUp(self) -> None:
        super().setUp()
        # 10 fps, 100 frames: square present during frames 40-69 (4.0s-7.0s).
        frames = [_square_frame(40 <= i < 70) for i in range(100)]
        self.video_path = _write_video(self.tmpdir, frames)
        self.labels = EVAL.Labels(
            video=str(self.video_path),
            clip_start=1.0,
            clip_end=None,
            events=(
                EVAL.Interval(4.0, 4.5, "square appears"),
                EVAL.Interval(7.0, 7.5, "square disappears"),
            ),
            static=(EVAL.Interval(1.0, 4.0), EVAL.Interval(4.5, 7.0)),
        )

    def test_open_clip_uses_video_end_when_unset(self) -> None:
        view = EVAL.open_clip(str(self.video_path), 1.0, None)
        self.assertEqual((view.start_frame, view.end_frame), (10, 100))

    def test_open_clip_rejects_empty_clip(self) -> None:
        with self.assertRaises(ValueError):
            EVAL.open_clip(str(self.video_path), 50.0, None)

    def test_open_clip_rejects_directory(self) -> None:
        with self.assertRaises(ValueError):
            EVAL.open_clip(str(self.tmpdir), 0.0, None)

    def test_decode_only_counts_view_frames(self) -> None:
        view = EVAL.open_clip(str(self.video_path), 1.0, 3.0)
        self.assertEqual(EVAL.decode_only(view), 20)

    def test_evaluate_without_yoloe(self) -> None:
        view = EVAL.open_clip(str(self.video_path), 1.0, None)
        report = EVAL.evaluate(
            view, self.labels, sample_every=5, yoloe_classes=None, repeats=1
        )
        self.assertEqual(report["decoded_frames"], 90)
        self.assertEqual(report["sampled_frames"], 18)
        methods = [run["method"] for run in report["runs"]]
        self.assertEqual(methods.count("pixel_change"), 16)
        metrics = Counter(
            run["params"]["metric"]
            for run in report["runs"]
            if run["method"] == "pixel_change"
        )
        self.assertEqual(metrics, {"mean": 8, "changed_fraction": 8})
        for run in report["runs"]:
            if run["method"] == "pixel_change":
                self.assertEqual(
                    "pixel_cutoff" in run["params"],
                    run["params"]["metric"] == "changed_fraction",
                )
        self.assertEqual(methods.count("background_subtraction"), 4)
        self.assertNotIn("yoloe_presence", methods)

        for run in report["runs"]:
            self.assertEqual(run["selected_indices"][0], 10)
            self.assertGreaterEqual(run["selection_seconds"], 0.0)
            self.assertEqual(set(run["read_seconds"]), {"auto", "seek", "sequential"})
            for seconds in run["read_seconds"].values():
                self.assertGreaterEqual(seconds, 0.0)
        self.assertEqual(report["read_strategies"], ["auto", "seek", "sequential"])

        pixel = [r for r in report["runs"] if r["method"] == "pixel_change"]
        for run in pixel:
            with self.subTest(params=run["params"]):
                self.assertEqual(run["selected_indices"], [10, 40, 70])
                self.assertEqual(run["event_recall"], 1.0)
                self.assertEqual(run["static_selections"], 0)
        self.assertIn("pixel_change", EVAL.format_report(report))

    def test_evaluate_with_fake_yoloe(self) -> None:
        view = EVAL.open_clip(str(self.video_path), 1.0, None)
        boxes = [_box(i) for i in range(40, 70, 5)]
        fake = _FakeDetect([{"type": "bear", "bboxes": boxes}])
        report = EVAL.evaluate(
            view,
            self.labels,
            sample_every=5,
            thresholds=(5.0,),
            foreground_fractions=(),
            yoloe_classes=["bear"],
            repeats=1,
            detect_fn=fake,
        )
        runs = {run["method"]: run for run in report["runs"]}
        self.assertEqual(runs["yoloe_count_change"]["selected_indices"], [10, 40, 70])
        self.assertEqual(
            runs["yoloe_presence"]["selected_indices"], [10, 40, 45, 50, 55, 60, 65]
        )
        self.assertEqual(runs["yoloe_presence"]["event_recall"], 0.5)
        self.assertEqual(fake.calls[0][1]["frame_stride"], 5)

    def test_evaluate_times_only_requested_read_strategies(self) -> None:
        view = EVAL.open_clip(str(self.video_path), 1.0, None)
        report = EVAL.evaluate(
            view, self.labels, thresholds=(5.0,), changed_fraction_thresholds=(),
            foreground_fractions=(), yoloe_classes=None, repeats=1,
            read_strategies=["sequential"],
        )
        self.assertEqual(report["read_strategies"], ["sequential"])
        self.assertEqual(set(report["runs"][0]["read_seconds"]), {"sequential"})
        self.assertIn("read sequential s", EVAL.format_report(report))

    def test_evaluate_rejects_bad_read_strategies(self) -> None:
        view = EVAL.open_clip(str(self.video_path), 1.0, None)
        for strategies in ([], ["fast"]):
            with self.subTest(strategies=strategies):
                with self.assertRaises(ValueError):
                    EVAL.evaluate(
                        view, self.labels, yoloe_classes=None, repeats=1,
                        read_strategies=strategies,
                    )

    def test_evaluate_rejects_labels_outside_clip(self) -> None:
        view = EVAL.open_clip(str(self.video_path), 1.0, 5.0)
        with self.assertRaises(ValueError):
            EVAL.evaluate(view, self.labels, yoloe_classes=None, repeats=1)

    def test_contact_sheets(self) -> None:
        view = EVAL.open_clip(str(self.video_path), 1.0, None)
        paths = EVAL.write_contact_sheets(
            view, self.tmpdir / "sheets", columns=2, rows=2, thumb_width=80
        )
        # 9 thumbnails (one per second over 9s), 4 per sheet -> 3 sheets.
        self.assertEqual(len(paths), 3)
        sheet = cv2.imread(str(paths[0]))
        self.assertEqual(sheet.shape[:2], (2 * 48, 2 * 80))

    def test_contact_sheets_reject_bad_layout(self) -> None:
        view = EVAL.open_clip(str(self.video_path), 1.0, None)
        with self.assertRaises(ValueError):
            EVAL.write_contact_sheets(view, self.tmpdir, columns=0)

    def test_main_writes_json_report(self) -> None:
        labels_path = _write_labels(
            self.tmpdir,
            {
                "video": str(self.video_path),
                "clip_start": 1.0,
                "clip_end": None,
                "events": [{"start": 4.0, "end": 4.5, "description": "enter"}],
                "static": [{"start": 1.0, "end": 4.0}],
            },
        )
        output = self.tmpdir / "report.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = EVAL.main(
                [
                    "--labels", str(labels_path),
                    "--skip-yoloe",
                    "--repeats", "1",
                    "--pixel-cutoff", "40",
                    "--read-strategies", "auto, seek",
                    "--output", str(output),
                ]
            )
        self.assertEqual(exit_code, 0)
        report = json.loads(output.read_text())
        self.assertEqual(report["start_frame"], 10)
        cutoffs = {
            run["params"]["pixel_cutoff"]
            for run in report["runs"]
            if run["params"].get("metric") == "changed_fraction"
        }
        self.assertEqual(cutoffs, {40.0})
        self.assertEqual(report["read_strategies"], ["auto", "seek"])
        self.assertIn("decode-only", stdout.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
