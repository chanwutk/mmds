from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mmds.utilities.video import (  # noqa: E402
    FRAME_READ_STRATEGIES,
    PIXEL_DIFF_REFERENCES,
    Video,
    VideoView,
    _pixel_diff_gray,
    _select_by_pixel_change,
    iter_sampled_frames,
    open_video,
    read_frames_at_indices,
    read_frames_sequentially,
    select_frames_by_pixel_change,
)

WIDTH, HEIGHT, FPS = 160, 96, 10.0


def _black() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def _with_square(x: int = 40, y: int = 20, size: int = 40) -> np.ndarray:
    frame = _black()
    frame[y : y + size, x : x + size] = 255
    return frame


def _write_video(
    directory: Path,
    num_frames: int,
    frame_at: Callable[[int], np.ndarray],
    name: str = "clip.avi",
) -> Path:
    path = directory / name
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV MJPG VideoWriter is unavailable")
    try:
        for index in range(num_frames):
            writer.write(frame_at(index))
    finally:
        writer.release()
    return path


def _gray(value: int) -> np.ndarray:
    return np.full((4, 4), value, dtype=np.uint8)


def _indexed(values: list[int], start: int = 0, step: int = 1):
    return [(start + i * step, _gray(v)) for i, v in enumerate(values)]


class _ExplodingIterable:
    """Fails the test if anything tries to consume it."""

    def __iter__(self):
        raise AssertionError("input was consumed before validation")


class SelectByPixelChangeTests(unittest.TestCase):
    def test_empty_input_selects_nothing(self) -> None:
        self.assertEqual(_select_by_pixel_change([], 10.0, "previous_sample"), [])

    def test_first_frame_always_selected(self) -> None:
        self.assertEqual(
            _select_by_pixel_change(_indexed([7], start=42), 255.0, "previous_sample"),
            [42],
        )

    def test_static_frames_select_only_first(self) -> None:
        for reference in PIXEL_DIFF_REFERENCES:
            with self.subTest(reference=reference):
                self.assertEqual(
                    _select_by_pixel_change(_indexed([50] * 6), 1.0, reference),
                    [0],
                )

    def test_abrupt_change_selected_at_its_index(self) -> None:
        frames = _indexed([0, 0, 0, 200, 200, 200], start=100, step=5)
        for reference in PIXEL_DIFF_REFERENCES:
            with self.subTest(reference=reference):
                self.assertEqual(
                    _select_by_pixel_change(frames, 15.0, reference), [100, 115]
                )

    def test_change_back_is_also_selected(self) -> None:
        frames = _indexed([0, 200, 200, 0])
        self.assertEqual(
            _select_by_pixel_change(frames, 15.0, "previous_sample"), [0, 1, 3]
        )

    def test_score_equal_to_threshold_is_selected(self) -> None:
        frames = _indexed([0, 10])
        self.assertEqual(
            _select_by_pixel_change(frames, 10.0, "previous_sample"), [0, 1]
        )

    def test_score_just_below_threshold_is_not_selected(self) -> None:
        frames = _indexed([0, 10])
        self.assertEqual(
            _select_by_pixel_change(frames, 10.0001, "previous_sample"), [0]
        )

    def test_threshold_zero_keeps_every_frame_including_duplicates(self) -> None:
        frames = _indexed([5, 5, 5, 9])
        for reference in PIXEL_DIFF_REFERENCES:
            with self.subTest(reference=reference):
                self.assertEqual(
                    _select_by_pixel_change(frames, 0, reference), [0, 1, 2, 3]
                )

    def test_integer_threshold_is_accepted(self) -> None:
        self.assertEqual(
            _select_by_pixel_change(_indexed([0, 20]), 15, "previous_sample"), [0, 1]
        )

    def test_slow_drift_missed_by_previous_sample(self) -> None:
        frames = _indexed([0, 4, 8, 12, 16, 20])
        self.assertEqual(
            _select_by_pixel_change(frames, 10.0, "previous_sample"), [0]
        )

    def test_slow_drift_caught_by_last_selected(self) -> None:
        frames = _indexed([0, 4, 8, 12, 16, 20, 24])
        # 12 - 0 >= 10 selects index 3; then 24 - 12 >= 10 selects index 6.
        self.assertEqual(
            _select_by_pixel_change(frames, 10.0, "last_selected"), [0, 3, 6]
        )

    def test_last_selected_keeps_reference_until_selection(self) -> None:
        # 9 vs 0 is below threshold, so 18 is compared to 0 (not 9) and selected.
        frames = _indexed([0, 9, 18])
        self.assertEqual(
            _select_by_pixel_change(frames, 10.0, "last_selected"), [0, 2]
        )
        self.assertEqual(
            _select_by_pixel_change(frames, 10.0, "previous_sample"), [0]
        )

    def test_score_is_mean_over_all_pixels(self) -> None:
        reference = np.zeros((4, 4), dtype=np.uint8)
        partial = reference.copy()
        partial[0, :] = 200  # one quarter of pixels changed -> mean 50
        frames = [(0, reference), (1, partial)]
        self.assertEqual(
            _select_by_pixel_change(frames, 50.0, "previous_sample"), [0, 1]
        )
        self.assertEqual(
            _select_by_pixel_change(frames, 50.01, "previous_sample"), [0]
        )

    def test_absdiff_does_not_wrap_for_darker_frames(self) -> None:
        frames = _indexed([200, 0])
        self.assertEqual(
            _select_by_pixel_change(frames, 199.0, "previous_sample"), [0, 1]
        )

    def test_invalid_thresholds_rejected(self) -> None:
        cases = [
            (True, TypeError),
            ("15", TypeError),
            (None, TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (-0.5, ValueError),
        ]
        for threshold, error in cases:
            with self.subTest(threshold=threshold):
                with self.assertRaises(error):
                    _select_by_pixel_change([], threshold, "previous_sample")

    def test_invalid_reference_rejected(self) -> None:
        for reference in ("previous", "", None, "LAST_SELECTED"):
            with self.subTest(reference=reference):
                with self.assertRaises(ValueError):
                    _select_by_pixel_change([], 10.0, reference)

    def test_validation_happens_before_consuming_input(self) -> None:
        with self.assertRaises(ValueError):
            _select_by_pixel_change(_ExplodingIterable(), -1.0, "previous_sample")
        with self.assertRaises(ValueError):
            _select_by_pixel_change(_ExplodingIterable(), 1.0, "bogus")


def _changed(num_pixels: int, by: int, shape=(100, 100), base: int = 50):
    """A (reference, frame) pair where the first *num_pixels* change by *by*."""
    reference = np.full(shape, base, dtype=np.uint8)
    frame = reference.copy()
    frame.reshape(-1)[:num_pixels] = base + by
    return [(0, reference), (1, frame)]


class ChangedFractionTests(unittest.TestCase):
    def select(self, frames, threshold, reference="previous_sample", cutoff=25.0):
        return _select_by_pixel_change(
            frames, threshold, reference, "changed_fraction", cutoff
        )

    def test_two_percent_of_pixels_scores_point_zero_two(self) -> None:
        frames = _changed(200, by=100)  # 200 / 10_000 = 0.02
        self.assertEqual(self.select(frames, 0.02), [0, 1])
        self.assertEqual(self.select(frames, 0.0201), [0])

    def test_difference_equal_to_cutoff_is_not_counted(self) -> None:
        self.assertEqual(self.select(_changed(5000, by=25), 0.01, cutoff=25.0), [0])
        self.assertEqual(self.select(_changed(5000, by=26), 0.01, cutoff=25.0), [0, 1])

    def test_small_change_everywhere_ignored_unlike_mean(self) -> None:
        frames = _changed(10_000, by=10)  # every pixel shifts by 10 gray levels
        self.assertEqual(self.select(frames, 0.001), [0])
        self.assertEqual(
            _select_by_pixel_change(frames, 5.0, "previous_sample", "mean"), [0, 1]
        )

    def test_small_sharp_patch_caught_unlike_mean(self) -> None:
        frames = _changed(100, by=200)  # 1% of pixels, mean difference 2.0
        self.assertEqual(self.select(frames, 0.005), [0, 1])
        self.assertEqual(
            _select_by_pixel_change(frames, 5.0, "previous_sample", "mean"), [0]
        )

    def test_absdiff_counts_darkening_pixels(self) -> None:
        reference = np.full((10, 10), 200, dtype=np.uint8)
        frame = reference.copy()
        frame[0, :] = 0
        self.assertEqual(self.select([(0, reference), (1, frame)], 0.1), [0, 1])

    def test_zero_cutoff_counts_any_nonzero_difference(self) -> None:
        self.assertEqual(self.select(_changed(1, by=1), 0.0001, cutoff=0), [0, 1])

    def test_threshold_one_requires_every_pixel(self) -> None:
        self.assertEqual(self.select(_changed(10_000, by=100), 1.0), [0, 1])
        self.assertEqual(self.select(_changed(9_999, by=100), 1.0), [0])

    def test_threshold_zero_keeps_every_frame(self) -> None:
        self.assertEqual(self.select(_indexed([5, 5, 5]), 0.0), [0, 1, 2])

    def test_reference_modes_with_fraction(self) -> None:
        # Each step brightens 10 more gray levels: only last_selected
        # accumulates past the 25-level cutoff.
        frames = _indexed([0, 10, 20, 30, 40])
        self.assertEqual(self.select(frames, 0.5, "previous_sample"), [0])
        self.assertEqual(self.select(frames, 0.5, "last_selected"), [0, 3])

    def test_default_metric_is_mean(self) -> None:
        frames = _changed(100, by=200)
        self.assertEqual(_select_by_pixel_change(frames, 2.0, "previous_sample"), [0, 1])

    def test_invalid_metric_rejected(self) -> None:
        for metric in ("fraction", "", None, "MEAN"):
            with self.subTest(metric=metric):
                with self.assertRaises(ValueError):
                    _select_by_pixel_change([], 0.1, "previous_sample", metric)

    def test_invalid_pixel_cutoff_rejected(self) -> None:
        cases = [
            (True, TypeError),
            ("25", TypeError),
            (None, TypeError),
            (float("nan"), ValueError),
            (-1.0, ValueError),
            (255, ValueError),
        ]
        for cutoff, error in cases:
            with self.subTest(cutoff=cutoff):
                with self.assertRaises(error):
                    self.select([], 0.1, cutoff=cutoff)

    def test_fraction_threshold_above_one_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.select([], 1.5)
        self.assertEqual(
            _select_by_pixel_change([], 15.0, "previous_sample", "mean"), []
        )

    def test_validation_happens_before_consuming_input(self) -> None:
        with self.assertRaises(ValueError):
            _select_by_pixel_change(_ExplodingIterable(), 0.1, "previous_sample", "x")
        with self.assertRaises(ValueError):
            self.select(_ExplodingIterable(), 2.0)
        with self.assertRaises(ValueError):
            self.select(_ExplodingIterable(), 0.1, cutoff=-5)


class PixelDiffGrayTests(unittest.TestCase):
    def test_bgr_frame_becomes_fixed_size_gray(self) -> None:
        out = _pixel_diff_gray(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.assertEqual(out.shape, (180, 320))
        self.assertEqual(out.dtype, np.uint8)

    def test_uniform_frame_stays_uniform(self) -> None:
        out = _pixel_diff_gray(np.full((90, 160, 3), 77, dtype=np.uint8))
        self.assertTrue(np.all(out == 77))

    def test_non_16_9_and_tiny_frames_are_resized(self) -> None:
        for shape in ((480, 640, 3), (10, 10, 3)):
            with self.subTest(shape=shape):
                self.assertEqual(
                    _pixel_diff_gray(np.zeros(shape, dtype=np.uint8)).shape, (180, 320)
                )

    def test_grayscale_inputs_accepted(self) -> None:
        for shape in ((90, 160), (90, 160, 1)):
            with self.subTest(shape=shape):
                out = _pixel_diff_gray(np.full(shape, 33, dtype=np.uint8))
                self.assertEqual(out.shape, (180, 320))
                self.assertTrue(np.all(out == 33))

    def test_bgra_input_accepted(self) -> None:
        out = _pixel_diff_gray(np.full((90, 160, 4), 120, dtype=np.uint8))
        self.assertEqual(out.shape, (180, 320))

    def test_unsupported_shapes_rejected(self) -> None:
        for shape in ((90, 160, 2), (90,), (2, 90, 160, 3)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    _pixel_diff_gray(np.zeros(shape, dtype=np.uint8))


class _VideoFileTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)

    def open(self, num_frames: int, frame_at: Callable[[int], np.ndarray]) -> Video:
        video = open_video(str(_write_video(self.tmpdir, num_frames, frame_at)))
        assert isinstance(video, Video)
        self.assertEqual(video.num_frames, num_frames)
        return video


class IterSampledFramesTests(_VideoFileTestCase):
    def test_sample_every_one_yields_every_index(self) -> None:
        video = self.open(6, lambda i: _black())
        self.assertEqual([i for i, _ in iter_sampled_frames(video, 1)], list(range(6)))

    def test_default_sample_every_is_one(self) -> None:
        video = self.open(4, lambda i: _black())
        self.assertEqual([i for i, _ in iter_sampled_frames(video)], [0, 1, 2, 3])

    def test_stride_on_full_video(self) -> None:
        video = self.open(10, lambda i: _black())
        self.assertEqual([i for i, _ in iter_sampled_frames(video, 3)], [0, 3, 6, 9])

    def test_view_yields_original_indices(self) -> None:
        video = self.open(30, lambda i: _black())
        view = VideoView(video, start=1.2, end=2.5)  # frames 12..24
        self.assertEqual(
            [i for i, _ in iter_sampled_frames(view, 4)], [12, 16, 20, 24]
        )

    def test_yielded_frames_match_source_frames(self) -> None:
        video = self.open(20, lambda i: np.full((HEIGHT, WIDTH, 3), i * 10, np.uint8))
        view = VideoView(video, start=0.5, end=1.5)
        for index, frame in iter_sampled_frames(view, 3):
            with self.subTest(index=index):
                self.assertAlmostEqual(float(frame.mean()), index * 10, delta=2)

    def test_stride_longer_than_video_yields_first_only(self) -> None:
        video = self.open(5, lambda i: _black())
        self.assertEqual([i for i, _ in iter_sampled_frames(video, 100)], [0])

    def test_empty_view_yields_nothing(self) -> None:
        video = self.open(10, lambda i: _black())
        self.assertEqual(list(iter_sampled_frames(VideoView(video, 0.5, 0.5), 1)), [])

    def test_invalid_arguments_rejected_eagerly(self) -> None:
        video = self.open(3, lambda i: _black())
        cases = [
            (video, True, TypeError),
            (video, 2.0, TypeError),
            (video, "5", TypeError),
            (video, 0, ValueError),
            (video, -3, ValueError),
            ("clip.mp4", 1, TypeError),
            ([_black()], 1, TypeError),
        ]
        for source, sample_every, error in cases:
            with self.subTest(source=type(source).__name__, sample_every=sample_every):
                with self.assertRaises(error):
                    iter_sampled_frames(source, sample_every)  # not iterated


def _unique_frame(i: int) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (i * 3) % 200, dtype=np.uint8)
    x = (i * 7) % (WIDTH - 16)
    frame[10:26, x : x + 16] = 255
    return frame


class FrameReadStrategyFileTests(_VideoFileTestCase):
    """All retrieval strategies must return the same pixels on a real file."""

    selections = {
        "sparse": [3, 47, 88],
        "dense": list(range(10, 60, 2)),
        "clustered": [5, 6, 7, 8, 70, 71, 72, 73],
        "unsorted_with_junk": [88, -1, 12, 12, True, 40],
        "past_end": [95, 99, 100, 150],
    }

    def setUp(self) -> None:
        super().setUp()
        self.video = self.open(100, _unique_frame)
        self.decoded = dict(enumerate(self.video))

    def test_strategies_return_identical_frames(self) -> None:
        for name, indices in self.selections.items():
            expected = sorted(
                {i for i in indices if isinstance(i, int) and not isinstance(i, bool)
                 and 0 <= i < 100}
            )
            for strategy in FRAME_READ_STRATEGIES:
                for max_gap in (0, 3, 100):
                    with self.subTest(selection=name, strategy=strategy, max_gap=max_gap):
                        frames = read_frames_at_indices(
                            self.video.path, indices, strategy=strategy, max_gap=max_gap
                        )
                        self.assertEqual(list(frames), expected)
                        for index in expected:
                            np.testing.assert_array_equal(
                                frames[index], self.decoded[index]
                            )

    def test_read_frames_sequentially_matches_seek(self) -> None:
        indices = [4, 30, 31, 77]
        sequential = read_frames_sequentially(self.video.path, indices)
        seek = read_frames_at_indices(self.video.path, indices, strategy="seek")
        self.assertEqual(list(sequential), list(seek))
        for index in indices:
            np.testing.assert_array_equal(sequential[index], seek[index])

    def test_view_selection_retrieved_sequentially_from_late_start(self) -> None:
        view = VideoView(self.video, start=6.0, end=9.0)  # frames 60..89
        indices = select_frames_by_pixel_change(view, 0.0, 7)
        self.assertEqual(indices, [60, 67, 74, 81, 88])
        frames = read_frames_sequentially(view.path, indices)
        for index in indices:
            np.testing.assert_array_equal(frames[index], self.decoded[index])


class SelectFramesByPixelChangeTests(_VideoFileTestCase):
    def test_static_video_selects_first_frame_only(self) -> None:
        video = self.open(20, lambda i: _with_square())
        self.assertEqual(select_frames_by_pixel_change(video, 1.0, 1), [0])

    def test_object_entrance_selected_at_original_index(self) -> None:
        video = self.open(20, lambda i: _with_square() if i >= 12 else _black())
        self.assertEqual(select_frames_by_pixel_change(video, 5.0, 1), [0, 12])

    def test_change_between_samples_caught_at_next_sample(self) -> None:
        video = self.open(20, lambda i: _with_square() if i >= 7 else _black())
        self.assertEqual(select_frames_by_pixel_change(video, 5.0, 5), [0, 10])

    def test_view_returns_original_indices(self) -> None:
        video = self.open(40, lambda i: _with_square() if i >= 23 else _black())
        view = VideoView(video, start=1.0, end=3.5)  # frames 10..34
        self.assertEqual(select_frames_by_pixel_change(view, 5.0, 5), [10, 25])

    def test_indices_sorted_unique_in_range_and_on_stride(self) -> None:
        def moving(i: int) -> np.ndarray:
            return _with_square(x=(i * 7) % (WIDTH - 40))

        video = self.open(60, moving)
        view = VideoView(video, start=1.3, end=5.1)
        indices = select_frames_by_pixel_change(view, 0.5, 4)
        self.assertGreater(len(indices), 1)
        self.assertEqual(indices, sorted(set(indices)))
        for index in indices:
            self.assertGreaterEqual(index, view.start_frame)
            self.assertLess(index, view.end_frame)
            self.assertEqual((index - view.start_frame) % 4, 0)

    def test_changed_fraction_on_real_video(self) -> None:
        # The 40x40 square covers ~10% of the frame.
        video = self.open(30, lambda i: _with_square() if i >= 17 else _black())
        view = VideoView(video, start=0.5, end=3.0)
        self.assertEqual(
            select_frames_by_pixel_change(
                view, 0.05, 2, metric="changed_fraction", pixel_cutoff=40.0
            ),
            [5, 17],
        )
        self.assertEqual(
            select_frames_by_pixel_change(view, 0.2, 2, metric="changed_fraction"),
            [5],
        )

    def test_changed_fraction_validation_in_public_function(self) -> None:
        video = self.open(3, lambda i: _black())
        with self.assertRaises(ValueError):
            select_frames_by_pixel_change(video, metric="changed_fraction")  # 15 > 1
        with self.assertRaises(ValueError):
            select_frames_by_pixel_change(video, metric="median")
        with self.assertRaises(TypeError):
            select_frames_by_pixel_change(
                video, 0.1, metric="changed_fraction", pixel_cutoff="25"
            )

    def test_threshold_zero_keeps_every_sampled_frame(self) -> None:
        video = self.open(12, lambda i: _black())
        self.assertEqual(select_frames_by_pixel_change(video, 0.0, 3), [0, 3, 6, 9])

    def test_reference_modes_on_slow_brightening(self) -> None:
        video = self.open(
            9, lambda i: np.full((HEIGHT, WIDTH, 3), 50 + i * 4, dtype=np.uint8)
        )
        self.assertEqual(
            select_frames_by_pixel_change(video, 10.0, 1, "previous_sample"), [0]
        )
        last_selected = select_frames_by_pixel_change(video, 10.0, 1, "last_selected")
        self.assertEqual(last_selected[0], 0)
        self.assertGreaterEqual(len(last_selected), 2)

    def test_round_trip_with_read_frames_at_indices(self) -> None:
        video = self.open(30, lambda i: _with_square(x=10 + 20 * (i // 6)))
        view = VideoView(video, start=0.4, end=2.8)
        indices = select_frames_by_pixel_change(view, 2.0, 2)
        self.assertGreater(len(indices), 1)
        frames = read_frames_at_indices(view.path, indices)
        self.assertEqual(sorted(frames), indices)
        by_index = dict(iter_sampled_frames(view, 1))
        for index in indices:
            with self.subTest(index=index):
                np.testing.assert_array_equal(frames[index], by_index[index])

    def test_empty_view_returns_empty_list(self) -> None:
        video = self.open(10, lambda i: _black())
        self.assertEqual(select_frames_by_pixel_change(VideoView(video, 0.3, 0.3)), [])

    def test_defaults(self) -> None:
        video = self.open(11, lambda i: _black())
        self.assertEqual(select_frames_by_pixel_change(video), [0])

    def test_invalid_arguments_rejected(self) -> None:
        video = self.open(3, lambda i: _black())
        with self.assertRaises(ValueError):
            select_frames_by_pixel_change(video, sample_every=0)
        with self.assertRaises(TypeError):
            select_frames_by_pixel_change(video, sample_every=True)
        with self.assertRaises(ValueError):
            select_frames_by_pixel_change(video, threshold=-1.0)
        with self.assertRaises(TypeError):
            select_frames_by_pixel_change(video, threshold="high")
        with self.assertRaises(ValueError):
            select_frames_by_pixel_change(video, reference="keyframe")
        with self.assertRaises(TypeError):
            select_frames_by_pixel_change(str(video.path))

    def test_view_iteration_still_yields_every_frame(self) -> None:
        video = self.open(30, lambda i: _with_square() if i >= 15 else _black())
        view = VideoView(video, start=0.5, end=2.5)
        select_frames_by_pixel_change(view, 5.0, 5)
        self.assertEqual(len(list(view)), view.num_frames)
        self.assertEqual(view.num_frames, 20)


if __name__ == "__main__":
    unittest.main()
