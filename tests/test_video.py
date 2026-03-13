from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mmds.utilities.video import (  # noqa: E402
    Video,
    VideoView,
    _cache_path,
    _download,
    _is_direct_video_url,
    _resolve,
    _yt_download,
    open_video,
)


def _make_cap(*, width=640, height=480, fps=30.0, num_frames=100, opened=True):
    """Return a MagicMock VideoCapture with common property stubs."""
    cap = MagicMock()
    cap.isOpened.return_value = opened
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: float(width),
        cv2.CAP_PROP_FRAME_HEIGHT: float(height),
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_COUNT: float(num_frames),
    }.get(prop, 0.0)
    return cap


class CachePathTests(unittest.TestCase):
    def test_same_url_returns_same_path(self) -> None:
        url = "https://example.com/clip.mp4"
        self.assertEqual(_cache_path(url), _cache_path(url))

    def test_different_urls_return_different_paths(self) -> None:
        self.assertNotEqual(
            _cache_path("https://example.com/a.mp4"),
            _cache_path("https://example.com/b.mp4"),
        )

    def test_filename_is_preserved(self) -> None:
        self.assertEqual(_cache_path("https://example.com/clip.mp4").name, "clip.mp4")

    def test_url_with_no_filename_falls_back(self) -> None:
        # urlparse path is empty → falls back to "video"
        p = _cache_path("https://example.com/")
        self.assertEqual(p.name, "video")

    def test_path_is_under_cache_dir(self) -> None:
        from mmds.utilities.video import _CACHE_DIR

        p = _cache_path("https://example.com/clip.mp4")
        self.assertTrue(str(p).startswith(str(_CACHE_DIR)))


class ResolveTests(unittest.TestCase):
    def test_plain_local_path(self) -> None:
        self.assertEqual(_resolve("/tmp/video.mp4"), Path("/tmp/video.mp4"))

    def test_file_url(self) -> None:
        self.assertEqual(_resolve("file:///tmp/video.mp4"), Path("/tmp/video.mp4"))

    def test_http_url_triggers_download(self) -> None:
        url = "https://example.com/clip.mp4"
        with patch(
            "mmds.utilities.video._download", return_value=Path("/cache/clip.mp4")
        ) as mock_dl:
            result = _resolve(url)
        mock_dl.assert_called_once_with(url)
        self.assertEqual(result, Path("/cache/clip.mp4"))


class IsDirectVideoUrlTests(unittest.TestCase):
    def test_mp4_extension_is_direct(self) -> None:
        self.assertTrue(_is_direct_video_url("https://example.com/clip.mp4"))

    def test_avi_extension_is_direct(self) -> None:
        self.assertTrue(_is_direct_video_url("https://example.com/clip.avi"))

    def test_webm_extension_is_direct(self) -> None:
        self.assertTrue(_is_direct_video_url("https://cdn.example.com/video.webm"))

    def test_youtube_url_is_not_direct(self) -> None:
        self.assertFalse(
            _is_direct_video_url("https://www.youtube.com/watch?v=s5iU3nLOvi8")
        )

    def test_path_with_no_extension_is_not_direct(self) -> None:
        self.assertFalse(_is_direct_video_url("https://example.com/video"))

    def test_html_extension_is_not_direct(self) -> None:
        self.assertFalse(_is_direct_video_url("https://example.com/page.html"))

    def test_extension_check_is_case_insensitive(self) -> None:
        self.assertTrue(_is_direct_video_url("https://example.com/CLIP.MP4"))


class DownloadTests(unittest.TestCase):
    def test_returns_cached_file_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://example.com/clip.mp4"
            dest = Path(tmpdir) / "clip.mp4"
            dest.write_bytes(b"fake")

            with patch("mmds.utilities.video._cache_path", return_value=dest):
                with patch(
                    "mmds.utilities.video.urllib.request.urlretrieve"
                ) as mock_dl:
                    result = _download(url)
                    mock_dl.assert_not_called()

            self.assertEqual(result, dest)

    def test_downloads_when_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://example.com/new.mp4"
            dest = Path(tmpdir) / "subdir" / "new.mp4"

            with patch("mmds.utilities.video._cache_path", return_value=dest):
                with patch(
                    "mmds.utilities.video.urllib.request.urlretrieve"
                ) as mock_dl:
                    result = _download(url)
                    mock_dl.assert_called_once_with(url, dest)

            self.assertEqual(result, dest)
            self.assertTrue(dest.parent.exists())

    def test_platform_url_delegates_to_yt_download(self) -> None:
        url = "https://www.youtube.com/watch?v=s5iU3nLOvi8"
        fake_path = Path("/cache/abc123/video.mp4")
        with patch(
            "mmds.utilities.video._yt_download", return_value=fake_path
        ) as mock_yt:
            with patch("mmds.utilities.video.urllib.request.urlretrieve") as mock_dl:
                result = _download(url)
        mock_yt.assert_called_once_with(url)
        mock_dl.assert_not_called()
        self.assertEqual(result, fake_path)

    def test_direct_url_does_not_use_yt_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://example.com/clip.mp4"
            dest = Path(tmpdir) / "clip.mp4"
            with patch("mmds.utilities.video._cache_path", return_value=dest):
                with patch("mmds.utilities.video.urllib.request.urlretrieve"):
                    with patch("mmds.utilities.video._yt_download") as mock_yt:
                        _download(url)
            mock_yt.assert_not_called()


class YtDownloadTests(unittest.TestCase):
    def test_returns_cached_file_without_calling_yt_dlp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://www.youtube.com/watch?v=TESTID"
            url_hash = __import__("hashlib").sha256(url.encode()).hexdigest()[:16]
            cache_subdir = Path(tmpdir) / url_hash
            cache_subdir.mkdir()
            cached_file = cache_subdir / "Wildlife.mp4"
            cached_file.write_bytes(b"fake-video")

            with patch("mmds.utilities.video._CACHE_DIR", Path(tmpdir)):
                with patch("mmds.utilities.video.yt_dlp", create=True) as mock_yt_dlp:
                    result = _yt_download(url)
                    mock_yt_dlp.YoutubeDL.assert_not_called()

            self.assertEqual(result, cached_file)

    def test_calls_yt_dlp_when_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://www.youtube.com/watch?v=NEWID"
            url_hash = __import__("hashlib").sha256(url.encode()).hexdigest()[:16]
            cache_subdir = Path(tmpdir) / url_hash

            # Simulate yt-dlp writing a file during download.
            def fake_download(urls: list[str]) -> None:
                (cache_subdir / "Swan Valley Wildlife.mp4").write_bytes(b"video")

            mock_ydl = MagicMock()
            mock_ydl.__enter__ = lambda s: mock_ydl
            mock_ydl.__exit__ = MagicMock(return_value=False)
            mock_ydl.download.side_effect = fake_download

            mock_yt_dlp_module = MagicMock()
            mock_yt_dlp_module.YoutubeDL.return_value = mock_ydl

            with patch("mmds.utilities.video._CACHE_DIR", Path(tmpdir)):
                with patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp_module}):
                    result = _yt_download(url)

            mock_ydl.download.assert_called_once_with([url])
            self.assertEqual(result.suffix, ".mp4")

    def test_raises_if_yt_dlp_produces_no_video_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://www.youtube.com/watch?v=BADID"

            mock_ydl = MagicMock()
            mock_ydl.__enter__ = lambda s: mock_ydl
            mock_ydl.__exit__ = MagicMock(return_value=False)
            mock_ydl.download.return_value = None  # writes nothing

            mock_yt_dlp_module = MagicMock()
            mock_yt_dlp_module.YoutubeDL.return_value = mock_ydl

            with patch("mmds.utilities.video._CACHE_DIR", Path(tmpdir)):
                with patch.dict("sys.modules", {"yt_dlp": mock_yt_dlp_module}):
                    with self.assertRaises(ValueError, msg="yt-dlp did not produce"):
                        _yt_download(url)


class VideoMetadataTests(unittest.TestCase):
    def test_properties_are_read_from_capture(self) -> None:
        cap = _make_cap(width=1920, height=1080, fps=24.0, num_frames=240)
        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=cap):
            v = Video(Path("/tmp/test.mp4"))

        self.assertEqual(v.width, 1920)
        self.assertEqual(v.height, 1080)
        self.assertEqual(v.fps, 24.0)
        self.assertEqual(v.num_frames, 240)
        self.assertEqual(len(v), 240)

    def test_capture_is_released_after_init(self) -> None:
        cap = _make_cap()
        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=cap):
            Video(Path("/tmp/test.mp4"))
        cap.release.assert_called_once()

    def test_capture_is_released_on_init_error(self) -> None:
        cap = _make_cap()
        cap.get.side_effect = RuntimeError("boom")
        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=cap):
            with self.assertRaises(RuntimeError):
                Video(Path("/tmp/test.mp4"))
        cap.release.assert_called_once()

    def test_raises_for_unopenable_file(self) -> None:
        cap = _make_cap(opened=False)
        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=cap):
            with self.assertRaises(ValueError, msg="Cannot open video"):
                Video(Path("/nonexistent.mp4"))

    def test_path_property(self) -> None:
        cap = _make_cap()
        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=cap):
            v = Video(Path("/tmp/test.mp4"))
        self.assertEqual(v.path, Path("/tmp/test.mp4"))

    def test_repr_contains_key_info(self) -> None:
        cap = _make_cap(width=640, height=480, fps=25.0, num_frames=50)
        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=cap):
            v = Video(Path("/tmp/test.mp4"))
        r = repr(v)
        self.assertIn("test.mp4", r)
        self.assertIn("640x480", r)
        self.assertIn("25.00 fps", r)
        self.assertIn("50 frames", r)


class VideoIterationTests(unittest.TestCase):
    def _make_frame(self, value: int) -> np.ndarray:
        return np.full((480, 640, 3), value, dtype=np.uint8)

    def test_iterates_over_all_frames(self) -> None:
        frames = [self._make_frame(i) for i in range(3)]
        read_returns = [(True, f) for f in frames] + [(False, None)]

        init_cap = _make_cap(num_frames=3)
        iter_cap = MagicMock()
        iter_cap.read.side_effect = read_returns

        with patch(
            "mmds.utilities.video.cv2.VideoCapture", side_effect=[init_cap, iter_cap]
        ):
            v = Video(Path("/tmp/test.mp4"))
            result = list(v)

        self.assertEqual(len(result), 3)
        for got, expected in zip(result, frames):
            np.testing.assert_array_equal(got, expected)

    def test_capture_is_released_after_full_iteration(self) -> None:
        init_cap = _make_cap()
        iter_cap = MagicMock()
        iter_cap.read.return_value = (False, None)

        with patch(
            "mmds.utilities.video.cv2.VideoCapture", side_effect=[init_cap, iter_cap]
        ):
            v = Video(Path("/tmp/test.mp4"))
            list(v)

        iter_cap.release.assert_called_once()

    def test_capture_is_released_on_early_exit(self) -> None:
        frame = self._make_frame(0)
        init_cap = _make_cap()
        iter_cap = MagicMock()
        iter_cap.read.return_value = (True, frame)

        with patch(
            "mmds.utilities.video.cv2.VideoCapture", side_effect=[init_cap, iter_cap]
        ):
            v = Video(Path("/tmp/test.mp4"))
            gen = iter(v)
            next(gen)
            gen.close()  # simulate early exit

        iter_cap.release.assert_called_once()

    def test_can_iterate_twice(self) -> None:
        frame = self._make_frame(42)
        init_cap = _make_cap(num_frames=1)
        iter_cap1 = MagicMock()
        iter_cap1.read.side_effect = [(True, frame), (False, None)]
        iter_cap2 = MagicMock()
        iter_cap2.read.side_effect = [(True, frame), (False, None)]

        with patch(
            "mmds.utilities.video.cv2.VideoCapture",
            side_effect=[init_cap, iter_cap1, iter_cap2],
        ):
            v = Video(Path("/tmp/test.mp4"))
            self.assertEqual(len(list(v)), 1)
            self.assertEqual(len(list(v)), 1)


class VideoViewTests(unittest.TestCase):
    def _make_video(self, *, width=640, height=480, fps=30.0, num_frames=300) -> Video:
        """Return a mock-constructed Video with the given metadata."""
        cap = _make_cap(width=width, height=height, fps=fps, num_frames=num_frames)
        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=cap):
            return Video(Path("/tmp/test.mp4"))

    def test_num_frames_computed_from_range(self) -> None:
        video = self._make_video(fps=30.0, num_frames=300)
        view = VideoView(video, start=1.0, end=3.0)
        # 1.0s * 30fps = frame 30; 3.0s * 30fps = frame 90 → 60 frames
        self.assertEqual(view.num_frames, 60)

    def test_len_equals_num_frames(self) -> None:
        video = self._make_video(fps=30.0, num_frames=300)
        view = VideoView(video, start=0.0, end=2.0)
        self.assertEqual(len(view), view.num_frames)

    def test_delegates_width_height_fps_path(self) -> None:
        video = self._make_video(width=1920, height=1080, fps=24.0, num_frames=240)
        view = VideoView(video, start=0.0, end=5.0)
        self.assertEqual(view.width, 1920)
        self.assertEqual(view.height, 1080)
        self.assertEqual(view.fps, 24.0)
        self.assertEqual(view.path, Path("/tmp/test.mp4"))

    def test_start_frame_and_end_frame_properties(self) -> None:
        video = self._make_video(fps=10.0, num_frames=100)
        view = VideoView(video, start=2.0, end=5.0)
        self.assertEqual(view.start_frame, 20)
        self.assertEqual(view.end_frame, 50)

    def test_end_clamped_to_video_length(self) -> None:
        video = self._make_video(fps=30.0, num_frames=100)
        view = VideoView(video, start=0.0, end=999.0)
        self.assertEqual(view.end_frame, 100)
        self.assertEqual(view.num_frames, 100)

    def test_zero_length_view(self) -> None:
        video = self._make_video(fps=30.0, num_frames=300)
        view = VideoView(video, start=5.0, end=5.0)
        self.assertEqual(view.num_frames, 0)
        self.assertEqual(len(view), 0)

    def test_iterates_only_range_frames(self) -> None:
        video = self._make_video(fps=10.0, num_frames=100)
        view = VideoView(video, start=2.0, end=4.0)
        # start_frame=20, end_frame=40 → 20 frames

        frames = [np.full((480, 640, 3), i, dtype=np.uint8) for i in range(20)]
        read_returns = [(True, f) for f in frames] + [(False, None)]

        iter_cap = MagicMock()
        iter_cap.read.side_effect = read_returns

        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=iter_cap):
            result = list(view)

        self.assertEqual(len(result), 20)
        # Verify seek was called to start_frame
        iter_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 20)

    def test_capture_seeks_to_start_frame(self) -> None:
        video = self._make_video(fps=30.0, num_frames=300)
        view = VideoView(video, start=3.0, end=4.0)
        # start_frame = 90

        iter_cap = MagicMock()
        iter_cap.read.return_value = (False, None)

        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=iter_cap):
            list(view)  # exhaust the iterator

        iter_cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 90)

    def test_capture_is_released_after_iteration(self) -> None:
        video = self._make_video(fps=10.0, num_frames=100)
        view = VideoView(video, start=0.0, end=1.0)

        iter_cap = MagicMock()
        iter_cap.read.return_value = (False, None)

        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=iter_cap):
            list(view)

        iter_cap.release.assert_called_once()

    def test_capture_is_released_on_early_exit(self) -> None:
        video = self._make_video(fps=10.0, num_frames=100)
        view = VideoView(video, start=0.0, end=5.0)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        iter_cap = MagicMock()
        iter_cap.read.return_value = (True, frame)

        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=iter_cap):
            gen = iter(view)
            next(gen)
            gen.close()

        iter_cap.release.assert_called_once()

    def test_repr_contains_frame_range(self) -> None:
        video = self._make_video(fps=10.0, num_frames=100)
        view = VideoView(video, start=2.0, end=5.0)
        r = repr(view)
        self.assertIn("test.mp4", r)
        self.assertIn("20", r)
        self.assertIn("50", r)
        self.assertIn("30 frames", r)


class OpenVideoTests(unittest.TestCase):
    def test_single_local_file(self) -> None:
        cap = _make_cap()
        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=cap):
            result = open_video("/tmp/clip.mp4")
        self.assertIsInstance(result, Video)

    def test_file_url(self) -> None:
        cap = _make_cap()
        with patch("mmds.utilities.video.cv2.VideoCapture", return_value=cap):
            result = open_video("file:///tmp/clip.mp4")
        self.assertIsInstance(result, Video)

    def test_http_url_downloads_then_opens(self) -> None:
        cap = _make_cap()
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with patch("mmds.utilities.video._download", return_value=Path(f.name)):
                with patch("mmds.utilities.video.cv2.VideoCapture", return_value=cap):
                    result = open_video("https://example.com/clip.mp4")
        self.assertIsInstance(result, Video)

    def test_directory_returns_list_of_videos(self) -> None:
        cap = _make_cap()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "a.mp4").touch()
            (tmpdir / "b.avi").touch()
            (tmpdir / "c.txt").touch()  # should be skipped

            with patch("mmds.utilities.video.cv2.VideoCapture", return_value=cap):
                result = open_video(str(tmpdir))

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        names = {v.path.name for v in result}
        self.assertEqual(names, {"a.mp4", "b.avi"})

    def test_directory_result_is_sorted_by_name(self) -> None:
        cap = _make_cap()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for name in ("c.mp4", "a.mp4", "b.mp4"):
                (tmpdir / name).touch()

            with patch("mmds.utilities.video.cv2.VideoCapture", return_value=cap):
                result = open_video(str(tmpdir))

        self.assertEqual([v.path.name for v in result], ["a.mp4", "b.mp4", "c.mp4"])

    def test_empty_directory_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = open_video(tmpdir)
        self.assertEqual(result, [])

    def test_directory_with_no_video_files_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "readme.txt").touch()
            (tmpdir / "data.json").touch()
            result = open_video(str(tmpdir))
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
