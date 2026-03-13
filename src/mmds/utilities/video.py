"""Internal video utility for reading video files and directories.

Usage::

    from mmds.utilities.video import open_video, VideoView

    # Single file (local path or http/https URL)
    video = open_video("clip.mp4")
    print(video.width, video.height, video.fps, video.num_frames)
    for frame in video:   # numpy BGR frames via OpenCV
        ...

    # Time-range view (only iterates frames in [start, end) seconds)
    view = VideoView(video, start=10.0, end=20.0)
    for frame in view:   # only frames between 10s and 20s
        ...

    # Directory — returns a list of Video objects, sorted by filename
    videos = open_video("/data/clips/")
"""

from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


_CACHE_DIR = Path.home() / ".cache" / "mmds" / "videos"
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


# ---------------------------------------------------------------------------
# Download / cache helpers
# ---------------------------------------------------------------------------


def _cache_path(url: str) -> Path:
    """Return the local cache path for *url* without downloading.

    Only valid for direct-file URLs whose path ends in a known video extension.
    For platform URLs (e.g. YouTube) use :func:`_yt_download` instead.
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    filename = Path(urllib.parse.urlparse(url).path).name or "video"
    return _CACHE_DIR / url_hash / filename


def _is_direct_video_url(url: str) -> bool:
    """Return True if *url* has a path that ends with a known video extension."""
    path = urllib.parse.urlparse(url).path
    return Path(path).suffix.lower() in _VIDEO_EXTENSIONS


def _yt_download(url: str) -> Path:
    """Download *url* via yt-dlp and return the local path.

    yt-dlp is used for platform URLs (e.g. YouTube) where the URL path does
    not directly name a video file.  The download is cached under
    ``~/.cache/mmds/videos/<sha256-16>/`` and re-used on subsequent calls.
    The actual filename (including extension chosen by yt-dlp) is discovered
    by listing the directory after the download completes.
    """
    import yt_dlp  # deferred: heavy import

    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    cache_subdir = _CACHE_DIR / url_hash

    # Re-use an already-downloaded file if present.
    if cache_subdir.exists():
        existing = [
            p
            for p in cache_subdir.iterdir()
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
        ]
        if existing:
            return existing[0]

    cache_subdir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(cache_subdir / "%(title)s.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        # Request a pre-muxed format (single file, no ffmpeg merge required).
        # mp4 is preferred because OpenCV reads it reliably; webm is the fallback.
        "format": "best[ext=mp4]/best[ext=webm]/best",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    downloaded = [
        p
        for p in cache_subdir.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
    ]
    if not downloaded:
        raise ValueError(f"yt-dlp did not produce a video file for: {url}")
    return downloaded[0]


def _download(url: str) -> Path:
    """Download *url* into the local cache and return its path.

    Re-uses the cached copy on subsequent calls without issuing a new request.
    Cache lives at ``~/.cache/mmds/videos/<sha256-16>/<filename>``.

    Direct video file URLs (path ends with a known extension such as ``.mp4``)
    are fetched with :mod:`urllib.request`.  Platform URLs (e.g. YouTube,
    Vimeo) whose URL path does not end in a video extension are downloaded via
    ``yt-dlp``.
    """
    if _is_direct_video_url(url):
        dest = _cache_path(url)
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        return dest
    return _yt_download(url)


def _resolve(path_or_url: str) -> Path:
    """Return a local *Path* for *path_or_url*, downloading if necessary."""
    parsed = urllib.parse.urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        return _download(path_or_url)
    if parsed.scheme == "file":
        return Path(urllib.request.url2pathname(parsed.path))
    return Path(path_or_url)


# ---------------------------------------------------------------------------
# Video class
# ---------------------------------------------------------------------------


class Video:
    """A handle to a single video file.

    Metadata (``width``, ``height``, ``fps``, ``num_frames``) is read once at
    construction time.  Iterating over the object yields BGR ``numpy`` frames.
    The underlying ``VideoCapture`` is opened and released for each iteration,
    so multiple passes over the same ``Video`` are safe.

    Example::

        video = Video(Path("clip.mp4"))
        print(video.width, video.height, video.fps, video.num_frames)
        for frame in video:
            process(frame)
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        try:
            self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps: float = cap.get(cv2.CAP_PROP_FPS)
            self._num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return self._num_frames

    def __iter__(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(str(self._path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    def __repr__(self) -> str:
        return (
            f"Video({self._path.name!r}, "
            f"{self._width}x{self._height}, "
            f"{self._fps:.2f} fps, "
            f"{self._num_frames} frames)"
        )


class VideoView:
    """A time-range view over a :class:`Video`.

    Iterates only frames whose timestamp falls within ``[start, end)``
    seconds.  Uses ``cv2.CAP_PROP_POS_FRAMES`` to seek directly to the
    start frame, so creating a view over a late section of a long video
    is O(1) rather than O(n).

    The same metadata properties as :class:`Video` are exposed (``width``,
    ``height``, ``fps``, ``path``).  ``num_frames`` and ``__len__`` reflect
    the *view* length, not the full video length.

    Example::

        video = Video(Path("clip.mp4"))
        view = VideoView(video, start=10.0, end=20.0)
        print(view.num_frames)  # ~300 for 30fps
        for frame in view:
            process(frame)
    """

    def __init__(self, video: Video, start: float, end: float) -> None:
        self._video = video
        fps = video.fps or 30.0
        self._start_frame = int(start * fps)
        self._end_frame = min(int(end * fps), video.num_frames)

    @property
    def width(self) -> int:
        return self._video.width

    @property
    def height(self) -> int:
        return self._video.height

    @property
    def fps(self) -> float:
        return self._video.fps

    @property
    def num_frames(self) -> int:
        return max(0, self._end_frame - self._start_frame)

    @property
    def path(self) -> Path:
        return self._video.path

    @property
    def start_frame(self) -> int:
        return self._start_frame

    @property
    def end_frame(self) -> int:
        return self._end_frame

    def __len__(self) -> int:
        return self.num_frames

    def __iter__(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(str(self._video.path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
            for _ in range(self.num_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    def __repr__(self) -> str:
        return (
            f"VideoView({self._video.path.name!r}, "
            f"frames {self._start_frame}\u2013{self._end_frame}, "
            f"{self.num_frames} frames)"
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def open_video(path_or_url: str) -> Video | list[Video]:
    """Open a video file or a directory of video files.

    Args:
        path_or_url: One of:
            - A local file path (``/data/clip.mp4``, ``clip.mp4``)
            - A ``file://`` URL
            - An ``http://`` / ``https://`` URL (downloaded and cached)
            - A local directory path (returns a sorted list of :class:`Video`)

    Returns:
        A single :class:`Video` for file inputs, or a sorted ``list[Video]``
        for directory inputs (sorted by filename; non-video files are skipped).

    The download cache lives at ``~/.cache/mmds/videos/`` and is keyed by the
    SHA-256 of the URL, so the same URL is never downloaded twice.
    """
    parsed = urllib.parse.urlparse(path_or_url)
    if parsed.scheme in ("", "file"):
        local = (
            Path(urllib.request.url2pathname(parsed.path))
            if parsed.scheme == "file"
            else Path(path_or_url)
        )
        if local.is_dir():
            paths = sorted(
                p
                for p in local.iterdir()
                if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
            )
            return [Video(p) for p in paths]
    return Video(_resolve(path_or_url))
