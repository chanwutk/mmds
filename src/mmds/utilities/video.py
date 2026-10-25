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

    # Optional pixel-change filtering — returns original-video frame indices
    indices = select_frames_by_pixel_change(view, threshold=15.0, sample_every=5)
    frames = read_frames_at_indices(view.path, indices)
"""

from __future__ import annotations

import hashlib
import math
import numbers
import urllib.parse
import urllib.request
from pathlib import Path
from collections.abc import Iterable
from typing import Iterator

import cv2
import numpy as np


_CACHE_DIR = Path.home() / ".cache" / "mmds" / "videos"
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}

# Pixel-change frame selection compares frames at this (width, height) after a
# Gaussian blur, so scores are comparable across source resolutions.
_PIXEL_DIFF_SIZE = (320, 180)
_PIXEL_DIFF_BLUR = (5, 5)
PIXEL_DIFF_REFERENCES = ("previous_sample", "last_selected")
PIXEL_DIFF_METRICS = ("mean", "changed_fraction")
DEFAULT_PIXEL_CUTOFF = 25.0

# A seek decodes forward from the nearest keyframe and cost ~60-95 ms per frame
# on 720p H.264, about the same as grab()-ing ~100-120 frames, so "auto"
# retrieval reads forward across gaps up to this many frames.
FRAME_READ_STRATEGIES = ("auto", "seek", "sequential")
DEFAULT_MAX_SEQUENTIAL_GAP = 100


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


def _validate_read_strategy(strategy: object) -> str:
    if strategy not in FRAME_READ_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {FRAME_READ_STRATEGIES}, got {strategy!r}"
        )
    return strategy  # type: ignore[return-value]


def _validate_max_gap(max_gap: object) -> int:
    if not isinstance(max_gap, int) or isinstance(max_gap, bool):
        raise TypeError(f"max_gap must be an int, got {type(max_gap).__name__}")
    if max_gap < 0:
        raise ValueError("max_gap must be >= 0")
    return max_gap


def _clean_frame_indices(frame_indices: Iterable[int]) -> list[int]:
    """Sorted unique non-negative ints; anything else is dropped."""
    return sorted(
        {
            frame_index
            for frame_index in frame_indices
            if isinstance(frame_index, int)
            and not isinstance(frame_index, bool)
            and frame_index >= 0
        }
    )


def read_frames_at_indices(
    video_path: str | Path,
    frame_indices: Iterable[int],
    *,
    strategy: str = "auto",
    max_gap: int = DEFAULT_MAX_SEQUENTIAL_GAP,
) -> dict[int, np.ndarray]:
    """Read requested frames with one capture, in ascending index order.

    Duplicate indices are decoded once. Negative and non-integer indices are
    ignored, and frames that cannot be decoded (including indices past the end
    of the video) are omitted from the result.

    ``strategy`` controls how the capture moves between requested frames:

    - ``"seek"``: seek to every requested frame. Each seek decodes forward from
      the nearest keyframe, so it costs roughly the same however far it jumps.
    - ``"sequential"``: seek once to the smallest index, then advance with
      ``grab()`` (no BGR conversion) and decode only requested frames.
    - ``"auto"``: advance with ``grab()`` when at most ``max_gap`` frames lie
      between the current position and the next requested frame; otherwise
      seek. Suits clustered selections (dense bursts separated by long gaps).

    After a failed ``grab()`` or ``read()``, the next requested frame is
    reached by seeking. All strategies return the same frames.
    """
    strategy = _validate_read_strategy(strategy)
    max_gap = _validate_max_gap(max_gap)
    ordered_indices = _clean_frame_indices(frame_indices)
    if not ordered_indices:
        return {}

    cap = cv2.VideoCapture(str(video_path))
    try:
        frames: dict[int, np.ndarray] = {}
        # Index of the frame the next cap.read() returns; None forces a seek.
        position: int | None = None
        for frame_index in ordered_indices:
            gap = None if position is None else frame_index - position
            read_forward = gap is not None and (
                strategy == "sequential"
                or (strategy == "auto" and gap <= max_gap)
            )
            if not read_forward:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                position = frame_index
            while position < frame_index and cap.grab():
                position += 1
            if position < frame_index:
                position = None
                continue
            ok, frame = cap.read()
            if ok:
                frames[frame_index] = frame
                position = frame_index + 1
            else:
                position = None
        return frames
    finally:
        cap.release()


def read_frames_sequentially(
    video_path: str | Path,
    frame_indices: Iterable[int],
) -> dict[int, np.ndarray]:
    """Shorthand for ``read_frames_at_indices(..., strategy="sequential")``."""
    return read_frames_at_indices(video_path, frame_indices, strategy="sequential")


# ---------------------------------------------------------------------------
# Frame sampling and pixel-change selection
# ---------------------------------------------------------------------------


def _validate_sample_every(sample_every: object) -> int:
    if not isinstance(sample_every, int) or isinstance(sample_every, bool):
        raise TypeError(
            f"sample_every must be an int, got {type(sample_every).__name__}"
        )
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1")
    return sample_every


def iter_sampled_frames(
    video: Video | VideoView,
    sample_every: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(original_frame_index, frame)`` for every ``sample_every``-th frame.

    Sampling starts at the first frame of *video* (``start_frame`` for a
    :class:`VideoView`). Indices always refer to the underlying source video,
    never to positions within the view. Arguments are validated eagerly, before
    any frame is decoded.
    """
    if not isinstance(video, (Video, VideoView)):
        raise TypeError(
            f"video must be a Video or VideoView, got {type(video).__name__}"
        )
    _validate_sample_every(sample_every)
    start_index = video.start_frame if isinstance(video, VideoView) else 0
    return _iter_sampled_frames(video, sample_every, start_index)


def _iter_sampled_frames(
    video: Video | VideoView,
    sample_every: int,
    start_index: int,
) -> Iterator[tuple[int, np.ndarray]]:
    for local_index, frame in enumerate(video):
        if local_index % sample_every == 0:
            yield start_index + local_index, frame


def _pixel_diff_gray(frame: np.ndarray) -> np.ndarray:
    """Downscale, grayscale, and blur *frame* for pixel-difference scoring."""
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 1:
        gray = frame[:, :, 0]
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported frame shape for pixel diff: {frame.shape}")
    gray = cv2.resize(gray, _PIXEL_DIFF_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, _PIXEL_DIFF_BLUR, 0)


def _validate_threshold(threshold: object) -> float:
    if not isinstance(threshold, numbers.Real) or isinstance(threshold, bool):
        raise TypeError(
            f"threshold must be a real number, got {type(threshold).__name__}"
        )
    value = float(threshold)
    if not math.isfinite(value) or value < 0:
        raise ValueError("threshold must be a finite number >= 0")
    return value


def _validate_reference(reference: object) -> str:
    if reference not in PIXEL_DIFF_REFERENCES:
        raise ValueError(
            f"reference must be one of {PIXEL_DIFF_REFERENCES}, got {reference!r}"
        )
    return reference  # type: ignore[return-value]


def _validate_metric(metric: object) -> str:
    if metric not in PIXEL_DIFF_METRICS:
        raise ValueError(
            f"metric must be one of {PIXEL_DIFF_METRICS}, got {metric!r}"
        )
    return metric  # type: ignore[return-value]


def _validate_pixel_cutoff(pixel_cutoff: object) -> float:
    if not isinstance(pixel_cutoff, numbers.Real) or isinstance(pixel_cutoff, bool):
        raise TypeError(
            f"pixel_cutoff must be a real number, got {type(pixel_cutoff).__name__}"
        )
    value = float(pixel_cutoff)
    if not math.isfinite(value) or not 0 <= value < 255:
        raise ValueError("pixel_cutoff must be a finite number in [0, 255)")
    return value


def _pixel_diff_score(
    gray: np.ndarray,
    reference_gray: np.ndarray,
    metric: str,
    pixel_cutoff: float,
) -> float:
    difference = cv2.absdiff(gray, reference_gray)
    if metric == "mean":
        return float(np.mean(difference))
    return np.count_nonzero(difference > pixel_cutoff) / difference.size


def _select_by_pixel_change(
    indexed_grays: Iterable[tuple[int, np.ndarray]],
    threshold: float,
    reference: str,
    metric: str = "mean",
    pixel_cutoff: float = DEFAULT_PIXEL_CUTOFF,
) -> list[int]:
    """Select indices whose preprocessed frame differs enough from the reference.

    The first frame is always selected. A frame is selected when its score
    against the reference frame is ``>= threshold``. With ``metric="mean"`` the
    score is the mean absolute pixel difference (0-255). With
    ``"changed_fraction"`` it is the fraction of pixels whose absolute
    difference is ``> pixel_cutoff`` (0-1), so ``threshold`` must be ``<= 1``.

    With ``reference="previous_sample"`` the reference is the previous sampled
    frame; with ``"last_selected"`` it is the most recently selected frame, so
    slow changes accumulate until they cross the threshold.

    Arguments are validated before *indexed_grays* is consumed, so a lazy
    iterator decodes nothing when validation fails.
    """
    threshold = _validate_threshold(threshold)
    reference = _validate_reference(reference)
    metric = _validate_metric(metric)
    pixel_cutoff = _validate_pixel_cutoff(pixel_cutoff)
    if metric == "changed_fraction" and threshold > 1:
        raise ValueError("threshold must be <= 1 for metric='changed_fraction'")

    selected: list[int] = []
    reference_gray: np.ndarray | None = None
    for frame_index, gray in indexed_grays:
        if reference_gray is None:
            selected.append(frame_index)
            reference_gray = gray
            continue
        score = _pixel_diff_score(gray, reference_gray, metric, pixel_cutoff)
        is_selected = score >= threshold
        if is_selected:
            selected.append(frame_index)
        if reference == "previous_sample" or is_selected:
            reference_gray = gray
    return selected


def select_frames_by_pixel_change(
    video: Video | VideoView,
    threshold: float = 15.0,
    sample_every: int = 5,
    reference: str = "previous_sample",
    metric: str = "mean",
    pixel_cutoff: float = DEFAULT_PIXEL_CUTOFF,
) -> list[int]:
    """Select original frame indices where the picture changes noticeably.

    Every ``sample_every``-th frame is downscaled to 320x180, converted to
    grayscale, blurred, and compared with a reference frame (see
    :func:`_select_by_pixel_change` for ``reference`` and ``metric``
    semantics). The first sampled frame is always included, and
    ``threshold=0`` keeps every sampled frame.

    - ``metric="mean"``: ``threshold`` is a mean absolute difference (0-255).
    - ``metric="changed_fraction"``: ``threshold`` is the fraction (0-1) of
      pixels whose difference exceeds ``pixel_cutoff``; e.g. ``0.02`` selects a
      frame once 2% of pixels change by more than ``pixel_cutoff`` gray levels.

    Returned indices are sorted, unique, and relative to the original video,
    even when *video* is a :class:`VideoView`. Pass them to
    :func:`read_frames_at_indices` to retrieve full-resolution frames.

    This is an experimental baseline: every frame is still decoded, and
    successive-frame comparison (the default) can miss slow changes.
    """
    sampled = iter_sampled_frames(video, sample_every)
    grays = (
        (frame_index, _pixel_diff_gray(frame)) for frame_index, frame in sampled
    )
    return _select_by_pixel_change(grays, threshold, reference, metric, pixel_cutoff)


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
