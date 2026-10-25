"""Motion-guided frame pruning for a cat-counting query on trail-camera footage.

Query: "How many cats are visible at the same time, and when does the second
cat appear?"

Original plan::

    Video -> View (uniform samples) -> Map (MLLM counts cats per frame)
          -> Reduce (max simultaneous count, first timestamp with >= 2 cats)

Optimized plan::

    Video -> View (samples) -> PixelChangeFilter (changed_fraction, last_selected)
          -> Map -> Reduce

Refined plan: the optimized plan, then the MLLM re-checks only the sampled
frames between the previous kept frame and the first kept frame with >= 2
cats, so the timestamp can move earlier.

``PixelChangeFilter`` is a physical step, not a logical VRA operator; these
plans are plain Python over ``mmds.utilities.video`` and are not wired into the
DSL or optimizer.

Every plan uses the same sampled frames, so the optimized plan's frames are a
subset of the original plan's. With a consistent MLLM its "first frame with
>= 2 cats" is therefore never earlier than the original plan's, which is never
earlier than the true entrance. The refined plan recovers the original plan's
timestamp whenever that frame lies in the gap just before the first kept
two-cat frame.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mmds.model import PromptSpec, ResolvedPrompt  # noqa: E402
from mmds.utilities.video import (  # noqa: E402
    DEFAULT_PIXEL_CUTOFF,
    VideoView,
    _validate_sample_every,
    read_frames_at_indices,
    select_frames_by_pixel_change,
)

QUESTION = (
    "How many cats are visible at the same time, and when does the second cat appear?"
)
COUNT_PROMPT = (
    "This is a single frame from a stationary night-vision trail camera. "
    "Count the cats visible in the frame, including cats that are only partly "
    "visible at the edges. Return 0 if there are none."
)
COUNT_SCHEMA = {"cat_count": "integer"}
DEFAULT_SAMPLE_EVERY = 5
DEFAULT_THRESHOLD = 0.01
MIN_COUNT = 2
# 64 decoded 720p BGR frames hold about 177 MB.
FRAME_BATCH_SIZE = 64

FrameCounter = Callable[[int, np.ndarray], int]


# ---------------------------------------------------------------------------
# Plan steps
# ---------------------------------------------------------------------------


def uniform_view(view: VideoView, sample_every: int) -> list[int]:
    """Original frame indices of every ``sample_every``-th frame of *view*."""
    _validate_sample_every(sample_every)
    return list(range(view.start_frame, view.end_frame, sample_every))


def pixel_change_filter(
    view: VideoView,
    *,
    sample_every: int,
    threshold: float = DEFAULT_THRESHOLD,
    pixel_cutoff: float = DEFAULT_PIXEL_CUTOFF,
) -> list[int]:
    """Keep sampled frames whose changed-pixel fraction vs the last kept frame
    is ``>= threshold``."""
    return select_frames_by_pixel_change(
        view,
        threshold=threshold,
        sample_every=sample_every,
        reference="last_selected",
        metric="changed_fraction",
        pixel_cutoff=pixel_cutoff,
    )


def _validate_count(frame_index: int, count: Any) -> int:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(
            f"frame counter returned {count!r} for frame {frame_index}; "
            "expected an int >= 0"
        )
    return count


def map_count(
    video_path: str | Path,
    frame_indices: Iterable[int],
    counter: FrameCounter,
    *,
    stop_at_count: int | None = None,
    batch_size: int = FRAME_BATCH_SIZE,
    timings: dict[str, float] | None = None,
) -> dict[int, int]:
    """Count cats in each frame, in ascending index order.

    Frames are read in batches of ``batch_size`` to bound memory. Frames that
    cannot be decoded are skipped. With ``stop_at_count``, counting stops after
    the first frame whose count is ``>= stop_at_count``. Read and counter time
    are added to ``timings["read"]`` and ``timings["mllm"]``.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be an int >= 1")
    if stop_at_count is not None and (
        isinstance(stop_at_count, bool) or not isinstance(stop_at_count, int)
        or stop_at_count < 1
    ):
        raise ValueError("stop_at_count must be an int >= 1 or None")
    if timings is None:
        timings = {}
    timings.setdefault("read", 0.0)
    timings.setdefault("mllm", 0.0)

    ordered = sorted(set(frame_indices))
    counts: dict[int, int] = {}
    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset : offset + batch_size]
        started = time.perf_counter()
        frames = read_frames_at_indices(video_path, batch)
        timings["read"] += time.perf_counter() - started
        for frame_index in batch:
            if frame_index not in frames:
                continue
            started = time.perf_counter()
            count = _validate_count(frame_index, counter(frame_index, frames[frame_index]))
            timings["mllm"] += time.perf_counter() - started
            counts[frame_index] = count
            if stop_at_count is not None and count >= stop_at_count:
                return counts
    return counts


@dataclass(frozen=True)
class Answer:
    max_simultaneous: int | None
    first_index: int | None
    fps: float

    @property
    def first_seconds(self) -> float | None:
        return None if self.first_index is None else self.first_index / self.fps


def reduce_answer(
    counts: Mapping[int, int], fps: float, min_count: int = MIN_COUNT
) -> Answer:
    """Max count over all frames, and the earliest frame with ``>= min_count``."""
    if not fps or fps <= 0:
        raise ValueError("fps must be > 0")
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise ValueError("min_count must be an int >= 1")
    if not counts:
        return Answer(None, None, fps)
    hits = [i for i, count in counts.items() if count >= min_count]
    return Answer(max(counts.values()), min(hits) if hits else None, fps)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@dataclass
class PlanResult:
    name: str
    answer: Answer
    sampled: int
    kept_indices: list[int]
    counts: dict[int, int]
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def frames_sent(self) -> int:
        """Frames passed to the counter (one MLLM request each, before caching)."""
        return len(self.counts)


def run_original(
    view: VideoView, counter: FrameCounter, *, sample_every: int = DEFAULT_SAMPLE_EVERY
) -> PlanResult:
    sampled = uniform_view(view, sample_every)
    seconds: dict[str, float] = {"select": 0.0}
    counts = map_count(view.path, sampled, counter, timings=seconds)
    return PlanResult(
        "original (uniform)", reduce_answer(counts, view.fps), len(sampled),
        sampled, counts, seconds,
    )


def run_pruned(
    view: VideoView,
    counter: FrameCounter,
    *,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    threshold: float = DEFAULT_THRESHOLD,
    pixel_cutoff: float = DEFAULT_PIXEL_CUTOFF,
) -> PlanResult:
    started = time.perf_counter()
    kept = pixel_change_filter(
        view, sample_every=sample_every, threshold=threshold, pixel_cutoff=pixel_cutoff
    )
    seconds = {"select": time.perf_counter() - started}
    counts = map_count(view.path, kept, counter, timings=seconds)
    return PlanResult(
        "optimized (pixel-change)", reduce_answer(counts, view.fps),
        len(uniform_view(view, sample_every)), kept, counts, seconds,
    )


def refine_first_occurrence(
    view: VideoView,
    pruned: PlanResult,
    counter: FrameCounter,
    *,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    min_count: int = MIN_COUNT,
) -> PlanResult:
    """Re-check sampled frames just before the first kept frame with >= min_count.

    Candidates are the sampled frames strictly between the previous kept frame
    (or the view start) and the first kept hit, scanned in order until one has
    ``>= min_count``. If the pruned plan found no hit there is nothing to
    localize and the result equals the pruned plan's.
    """
    first = pruned.answer.first_index
    seconds = {"select": pruned.seconds.get("select", 0.0),
               "read": pruned.seconds.get("read", 0.0),
               "mllm": pruned.seconds.get("mllm", 0.0)}
    counts = dict(pruned.counts)
    if first is not None:
        earlier_kept = [i for i in pruned.kept_indices if i < first]
        previous = max(earlier_kept) if earlier_kept else None
        candidates = [
            i for i in uniform_view(view, sample_every)
            if i < first and (previous is None or i > previous) and i not in counts
        ]
        counts.update(
            map_count(view.path, candidates, counter, stop_at_count=min_count,
                      timings=seconds)
        )
    return PlanResult(
        "optimized + refinement", reduce_answer(counts, view.fps, min_count),
        pruned.sampled, pruned.kept_indices, counts, seconds,
    )


# ---------------------------------------------------------------------------
# Gemini frame counter
# ---------------------------------------------------------------------------


class GeminiFrameCounter:
    """Count cats in one frame with Gemini, caching answers on disk.

    Cache keys combine ``video_id``, model, and frame index, so cached answers
    are reused across runs and plans. Transient API errors are retried with
    exponential backoff.
    """

    def __init__(
        self,
        video_id: str,
        *,
        executor: Any | None = None,
        model: str | None = None,
        cache_path: str | Path | None = None,
        jpeg_quality: int = 90,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not video_id:
            raise ValueError("video_id must be a non-empty string")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if executor is None:
            from mmds import GeminiPromptExecutor

            executor = (
                GeminiPromptExecutor(model=model) if model else GeminiPromptExecutor()
            )
        self.executor = executor
        self.model = model or getattr(executor, "model", "unknown")
        self.video_id = video_id
        self.cache_path = Path(cache_path) if cache_path else None
        self.jpeg_quality = jpeg_quality
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self.api_calls = 0
        self.cache_hits = 0
        self.api_seconds = 0.0
        self._cache: dict[str, int] = {}
        if self.cache_path and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())

    def _key(self, frame_index: int) -> str:
        return f"{self.video_id}|{self.model}|{frame_index}"

    def __call__(self, frame_index: int, frame: np.ndarray) -> int:
        key = self._key(frame_index)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise ValueError(f"could not JPEG-encode frame {frame_index}")
        # The executor only forwards inline bytes for video-typed values; Gemini
        # accepts any inline mime type, including image/jpeg.
        image = {"type": "Video", "bytes": encoded.tobytes(), "mime_type": "image/jpeg"}
        prompt = PromptSpec(parts=(COUNT_PROMPT,), output_schema=COUNT_SCHEMA)
        resolved = ResolvedPrompt(parts=(COUNT_PROMPT, image), output_schema=COUNT_SCHEMA)

        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                response = self.executor.execute("map", prompt, resolved, None, {})
            except Exception:
                if attempt == self.max_attempts:
                    raise
                self._sleep(self.backoff_seconds * 2 ** (attempt - 1))
                continue
            finally:
                self.api_seconds += time.perf_counter() - started
                self.api_calls += 1
            break

        if not isinstance(response, Mapping) or "cat_count" not in response:
            raise ValueError(f"unexpected Gemini response for frame {frame_index}: {response!r}")
        count = _validate_count(frame_index, response["cat_count"])
        self._cache[key] = count
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, sort_keys=True))
        return count


def run_all(
    view: VideoView,
    counter: FrameCounter,
    *,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    threshold: float = DEFAULT_THRESHOLD,
    pixel_cutoff: float = DEFAULT_PIXEL_CUTOFF,
) -> list[PlanResult]:
    """Run the original, optimized, and refined plans on *view*."""
    original = run_original(view, counter, sample_every=sample_every)
    pruned = run_pruned(
        view, counter, sample_every=sample_every, threshold=threshold,
        pixel_cutoff=pixel_cutoff,
    )
    refined = refine_first_occurrence(view, pruned, counter, sample_every=sample_every)
    return [original, pruned, refined]


def describe(result: PlanResult) -> str:
    answer = result.answer
    when = "never" if answer.first_seconds is None else f"{answer.first_seconds:.1f}s"
    return (
        f"{result.name}: max {answer.max_simultaneous} cats at once; "
        f"second cat first seen at {when}; {result.frames_sent} frames sent"
    )
