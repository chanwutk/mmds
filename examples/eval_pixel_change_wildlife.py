"""Evaluate pixel-change frame selection on stationary wildlife footage.

Compares three experimental frame selectors against hand-labeled change
intervals:

- pixel change: ``select_frames_by_pixel_change`` semantics, swept over
  thresholds, both reference modes, and both metrics (mean absolute difference
  and changed-pixel fraction)
- background subtraction: OpenCV MOG2 foreground fraction
- YOLOE: frames with detections, and frames where per-class counts change

Every selector sees the same sampled frames (``--sample-every``) and returns
original-video frame indices. Costs are timed separately: decoding the clip,
selecting frames (excluding decoding), and re-reading the selected frames with
``read_frames_at_indices``.

Usage::

    PYTHONPATH=src:. ./.venv/bin/python examples/eval_pixel_change_wildlife.py
    PYTHONPATH=src:. ./.venv/bin/python examples/eval_pixel_change_wildlife.py \\
        --video /path/to/local.mp4 --output results.json
    # Thumbnail grids (1 per second) for drafting or checking the labels
    PYTHONPATH=src:. ./.venv/bin/python examples/eval_pixel_change_wildlife.py \\
        --contact-sheets /tmp/sheets
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mmds.utilities.video import (  # noqa: E402
    DEFAULT_PIXEL_CUTOFF,
    FRAME_READ_STRATEGIES,
    PIXEL_DIFF_REFERENCES,
    Video,
    VideoView,
    _pixel_diff_gray,
    _select_by_pixel_change,
    iter_sampled_frames,
    open_video,
    read_frames_at_indices,
)

VIDEO_URL = "https://www.youtube.com/watch?v=VMNReumMHeE"
CLIP_START = 13 * 60 + 49.0
DEFAULT_LABELS = ROOT / "data" / "wildlife_pixel_change_ground_truth.json"
DEFAULT_SAMPLE_EVERY = 5
PIXEL_THRESHOLDS = (2.0, 4.0, 8.0, 15.0)
CHANGED_FRACTION_THRESHOLDS = (0.002, 0.005, 0.01, 0.02)
FOREGROUND_FRACTIONS = (0.002, 0.005, 0.01, 0.02)
DEFAULT_YOLOE_MODEL = "yoloe-11s-seg.pt"
DEFAULT_YOLOE_CLASSES = ("animal", "cat", "possum", "fox", "bird")


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A labeled ``[start, end)`` span in absolute source-video seconds."""

    start: float
    end: float
    description: str = ""

    def contains(self, seconds: float) -> bool:
        return self.start <= seconds < self.end


@dataclass(frozen=True)
class Labels:
    video: str
    clip_start: float
    clip_end: float | None
    events: tuple[Interval, ...]
    static: tuple[Interval, ...]


def _finite_seconds(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{what} must be a finite number >= 0, got {value!r}")
    return seconds


def _parse_intervals(raw: Any, key: str) -> tuple[Interval, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"labels[{key!r}] must be a list")
    intervals: list[Interval] = []
    for position, item in enumerate(raw):
        where = f"labels[{key!r}][{position}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be an object")
        start = _finite_seconds(item.get("start"), f"{where}.start")
        end = _finite_seconds(item.get("end"), f"{where}.end")
        if start >= end:
            raise ValueError(f"{where} must have start < end")
        description = item.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"{where}.description must be a string")
        intervals.append(Interval(start, end, description))
    for previous, current in zip(intervals, intervals[1:]):
        if current.start < previous.end:
            raise ValueError(
                f"labels[{key!r}] must be sorted and non-overlapping "
                f"({previous.start}-{previous.end} vs {current.start}-{current.end})"
            )
    return tuple(intervals)


def _check_within(labels: Labels, clip_start: float, clip_end: float | None) -> None:
    for interval in labels.events + labels.static:
        if interval.start < clip_start or (
            clip_end is not None and interval.end > clip_end
        ):
            raise ValueError(
                f"interval {interval.start}-{interval.end} lies outside the clip "
                f"[{clip_start}, {clip_end}]"
            )


def load_labels(path: str | Path) -> Labels:
    """Load and validate hand-labeled change intervals."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("labels file must contain a JSON object")
    video = raw.get("video")
    if not isinstance(video, str) or not video:
        raise ValueError("labels['video'] must be a non-empty string")
    clip_start = _finite_seconds(raw.get("clip_start"), "labels['clip_start']")
    clip_end = raw.get("clip_end")
    if clip_end is not None:
        clip_end = _finite_seconds(clip_end, "labels['clip_end']")
        if clip_end <= clip_start:
            raise ValueError("labels['clip_end'] must be greater than clip_start")
    events = _parse_intervals(raw.get("events"), "events")
    if not events:
        raise ValueError("labels['events'] must contain at least one interval")
    static = _parse_intervals(raw.get("static", []), "static")
    for event in events:
        for quiet in static:
            if event.start < quiet.end and quiet.start < event.end:
                raise ValueError(
                    f"event {event.start}-{event.end} overlaps static interval "
                    f"{quiet.start}-{quiet.end}"
                )
    labels = Labels(video, clip_start, clip_end, events, static)
    _check_within(labels, clip_start, clip_end)
    return labels


def validate_labels_against_clip(
    labels: Labels, clip_start: float, clip_end: float
) -> None:
    """Fail if any labeled interval falls outside the clip actually decoded."""
    _check_within(labels, clip_start, clip_end)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_selection(
    selected: Sequence[int],
    *,
    sampled_count: int,
    fps: float,
    labels: Labels,
) -> dict[str, Any]:
    """Score selected frame indices against labeled intervals.

    - ``event_recall``: share of events with at least one selected frame inside
    - ``latency_seconds``: per event, first selected timestamp minus event start
      (``None`` when missed)
    - ``static_selections``: selected frames inside labeled static intervals,
      excluding the first selection, which every selector is forced to make
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if sampled_count < 0:
        raise ValueError("sampled_count must be >= 0")
    seconds = [index / fps for index in selected]

    latencies: list[float | None] = []
    missed: list[str] = []
    for event in labels.events:
        inside = [t for t in seconds if event.contains(t)]
        if inside:
            latencies.append(round(min(inside) - event.start, 3))
        else:
            latencies.append(None)
            missed.append(event.description or f"{event.start}-{event.end}")

    static_selections = sum(
        1 for t in seconds[1:] if any(quiet.contains(t) for quiet in labels.static)
    )
    static_minutes = sum(q.end - q.start for q in labels.static) / 60.0
    return {
        "selected": len(selected),
        "sampled": sampled_count,
        "kept_fraction": (len(selected) / sampled_count) if sampled_count else 0.0,
        "event_recall": (len(labels.events) - len(missed)) / len(labels.events),
        "missed_events": missed,
        "latency_seconds": latencies,
        "static_selections": static_selections,
        "static_selections_per_minute": (
            static_selections / static_minutes if static_minutes else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Baseline selectors
# ---------------------------------------------------------------------------


def select_by_background_subtraction(
    indexed_grays: Iterable[tuple[int, np.ndarray]],
    min_foreground_fraction: float,
    *,
    history: int = 500,
    var_threshold: float = 16.0,
) -> list[int]:
    """Select frames whose MOG2 foreground covers ``>= min_foreground_fraction``.

    The first frame is always selected (matching the pixel-change selector).
    Masks are opened with a 3x3 kernel to drop isolated noise pixels.
    """
    if isinstance(min_foreground_fraction, bool) or not (
        0.0 <= float(min_foreground_fraction) <= 1.0
    ):
        raise ValueError("min_foreground_fraction must be within [0, 1]")
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=False
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    selected: list[int] = []
    for position, (frame_index, gray) in enumerate(indexed_grays):
        mask = subtractor.apply(gray)
        if position == 0:
            selected.append(frame_index)
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if np.count_nonzero(mask) / mask.size >= min_foreground_fraction:
            selected.append(frame_index)
    return selected


DetectFn = Callable[..., list[dict[str, Any]]]


def _default_detect(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    from mmds.execution.ops.detect import _detect_in_video

    return _detect_in_video(*args, **kwargs)


def select_by_yoloe(
    video: Video | VideoView,
    sampled_indices: Sequence[int],
    classes: Sequence[str],
    *,
    sample_every: int,
    model: str = DEFAULT_YOLOE_MODEL,
    conf: float | None = None,
    detect_fn: DetectFn | None = None,
) -> dict[str, list[int]]:
    """Run YOLOE on the sampled frames and derive two selections.

    - ``presence``: frames with at least one detection
    - ``count_change``: frames whose per-class detection counts differ from the
      previous sampled frame

    Both always include the first sampled frame.
    """
    if not sampled_indices:
        return {"presence": [], "count_change": []}
    detect = detect_fn or _default_detect
    detections = detect(
        video, list(classes), model, frame_stride=sample_every, conf=conf
    )
    sampled = set(sampled_indices)
    counts: dict[int, Counter[str]] = {}
    for group in detections:
        for box in group["bboxes"]:
            frame_index = box["frame_idx"]
            if frame_index not in sampled:
                raise ValueError(
                    f"YOLOE returned frame {frame_index}, which was not sampled"
                )
            counts.setdefault(frame_index, Counter())[group["type"]] += 1

    first = sampled_indices[0]
    presence = [i for i in sampled_indices if i == first or counts.get(i)]
    count_change: list[int] = []
    previous: Counter[str] | None = None
    for frame_index in sampled_indices:
        current = counts.get(frame_index, Counter())
        if previous is None or current != previous:
            count_change.append(frame_index)
        previous = current
    return {"presence": presence, "count_change": count_change}


# ---------------------------------------------------------------------------
# Timing and decoding
# ---------------------------------------------------------------------------


def median_time(fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    """Run *fn* ``repeats`` times; return (median seconds, last result)."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be an int >= 1")
    durations: list[float] = []
    result: Any = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations), result


def decode_only(video: Video | VideoView) -> int:
    """Decode every frame and discard it; return the number decoded."""
    return sum(1 for _ in video)


def preprocess_sampled(
    video: Video | VideoView, sample_every: int
) -> list[tuple[int, np.ndarray]]:
    return [
        (frame_index, _pixel_diff_gray(frame))
        for frame_index, frame in iter_sampled_frames(video, sample_every)
    ]


# ---------------------------------------------------------------------------
# Contact sheets
# ---------------------------------------------------------------------------


def write_contact_sheets(
    video: Video | VideoView,
    out_dir: str | Path,
    *,
    every_seconds: float = 1.0,
    columns: int = 6,
    rows: int = 5,
    thumb_width: int = 320,
) -> list[Path]:
    """Write timestamped thumbnail grids, one thumbnail per ``every_seconds``."""
    if every_seconds <= 0 or columns < 1 or rows < 1 or thumb_width < 1:
        raise ValueError("every_seconds, columns, rows and thumb_width must be > 0")
    fps = video.fps or 30.0
    step = max(1, round(fps * every_seconds))
    thumb_height = max(1, round(thumb_width * video.height / max(video.width, 1)))
    blank = np.zeros((thumb_height, thumb_width, 3), dtype=np.uint8)

    thumbs: list[np.ndarray] = []
    for frame_index, frame in iter_sampled_frames(video, step):
        thumb = cv2.resize(frame, (thumb_width, thumb_height))
        label = f"{frame_index / fps:.1f}s f{frame_index}"
        cv2.putText(thumb, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(thumb, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_sheet = columns * rows
    paths: list[Path] = []
    for sheet_number, offset in enumerate(range(0, len(thumbs), per_sheet)):
        chunk = thumbs[offset : offset + per_sheet]
        chunk += [blank] * (per_sheet - len(chunk))
        grid = np.vstack(
            [np.hstack(chunk[r * columns : (r + 1) * columns]) for r in range(rows)]
        )
        path = out / f"sheet_{sheet_number:03d}.jpg"
        cv2.imwrite(str(path), grid)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    view: VideoView,
    labels: Labels,
    *,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    thresholds: Sequence[float] = PIXEL_THRESHOLDS,
    changed_fraction_thresholds: Sequence[float] = CHANGED_FRACTION_THRESHOLDS,
    pixel_cutoff: float = DEFAULT_PIXEL_CUTOFF,
    foreground_fractions: Sequence[float] = FOREGROUND_FRACTIONS,
    yoloe_classes: Sequence[str] | None = DEFAULT_YOLOE_CLASSES,
    yoloe_model: str = DEFAULT_YOLOE_MODEL,
    yoloe_conf: float | None = None,
    repeats: int = 3,
    read_strategies: Sequence[str] = FRAME_READ_STRATEGIES,
    detect_fn: DetectFn | None = None,
) -> dict[str, Any]:
    """Run every selector on *view* and return timings plus scores.

    ``selection_seconds`` excludes decoding: it is the selector's total time
    minus the decode-only time of the same clip. ``read_seconds`` maps each of
    ``read_strategies`` to the time ``read_frames_at_indices`` takes to fetch
    the selected frames. Pass ``yoloe_classes=None`` to skip YOLOE.
    """
    if not read_strategies:
        raise ValueError("read_strategies must not be empty")
    for strategy in read_strategies:
        if strategy not in FRAME_READ_STRATEGIES:
            raise ValueError(
                f"read strategy must be one of {FRAME_READ_STRATEGIES}, got {strategy!r}"
            )
    fps = view.fps
    if not fps or fps <= 0:
        raise ValueError("video fps must be > 0")
    validate_labels_against_clip(
        labels, view.start_frame / fps, view.end_frame / fps
    )

    decode_seconds, decoded_frames = median_time(lambda: decode_only(view), repeats)
    preprocess_seconds, cached = median_time(
        lambda: preprocess_sampled(view, sample_every), repeats
    )
    sampled_indices = [frame_index for frame_index, _ in cached]
    pixel_overhead = max(0.0, preprocess_seconds - decode_seconds)

    runs: list[dict[str, Any]] = []

    def add(method: str, params: dict[str, Any], selected: list[int],
            selection_seconds: float) -> None:
        read_seconds = {
            strategy: median_time(
                lambda s=strategy: read_frames_at_indices(
                    view.path, selected, strategy=s
                ),
                repeats,
            )[0]
            for strategy in read_strategies
        }
        runs.append(
            {
                "method": method,
                "params": params,
                "selection_seconds": selection_seconds,
                "read_seconds": read_seconds,
                "selected_indices": selected,
                **score_selection(
                    selected, sampled_count=len(sampled_indices), fps=fps,
                    labels=labels,
                ),
            }
        )

    pixel_sweeps: list[tuple[str, Sequence[float], dict[str, Any]]] = [
        ("mean", thresholds, {}),
        ("changed_fraction", changed_fraction_thresholds,
         {"pixel_cutoff": pixel_cutoff}),
    ]
    for metric, metric_thresholds, extra_params in pixel_sweeps:
        for reference in PIXEL_DIFF_REFERENCES:
            for threshold in metric_thresholds:
                seconds, selected = median_time(
                    lambda t=threshold, r=reference, m=metric: _select_by_pixel_change(
                        cached, t, r, m, pixel_cutoff
                    ),
                    repeats,
                )
                params = {"metric": metric, "reference": reference,
                          "threshold": threshold, **extra_params}
                add("pixel_change", params, selected, pixel_overhead + seconds)

    for fraction in foreground_fractions:
        seconds, selected = median_time(
            lambda f=fraction: select_by_background_subtraction(cached, f), repeats
        )
        add("background_subtraction", {"min_foreground_fraction": fraction},
            selected, pixel_overhead + seconds)

    if yoloe_classes:
        seconds, yoloe = median_time(
            lambda: select_by_yoloe(
                view, sampled_indices, yoloe_classes, sample_every=sample_every,
                model=yoloe_model, conf=yoloe_conf, detect_fn=detect_fn,
            ),
            repeats,
        )
        yoloe_seconds = max(0.0, seconds - decode_seconds)
        for variant in ("presence", "count_change"):
            add(f"yoloe_{variant}",
                {"classes": list(yoloe_classes), "conf": yoloe_conf},
                yoloe[variant], yoloe_seconds)

    return {
        "video": str(view.path),
        "fps": fps,
        "start_frame": view.start_frame,
        "end_frame": view.end_frame,
        "sample_every": sample_every,
        "repeats": repeats,
        "read_strategies": list(read_strategies),
        "decoded_frames": decoded_frames,
        "sampled_frames": len(sampled_indices),
        "decode_seconds": decode_seconds,
        "runs": runs,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Video: {report['video']}",
        f"Frames {report['start_frame']}-{report['end_frame']} @ {report['fps']:.2f} fps; "
        f"decoded {report['decoded_frames']}, sampled {report['sampled_frames']} "
        f"(every {report['sample_every']})",
        f"Decode-only: {report['decode_seconds']:.2f}s (median of {report['repeats']})",
        "",
        f"{'method':<24}{'params':<78}{'kept':>6}{'kept%':>7}{'recall':>8}"
        f"{'static/min':>11}{'select s':>10}"
        + "".join(f"{'read ' + s + ' s':>18}" for s in report["read_strategies"]),
    ]
    for run in report["runs"]:
        params = ", ".join(
            f"{k}={v}" for k, v in run["params"].items() if k != "classes"
        )
        lines.append(
            f"{run['method']:<24}{params:<78}{run['selected']:>6}"
            f"{100 * run['kept_fraction']:>6.1f}%{run['event_recall']:>8.2f}"
            f"{run['static_selections_per_minute']:>11.2f}"
            f"{run['selection_seconds']:>10.2f}"
            + "".join(
                f"{run['read_seconds'][s]:>18.2f}" for s in report["read_strategies"]
            )
        )
        if run["missed_events"]:
            lines.append(f"{'':<24}missed: {'; '.join(run['missed_events'])}")
    return "\n".join(lines)


def open_clip(
    video_source: str, clip_start: float, clip_end: float | None
) -> VideoView:
    """Open *video_source* as a view from *clip_start* to *clip_end* (or the end)."""
    video = open_video(video_source)
    if not isinstance(video, Video):
        raise ValueError(f"{video_source!r} must be a single video file")
    if not video.fps or video.fps <= 0:
        raise ValueError(f"{video_source!r} reports no frame rate")
    if clip_end is None:
        clip_end = video.num_frames / video.fps
    view = VideoView(video, clip_start, clip_end)
    if view.num_frames == 0:
        raise ValueError(
            f"clip [{clip_start}, {clip_end}] of {video_source!r} contains no frames"
        )
    return view


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--video", help="local path overriding the labels' video URL")
    parser.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--pixel-cutoff",
        type=float,
        default=DEFAULT_PIXEL_CUTOFF,
        help="gray-level cutoff for metric='changed_fraction' (default: %(default)s)",
    )
    parser.add_argument(
        "--read-strategies",
        default=",".join(FRAME_READ_STRATEGIES),
        help="comma-separated read_frames_at_indices strategies to time "
        "(default: %(default)s)",
    )
    parser.add_argument("--yoloe-classes", default=",".join(DEFAULT_YOLOE_CLASSES))
    parser.add_argument("--yoloe-model", default=DEFAULT_YOLOE_MODEL)
    parser.add_argument("--yoloe-conf", type=float, default=None)
    parser.add_argument("--skip-yoloe", action="store_true")
    parser.add_argument("--output", help="write the full JSON report here")
    parser.add_argument(
        "--contact-sheets",
        help="write 1-per-second thumbnail grids of the default clip here and exit "
        "(does not read --labels)",
    )
    args = parser.parse_args(argv)

    if args.contact_sheets:
        view = open_clip(args.video or VIDEO_URL, CLIP_START, None)
        for path in write_contact_sheets(view, args.contact_sheets):
            print(path)
        return 0

    labels = load_labels(args.labels)
    view = open_clip(args.video or labels.video, labels.clip_start, labels.clip_end)

    classes = None if args.skip_yoloe else [
        c.strip() for c in args.yoloe_classes.split(",") if c.strip()
    ]
    report = evaluate(
        view,
        labels,
        sample_every=args.sample_every,
        pixel_cutoff=args.pixel_cutoff,
        read_strategies=[
            s.strip() for s in args.read_strategies.split(",") if s.strip()
        ],
        yoloe_classes=classes,
        yoloe_model=args.yoloe_model,
        yoloe_conf=args.yoloe_conf,
        repeats=args.repeats,
    )
    print(format_report(report))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
