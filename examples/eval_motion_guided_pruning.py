"""Evaluate motion-guided frame pruning against uniform sampling.

Runs the three plans from ``examples/motion_guided_pruning.py`` on the
13:49-to-end trail-camera clip and scores each answer to "How many cats are
visible at the same time, and when does the second cat appear?" against
``query_answers`` in ``data/wildlife_pixel_change_ground_truth.json``:

- whether the maximum simultaneous count is correct
- timestamp error of the second cat (positive = later than the truth)
- frames sent to the MLLM, and the reduction versus the original plan
- whether the answer matches the original plan's answer

Gemini answers are cached per (video, model, frame), so reruns make no API
calls. Requires ``GEMINI_API_KEY`` on the first run.

Usage::

    PYTHONPATH=src:. ./.venv/bin/python examples/eval_motion_guided_pruning.py
    PYTHONPATH=src:. ./.venv/bin/python examples/eval_motion_guided_pruning.py \\
        --threshold 0.02 --output /tmp/pruning.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src", ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import eval_pixel_change_wildlife as wildlife  # noqa: E402
import motion_guided_pruning as mgp  # noqa: E402

from mmds.utilities.video import DEFAULT_PIXEL_CUTOFF, VideoView  # noqa: E402

DEFAULT_CACHE = Path.home() / ".cache" / "mmds" / "motion_guided_pruning" / "cat_counts.json"


@dataclass(frozen=True)
class QueryAnswers:
    max_simultaneous: int
    first_seconds: float


def load_query_answers(path: str | Path) -> QueryAnswers:
    """Read and validate ``query_answers`` from a labels file."""
    labels = wildlife.load_labels(path)
    raw = json.loads(Path(path).read_text()).get("query_answers")
    if not isinstance(raw, dict):
        raise ValueError("labels['query_answers'] must be an object")
    count = raw.get("max_simultaneous_cats")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("query_answers.max_simultaneous_cats must be an int >= 0")
    seconds = raw.get("second_cat_first_visible")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not (
        math.isfinite(seconds)
    ):
        raise ValueError("query_answers.second_cat_first_visible must be a finite number")
    if seconds < labels.clip_start or (
        labels.clip_end is not None and seconds >= labels.clip_end
    ):
        raise ValueError("query_answers.second_cat_first_visible lies outside the clip")
    return QueryAnswers(count, float(seconds))


def score_plan(
    result: mgp.PlanResult, truth: QueryAnswers, original: mgp.PlanResult
) -> dict[str, Any]:
    answer = result.answer
    error = (
        None if answer.first_seconds is None
        else round(answer.first_seconds - truth.first_seconds, 3)
    )
    return {
        "plan": result.name,
        "max_simultaneous": answer.max_simultaneous,
        "second_cat_seconds": answer.first_seconds,
        "max_correct": answer.max_simultaneous == truth.max_simultaneous,
        "timestamp_error_seconds": error,
        "frames_sent": result.frames_sent,
        "sampled": result.sampled,
        "reduction_vs_original": (
            1 - result.frames_sent / original.frames_sent if original.frames_sent else 0.0
        ),
        "matches_original_max": answer.max_simultaneous == original.answer.max_simultaneous,
        "matches_original_timestamp": answer.first_index == original.answer.first_index,
        "seconds": dict(result.seconds),
    }


def evaluate(
    view: VideoView,
    truth: QueryAnswers,
    counter: mgp.FrameCounter,
    *,
    sample_every: int = mgp.DEFAULT_SAMPLE_EVERY,
    threshold: float = mgp.DEFAULT_THRESHOLD,
    pixel_cutoff: float = DEFAULT_PIXEL_CUTOFF,
) -> dict[str, Any]:
    results = mgp.run_all(
        view, counter, sample_every=sample_every, threshold=threshold,
        pixel_cutoff=pixel_cutoff,
    )
    original = results[0]
    return {
        "question": mgp.QUESTION,
        "video": str(view.path),
        "start_frame": view.start_frame,
        "end_frame": view.end_frame,
        "fps": view.fps,
        "sample_every": sample_every,
        "threshold": threshold,
        "pixel_cutoff": pixel_cutoff,
        "truth": {
            "max_simultaneous": truth.max_simultaneous,
            "second_cat_seconds": truth.first_seconds,
        },
        "plans": [score_plan(result, truth, original) for result in results],
        "counts": {result.name: result.counts for result in results},
    }


def format_report(report: dict[str, Any]) -> str:
    truth = report["truth"]
    lines = [
        f"Q: {report['question']}",
        f"Truth: max {truth['max_simultaneous']} cats; second cat at "
        f"{truth['second_cat_seconds']:.1f}s",
        f"Sampling every {report['sample_every']} frames; pixel filter threshold "
        f"{report['threshold']} (cutoff {report['pixel_cutoff']})",
        "",
        f"{'plan':<26}{'max':>5}{'ok':>5}{'2nd cat s':>11}{'error s':>9}"
        f"{'frames':>8}{'saved':>8}{'same max':>10}{'same time':>11}",
    ]
    for plan in report["plans"]:
        when = plan["second_cat_seconds"]
        error = plan["timestamp_error_seconds"]
        lines.append(
            f"{plan['plan']:<26}{str(plan['max_simultaneous']):>5}"
            f"{'yes' if plan['max_correct'] else 'no':>5}"
            f"{'never' if when is None else f'{when:.1f}':>11}"
            f"{'-' if error is None else f'{error:+.1f}':>9}"
            f"{plan['frames_sent']:>8}{100 * plan['reduction_vs_original']:>7.1f}%"
            f"{'yes' if plan['matches_original_max'] else 'no':>10}"
            f"{'yes' if plan['matches_original_timestamp'] else 'no':>11}"
        )
    return "\n".join(lines)


def make_counter(
    video_id: str, *, model: str | None, cache_path: Path
) -> mgp.GeminiFrameCounter:
    return mgp.GeminiFrameCounter(video_id, model=model, cache_path=cache_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default=str(wildlife.DEFAULT_LABELS))
    parser.add_argument("--video", help="local path overriding the labels' video URL")
    parser.add_argument("--sample-every", type=int, default=mgp.DEFAULT_SAMPLE_EVERY)
    parser.add_argument("--threshold", type=float, default=mgp.DEFAULT_THRESHOLD)
    parser.add_argument("--pixel-cutoff", type=float, default=DEFAULT_PIXEL_CUTOFF)
    parser.add_argument("--model", default=None, help="Gemini model (executor default if unset)")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", help="write the full JSON report here")
    args = parser.parse_args(argv)

    labels = wildlife.load_labels(args.labels)
    truth = load_query_answers(args.labels)
    view = wildlife.open_clip(args.video or labels.video, labels.clip_start, labels.clip_end)
    counter = make_counter(view.path.name, model=args.model, cache_path=Path(args.cache))

    report = evaluate(
        view, truth, counter, sample_every=args.sample_every,
        threshold=args.threshold, pixel_cutoff=args.pixel_cutoff,
    )
    report["counter"] = {
        "model": getattr(counter, "model", None),
        "api_calls": getattr(counter, "api_calls", None),
        "cache_hits": getattr(counter, "cache_hits", None),
        "api_seconds": getattr(counter, "api_seconds", None),
    }
    print(format_report(report))
    print(
        f"\nCounter: {report['counter']['api_calls']} API calls "
        f"({report['counter']['api_seconds']:.1f}s), "
        f"{report['counter']['cache_hits']} cache hits"
    )
    if args.output:
        serializable = dict(report)
        serializable["counts"] = {
            name: {str(k): v for k, v in counts.items()}
            for name, counts in report["counts"].items()
        }
        Path(args.output).write_text(json.dumps(serializable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
