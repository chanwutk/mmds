"""Helpers shared by the query-type experiment scripts.

Each script asks one question over its own data. What they share lives here:
splitting a video into chunks, sampling frames for the detector, cutting and
re-encoding video, interval arithmetic and localization scores, summing
Gemini usage, pricing tokens and GPU time in dollars, scoring yes/no answers,
summarizing repeats, recording what
produced a result, picking and warming up the detector device, and writing
result files atomically.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# A trailing remainder shorter than this is merged into the previous chunk
# instead of becoming its own (near-empty) chunk.
MIN_TAIL_SECONDS = 30.0
DETECTOR_FPS = 5.0
# Same encoding as the paper's materialized soccer clips, so every video an
# experiment sends to a model is produced the same way.
ENCODING = {
    "video_codec": "libx264",
    "video_crf": 18,
    "video_preset": "veryfast",
    "pixel_format": "yuv420p",
    "audio_codec": "aac",
    "audio_bitrate": "128k",
}

Interval = tuple[float, float]


# ---------------------------------------------------------------------------
# Chunking and frame sampling
# ---------------------------------------------------------------------------


def make_chunks(
    duration: float,
    chunk_seconds: float,
    min_tail_seconds: float = MIN_TAIL_SECONDS,
) -> list[tuple[float, float]]:
    """Split ``[0, duration)`` into fixed-length chunks.

    A trailing remainder shorter than ``min_tail_seconds`` is merged into the
    previous chunk, so a 2700.003 s video yields nine chunks, not ten.
    """
    if not (math.isfinite(duration) and duration > 0):
        raise ValueError(f"duration must be a positive number, got {duration!r}")
    if not (math.isfinite(chunk_seconds) and chunk_seconds > 0):
        raise ValueError(f"chunk_seconds must be a positive number, got {chunk_seconds!r}")

    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        if chunks and end - start < min_tail_seconds:
            chunks[-1] = (chunks[-1][0], duration)
            break
        chunks.append((start, end))
        start = end
    return chunks


def detector_stride(fps: float, target_fps: float = DETECTOR_FPS) -> int:
    """Frame stride that samples ``fps`` video at exactly ``target_fps``."""
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"Video reports an invalid fps: {fps!r}")
    stride = round(fps / target_fps)
    if stride < 1 or not math.isclose(fps / stride, target_fps, rel_tol=0.01):
        raise ValueError(f"Cannot sample {fps} fps video at exactly {target_fps} fps.")
    return stride


# ---------------------------------------------------------------------------
# Video files
# ---------------------------------------------------------------------------


def encode_command(
    source: Path,
    target: Path,
    *,
    start: float | None = None,
    end: float | None = None,
    height: int | None = None,
    crf: int = ENCODING["video_crf"],
    audio_bitrate: str = ENCODING["audio_bitrate"],
) -> list[str]:
    """ffmpeg command that re-encodes ``source`` (or ``[start, end)`` of it).

    Every frame is kept as is (no frame-rate change), so frame ``i`` of a
    full re-encode is frame ``i`` of the source. A cut starts its timestamps
    at zero. ``height`` scales the picture, keeping the aspect ratio. ``crf``
    and ``audio_bitrate`` default to the paper's settings; a smaller file
    (higher ``crf``) is for sending a clip inline in a Gemini request.
    """
    if (start is None) != (end is None):
        raise ValueError("Give both start and end, or neither.")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None and end is not None:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Invalid cut [{start!r}, {end!r}).")
        command += ["-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}"]
    else:
        command += ["-i", str(source)]
    if height is not None:
        if height < 2 or height % 2:
            raise ValueError(f"height must be an even number >= 2, got {height!r}")
        command += ["-vf", f"scale=-2:{height}"]
    command += [
        "-fps_mode", "passthrough",
        "-c:v", ENCODING["video_codec"],
        "-crf", str(crf),
        "-preset", ENCODING["video_preset"],
        "-pix_fmt", ENCODING["pixel_format"],
        "-c:a", ENCODING["audio_codec"],
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(target),
    ]
    return command


def encode_video(source: Path, target: Path, **kwargs: Any) -> float:
    """Run :func:`encode_command`; return the seconds it took.

    The output is written next to ``target`` and renamed into place, so an
    interrupted run never leaves a truncated file under the final name.
    """
    tmp = target.with_name(target.stem + ".partial" + target.suffix)
    started = time.perf_counter()
    subprocess.run(encode_command(source, tmp, **kwargs), check=True)
    tmp.replace(target)
    return time.perf_counter() - started


def probe_video(path: Path) -> dict[str, Any]:
    """Frame count (from the container's packets), frame rate and duration."""
    output = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=nb_read_packets,avg_frame_rate,width,height:format=duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return parse_probe(json.loads(output))


def parse_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    streams = probe.get("streams") or []
    if not streams:
        raise ValueError("ffprobe found no video stream.")
    stream = streams[0]
    numerator, _, denominator = str(stream["avg_frame_rate"]).partition("/")
    fps = float(numerator) / float(denominator) if float(denominator or 1) else 0.0
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"ffprobe reported an invalid frame rate {stream['avg_frame_rate']!r}.")
    return {
        "frames": int(stream["nb_read_packets"]),
        "fps": fps,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration": float(probe["format"]["duration"]),
    }


# ---------------------------------------------------------------------------
# Intervals and localization scores
# ---------------------------------------------------------------------------


def merge_intervals(intervals: Iterable[Interval], gap: float = 0.0) -> list[Interval]:
    """Sort intervals and merge those that overlap or lie at most ``gap`` apart."""
    if not (math.isfinite(gap) and gap >= 0):
        raise ValueError(f"gap must be a non-negative number, got {gap!r}")
    merged: list[Interval] = []
    for start, end in sorted(_checked(intervals)):
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total_length(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersect_length(left: Iterable[Interval], right: Iterable[Interval]) -> float:
    """Total time covered by both interval sets."""
    a, b = merge_intervals(left), merge_intervals(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def subtract_intervals(intervals: Iterable[Interval], removed: Iterable[Interval]) -> list[Interval]:
    """The parts of ``intervals`` not covered by ``removed``."""
    cuts = merge_intervals(removed)
    result: list[Interval] = []
    for start, end in merge_intervals(intervals):
        cursor = start
        for cut_start, cut_end in cuts:
            if cut_end <= cursor or cut_start >= end:
                continue
            if cut_start > cursor:
                result.append((cursor, cut_start))
            cursor = max(cursor, cut_end)
        if cursor < end:
            result.append((cursor, end))
    return result


def clamp_intervals(intervals: Iterable[Interval], duration: float) -> tuple[list[Interval], int]:
    """Clip model-predicted intervals to ``[0, duration]``.

    Returns the kept intervals and how many were dropped because they were
    empty, reversed, or entirely outside the video once clipped.
    """
    kept: list[Interval] = []
    dropped = 0
    for start, end in intervals:
        if not (math.isfinite(start) and math.isfinite(end)):
            dropped += 1
            continue
        start, end = max(0.0, start), min(duration, end)
        if end <= start:
            dropped += 1
            continue
        kept.append((start, end))
    return kept, dropped


def tiou(a: Interval, b: Interval) -> float:
    """Temporal intersection over union; 0 when the union is empty."""
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


def f1_score(precision: float | None, recall: float | None) -> float | None:
    """F1, which is 0 when either rate is 0 (even if the other is undefined).

    So a plan that predicts nothing while labels exist scores 0, not ``None``;
    ``None`` is left for cases with nothing to score at all.
    """
    if precision == 0 or recall == 0:
        return 0.0
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall)


def seconds_scores(films: Iterable[tuple[Sequence[Interval], Sequence[Interval], Sequence[Interval]]]) -> dict[str, Any]:
    """Precision, recall and F1 over seconds, summed over films.

    Each film gives (predicted, labeled, ignored) intervals. Ignored time is
    removed from both sides first, so it can be neither a hit nor a miss.
    """
    overlap = predicted = labeled = 0.0
    for predicted_intervals, labeled_intervals, ignored in films:
        p = subtract_intervals(predicted_intervals, ignored)
        g = subtract_intervals(labeled_intervals, ignored)
        overlap += intersect_length(p, g)
        predicted += total_length(p)
        labeled += total_length(g)
    precision = overlap / predicted if predicted else None
    recall = overlap / labeled if labeled else None
    return {
        "overlap_seconds": overlap,
        "predicted_seconds": predicted,
        "labeled_seconds": labeled,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def interval_scores(
    films: Iterable[tuple[Sequence[Interval], Sequence[Interval], Sequence[Interval]]],
    threshold: float,
) -> dict[str, Any]:
    """The paper's interval metric: one-to-one matches at tIoU >= ``threshold``.

    Each film gives (predicted, labeled, ignored) intervals. Predictions that
    lie entirely inside ignored time are not counted. Matching is greedy by
    descending tIoU, summed over films.
    """
    tp = n_predicted = n_labeled = 0
    for predicted_intervals, labeled_intervals, ignored in films:
        predicted = [p for p in predicted_intervals if total_length(subtract_intervals([p], ignored)) > 0]
        labeled = list(labeled_intervals)
        candidates: list[tuple[float, int, int]] = []
        for i, p in enumerate(predicted):
            for j, g in enumerate(labeled):
                score = tiou(p, g)
                if score > 0 and score >= threshold:
                    candidates.append((score, i, j))
        candidates.sort(reverse=True)
        used_p: set[int] = set()
        used_g: set[int] = set()
        for _, i, j in candidates:
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
        tp += len(used_p)
        n_predicted += len(predicted)
        n_labeled += len(labeled)
    precision = tp / n_predicted if n_predicted else None
    recall = tp / n_labeled if n_labeled else None
    return {
        "tp": tp,
        "fp": n_predicted - tp,
        "fn": n_labeled - tp,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def _checked(intervals: Iterable[Interval]) -> list[Interval]:
    checked = []
    for start, end in intervals:
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            raise ValueError(f"Invalid interval ({start!r}, {end!r}).")
        checked.append((float(start), float(end)))
    return checked


# ---------------------------------------------------------------------------
# Scoring and usage
# ---------------------------------------------------------------------------


def binary_scores(pairs: Iterable[tuple[bool, bool]]) -> dict[str, Any]:
    """Confusion counts and rates for (predicted, actual) yes/no pairs.

    A rate whose denominator is zero is ``None`` rather than a made-up number.
    """
    tp = fp = fn = tn = 0
    for predicted, actual in pairs:
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    total = tp + fp + fn + tn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "accuracy": (tp + tn) / total if total else None,
    }


def summarize_usage(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum Gemini usage records; calls without usage are counted, not guessed."""
    summary: dict[str, Any] = {
        "calls": 0,
        "calls_without_usage": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "input_tokens_by_modality": {},
        "calls_by_op": {},
        "input_tokens_by_op": {},
    }
    for record in records:
        summary["calls"] += 1
        op = str(record.get("op_type", "unknown"))
        summary["calls_by_op"][op] = summary["calls_by_op"].get(op, 0) + 1
        usage = record.get("usage")
        if usage is None:
            summary["calls_without_usage"] += 1
            continue
        prompt_tokens = int(usage.get("prompt_token_count") or 0)
        summary["input_tokens_by_op"][op] = summary["input_tokens_by_op"].get(op, 0) + prompt_tokens
        summary["input_tokens"] += prompt_tokens
        summary["output_tokens"] += int(usage.get("candidates_token_count") or 0)
        summary["thinking_tokens"] += int(usage.get("thoughts_token_count") or 0)
        for modality, count in _tokens_by_modality(usage.get("prompt_tokens_details")).items():
            by_modality = summary["input_tokens_by_modality"]
            by_modality[modality] = by_modality.get(modality, 0) + count
    return summary


def _tokens_by_modality(details: Iterable[Mapping[str, Any]] | None) -> dict[str, int]:
    """Token counts per modality name (``"MediaModality.VIDEO"`` and ``"VIDEO"`` are both ``"VIDEO"``)."""
    counts: dict[str, int] = {}
    for detail in details or []:
        modality = str(detail.get("modality", "UNKNOWN")).rsplit(".", 1)[-1].upper()
        counts[modality] = counts.get(modality, 0) + int(detail.get("token_count") or 0)
    return counts


# ---------------------------------------------------------------------------
# Dollar cost
# ---------------------------------------------------------------------------

# Dollars per 1M tokens: Gemini API paid tier, standard (not batch) rates, from
# ai.google.dev/gemini-api/docs/pricing on 2026-10-25. Input tokens Gemini
# serves from its automatic cache cost the "cached" rate instead of "input";
# output includes thinking tokens.
GEMINI_PRICES: dict[str, dict[str, Any]] = {
    "gemini-3.1-flash-lite": {
        "input": {"TEXT": 0.25, "IMAGE": 0.25, "VIDEO": 0.25, "AUDIO": 0.50},
        "cached": {"TEXT": 0.025, "IMAGE": 0.025, "VIDEO": 0.025, "AUDIO": 0.05},
        "output": 1.50,
    },
}
# Dollars per GPU-hour: AWS g4dn.xlarge (1x NVIDIA T4, 4 vCPUs), on-demand in
# us-east-1, $0.526/h on 2026-10-25. The detector times were measured on a
# TITAN Xp, which no cloud rents; pricing them at this rate assumes a T4 runs
# YOLOE about as fast. Break-even rates (``break_even_gpu_rate``) show how much
# the comparison depends on it.
GPU_DOLLARS_PER_HOUR = 0.526


def gemini_dollars(record: Mapping[str, Any]) -> float:
    """What one Gemini call cost, from its usage record.

    Input is priced per modality, cached input at the cache rate, and output
    (thinking included) at the output rate. A model without a price, or counts
    that do not add up, raise instead of being guessed.
    """
    model = record.get("model")
    if model not in GEMINI_PRICES:
        raise ValueError(f"No price for Gemini model {model!r}; add it to GEMINI_PRICES.")
    prices = GEMINI_PRICES[model]
    usage = record.get("usage")
    if usage is None:
        raise ValueError(f"A {model} call has no usage, so its cost is unknown.")
    prompt = _tokens_by_modality(usage.get("prompt_tokens_details"))
    cached = _tokens_by_modality(usage.get("cache_tokens_details"))
    for name, counts, total in (
        ("input", prompt, usage.get("prompt_token_count")),
        ("cached", cached, usage.get("cached_content_token_count")),
    ):
        if sum(counts.values()) != int(total or 0):
            raise ValueError(f"{model}: {name} tokens by modality {counts} do not add up to {total}.")
    dollars = 0.0
    for modality, count in prompt.items():
        if modality not in prices["input"]:
            raise ValueError(f"No {model} price for {modality} input.")
        hit = cached.get(modality, 0)
        if hit > count:
            raise ValueError(f"{model}: {hit} cached {modality} tokens, but only {count} input ones.")
        dollars += (count - hit) * prices["input"][modality] + hit * prices["cached"][modality]
    output = int(usage.get("candidates_token_count") or 0) + int(usage.get("thoughts_token_count") or 0)
    return (dollars + output * prices["output"]) / 1e6


def usage_dollars(records: Iterable[Mapping[str, Any]]) -> float | None:
    """What a list of Gemini calls cost; ``None`` if any call has no usage."""
    records = list(records)
    if any(record.get("usage") is None for record in records):
        return None
    return sum(gemini_dollars(record) for record in records)


def gpu_dollars(seconds: float) -> float:
    """Detector time in dollars at ``GPU_DOLLARS_PER_HOUR``."""
    return seconds / 3600 * GPU_DOLLARS_PER_HOUR


def plan_dollars(gemini: float | None, gpu_seconds: float) -> dict[str, float | None]:
    """A plan's Gemini, GPU and total dollars (total ``None`` when Gemini's is unknown)."""
    gpu = gpu_dollars(gpu_seconds)
    return {"gemini_dollars": gemini, "gpu_dollars": gpu, "dollars": None if gemini is None else gemini + gpu}


def break_even_gpu_rate(reference_dollars: float, gemini: float, gpu_seconds: float) -> float | None:
    """The GPU price per hour at which a plan costs as much as the reference plan.

    Below it, the plan is cheaper. ``None`` for a plan that uses no GPU time;
    negative when its Gemini cost alone exceeds the reference.
    """
    if gpu_seconds <= 0:
        return None
    return (reference_dollars - gemini) / (gpu_seconds / 3600)


def cost_comparison(
    reference_plan: str,
    reference_dollars: float | None,
    gpu_plans: Mapping[str, tuple[float | None, float]],
) -> dict[str, Any]:
    """The GPU price used, and each GPU plan's break-even price against ``reference_plan``.

    ``gpu_plans`` maps a plan to its (Gemini dollars, GPU seconds). Plans whose
    cost is unknown, or any plan when the reference's is, get no rate.
    """
    rates = {}
    if reference_dollars is not None:
        for plan, (gemini, gpu_seconds) in gpu_plans.items():
            if gemini is not None:
                rates[plan] = break_even_gpu_rate(reference_dollars, gemini, gpu_seconds)
    return {
        "gpu_dollars_per_hour": GPU_DOLLARS_PER_HOUR,
        "reference_plan": reference_plan,
        "break_even_gpu_dollars_per_hour": rates,
    }


def format_cost_comparison(costs: Mapping[str, Any]) -> list[str]:
    """Report lines for a ``cost_comparison``; none when there is no rate to show."""
    rates = costs.get("break_even_gpu_dollars_per_hour") or {}
    if not rates:
        return []
    return [
        "",
        (
            f"GPU time is priced at ${costs['gpu_dollars_per_hour']}/h. Each plan that uses the GPU costs as much as "
            f"{costs['reference_plan']} at this price per GPU-hour (and less below it):"
        ),
        "",
        *(f"- {plan}: ${fmt(rate, 2)}/h" for plan, rate in rates.items()),
    ]


# ---------------------------------------------------------------------------
# Summaries over repeats, and report numbers
# ---------------------------------------------------------------------------


def mean_sd(values: Iterable[float | None]) -> dict[str, float | None]:
    """Mean and sample standard deviation, ignoring ``None``; sd is 0 for one value."""
    present = [value for value in values if value is not None]
    if not present:
        return {"mean": None, "sd": None}
    return {"mean": statistics.fmean(present), "sd": statistics.stdev(present) if len(present) > 1 else 0.0}


def aggregate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``mean_sd`` of every number across runs that share one shape, nested dicts included."""
    if not runs:
        raise ValueError("Nothing to aggregate.")
    result: dict[str, Any] = {}
    for key, first in runs[0].items():
        if isinstance(first, Mapping):
            result[key] = aggregate([run[key] for run in runs])
        else:
            result[key] = mean_sd(run[key] for run in runs)
    return result


def fmt(value: Mapping[str, Any] | float | None, digits: int = 3) -> str:
    """A report number: ``"mean ± sd"`` for a ``mean_sd`` result, plain otherwise, ``"n/a"`` if missing."""
    if isinstance(value, Mapping):
        if value.get("mean") is None:
            return "n/a"
        return f"{value['mean']:.{digits}f} ± {value['sd']:.{digits}f}"
    return "n/a" if value is None else f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _command_output(command: Sequence[str]) -> str | None:
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return output.strip().splitlines()[0] if output.strip() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_info(weights: Iterable[Path] = ()) -> dict[str, Any]:
    """What produced a result: code version, tool and package versions, machine, time."""
    from importlib import metadata

    packages = {}
    for name in ("google-genai", "ultralytics", "torch", "opencv-python"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    commit = _command_output(["git", "rev-parse", "HEAD"])
    # Uncommitted changes to tracked files; unknown (None) without git.
    dirty = None if commit is None else _command_output(["git", "status", "--porcelain", "--untracked-files=no"]) is not None
    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "packages": packages,
        "ffmpeg": _command_output(["ffmpeg", "-version"]),
        "machine": f"{platform.node()} ({platform.platform()})",
        "weights_sha256": {path.name: _sha256(path) for path in weights if path.exists()},
    }


def record_run_info(folder: Path, weights: Iterable[Path] = ()) -> None:
    """Append this invocation's ``run_info`` to ``folder/_run_info.jsonl``.

    One line per invocation, so a run resumed later with other code or on
    another machine keeps both records.
    """
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "_run_info.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run_info(weights)) + "\n")


def load_average() -> list[float] | None:
    """The machine's 1-, 5- and 15-minute load averages; ``None`` where the OS has none.

    Recorded with each detector timing: other work on a shared machine slows
    a timed run down, and this shows when it did.
    """
    try:
        return [round(value, 2) for value in os.getloadavg()]
    except (AttributeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Gemini runs
# ---------------------------------------------------------------------------


def require_gemini_key() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to run the Gemini modes.")


def usage_collector() -> tuple[list[dict[str, Any]], Callable[[dict[str, Any]], None]]:
    """Return a record list and a thread-safe sink that appends to it."""
    records: list[dict[str, Any]] = []
    lock = threading.Lock()

    def sink(record: dict[str, Any]) -> None:
        with lock:
            records.append(record)

    return records, sink


def append_failed_usage(path: Path, label: Mapping[str, Any], records: Iterable[Mapping[str, Any]]) -> None:
    """Keep the usage of calls billed before a failure, tagged with ``label``."""
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({**label, **record}) + "\n")


# ---------------------------------------------------------------------------
# Detector device and warm-up
# ---------------------------------------------------------------------------


def detector_device(*, allow_cpu: bool) -> tuple[str, str | None]:
    """Return Detect's device and GPU name; refuse to time CPU as GPU time.

    This is the same check Detect uses to pick its device (CUDA, then Apple
    MPS, then CPU), so a silent CPU fallback fails loudly unless ``allow_cpu``
    is set. Only CUDA reports a GPU name.
    """
    from mmds.execution.ops.detect import _get_device

    device = _get_device()
    if device == "cpu" and not allow_cpu:
        raise SystemExit(
            "Neither CUDA nor MPS is usable; refusing to record CPU time as GPU time "
            "(use --allow-cpu to override)."
        )
    gpu_name = None
    if device == "cuda":
        import torch

        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return device, gpu_name


def warm_up_detector(
    run_plan: Callable[[str], Any],
    first_row: Mapping[str, Any],
    result_dir: Path,
    device: str,
    gpu_name: str | None,
) -> None:
    """Load YOLOE and run it on one second of video before anything is timed.

    ``run_plan`` executes the script's detector plan on an input file path.
    The warm-up time, which includes loading the model, is saved once.
    """
    warmup_row = {**first_row, "video": {**first_row["video"], "start": 0.0, "end": 1.0}}
    warmup_file = write_jsonl(result_dir / "_warmup.jsonl", [warmup_row])
    started = time.perf_counter()
    try:
        run_plan(str(warmup_file))
    finally:
        warmup_file.unlink()
    seconds = time.perf_counter() - started
    write_json_atomic(
        result_dir / "_warmup.json",
        {"seconds_including_model_load": seconds, "device": device, "gpu": gpu_name},
    )


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write ``rows`` as JSON lines and return ``path``.

    ``Input(...)`` reads a file, so every one-off plan input is written here.
    """
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path
