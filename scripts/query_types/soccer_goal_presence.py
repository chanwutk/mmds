"""Presence of an event: "Is a goal scored in this chunk?" on SoccerNet.

Every mode answers the same question for the same 5-minute chunks:

- ``naive``: Gemini watches the video chunk.
- ``transcript``: the ``ModalitySubstitution`` rewrite of the naive plan;
  Gemini reads the chunk's commentary transcript instead of the video.
- ``detector``: detector substitution; YOLOE looks for a soccer ball inside
  a goal net, and the chunk counts as a goal if any frame has one.

The expected outcome is that the transcript keeps the evidence (commentators
announce goals) while the detector loses it (a goal is an event, not an
object). Each mode records accuracy and cost: model-call tokens for the
Gemini modes, detector wall-clock seconds for the detector mode.

Steps (run from the repo root with ``PYTHONPATH=src:.``)::

    python -m scripts.query_types.soccer_goal_presence prepare --soccernet-root data/soccernet
    python -m scripts.query_types.soccer_goal_presence run --mode naive
    python -m scripts.query_types.soccer_goal_presence run --mode transcript
    python -m scripts.query_types.soccer_goal_presence run --mode detector   # on the GPU server
    python -m scripts.query_types.soccer_goal_presence score

``prepare`` writes one chunk file per half. ``run`` writes one result file
per half and skips halves that already have one, so a failed run resumes
where it stopped. The Gemini modes read the API key from ``GEMINI_API_KEY``
(or ``GOOGLE_API_KEY``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mmds import Detect, Input, Map, Record, execute
from mmds.model import DatasetExpr
from mmds.optimizers.rewriter import ModalitySubstitution, PlanIndex, apply_rewrite
from mmds.render import program_from_plan

DEFAULT_OUT_DIR = Path("data/query_types/soccer_goal_presence")
CHUNK_SECONDS = 300.0
# A trailing remainder shorter than this is merged into the previous chunk
# instead of becoming its own (near-empty) chunk.
MIN_TAIL_SECONDS = 30.0
VIDEO_RESOLUTION = "224p"
NO_SPEECH = "[no commentary in this segment]"

SCHEMA = {"goal_scored": "boolean"}
NAIVE_PROMPT = (
    "You are watching a segment of a broadcast soccer match. Was a goal scored "
    "during this segment? Answer true only if a goal that counts is scored "
    "during this segment. Answer false for near misses, disallowed goals, and "
    "replays of goals scored before this segment.\n\nVideo segment:\n"
)
TRANSCRIPT_PROMPT = (
    "You are reading the live commentary transcript of a segment of a "
    "broadcast soccer match. Was a goal scored during this segment? Answer "
    "true only if the commentary indicates that a goal that counts was scored "
    "during this segment. Answer false for near misses, disallowed goals, and "
    "discussion of goals scored before this segment."
)

DETECTOR_MODEL = "yoloe-11s-seg.pt"
BALL_CLASS = "soccer ball"
NET_CLASS = "goal net"
DETECTOR_CLASSES = (BALL_CLASS, NET_CLASS)
DETECTOR_FPS = 5.0

MODES = ("naive", "transcript", "detector")


# ---------------------------------------------------------------------------
# Ground truth, chunking, and transcripts
# ---------------------------------------------------------------------------


def goal_times(labels: Mapping[str, Any], half: int) -> list[float]:
    """Return goal times in seconds from the start of ``half``'s video."""
    times: list[float] = []
    for annotation in labels.get("annotations", []):
        if annotation.get("label") != "Goal":
            continue
        game_time = annotation.get("gameTime")
        position = annotation.get("position")
        try:
            label_half = int(str(game_time).split(" - ", 1)[0])
            seconds = int(position) / 1000.0
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed Goal annotation: gameTime={game_time!r}, position={position!r}"
            ) from exc
        if label_half == half:
            times.append(seconds)
    return sorted(times)


def make_chunks(
    duration: float,
    chunk_seconds: float = CHUNK_SECONDS,
    min_tail_seconds: float = MIN_TAIL_SECONDS,
) -> list[tuple[float, float]]:
    """Split ``[0, duration)`` into fixed-length chunks.

    A trailing remainder shorter than ``min_tail_seconds`` is merged into the
    previous chunk, so a 2700.003 s half yields nine chunks, not ten.
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


def transcript_for_window(
    segments: Iterable[Mapping[str, Any]], start: float, end: float
) -> str:
    """Join the transcript segments whose midpoint falls in ``[start, end)``.

    Using the midpoint assigns a segment that straddles a chunk boundary to
    exactly one chunk.
    """
    texts: list[str] = []
    for segment in segments:
        midpoint = (float(segment["start"]) + float(segment["end"])) / 2
        text = str(segment.get("text", "")).strip()
        if text and start <= midpoint < end:
            texts.append(text)
    return " ".join(texts)


def chunk_rows(
    *,
    game_id: str,
    half: int,
    video_path: str,
    duration: float,
    labels: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    chunk_seconds: float = CHUNK_SECONDS,
) -> list[dict[str, Any]]:
    """Build one input row per chunk, with its ground-truth goal count."""
    goals = goal_times(labels, half)
    chunks = make_chunks(duration, chunk_seconds)
    rows: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(chunks):
        # The last chunk is open-ended so a goal or segment timestamped at or
        # just past the reported duration is not silently dropped.
        upper = end if index < len(chunks) - 1 else math.inf
        rows.append(
            {
                "chunk_id": f"{game_id}|{half}|{index:02d}",
                "game_id": game_id,
                "half": half,
                "start": start,
                "end": end,
                "video": {"type": "VideoView", "path": video_path, "start": start, "end": end},
                "transcript": transcript_for_window(segments, start, upper) or NO_SPEECH,
                "gt_goal_count": sum(1 for goal in goals if start <= goal < upper),
            }
        )
    return rows


def select_games(
    manifest: Mapping[str, Any],
    transcript_index: Mapping[str, Any],
    root: Path,
    count: int,
) -> list[Mapping[str, Any]]:
    """Return the first ``count`` games (by id) that are fully available.

    A game is available when both halves are transcribed and its labels and
    both videos exist under ``root``.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    transcribed = {
        (half["game_id"], int(half["half"]))
        for half in transcript_index.get("halves", [])
        if half.get("status") == "transcribed"
    }
    available = []
    for game in sorted(manifest.get("games", []), key=lambda g: g["game_id"]):
        videos = game.get("videos", {}).get(VIDEO_RESOLUTION, {})
        if not all((game["game_id"], half) in transcribed for half in (1, 2)):
            continue
        if not all(str(half) in videos for half in (1, 2)):
            continue
        paths = [root / game["labels"]["path"]] + [root / videos[str(h)]["path"] for h in (1, 2)]
        if all(path.exists() for path in paths):
            available.append(game)
    if len(available) < count:
        raise ValueError(f"Requested {count} games but only {len(available)} are available under {root}.")
    return available[:count]


def prepare(root: Path, out_dir: Path, games: int, chunk_seconds: float = CHUNK_SECONDS) -> list[Path]:
    """Write one chunk file per half and return their paths."""
    manifest = _read_json(root / "manifest.json")
    transcript_index = _read_json(root / "transcripts" / "index.json")
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for game_number, game in enumerate(select_games(manifest, transcript_index, root, games)):
        labels = _read_json(root / game["labels"]["path"])
        for half in (1, 2):
            video = game["videos"][VIDEO_RESOLUTION][str(half)]
            transcript = _read_json(root / "transcripts" / game["game_id"] / f"{half}.whisper.json")
            rows = chunk_rows(
                game_id=game["game_id"],
                half=half,
                video_path=str((root / video["path"]).resolve()),
                duration=float(video["duration_seconds"]),
                labels=labels,
                segments=transcript["segments"],
                chunk_seconds=chunk_seconds,
            )
            path = chunk_dir / f"g{game_number:02d}_h{half}.jsonl"
            _write_text_atomic(path, "".join(json.dumps(row) + "\n" for row in rows))
            written.append(path)
    return written


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def naive_plan(input_path: str) -> DatasetExpr:
    return Map(Input(input_path), [NAIVE_PROMPT, Record["video"]], schema=SCHEMA)


def transcript_plan(input_path: str) -> DatasetExpr:
    """Apply the ModalitySubstitution directive to the naive plan."""
    program = program_from_plan(naive_plan(input_path))
    directive = ModalitySubstitution()
    matches = directive.find_matches(PlanIndex.build(program.output_expr))
    if len(matches) != 1:
        raise ValueError(f"Expected one ModalitySubstitution match, found {len(matches)}.")
    rewritten = apply_rewrite(
        program,
        directive=directive,
        match=matches[0],
        params={
            "video_field": "video",
            "transcript_field": "transcript",
            "rewritten_prompt": TRANSCRIPT_PROMPT,
        },
    )
    return rewritten.output_expr


def detector_plan(input_path: str, frame_stride: int) -> DatasetExpr:
    return Detect(
        Input(input_path),
        "video",
        list(DETECTOR_CLASSES),
        model=DETECTOR_MODEL,
        frame_stride=frame_stride,
    )


def detector_stride(fps: float, target_fps: float = DETECTOR_FPS) -> int:
    """Frame stride that samples ``fps`` video at exactly ``target_fps``."""
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"Video reports an invalid fps: {fps!r}")
    stride = round(fps / target_fps)
    if stride < 1 or not math.isclose(fps / stride, target_fps, rel_tol=0.01):
        raise ValueError(f"Cannot sample {fps} fps video at exactly {target_fps} fps.")
    return stride


# ---------------------------------------------------------------------------
# Predictions and scoring
# ---------------------------------------------------------------------------


def ball_in_net_counts(detections: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count sampled frames with a ball, a net, and a ball centred inside a net."""
    by_frame: dict[int, dict[str, list[Sequence[float]]]] = {}
    for entry in detections:
        for box in entry.get("bboxes", []):
            by_frame.setdefault(int(box["frame_idx"]), {}).setdefault(entry["type"], []).append(box["bbox"])

    counts = {"frames_with_ball": 0, "frames_with_net": 0, "frames_ball_in_net": 0}
    for classes in by_frame.values():
        balls = classes.get(BALL_CLASS, [])
        nets = classes.get(NET_CLASS, [])
        counts["frames_with_ball"] += bool(balls)
        counts["frames_with_net"] += bool(nets)
        counts["frames_ball_in_net"] += any(
            _center_inside(ball, net) for ball in balls for net in nets
        )
    return counts


def _center_inside(inner: Sequence[float], outer: Sequence[float]) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def score(pairs: Iterable[tuple[bool, int]]) -> dict[str, Any]:
    """Score (predicted goal, ground-truth goal count) pairs per chunk."""
    tp = fp = fn = tn = 0
    for predicted, gt_count in pairs:
        actual = gt_count > 0
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
        "chunks": total,
        "goal_chunks": tp + fn,
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
    }
    for record in records:
        summary["calls"] += 1
        usage = record.get("usage")
        if usage is None:
            summary["calls_without_usage"] += 1
            continue
        summary["input_tokens"] += int(usage.get("prompt_token_count") or 0)
        summary["output_tokens"] += int(usage.get("candidates_token_count") or 0)
        summary["thinking_tokens"] += int(usage.get("thoughts_token_count") or 0)
        for detail in usage.get("prompt_tokens_details") or []:
            modality = str(detail.get("modality", "UNKNOWN")).rsplit(".", 1)[-1].upper()
            by_modality = summary["input_tokens_by_modality"]
            by_modality[modality] = by_modality.get(modality, 0) + int(detail.get("token_count") or 0)
    return summary


# ---------------------------------------------------------------------------
# Running one half
# ---------------------------------------------------------------------------


def run_prompt_half(
    mode: str,
    chunk_file: Path,
    executor: Any,
    usage_records: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    """Run a Gemini mode on one half; ``usage_records`` is filled by the executor's sink."""
    plans: dict[str, Callable[[str], DatasetExpr]] = {"naive": naive_plan, "transcript": transcript_plan}
    if mode not in plans:
        raise ValueError(f"Not a prompt mode: {mode!r}")
    usage_records.clear()
    rows = execute(plans[mode](str(chunk_file)), executor)
    outputs = []
    for row in rows:
        answer = row.get("goal_scored")
        if not isinstance(answer, bool):
            raise ValueError(f"{row['chunk_id']}: expected a boolean goal_scored, got {answer!r}")
        outputs.append({"chunk_id": row["chunk_id"], "gt_goal_count": row["gt_goal_count"], "prediction": answer})
    return {
        "mode": mode,
        "model": model,
        "chunk_file": chunk_file.name,
        "rows": outputs,
        "usage": list(usage_records),
    }


def run_detector_half(chunk_file: Path, frame_stride: int, device: str, gpu_name: str | None) -> dict[str, Any]:
    """Run the detector mode on one half, timing decode + inference (no model load)."""
    started = time.perf_counter()
    rows = execute(detector_plan(str(chunk_file), frame_stride))
    seconds = time.perf_counter() - started
    outputs = []
    for row in rows:
        counts = ball_in_net_counts(row.get("detections", []))
        outputs.append(
            {
                "chunk_id": row["chunk_id"],
                "gt_goal_count": row["gt_goal_count"],
                "prediction": counts["frames_ball_in_net"] > 0,
                **counts,
            }
        )
    return {
        "mode": "detector",
        "model": DETECTOR_MODEL,
        "chunk_file": chunk_file.name,
        "frame_stride": frame_stride,
        "device": device,
        "gpu": gpu_name,
        "detector_seconds": seconds,
        "rows": outputs,
    }


def run(mode: str, out_dir: Path, *, model: str, allow_cpu: bool = False) -> None:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}.")
    chunk_files = sorted((out_dir / "chunks").glob("*.jsonl"))
    if not chunk_files:
        raise SystemExit(f"No chunk files in {out_dir / 'chunks'}; run 'prepare' first.")
    result_dir = out_dir / "results" / mode
    result_dir.mkdir(parents=True, exist_ok=True)
    pending = [f for f in chunk_files if not (result_dir / f"{f.stem}.json").exists()]
    print(f"{mode}: {len(chunk_files) - len(pending)} halves done, {len(pending)} to run")
    if not pending:
        return

    if mode == "detector":
        _run_detector(pending, result_dir, allow_cpu=allow_cpu)
    else:
        _run_prompt_mode(mode, pending, result_dir, model=model)


def _run_prompt_mode(mode: str, chunk_files: Sequence[Path], result_dir: Path, *, model: str) -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to run the Gemini modes.")
    from mmds import GeminiPromptExecutor

    records: list[dict[str, Any]] = []
    lock = threading.Lock()

    def sink(record: dict[str, Any]) -> None:
        with lock:
            records.append(record)

    executor = GeminiPromptExecutor(model=model, usage_sink=sink)
    for chunk_file in chunk_files:
        print(f"  {mode} {chunk_file.name} ...", flush=True)
        try:
            payload = run_prompt_half(mode, chunk_file, executor, records, model)
        except Exception:
            # Calls made before the failure were billed; keep their usage.
            with (result_dir / "_failed_usage.jsonl").open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps({"chunk_file": chunk_file.name, **record}) + "\n")
            raise
        _write_json_atomic(result_dir / f"{chunk_file.stem}.json", payload)


def _run_detector(chunk_files: Sequence[Path], result_dir: Path, *, allow_cpu: bool) -> None:
    # The same check Detect uses to pick its device: fail loudly rather than
    # time a silent CPU fallback as GPU time.
    from mmds.execution.ops.detect import _get_device
    from mmds.utilities.video import open_video

    device = _get_device()
    if device != "cuda" and not allow_cpu:
        raise SystemExit("CUDA is not usable; refusing to record CPU time as GPU time (use --allow-cpu to override).")
    gpu_name = None
    if device == "cuda":
        import torch

        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    _warm_up_detector(chunk_files[0], result_dir, device, gpu_name)
    for chunk_file in chunk_files:
        first_row = json.loads(chunk_file.read_text(encoding="utf-8").splitlines()[0])
        stride = detector_stride(open_video(first_row["video"]["path"]).fps)
        print(f"  detector {chunk_file.name} (stride {stride}) ...", flush=True)
        payload = run_detector_half(chunk_file, stride, device, gpu_name)
        _write_json_atomic(result_dir / f"{chunk_file.stem}.json", payload)


def _warm_up_detector(chunk_file: Path, result_dir: Path, device: str, gpu_name: str | None) -> None:
    """Load YOLOE and run it on one second of video before anything is timed."""
    first_row = json.loads(chunk_file.read_text(encoding="utf-8").splitlines()[0])
    warmup_row = {**first_row, "video": {**first_row["video"], "start": 0.0, "end": 1.0}}
    warmup_file = result_dir / "_warmup.jsonl"
    warmup_file.write_text(json.dumps(warmup_row) + "\n", encoding="utf-8")
    started = time.perf_counter()
    execute(detector_plan(str(warmup_file), 1))
    seconds = time.perf_counter() - started
    warmup_file.unlink()
    _write_json_atomic(
        result_dir / "_warmup.json",
        {"seconds_including_model_load": seconds, "device": device, "gpu": gpu_name},
    )


# ---------------------------------------------------------------------------
# Scoring a results directory
# ---------------------------------------------------------------------------


def score_results(out_dir: Path) -> list[dict[str, Any]]:
    """Score every mode that has results; all modes must cover the same chunks."""
    summaries: list[dict[str, Any]] = []
    chunk_sets: dict[str, set[str]] = {}
    for mode in MODES:
        files = sorted((out_dir / "results" / mode).glob("g*.json"))
        if not files:
            continue
        payloads = [_read_json(path) for path in files]
        rows = [row for payload in payloads for row in payload["rows"]]
        chunk_sets[mode] = {row["chunk_id"] for row in rows}
        summary = {
            "mode": mode,
            "halves": len(payloads),
            **score((row["prediction"], row["gt_goal_count"]) for row in rows),
        }
        if mode == "detector":
            summary["detector_seconds"] = sum(p["detector_seconds"] for p in payloads)
            summary["gpu"] = sorted({str(p.get("gpu")) for p in payloads})
        else:
            summary.update(summarize_usage(record for p in payloads for record in p["usage"]))
            summary["model"] = sorted({p["model"] for p in payloads})
        summaries.append(summary)

    if len({frozenset(chunks) for chunks in chunk_sets.values()}) > 1:
        sizes = {mode: len(chunks) for mode, chunks in chunk_sets.items()}
        raise ValueError(f"Modes cover different chunks {sizes}; finish every mode before scoring.")
    return summaries


def format_table(summaries: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| Mode | Chunks | Goal chunks | TP | FP | FN | TN | Precision | Recall | Accuracy "
        "| Model calls | Input tokens | Output tokens | Detector s |"
    )
    columns = header.count("|") - 1
    lines = [header, "|" + "---|" * columns]
    for s in summaries:
        lines.append(
            "| {mode} | {chunks} | {goal_chunks} | {tp} | {fp} | {fn} | {tn} | {p} | {r} | {a} "
            "| {calls} | {inp} | {out} | {det} |".format(
                mode=s["mode"],
                chunks=s["chunks"],
                goal_chunks=s["goal_chunks"],
                tp=s["tp"],
                fp=s["fp"],
                fn=s["fn"],
                tn=s["tn"],
                p=_fmt(s["precision"]),
                r=_fmt(s["recall"]),
                a=_fmt(s["accuracy"]),
                calls=s.get("calls", 0),
                inp=f"{s['input_tokens']:,}" if "input_tokens" in s else "0",
                out=f"{s['output_tokens'] + s['thinking_tokens']:,}" if "output_tokens" in s else "0",
                det=f"{s['detector_seconds']:.1f}" if "detector_seconds" in s else "0",
            )
        )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


# ---------------------------------------------------------------------------
# Files and CLI
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="write chunk files for the selected games")
    prepare_parser.add_argument("--soccernet-root", type=Path, required=True)
    prepare_parser.add_argument("--games", type=int, default=4)

    run_parser = commands.add_parser("run", help="run one mode over every chunk file")
    run_parser.add_argument("--mode", choices=MODES, required=True)
    run_parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    run_parser.add_argument("--allow-cpu", action="store_true")

    commands.add_parser("score", help="print and save the results table")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        written = prepare(args.soccernet_root, args.out_dir, args.games)
        print(f"Wrote {len(written)} chunk files to {args.out_dir / 'chunks'}")
    elif args.command == "run":
        run(args.mode, args.out_dir, model=args.model, allow_cpu=args.allow_cpu)
    else:
        summaries = score_results(args.out_dir)
        _write_json_atomic(args.out_dir / "summary.json", summaries)
        print(format_table(summaries))


if __name__ == "__main__":
    main()
