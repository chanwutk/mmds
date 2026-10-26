"""YOLOE on WLD films: scan variants, the detection rule, hit windows, and the early-stop timing check.

A scan runs YOLOE over a whole film, sampling frames at ``DETECTOR_FPS`` with
a low confidence floor (``SCAN_CONF``), and saves each sampled frame's best
box confidence per queried animal plus how long the scan took. Every YOLOE
plan in both experiments is computed from these scans.

=============  ===============  =================  ===========================
variant        model            video              prompt
=============  ===============  =================  ===========================
small          yoloe-11s-seg    360p copy          the queried animal's name
small_vocab    yoloe-11s-seg    360p copy          all of DETECTOR_VOCABULARY
large          yoloe-11l-seg    original download  the queried animal's name
large_vocab    yoloe-11l-seg    original download  all of DETECTOR_VOCABULARY
=============  ===============  =================  ===========================

A one-name variant scans once per (film, animal); a vocabulary variant scans
once per film and keeps every query animal's boxes, so an animal on screen
competes for its best-matching name. ``small`` is the pre-registered detector
(YOLOE's defaults). ``large`` is the only variant that found every present
animal, so it is the filter in front of Gemini (``FILTER_VARIANT``).

The rule says "yes" once ``persistence`` consecutive sampled frames have a
box at ``conf`` or higher; ``MAIN_RULE`` is YOLOE's default confidence and one
frame. Stopping the scan where the rule first fires cannot change the
answer. Its time is estimated from the full scan, assuming every sampled
frame costs the same, and ``validate`` checks that estimate against a real
early stop (``Detect(stop_after_n=1)``) on five pairs.

Run on a GPU, one scan at a time (``run_yoloe_gpu.sh`` does this). Scans side
by side on one machine compete for its CPUs: six at once doubled every scan's
time. Each timing records the machine's load average when it starts::

    PYTHONPATH=src:. python -m scripts.query_types.wld_yoloe scan --variant large
    PYTHONPATH=src:. python -m scripts.query_types.wld_yoloe validate
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmds import Detect, Input, execute
from mmds.model import DatasetExpr
from scripts.query_types.common import (
    Interval,
    clamp_intervals,
    detector_device,
    detector_stride,
    load_average,
    merge_intervals,
    read_json,
    record_run_info,
    warm_up_detector,
    write_json_atomic,
    write_jsonl,
)
from scripts.query_types.wld import DETECTOR_VOCABULARY, MERGE_GAP_SECONDS, OUT_ROOT, WLD_ROOT, Pair, load_pairs

SCAN_CONF = 0.1
MAIN_RULE = (0.25, 1)  # (confidence, persistence): YOLOE's default confidence, one frame
SWEEP_CONFS = (0.1, 0.25, 0.4, 0.6)
SWEEP_PERSISTENCE = (1, 2, 3)
# Seconds of context added on each side of a hit before Gemini checks it.
PAD_SECONDS = 10.0


@dataclass(frozen=True)
class Variant:
    name: str
    model: str
    native: bool  # scan the original download rather than the 360p copy
    vocabulary: bool  # prompt with all of DETECTOR_VOCABULARY, one scan per film


VARIANTS: dict[str, Variant] = {
    variant.name: variant
    for variant in (
        Variant("small", "yoloe-11s-seg.pt", native=False, vocabulary=False),
        Variant("small_vocab", "yoloe-11s-seg.pt", native=False, vocabulary=True),
        Variant("large", "yoloe-11l-seg.pt", native=True, vocabulary=False),
        Variant("large_vocab", "yoloe-11l-seg.pt", native=True, vocabulary=True),
    )
}
MAIN_VARIANT = "small"
FILTER_VARIANT = "large"


# ---------------------------------------------------------------------------
# Plans and inputs
# ---------------------------------------------------------------------------


def video_path(pair: Pair, variant: Variant) -> str:
    """The prepared 360p film, or the original download it was made from.

    Both have the same frames (``wld prepare`` keeps every frame), so frame
    indices, and therefore the labels, mean the same thing in both.
    """
    if not variant.native:
        return pair.path
    prepared = Path(pair.path)
    return str(prepared.parent.parent / "videos" / prepared.name)


def detect_plan(
    input_path: str,
    variant: Variant,
    animal: str,
    frame_stride: int,
    *,
    conf: float = SCAN_CONF,
    stop_after_n: int | None = None,
) -> DatasetExpr:
    classes = list(DETECTOR_VOCABULARY) if variant.vocabulary else [animal]
    return Detect(
        Input(input_path), "video", classes,
        model=variant.model, frame_stride=frame_stride, conf=conf, stop_after_n=stop_after_n,
    )


def frame_max_conf(detections: Iterable[Mapping[str, Any]], animal: str | None = None) -> dict[int, float]:
    """Best box confidence per frame that has any box (of ``animal``, if given)."""
    best: dict[int, float] = {}
    for entry in detections:
        if animal is not None and entry.get("type") != animal:
            continue
        for box in entry.get("bboxes", []):
            frame = int(box["frame_idx"])
            best[frame] = max(best.get(frame, 0.0), float(box["confidence"]))
    return best


def sampled_count(frames: int, stride: int) -> int:
    return (frames + stride - 1) // stride


# ---------------------------------------------------------------------------
# Scans
# ---------------------------------------------------------------------------


def scan_dir(variant_name: str, out_root: Path = OUT_ROOT) -> Path:
    return out_root / "yoloe" / variant_name


def scan_units(variant: Variant, pairs: Sequence[Pair]) -> list[tuple[str, list[Pair]]]:
    """Each scan as ``(result file name, the pairs it answers)``.

    A one-name variant scans each pair on its own; a vocabulary variant scans
    each film once, answering all of that film's pairs.
    """
    if not variant.vocabulary:
        return [(pair.file_stem, [pair]) for pair in pairs]
    films: dict[str, list[Pair]] = {}
    for pair in pairs:
        films.setdefault(pair.film_id, []).append(pair)
    return list(films.items())


def run_scan(variant_name: str, pairs: Sequence[Pair], *, allow_cpu: bool, out_root: Path = OUT_ROOT) -> None:
    """Scan every unit that has no result file yet, one after another."""
    variant = VARIANTS[variant_name]
    units = scan_units(variant, pairs)
    result_dir = scan_dir(variant_name, out_root)
    scratch = out_root / "scratch" / "yoloe" / variant_name
    pending = [unit for unit in units if not (result_dir / f"{unit[0]}.json").exists()]
    print(f"{variant_name}: {len(units) - len(pending)} done, {len(pending)} to run", flush=True)
    if not pending:
        return
    result_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    record_run_info(result_dir, weights=[Path(variant.model)])
    device, gpu_name = detector_device(allow_cpu=allow_cpu)
    first = pending[0][1][0]
    warm_up_detector(
        lambda path: execute(detect_plan(path, variant, first.animal, 1)),
        {"pair_id": first.pair_id, "video": {"path": video_path(first, variant)}},
        scratch,
        device,
        gpu_name,
    )
    for name, covered in pending:
        pair = covered[0]
        stride = detector_stride(pair.fps)
        video = video_path(pair, variant)
        input_path = write_jsonl(scratch / f"{name}.jsonl", [{"pair_id": pair.pair_id, "video": {"path": video}}])
        print(f"  {name} (stride {stride}) ...", flush=True)
        load = load_average()
        started = time.perf_counter()
        rows = execute(detect_plan(str(input_path), variant, pair.animal, stride))
        seconds = time.perf_counter() - started
        detections = rows[0].get("detections", [])
        write_json_atomic(
            result_dir / f"{name}.json",
            {
                "variant": variant_name,
                "model": variant.model,
                "video": video,
                "stride": stride,
                "sampled_frames": sampled_count(pair.frames, stride),
                "seconds": seconds,
                "load_average": load,
                "device": device,
                "gpu": gpu_name,
                "frame_max_conf": {
                    p.animal: {str(f): c for f, c in sorted(frame_max_conf(detections, p.animal).items())}
                    for p in covered
                },
            },
        )


def load_scans(variant_name: str, pairs: Sequence[Pair], out_root: Path = OUT_ROOT) -> dict[str, dict[str, Any]]:
    """One scan per pair id, for the pairs this variant has scanned.

    A vocabulary scan serves every animal of its film, and the pair's time
    is the whole pass: answering one question still needs the pass, and the
    extra names barely change its cost.
    """
    result_dir = scan_dir(variant_name, out_root)
    scans = {}
    for name, covered in scan_units(VARIANTS[variant_name], pairs):
        path = result_dir / f"{name}.json"
        if not path.exists():
            continue
        scan = read_json(path)
        for pair in covered:
            scans[pair.pair_id] = {**scan, "frame_max_conf": scan["frame_max_conf"][pair.animal]}
    return scans


# ---------------------------------------------------------------------------
# The rule, and hits as windows
# ---------------------------------------------------------------------------


def rule_fires(max_conf: Mapping[int, float], stride: int, conf: float, persistence: int) -> int | None:
    """Sampled-frame index where ``persistence`` consecutive sampled frames
    first all have a box at ``conf`` or higher; ``None`` if that never happens."""
    if persistence < 1:
        raise ValueError("persistence must be at least 1")
    hits = []
    for frame, value in max_conf.items():
        if frame % stride:
            raise ValueError(f"Frame {frame} is not a sampled frame for stride {stride}.")
        if value >= conf:
            hits.append(frame // stride)
    run = 0
    previous = None
    for index in sorted(hits):
        run = run + 1 if previous is not None and index == previous + 1 else 1
        previous = index
        if run >= persistence:
            return index
    return None


def yoloe_outcomes(scan: Mapping[str, Any], conf: float, persistence: int) -> dict[str, dict[str, Any]]:
    """The full-scan and early-stop YOLOE plans for one pair, from its scan.

    Early-stop time is the scan time up to and including the frame where the
    rule fires, assuming every sampled frame costs the same.
    """
    max_conf = {int(frame): value for frame, value in scan["frame_max_conf"].items()}
    fired = rule_fires(max_conf, scan["stride"], conf, persistence)
    fraction = 1.0 if fired is None else (fired + 1) / scan["sampled_frames"]
    return {
        "yoloe_full": {"answer": fired is not None, "seconds": scan["seconds"], "fraction_scanned": 1.0},
        "yoloe_early": {"answer": fired is not None, "seconds": scan["seconds"] * fraction, "fraction_scanned": fraction},
    }


def hits_to_intervals(frames: Sequence[int], fps: float, stride: int) -> list[Interval]:
    """A hit at sampled frame ``f`` covers frames ``f`` to ``f + stride``; merge short gaps."""
    return merge_intervals(((f / fps, (f + stride) / fps) for f in frames), MERGE_GAP_SECONDS)


def scan_hits(scan: Mapping[str, Any], fps: float) -> list[Interval]:
    """A scan's hits as intervals, at the main rule's confidence."""
    frames = sorted(int(frame) for frame, conf in scan["frame_max_conf"].items() if conf >= MAIN_RULE[0])
    return hits_to_intervals(frames, fps, scan["stride"])


def pad_windows(intervals: Sequence[Interval], duration: float, pad: float = PAD_SECONDS) -> list[Interval]:
    """Pad each interval, clip it to the film, and merge overlaps."""
    padded, _ = clamp_intervals([(start - pad, end + pad) for start, end in intervals], duration)
    return merge_intervals(padded)


def ranked_windows(scan: Mapping[str, Any], pair: Pair) -> list[tuple[Interval, float]]:
    """A scan's padded hit windows with their best box confidence, most confident first."""
    ranked = []
    for start, end in pad_windows(scan_hits(scan, pair.fps), pair.duration):
        best = max(
            (conf for frame, conf in scan["frame_max_conf"].items() if start <= int(frame) / pair.fps < end),
            default=0.0,
        )
        ranked.append(((start, end), best))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


# ---------------------------------------------------------------------------
# Early-stop timing check
# ---------------------------------------------------------------------------


def pick_validation_pairs(pairs: Sequence[Pair], scans: Mapping[str, Mapping[str, Any]]) -> list[Pair]:
    """Three pairs where the main rule fires early, midway and late, and the
    two longest films where the animal is absent and the rule never fires."""
    fired: list[tuple[float, Pair]] = []
    silent: list[Pair] = []
    for pair in pairs:
        outcome = yoloe_outcomes(scans[pair.pair_id], *MAIN_RULE)["yoloe_early"]
        if pair.truth == "present" and outcome["answer"]:
            fired.append((outcome["fraction_scanned"], pair))
        elif pair.truth == "absent" and not outcome["answer"]:
            silent.append(pair)
    if len(fired) < 3 or len(silent) < 2:
        raise ValueError(f"Need 3 detected and 2 silent pairs, have {len(fired)} and {len(silent)}.")
    fired.sort(key=lambda item: (item[0], item[1].pair_id))
    chosen = [fired[round(q * (len(fired) - 1))][1] for q in (0.1, 0.5, 0.9)]
    silent.sort(key=lambda p: (-p.duration, p.pair_id))
    return chosen + silent[:2]


def run_validation(pairs: Sequence[Pair], *, allow_cpu: bool, out_root: Path = OUT_ROOT) -> dict[str, Any]:
    """Run the real early stop on five pairs and compare it with the estimate from the main variant's scans."""
    variant = VARIANTS[MAIN_VARIANT]
    conf, persistence = MAIN_RULE
    if persistence != 1:
        raise ValueError("Detect can only stop after one frame; the main rule must have persistence 1.")
    scans = load_scans(MAIN_VARIANT, pairs, out_root)
    chosen = pick_validation_pairs(pairs, scans)
    scratch = out_root / "scratch" / "yoloe" / "validate"
    scratch.mkdir(parents=True, exist_ok=True)
    device, gpu_name = detector_device(allow_cpu=allow_cpu)
    first = chosen[0]
    warm_up_detector(
        lambda path: execute(detect_plan(path, variant, first.animal, 1, conf=conf, stop_after_n=1)),
        {"pair_id": first.pair_id, "video": {"path": video_path(first, variant)}},
        scratch,
        device,
        gpu_name,
    )
    checks = []
    for pair in chosen:
        stride = detector_stride(pair.fps)
        row = {"pair_id": pair.pair_id, "video": {"path": video_path(pair, variant)}}
        input_path = write_jsonl(scratch / f"{pair.file_stem}.jsonl", [row])
        load = load_average()
        started = time.perf_counter()
        rows = execute(detect_plan(str(input_path), variant, pair.animal, stride, conf=conf, stop_after_n=1))
        live_seconds = time.perf_counter() - started
        estimate = yoloe_outcomes(scans[pair.pair_id], *MAIN_RULE)["yoloe_early"]
        live_answer = any(entry.get("bboxes") for entry in rows[0].get("detections", []))
        checks.append(
            {
                "pair_id": pair.pair_id,
                "truth": pair.truth,
                "live_answer": live_answer,
                "estimated_answer": estimate["answer"],
                "answers_match": live_answer == estimate["answer"],
                "live_seconds": live_seconds,
                "estimated_seconds": estimate["seconds"],
                "relative_error": (estimate["seconds"] - live_seconds) / live_seconds if live_seconds else None,
                "load_average": load,
            }
        )
    report = {"variant": MAIN_VARIANT, "device": device, "gpu": gpu_name, "pairs": checks}
    write_json_atomic(out_root / "yoloe" / "validation.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--wld-root", type=Path, default=WLD_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="scan every pair (or film) with one variant")
    scan.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    validate = commands.add_parser("validate", help="time the real early stop on five pairs")
    for sub in (scan, validate):
        sub.add_argument("--allow-cpu", action="store_true")

    args = parser.parse_args(argv)
    pairs = load_pairs(args.wld_root)
    if args.command == "scan":
        run_scan(args.variant, pairs, allow_cpu=args.allow_cpu)
    else:
        print(json.dumps(run_validation(pairs, allow_cpu=args.allow_cpu), indent=2))


if __name__ == "__main__":
    main()
