"""Experiment 1: "Is a {animal} visible anywhere in this film?" on WLD.

Every plan answers the same yes/no question for every (film, animal) pair:

- ``mllm_full``: Gemini watches the whole film once.
- ``mllm_early``: Gemini watches 2-minute windows of the film in order and
  stops at the first "yes".
- ``yoloe_mllm``: YOLOE as a filter (``wld_yoloe.FILTER_VARIANT``). Gemini
  watches only YOLOE's padded hit windows, most confident first, and stops at
  the first "yes"; no window means "no". Its time is the YOLOE scan plus
  Gemini's calls; the scan's time is read from the scan files when scoring,
  so a rescan updates every repeat.
- ``yoloe_full`` and ``yoloe_early``: YOLOE alone, for every scan variant,
  computed from the scans in ``wld_yoloe`` (no Gemini).

Gemini reads each film from its public YouTube URL (``Pair.video``), with a
start/end window for 2-minute chunks and YOLOE windows. Nothing is uploaded,
so the Files API storage quota (which deleting files did not free) is never
used, and Gemini's time includes fetching the video but no upload. Gemini
plans run ``--repeat`` 1, 2, 3 for a mean and spread.

Steps, from the repo root with ``PYTHONPATH=src:.``, after ``wld prepare``
and the YOLOE scans (see WLD_README.md)::

    python -m scripts.query_types.wld_presence run --plan mllm_full --repeat 1
    python -m scripts.query_types.wld_presence run --plan mllm_early --repeat 1
    python -m scripts.query_types.wld_presence run --plan yoloe_mllm --repeat 1
    python -m scripts.query_types.wld_presence score
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mmds import Input, Map, Record
from mmds.model import DatasetExpr
from scripts.query_types.common import (
    aggregate,
    binary_scores,
    cost_comparison,
    f1_score,
    fmt,
    format_cost_comparison,
    gpu_dollars,
    make_chunks,
    plan_dollars,
    read_json,
    summarize_usage,
    usage_dollars,
    write_json_atomic,
)
from scripts.query_types.wld import OUT_ROOT, WLD_ROOT, Pair, load_pairs
from scripts.query_types.wld_gemini import RETRY_WAITS, ExecutorFactory, Task, ask_row, run_pairs
from scripts.query_types.wld_yoloe import (
    FILTER_VARIANT,
    MAIN_RULE,
    SWEEP_CONFS,
    SWEEP_PERSISTENCE,
    VARIANTS,
    load_scans,
    ranked_windows,
    yoloe_outcomes,
)

CHUNK_SECONDS = 120.0
GEMINI_PLANS = ("mllm_full", "mllm_early", "yoloe_mllm")
# The naive plan every other plan's cost is compared with.
REFERENCE_PLAN = "mllm_full"
SCHEMA = {"visible": "boolean"}
PROMPT = [
    "Is a ",
    Record["animal"],
    " visible on screen at any moment in this video? Count only a real ",
    Record["animal"],
    " that appears in the footage, not one that is only mentioned in the narration. "
    "Answer true or false.\n",
    Record["video"],
]


# ---------------------------------------------------------------------------
# Gemini plans
# ---------------------------------------------------------------------------


def presence_plan(input_path: str) -> DatasetExpr:
    return Map(Input(input_path), PROMPT, schema=SCHEMA, name="presence")


def ask(executor: Any, pair: Pair, video: Mapping[str, Any], scratch: Path) -> bool:
    """Ask the presence question about one video prompt value."""
    row = {"pair_id": pair.pair_id, "animal": pair.animal, "video": dict(video)}
    answer = ask_row(presence_plan, row, scratch / f"{pair.file_stem}.jsonl", executor).get("visible")
    if not isinstance(answer, bool):
        raise ValueError(f"{pair.pair_id}: expected a boolean 'visible', got {answer!r}.")
    return answer


def whole_film(pair: Pair, executor: Any, scratch: Path) -> dict[str, Any]:
    started = time.perf_counter()
    answer = ask(executor, pair, pair.video(), scratch)
    return {"answer": answer, "seconds": time.perf_counter() - started, "video_seconds_sent": pair.duration}


def early_stop(pair: Pair, executor: Any, scratch: Path) -> dict[str, Any]:
    """Ask about 2-minute windows of the film in order; stop at the first yes."""
    started = time.perf_counter()
    chunks = make_chunks(pair.duration, CHUNK_SECONDS)
    answer = False
    asked = 0
    for start, end in chunks:
        asked += 1
        answer = ask(executor, pair, pair.video(start, end), scratch)
        if answer:
            break
    return {
        "answer": answer,
        "seconds": time.perf_counter() - started,
        "chunks_asked": asked,
        "chunks_total": len(chunks),
        "video_seconds_sent": chunks[asked - 1][1],
    }


def yoloe_then_gemini(scans: Mapping[str, Mapping[str, Any]]) -> Task:
    """Ask Gemini about YOLOE's windows, most confident first; stop at the first yes."""

    def task(pair: Pair, executor: Any, scratch: Path) -> dict[str, Any]:
        scan = scans[pair.pair_id]
        started = time.perf_counter()
        windows = ranked_windows(scan, pair)
        answer = False
        sent = 0.0
        asked = 0
        for (start, end), _ in windows:
            asked += 1
            sent += end - start
            if ask(executor, pair, pair.video(start, end), scratch):
                answer = True
                break
        return {
            "answer": answer,
            # Gemini's time only. Scoring adds the YOLOE scan's time from the
            # scan files, so a rescan updates every repeat.
            "gemini_seconds": time.perf_counter() - started,
            "windows_asked": asked,
            "windows_total": len(windows),
            "video_seconds_sent": sent,
        }

    return task


def filter_scans(pairs: Sequence[Pair], out_root: Path = OUT_ROOT) -> dict[str, dict[str, Any]]:
    """The filter variant's scan for every pair; stops if any pair has none."""
    scans = load_scans(FILTER_VARIANT, pairs, out_root)
    missing = [p.pair_id for p in pairs if p.pair_id not in scans]
    if missing:
        raise SystemExit(f"yoloe_mllm needs '{FILTER_VARIANT}' scans for {len(missing)} pairs (e.g. {missing[0]}).")
    return scans


def plan_dir(plan: str, repeat: int, out_root: Path = OUT_ROOT) -> Path:
    return out_root / "presence" / f"run{repeat}" / plan


def run(
    plan: str,
    pairs: Sequence[Pair],
    repeat: int,
    *,
    out_root: Path = OUT_ROOT,
    executor_factory: ExecutorFactory | None = None,
    waits: Sequence[float] = RETRY_WAITS,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run one Gemini plan over every pair not yet done in this repeat."""
    if plan not in GEMINI_PLANS:
        raise ValueError(f"Unknown plan {plan!r}; choose from {GEMINI_PLANS}.")
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if plan == "yoloe_mllm":
        task = yoloe_then_gemini(filter_scans(pairs, out_root))
    else:
        task = whole_film if plan == "mllm_full" else early_stop
    run_pairs(
        pairs, plan_dir(plan, repeat, out_root), task,
        scratch=out_root / "scratch" / "presence" / plan,
        executor_factory=executor_factory, waits=waits, sleep=sleep,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_answers(pairs: Sequence[Pair], answers: Mapping[str, bool], *, strict: bool = False) -> dict[str, Any]:
    """Score yes/no answers. Pairs with a brief appearance are left out,
    unless ``strict``, where any annotated frame counts as present."""
    scored = []
    for pair in pairs:
        if strict:
            actual = pair.truth_any_frame
        elif pair.truth == "brief":
            continue
        else:
            actual = pair.truth == "present"
        scored.append((answers[pair.pair_id], actual))
    scores = binary_scores(scored)
    return {"pairs": len(scored), **scores, "f1": f1_score(scores["precision"], scores["recall"])}


def score_gemini_plan(plan: str, pairs: Sequence[Pair], out_root: Path = OUT_ROOT) -> dict[str, Any] | None:
    """Mean and spread over complete repeats; incomplete repeats are listed, not scored.

    ``yoloe_mllm``'s time is the YOLOE scan's, read from the current scan files,
    plus Gemini's own time on the windows; its dollars include the scan's GPU time.
    """
    runs = []
    incomplete = []
    scans: dict[str, dict[str, Any]] | None = None
    for run_dir in sorted((out_root / "presence").glob("run*")):
        files = [f for f in (run_dir / plan).glob("*.json") if not f.name.startswith("_")]
        results = {r["pair_id"]: r for r in map(read_json, files)}
        if not results:
            continue
        if set(results) != {pair.pair_id for pair in pairs}:
            incomplete.append({"run": run_dir.name, "pairs_done": len(results)})
            continue
        if plan == "yoloe_mllm":
            if scans is None:
                scans = filter_scans(pairs, out_root)
            yoloe_seconds = sum(scans[pair_id]["seconds"] for pair_id in results)
            gemini_seconds = sum(r["gemini_seconds"] for r in results.values())
            timing = {"seconds": yoloe_seconds + gemini_seconds, "yoloe_seconds": yoloe_seconds,
                      "gemini_seconds": gemini_seconds}
        else:
            timing = {"seconds": sum(r["seconds"] for r in results.values())}
        answers = {pair_id: result["answer"] for pair_id, result in results.items()}
        records = [record for result in results.values() for record in result["usage"]]
        usage = summarize_usage(records)
        runs.append(
            {
                "main": score_answers(pairs, answers),
                "strict": score_answers(pairs, answers, strict=True),
                **timing,
                "video_seconds_sent": sum(r["video_seconds_sent"] for r in results.values()),
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"] + usage["thinking_tokens"],
                "calls": usage["calls"],
                **plan_dollars(usage_dollars(records), timing.get("yoloe_seconds", 0.0)),
            }
        )
    if not runs and not incomplete:
        return None
    return {"plan": plan, "runs": len(runs), "incomplete": incomplete, **(aggregate(runs) if runs else {})}


def score_yoloe_variant(pairs: Sequence[Pair], scans: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """YOLOE alone at every rule of the sweep (scans are deterministic: one run)."""
    total_video = sum(pair.duration for pair in pairs)
    sweep = []
    for conf in SWEEP_CONFS:
        for persistence in SWEEP_PERSISTENCE:
            outcomes = {pair.pair_id: yoloe_outcomes(scans[pair.pair_id], conf, persistence) for pair in pairs}
            answers = {pair_id: o["yoloe_full"]["answer"] for pair_id, o in outcomes.items()}
            full_seconds = sum(o["yoloe_full"]["seconds"] for o in outcomes.values())
            early_seconds = sum(o["yoloe_early"]["seconds"] for o in outcomes.values())
            sweep.append(
                {
                    "conf": conf,
                    "persistence": persistence,
                    **score_answers(pairs, answers),
                    "strict_f1": score_answers(pairs, answers, strict=True)["f1"],
                    "full_seconds": full_seconds,
                    "early_seconds": early_seconds,
                    "full_dollars": gpu_dollars(full_seconds),
                    "early_dollars": gpu_dollars(early_seconds),
                    "early_video_fraction": sum(
                        outcomes[pair.pair_id]["yoloe_early"]["fraction_scanned"] * pair.duration for pair in pairs
                    ) / total_video,
                }
            )
    main = next(entry for entry in sweep if (entry["conf"], entry["persistence"]) == MAIN_RULE)
    devices = sorted({str(scan.get("gpu") or scan.get("device")) for scan in scans.values()})
    return {"main_rule": main, "sweep": sweep, "devices": devices}


def score_results(pairs: Sequence[Pair], out_root: Path = OUT_ROOT) -> dict[str, Any]:
    counts = {label: sum(1 for p in pairs if p.truth == label) for label in ("present", "brief", "absent")}
    summary: dict[str, Any] = {"pairs": len(pairs), "truth": counts, "gemini": {}, "yoloe": {}, "validation": None}
    for plan in GEMINI_PLANS:
        scored = score_gemini_plan(plan, pairs, out_root)
        if scored is not None:
            summary["gemini"][plan] = scored
    for variant in VARIANTS:
        scans = load_scans(variant, pairs, out_root)
        if len(scans) == len(pairs):
            summary["yoloe"][variant] = score_yoloe_variant(pairs, scans)
        elif scans:
            summary["yoloe"][variant] = {"incomplete": f"{len(scans)} of {len(pairs)} pairs scanned"}
    validation = out_root / "yoloe" / "validation.json"
    if validation.exists():
        summary["validation"] = read_json(validation)
    summary["costs"] = presence_costs(summary)
    return summary


def presence_costs(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Break-even GPU prices of the plans that use YOLOE, against ``REFERENCE_PLAN``."""
    scored_runs = {plan: s for plan, s in summary["gemini"].items() if s["runs"]}
    reference = scored_runs.get(REFERENCE_PLAN)
    gpu_plans: dict[str, tuple[float | None, float]] = {}
    if "yoloe_mllm" in scored_runs:
        filtered = scored_runs["yoloe_mllm"]
        gpu_plans["yoloe_mllm"] = (filtered["gemini_dollars"]["mean"], filtered["yoloe_seconds"]["mean"])
    for variant, scored in summary["yoloe"].items():
        if "main_rule" in scored:
            for plan in ("full", "early"):
                gpu_plans[f"yoloe_{plan} ({variant})"] = (0.0, scored["main_rule"][f"{plan}_seconds"])
    return cost_comparison(REFERENCE_PLAN, reference["dollars"]["mean"] if reference else None, gpu_plans)


def format_report(summary: Mapping[str, Any]) -> str:
    truth = summary["truth"]
    lines = [
        f"Pairs: {summary['pairs']} (present {truth['present']}, brief {truth['brief']}, absent {truth['absent']}; "
        "brief pairs are left out of the main scores)",
        "",
        (
            "| Plan | Runs | Precision | Recall | F1 | Strict F1 | Seconds | Video sent (s) | Input tokens "
            "| Gemini $ | GPU $ | Total $ |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for plan, s in summary["gemini"].items():
        if s["runs"]:
            lines.append(
                f"| {plan} | {s['runs']} | {fmt(s['main']['precision'])} | {fmt(s['main']['recall'])} "
                f"| {fmt(s['main']['f1'])} | {fmt(s['strict']['f1'])} | {fmt(s['seconds'], 1)} "
                f"| {fmt(s['video_seconds_sent'], 0)} | {fmt(s['input_tokens'], 0)} "
                f"| {fmt(s['gemini_dollars'], 3)} | {fmt(s['gpu_dollars'], 3)} | {fmt(s['dollars'], 3)} |"
            )
    scored = {variant: s for variant, s in summary["yoloe"].items() if "main_rule" in s}
    for variant, s in scored.items():
        main = s["main_rule"]
        for plan in ("full", "early"):
            lines.append(
                f"| yoloe_{plan} ({variant}) | 1 | {fmt(main['precision'])} | {fmt(main['recall'])} | {fmt(main['f1'])} "
                f"| {fmt(main['strict_f1'])} | {fmt(main[f'{plan}_seconds'], 1)} | n/a | 0 "
                f"| 0 | {fmt(main[f'{plan}_dollars'], 3)} | {fmt(main[f'{plan}_dollars'], 3)} |"
            )
    lines += format_cost_comparison(summary.get("costs") or {})
    for variant, s in scored.items():
        lines += [
            "",
            f"YOLOE sweep, {variant} (devices: {', '.join(s['devices'])}):",
            "",
            "| Confidence | Persistence | Precision | Recall | F1 | Early-stop seconds | Video scanned |",
            "|---|---|---|---|---|---|---|",
        ]
        for entry in s["sweep"]:
            lines.append(
                f"| {entry['conf']} | {entry['persistence']} | {fmt(entry['precision'])} | {fmt(entry['recall'])} "
                f"| {fmt(entry['f1'])} | {fmt(entry['early_seconds'], 1)} | {fmt(entry['early_video_fraction'])} |"
            )
    for plan, s in summary["gemini"].items():
        if s["incomplete"]:
            lines.append(f"\n{plan}: unfinished repeats, not scored: {s['incomplete']}")
    for variant, s in summary["yoloe"].items():
        if "incomplete" in s:
            lines.append(f"\nYOLOE {variant}: {s['incomplete']}, not scored")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--wld-root", type=Path, default=WLD_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run one Gemini plan over every pair")
    run_parser.add_argument("--plan", choices=GEMINI_PLANS, required=True)
    run_parser.add_argument("--repeat", type=int, default=1)
    commands.add_parser("score", help="print and save the results")

    args = parser.parse_args(argv)
    pairs = load_pairs(args.wld_root)
    if args.command == "run":
        run(args.plan, pairs, args.repeat)
    else:
        summary = score_results(pairs, OUT_ROOT)
        write_json_atomic(OUT_ROOT / "presence" / "summary.json", summary)
        print(format_report(summary))


if __name__ == "__main__":
    main()
