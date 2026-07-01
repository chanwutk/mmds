"""Run the full-match goal detection pipeline and evaluate against ground truth.

Usage:
    GEMINI_API_KEY=<key> PYTHONPATH=src:. .venv/bin/python scripts/run_match_benchmark.py
    GEMINI_API_KEY=<key> PYTHONPATH=src:. .venv/bin/python scripts/run_match_benchmark.py \\
        --output examples/match_results.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

from mmds import GeminiPromptExecutor, execute  # noqa: E402

DATA_FILE = ROOT / "data" / "matches.jsonl"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Gemini 2.5 Flash pricing
_INPUT_COST_PER_M = 0.15
_OUTPUT_COST_PER_M = 0.60

# A predicted timestamp counts as correct if within this many seconds of GT
MATCH_TOLERANCE_SEC = 60.0


def _format_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def _load_query(path: str):
    spec = importlib.util.spec_from_file_location("_query", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _evaluate(result_rows: list[dict], matches_data: list[dict]) -> dict:
    """Compare predicted goal timestamps to ground truth per half."""
    # Build GT lookup: {(game, half) -> [timestamp_sec]}
    gt: dict[tuple, list[float]] = {}
    for row in matches_data:
        key = (row["game"], row["half"])
        gt[key] = row.get("ground_truth_goals", [])

    tp = 0
    fp = 0
    fn_total = 0
    details = []

    # Group predictions by (game, half)
    from collections import defaultdict
    preds: dict[tuple, list[float]] = defaultdict(list)
    for r in result_rows:
        key = (r["game"], r["half"])
        preds[key].append(r["candidate_time_sec"])

    all_keys = set(gt.keys()) | set(preds.keys())
    for key in sorted(all_keys):
        game, half = key
        gt_times = sorted(gt.get(key, []))
        pred_times = sorted(preds.get(key, []))

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        for pi, pt in enumerate(pred_times):
            for gi, gt_t in enumerate(gt_times):
                if gi not in matched_gt and abs(pt - gt_t) <= MATCH_TOLERANCE_SEC:
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    break

        half_tp = len(matched_gt)
        half_fp = len(pred_times) - len(matched_pred)
        half_fn = len(gt_times) - len(matched_gt)
        tp += half_tp
        fp += half_fp
        fn_total += half_fn

        details.append({
            "game": Path(game).name,
            "half": half,
            "gt_goals": [_format_time(t) for t in gt_times],
            "predictions": [_format_time(t) for t in pred_times],
            "tp": half_tp, "fp": half_fp, "fn": half_fn,
        })

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn_total) if (tp + fn_total) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn_total,
        "precision": precision, "recall": recall, "f1": f1,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        sys.exit("Set GEMINI_API_KEY before running.")
    if not DATA_FILE.exists():
        sys.exit(f"{DATA_FILE} not found. Run prepare_match_data.py first.")

    matches_data = [json.loads(l) for l in DATA_FILE.read_text().splitlines() if l.strip()]
    total_gt = sum(len(r.get("ground_truth_goals", [])) for r in matches_data)
    print(f"Matches: {len(matches_data)} halves | Ground truth goals: {total_gt}\n")

    executor = GeminiPromptExecutor(
        api_key=GEMINI_API_KEY,
        model=GEMINI_MODEL,
        file_ready_timeout_seconds=300,
    )

    # ── Naive baseline ──────────────────────────────────────────────────────
    naive_mod = _load_query("examples/match_goal_detection_naive.py")
    print("Running naive pipeline (30s chunks, every clip to Gemini)…")
    t0 = time.time()
    naive_rows = execute(naive_mod.output, prompt_executor=executor)
    naive_time = time.time() - t0
    naive_calls = len([json.loads(l) for l in DATA_FILE.read_text().splitlines() if l.strip()])
    # Count how many chunks were generated (all sent to Gemini)
    from udfs.match_ops import NAIVE_CHUNK_SEC
    import subprocess as _sp
    naive_clip_count = 0
    for row in matches_data:
        probe = _sp.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", row["source"]],
            capture_output=True, text=True,
        )
        dur = float(probe.stdout.strip()) if probe.stdout.strip() else 45 * 60.0
        naive_clip_count += int(dur // NAIVE_CHUNK_SEC)

    print(f"  {naive_clip_count} clips sent to Gemini | "
          f"{len(naive_rows)} confirmed as goals | {naive_time:.1f}s\n")
    naive_metrics = _evaluate(naive_rows, matches_data)

    # ── Optimized (Whisper gate) ────────────────────────────────────────────
    opt_mod = _load_query("examples/match_goal_detection.py")
    print("Running optimized pipeline (Whisper gate → candidates only)…")
    t0 = time.time()
    opt_rows = execute(opt_mod.output, prompt_executor=executor)
    opt_time = time.time() - t0
    # Count candidates that were sent to Gemini
    from udfs.match_ops import find_goal_candidates
    opt_clip_count = sum(
        len(find_goal_candidates(row)["candidates"]) for row in matches_data
    )
    print(f"  {opt_clip_count} clips sent to Gemini | "
          f"{len(opt_rows)} confirmed as goals | {opt_time:.1f}s\n")
    opt_metrics = _evaluate(opt_rows, matches_data)

    savings_calls = (naive_clip_count - opt_clip_count) / naive_clip_count * 100

    # ── Table ───────────────────────────────────────────────────────────────
    W = 56
    def row(label: str, nv: object, ov: object, fmt: str = "") -> str:
        ns = format(nv, fmt) if fmt else str(nv)
        os_ = format(ov, fmt) if fmt else str(ov)
        return f"║ {label:<30} {ns:>10} {os_:>10} ║"

    print(f"\n╔{'═' * W}╗")
    print(f"║{'FULL-MATCH GOAL DETECTION BENCHMARK':^{W}}║")
    print(f"╠{'═' * W}╣")
    print(f"║ {'Metric':<30} {'Naive':>10} {'Whisper':>10} ║")
    print(f"╠{'═' * W}╣")
    print(row("Gemini calls (clips)", naive_clip_count, opt_clip_count))
    print(row("Call savings", "—", f"{savings_calls:.1f}%"))
    print(row("Confirmed as goals", len(naive_rows), len(opt_rows)))
    print(f"╠{'═' * W}╣")
    print(row("Precision", naive_metrics["precision"], opt_metrics["precision"], ".3f"))
    print(row("Recall",    naive_metrics["recall"],    opt_metrics["recall"],    ".3f"))
    print(row("F1",        naive_metrics["f1"],        opt_metrics["f1"],        ".3f"))
    print(row("TP / FP / FN",
              f"{naive_metrics['tp']}/{naive_metrics['fp']}/{naive_metrics['fn']}",
              f"{opt_metrics['tp']}/{opt_metrics['fp']}/{opt_metrics['fn']}"))
    print(f"╠{'═' * W}╣")
    print(row("Wall time (s)", f"{naive_time:.1f}", f"{opt_time:.1f}"))
    print(f"╚{'═' * W}╝")

    print(f"\nPer-half breakdown (Whisper):")
    for d in opt_metrics["details"]:
        print(f"  Half {d['half']}  {d['game']}")
        print(f"    GT:          {d['gt_goals']}")
        print(f"    Predictions: {d['predictions']}")
        print(f"    TP={d['tp']}  FP={d['fp']}  FN={d['fn']}")

    if args.output:
        result = {
            "model": GEMINI_MODEL,
            "tolerance_sec": MATCH_TOLERANCE_SEC,
            "dataset": {"halves": len(matches_data), "total_goals": total_gt},
            "naive": {
                "gemini_calls": naive_clip_count,
                "confirmed_goals": len(naive_rows),
                "wall_time_s": round(naive_time, 2),
                **{k: v for k, v in naive_metrics.items() if k != "details"},
                "per_half": naive_metrics["details"],
            },
            "whisper": {
                "gemini_calls": opt_clip_count,
                "confirmed_goals": len(opt_rows),
                "wall_time_s": round(opt_time, 2),
                "call_savings_pct": round(savings_calls, 2),
                **{k: v for k, v in opt_metrics.items() if k != "details"},
                "per_half": opt_metrics["details"],
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
