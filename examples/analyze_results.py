"""Comprehensive analysis of the soccer benchmark results.

Shows:
  - Text gate decision for every clip (passed / pruned) with the reason
  - Which goal clips the gate missed (false negatives at the gate)
  - Per-game breakdown: goals, non-goals, gate pass rate
  - API call log details (tokens, cost, Gemini response) if call log exists

Usage:
    PYTHONPATH=src:. .venv/bin/python examples/analyze_results.py
    PYTHONPATH=src:. .venv/bin/python examples/analyze_results.py --calls examples/soccer_results_5_calls.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from udfs.soccer_ops import (  # noqa: E402
    _GOAL_SIGNAL_WORDS,
    _GOAL_SIGNAL_PHRASES,
    is_goal_candidate,
)

DATA_FILE = ROOT / "data" / "soccer_goal_clips.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gate_reason(row: dict) -> str:
    text = str(row.get("transcript") or "").strip().lower()
    if not text:
        return "no transcript → conservative pass"
    tokens = {w.strip(".,!?;:'\"()-") for w in text.split()}
    matched = tokens & _GOAL_SIGNAL_WORDS
    if matched:
        return f"word match: {sorted(matched)}"
    for phrase in _GOAL_SIGNAL_PHRASES:
        if phrase in text:
            return f"phrase match: '{phrase}'"
    return "no signal → pruned"


def _short_game(game: str) -> str:
    return Path(game).name if game else "?"


# ---------------------------------------------------------------------------
# Gate analysis
# ---------------------------------------------------------------------------

def gate_analysis(rows: list[dict]) -> None:
    passed = [r for r in rows if is_goal_candidate(r)]
    pruned = [r for r in rows if not is_goal_candidate(r)]

    missed_goals = [r for r in pruned if r["true_label"]]
    gate_fp      = [r for r in passed if not r["true_label"]]  # non-goals that passed

    print(f"{'=' * 80}")
    print(f"TEXT GATE SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total clips    : {len(rows)}")
    print(f"  Passed to LLM  : {len(passed)}  ({len(passed)/len(rows)*100:.1f}%)")
    print(f"  Pruned         : {len(pruned)}  ({len(pruned)/len(rows)*100:.1f}%)")
    print(f"  Goal recall    : {len(passed) - len(gate_fp)} / "
          f"{sum(r['true_label'] for r in rows)} goals reached LLM")
    print(f"  Missed goals   : {len(missed_goals)}")
    print()

    # --- Per-game breakdown ---
    print(f"{'=' * 80}")
    print(f"PER-GAME BREAKDOWN")
    print(f"{'=' * 80}")
    by_game: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_game[r.get("game", "?")].append(r)

    for game, clips in sorted(by_game.items()):
        goals = [c for c in clips if c["true_label"]]
        goals_passed = [c for c in goals if is_goal_candidate(c)]
        non_goals_passed = [c for c in clips if not c["true_label"] and is_goal_candidate(c)]
        print(f"  {_short_game(game)}")
        print(f"    Goals:       {len(goals)}  ({len(goals_passed)} passed gate, "
              f"{len(goals) - len(goals_passed)} missed)")
        print(f"    Non-goals:   {len(clips) - len(goals)}  "
              f"({len(non_goals_passed)} passed gate = potential LLM false positives)")
        print()

    # --- Missed goals ---
    if missed_goals:
        print(f"{'=' * 80}")
        print(f"GOALS MISSED BY TEXT GATE  (fn={len(missed_goals)})")
        print(f"{'=' * 80}")
        for r in missed_goals:
            clip = Path(r["video"]["source"]).name
            t = (r.get("transcript") or "")
            print(f"  {clip}  game: {_short_game(r.get('game','?'))}")
            print(f"    Transcript: {t!r}")
            print()

    # --- All clips detail ---
    print(f"{'=' * 80}")
    print(f"ALL CLIPS — GATE DECISIONS")
    print(f"{'=' * 80}")
    for r in rows:
        clip = Path(r["video"]["source"]).name
        passed_gate = is_goal_candidate(r)
        label = "GOAL" if r["true_label"] else "non-goal"
        status = "PASS" if passed_gate else "PRUNE"
        reason = _gate_reason(r)
        t = (r.get("transcript") or "")[:80] or "[empty]"
        flag = ""
        if r["true_label"] and not passed_gate:
            flag = "  ← MISSED GOAL"
        if not r["true_label"] and passed_gate and "no transcript" not in reason:
            flag = "  ← gate FP"
        print(f"  {clip}  [{label}]  [{status}]  {reason}{flag}")
        print(f"    {t!r}")
        print()


# ---------------------------------------------------------------------------
# API call log analysis
# ---------------------------------------------------------------------------

def call_log_analysis(log_path: Path) -> None:
    calls = json.loads(log_path.read_text())
    print(f"{'=' * 80}")
    print(f"API CALL LOG  ({len(calls)} total calls)")
    print(f"{'=' * 80}")

    for pipeline in ("Baseline", "Optimized"):
        pcalls = [c for c in calls if c["pipeline"] == pipeline]
        if not pcalls:
            continue
        total_in  = sum(c["input_tokens"] for c in pcalls)
        total_out = sum(c["output_tokens"] for c in pcalls)
        total_cost = sum(c["cost_usd"] for c in pcalls)
        print(f"\n  Pipeline: {pipeline}")
        print(f"    Calls       : {len(pcalls)}")
        print(f"    Input tokens: {total_in:,}  (avg {total_in//len(pcalls):,}/call)")
        print(f"    Output tokens:{total_out:,}  (avg {total_out//len(pcalls):,}/call)")
        print(f"    Total cost  : ${total_cost:.4f}  (avg ${total_cost/len(pcalls):.5f}/call)")
        print()

        fp_calls = [c for c in pcalls if c["true_label"] is False and c["response"].lower().startswith("true")]
        fn_calls = [c for c in pcalls if c["true_label"] is True  and c["response"].lower().startswith("false")]
        print(f"    False positives (Gemini said goal on non-goal): {len(fp_calls)}")
        for c in fp_calls:
            print(f"      {c['clip_id']}  game: {_short_game(c['game'])}  tokens={c['input_tokens']}")
        print(f"    False negatives (Gemini missed real goal): {len(fn_calls)}")
        for c in fn_calls:
            print(f"      {c['clip_id']}  game: {_short_game(c['game'])}  tokens={c['input_tokens']}")

    print(f"\n{'=' * 80}")
    print(f"PER-CALL DETAIL")
    print(f"{'=' * 80}")
    for c in calls:
        label = "GOAL" if c["true_label"] else "non-goal"
        correct = (c["response"].lower().startswith("true")) == c["true_label"]
        mark = "✓" if correct else "✗"
        print(f"  [{c['pipeline']:<10}] {c['clip_id']}  [{label}]  "
              f"response={c['response']:<5}  {mark}  "
              f"in={c['input_tokens']:>6} out={c['output_tokens']}  ${c['cost_usd']:.5f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=Path, default=None,
                        help="Path to _calls.json log from run_soccer_benchmark.py")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        sys.exit(f"{DATA_FILE} not found. Run prepare_soccer_data.py first.")

    rows = [json.loads(l) for l in DATA_FILE.read_text().splitlines() if l.strip()]
    gate_analysis(rows)

    if args.calls:
        if not args.calls.exists():
            print(f"Call log not found: {args.calls}")
        else:
            call_log_analysis(args.calls)


if __name__ == "__main__":
    main()
