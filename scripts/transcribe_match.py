"""Transcribe a full SoccerNet match with Whisper and find goal candidate timestamps.

Runs Whisper on both halves of a match, saves timestamped transcripts, then
scans for goal-signal vocabulary and compares predicted timestamps against
SoccerNet ground truth.

Usage:
    PYTHONPATH=src:. .venv/bin/python scripts/transcribe_match.py \\
        --game "england_epl/2015-2016/2016-02-13 - 20-30 Chelsea 5 - 1 Newcastle Utd"

Output:
    data/transcripts/<safe_game_name>_half1.json
    data/transcripts/<safe_game_name>_half2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOCCERNET_DIR = ROOT / "data" / "soccernet"
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"

from udfs.soccer_ops import _GOAL_SIGNAL_WORDS, _GOAL_SIGNAL_PHRASES, _WHISPER_EXTRA_WORDS, _WHISPER_EXTRA_PHRASES  # noqa: E402

_ALL_SIGNAL_WORDS = _GOAL_SIGNAL_WORDS | _WHISPER_EXTRA_WORDS
_ALL_SIGNAL_PHRASES = _GOAL_SIGNAL_PHRASES + _WHISPER_EXTRA_PHRASES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_name(game: str) -> str:
    return game.replace("/", "_").replace(" ", "_").replace("-", "_")


def _load_ground_truth(game_dir: Path) -> dict[str, list[float]]:
    """Return {half_str -> [timestamp_sec, ...]} for all visible goals."""
    labels_path = game_dir / "Labels-v2.json"
    if not labels_path.exists():
        return {"1": [], "2": []}
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    result: dict[str, list[float]] = {"1": [], "2": []}
    for ann in data.get("annotations", []):
        if ann.get("label") != "Goal" or ann.get("visibility", "visible") != "visible":
            continue
        half = ann["gameTime"].split(" - ")[0]
        if half in result:
            result[half].append(int(ann["position"]) / 1000.0)
    return result


# Phrases that contain goal-signal words but describe non-goal contexts.
# Checked as substrings — if any matches, the segment is suppressed.
_FALSE_POSITIVE_PHRASES = (
    "on goal",          # "shots on goal", "attempts on goal"
    "behind the goal",  # crowd/camera position
    "goal difference",  # league table talk
    "goal kick",        # restart after ball goes out
    "goal line",        # defensive clearance
    "no goal",          # disallowed goal announcement
    "own goal",         # handled separately if needed
)


def _scan_for_goals(segments: list[dict]) -> list[dict]:
    """Return segments that contain goal-signal vocabulary, filtering known FP patterns."""
    hits = []
    for seg in segments:
        text = seg["text"].strip().lower()

        # Suppress known false positive contexts first
        if any(fp in text for fp in _FALSE_POSITIVE_PHRASES):
            continue

        tokens = {w.strip(".,!?;:'\"()-") for w in text.split()}
        matched_word = tokens & _ALL_SIGNAL_WORDS
        matched_phrase = next((p for p in _ALL_SIGNAL_PHRASES if p in text), None)
        if matched_word or matched_phrase:
            hits.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "signal": sorted(matched_word) if matched_word else [matched_phrase],
            })
    return hits


def _cluster_hits(hits: list[dict], gap_sec: float = 30.0) -> list[dict]:
    """Merge nearby signal hits into single candidate goal events."""
    if not hits:
        return []
    clusters = []
    cur = dict(hits[0])
    for h in hits[1:]:
        if h["start"] - cur["end"] <= gap_sec:
            cur["end"] = max(cur["end"], h["end"])
            cur["signal"] = sorted(set(cur["signal"]) | set(h["signal"]))
            cur["text"] = cur["text"] + " | " + h["text"]
        else:
            clusters.append(cur)
            cur = dict(h)
    clusters.append(cur)
    return clusters


def _format_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_half(mkv_path: Path, half: str, out_path: Path, model) -> list[dict]:
    if out_path.exists():
        print(f"  Half {half}: loading cached transcript from {out_path.name}")
        return json.loads(out_path.read_text())["segments"]

    print(f"  Half {half}: transcribing {mkv_path.name} …", flush=True)
    t0 = time.time()
    result = model.transcribe(str(mkv_path), fp16=False, language="en",
                              verbose=False, condition_on_previous_text=True)
    elapsed = time.time() - t0

    segments = [
        {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"]}
        for s in result["segments"]
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "game": str(mkv_path.parent),
        "half": half,
        "language": result.get("language", "?"),
        "duration_s": round(elapsed, 1),
        "segments": segments,
    }, indent=2, ensure_ascii=False))

    print(f"    Done in {elapsed:.0f}s — {len(segments)} segments saved → {out_path.name}")
    return segments


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(game: str, game_dir: Path) -> None:
    ground_truth = _load_ground_truth(game_dir)
    safe = _safe_name(game)

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

    print(f"\nLoading Whisper model 'small'…")
    import whisper
    model = whisper.load_model("small")
    print("Model ready.\n")

    all_candidates: dict[str, list[dict]] = {}

    for half in ("1", "2"):
        mkv_path = game_dir / f"{half}_224p.mkv"
        if not mkv_path.exists():
            print(f"  Half {half}: {mkv_path.name} not found, skipping.")
            continue

        out_path = TRANSCRIPT_DIR / f"{safe}_half{half}.json"
        segments = transcribe_half(mkv_path, half, out_path, model)

        hits = _scan_for_goals(segments)
        candidates = _cluster_hits(hits)
        all_candidates[half] = candidates

    # ── Results ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"GOAL DETECTION RESULTS — {Path(game).name}")
    print(f"{'=' * 70}")

    total_gt = sum(len(v) for v in ground_truth.values())
    total_candidates = sum(len(v) for v in all_candidates.values())
    print(f"Ground truth goals : {total_gt}")
    print(f"Candidates found   : {total_candidates}")
    print()

    for half in ("1", "2"):
        gt_times = ground_truth.get(half, [])
        candidates = all_candidates.get(half, [])
        print(f"  HALF {half}  ({len(gt_times)} real goals, {len(candidates)} candidates)")
        print(f"  {'─' * 66}")

        matched_gt = set()
        for c in candidates:
            c_mid = (c["start"] + c["end"]) / 2
            # Match if within 60s of any ground truth goal
            match = next(
                (i for i, gt in enumerate(gt_times)
                 if abs(c_mid - gt) < 60 and i not in matched_gt),
                None,
            )
            if match is not None:
                matched_gt.add(match)
                status = f"✓ GOAL (GT: {_format_time(gt_times[match])})"
            else:
                status = "✗ false positive"

            print(f"  [{_format_time(c['start'])}–{_format_time(c['end'])}]  "
                  f"{status}  signal={c['signal']}")
            print(f"    {c['text'][:100]!r}")

        missed = [gt for i, gt in enumerate(gt_times) if i not in matched_gt]
        for gt in missed:
            print(f"  [{_format_time(gt)}]  ← MISSED GOAL (no signal in transcript)")
        print()

    # Summary — TP = unique GT goals matched, not number of matching candidates
    tp = 0
    fp = 0
    for half in ("1", "2"):
        gt_times = ground_truth.get(half, [])
        candidates = all_candidates.get(half, [])
        matched_gt: set[int] = set()
        for c in candidates:
            c_mid = (c["start"] + c["end"]) / 2
            match = next(
                (i for i, gt in enumerate(gt_times)
                 if abs(c_mid - gt) < 60 and i not in matched_gt),
                None,
            )
            if match is not None:
                matched_gt.add(match)
                tp += 1
            else:
                fp += 1
    fn = total_gt - tp
    recall = tp / total_gt if total_gt else 0.0
    precision = tp / total_candidates if total_candidates else 0.0
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Gate recall   : {recall:.1%}  ({tp}/{total_gt} goals found by transcript)")
    print(f"  Gate precision: {precision:.1%}  ({tp}/{total_candidates} candidates are near a real goal)")
    print(f"  Clips to send to Gemini : {total_candidates}")
    print(f"  Full-match naive approach: ~{int(90*60/30)} clips (1 per 30s across 90 min)")
    print(f"\nTranscripts saved → {TRANSCRIPT_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game", default="england_epl/2015-2016/2016-02-13 - 20-30 Chelsea 5 - 1 Newcastle Utd",
        help="Game path relative to data/soccernet/",
    )
    args = parser.parse_args()

    game_dir = SOCCERNET_DIR / args.game
    if not game_dir.exists():
        sys.exit(f"Game directory not found: {game_dir}")

    analyze(args.game, game_dir)


if __name__ == "__main__":
    main()
