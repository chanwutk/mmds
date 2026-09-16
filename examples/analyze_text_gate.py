"""Show which clips are pruned vs. passed by the is_goal_candidate text gate.

Usage:
    PYTHONPATH=src:. .venv/bin/python examples/analyze_text_gate.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "soccer_goal_clips.jsonl"

from udfs.soccer_ops import is_goal_candidate

rows = [json.loads(l) for l in DATA_FILE.read_text().splitlines() if l.strip()]

passed = [r for r in rows if is_goal_candidate(r)]
pruned = [r for r in rows if not is_goal_candidate(r)]

print(f"Total clips : {len(rows)}")
print(f"Passed gate : {len(passed)}  → sent to Gemini")
print(f"Pruned      : {len(pruned)}  → no LLM call")
print()

def label(r):
    return "GOAL ✓" if r["true_label"] else "non-goal"

def reason(r):
    text = str(r.get("transcript") or "").strip().lower()
    if not text:
        return "[no transcript — conservative pass]"
    from udfs.soccer_ops import _GOAL_SIGNAL_WORDS, _GOAL_SIGNAL_PHRASES
    tokens = {w.strip(".,!?;:'\"()-") for w in text.split()}
    matched_words = tokens & _GOAL_SIGNAL_WORDS
    if matched_words:
        return f"[word match: {matched_words}]"
    for phrase in _GOAL_SIGNAL_PHRASES:
        if phrase in text:
            return f"[phrase match: '{phrase}']"
    return "[no match]"

print("=" * 80)
print("PASSED — forwarded to Gemini")
print("=" * 80)
for r in passed:
    clip = Path(r["video"]["source"]).name
    t = (r.get("transcript") or "")[:90] or "[empty]"
    print(f"  {clip}  [{label(r)}]  {reason(r)}")
    print(f"    transcript: {t!r}")
    print()

print("=" * 80)
print("PRUNED — text gate blocked, no Gemini call")
print("=" * 80)
for r in pruned:
    clip = Path(r["video"]["source"]).name
    t = (r.get("transcript") or "")[:90]
    print(f"  {clip}  [{label(r)}]")
    print(f"    transcript: {t!r}")
    print()
