"""Soccer UDFs for two use-cases.

Case Study 3 (goal detection): ``detect_ball_sampled`` + ``has_ball_activity``
  Local YOLOE gate that samples one frame per second to check for a sports ball.
  This pre-filters clips before the expensive Gemini verification step.
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# Goal-candidate text gate (Case Study 3 — commentary transcript approach)
# ---------------------------------------------------------------------------

# Single words checked with exact word-boundary matching (after stripping punctuation).
# Multi-word phrases are checked as substrings.
_GOAL_SIGNAL_WORDS = frozenset({"goal", "scores", "equalizer", "levelled"})
_GOAL_SIGNAL_PHRASES = (
    "into the net",
    "into the goal",
    "back of the net",
    "into the corner",
    "into the top",
    "ricocheting in",
    "tucks it in",
    "slots it in",
    "pokes it in",
    "headed in",
    "makes it",
)

_BALL_FRAME_THRESHOLD = 2


def is_goal_candidate(row: dict) -> bool:
    """Return True if the clip should be forwarded to the expensive Gemini step.

    Three cases:
    - No transcript: caption coverage is absent for this window; we cannot
      rule out a goal, so pass through (conservative fallback).
    - Transcript present + goal-signal language: likely a goal, pass through.
    - Transcript present + no goal-signal language: safe to prune.

    Single-word signals ("goal", "scores", …) use word-boundary matching so
    that "goalkeeper" or "on goal" do not produce false positives.
    Multi-word phrases are matched as substrings.
    """
    text = str(row.get("transcript") or "").strip().lower()
    if not text:
        return True  # no caption data → cannot rule out goal, defer to Gemini
    tokens = {w.strip(".,!?;:'\"()-") for w in text.split()}
    if tokens & _GOAL_SIGNAL_WORDS:
        return True
    return any(phrase in text for phrase in _GOAL_SIGNAL_PHRASES)


def has_ball_activity(row: dict) -> bool:
    """Return True if the clip's local detection indicates active ball play.

    Compatible with both ``detect_ball_sampled`` output (``ball_detected`` key)
    and the MMDS ``Detect`` operator output (``detections`` list of class dicts).
    """
    if "ball_detected" in row:
        return bool(row["ball_detected"])
    for det in row.get("detections", []):
        if det.get("type") == "sports ball":
            return len(det.get("bboxes", [])) >= _BALL_FRAME_THRESHOLD
    return False
