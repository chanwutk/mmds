"""Optimized: cheap text gate on SoccerNet captions, then Gemini on survivors only.

Stage 1 (Filter, no LLM): is_goal_candidate scans the pre-loaded transcript for
  goal-signal phrases ("into the net", "scores", "equalizer", …).
  This runs in microseconds and requires no video decoding or API calls.

Stage 2 (Filter, LLM): Gemini visually verifies goal occurrence only for clips
  that passed the text gate.

LLM calls = K  where K << N  (non-goal clips with neutral commentary are pruned)

Research story: text commentary is orders of magnitude cheaper to scan than
calling a VLM.  Using it as a pre-filter is an instance of cross-modal
optimization transfer (Thrust 2): a cheap modality (text) gates an expensive
one (video VLM).
"""
from mmds import Filter, Input, Record
from udfs.soccer_ops import is_goal_candidate

clips = Input("data/soccer_goal_clips.jsonl")

# Stage 1: cheap keyword scan of pre-loaded transcript (no API cost, no video I/O)
candidates = Filter(clips, is_goal_candidate)

# Stage 2: LLM call only on surviving clips
output = Filter(
    candidates,
    [
        "Watch this soccer broadcast clip carefully.\n",
        "Clip: ",
        Record["video"],
        "\nIs a goal scored or celebrated in this clip? "
        "Answer true if yes, false if not.",
    ],
)
