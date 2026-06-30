"""Baseline: every clip is sent directly to Gemini for goal verification.

LLM calls = N  (total number of clips in the dataset)
"""
from mmds import Filter, Input, Record

clips = Input("data/soccer_goal_clips.jsonl")

output = Filter(
    clips,
    [
        "Watch this soccer broadcast clip carefully.\n",
        "Clip: ",
        Record["video"],
        "\nIs a goal scored or celebrated in this clip? "
        "Answer true if yes, false if not.",
    ],
)
