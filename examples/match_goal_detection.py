from udfs.match_ops import find_goal_candidates, extract_candidate_clip, gather_transcript_context
from mmds import Input, Map, Unnest, Filter, Gather
from mmds.model import Record

matches = Input("data/matches.jsonl")

# UDF: load Whisper transcript → find goal-signal timestamps → add "candidates" list
transcribed = Map(matches, find_goal_candidates)

# One row per candidate timestamp
candidates = Unnest(transcribed, "candidates")

# UDF: extract 20s MP4 clip around each candidate → add "video" field
clips = Map(candidates, extract_candidate_clip)

# Gather: attach ±60s of broadcast commentary around each candidate timestamp
enriched = Gather(clips, gather_transcript_context, "transcript_context")

# VLM: verify using both video evidence and surrounding commentary text
output = Filter(enriched, [
    "Watch this soccer broadcast clip carefully.\n",
    "Clip: ", Record["video"],
    "\nBroadcast commentary around this moment:\n", Record["transcript_context"],
    "\nBased on both the video and the commentary, is a goal actually scored "
    "or celebrated in this clip? Answer true if yes, false if not.",
])
