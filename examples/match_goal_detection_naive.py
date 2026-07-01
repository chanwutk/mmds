from udfs.match_ops import chunk_half, extract_chunk_clip
from mmds import Input, Map, Unnest, Filter
from mmds.model import Record

matches = Input("data/matches.jsonl")

# Divide each half into 30s chunks — no transcript gate, every chunk goes to Gemini
chunked = Map(matches, chunk_half)
chunks = Unnest(chunked, "chunks")
clips = Map(chunks, extract_chunk_clip)

output = Filter(clips, [
    "Watch this soccer broadcast clip carefully.\n",
    "Clip: ", Record["video"],
    "\nIs a goal scored or celebrated in this clip? Answer true if yes, false if not.",
])
