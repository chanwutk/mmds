"""Semantic baseline: Gemini counts the bears in the full Swan Valley video.

Question: how many bears are there? The input is the full 19:35 video in
data/swan_valley_full.jsonl, so this is one Gemini call over the whole clip.

Run:
    ./run examples/animals_bear_count_map.py
"""

from mmds import Input, Map, Record

videos = Input("data/swan_valley_full.jsonl")

output = Map(
    videos,
    [
        "Watch this video.\n",
        Record["video"],
        "\nHow many bears are there? Count each distinct bear once, even if "
        "it appears in more than one scene. Answer 0 if there are no bears.",
    ],
    schema={"bear_count": "integer"},
    name="bear_count",
)
