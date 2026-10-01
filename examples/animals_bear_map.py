"""Semantic baseline: the full Swan Valley compilation is sent to Gemini.

The input is the 19:35 YouTube video, not the short windows in data/animals.jsonl.
Question: is a bear visible anywhere in that video?

Run:
    ./run examples/animals_bear_map.py
"""

from mmds import Input, Map, Record

clips = Input("data/swan_valley_full.jsonl")

output = Map(
    clips,
    [
        "Watch this video.\n",
        Record["video"],
        "\nIs a bear clearly visible anywhere in this video? "
        "Answer true only when a bear is visible, otherwise false.",
    ],
    schema={"bear_present": "boolean"},
    name="bear_present",
)
