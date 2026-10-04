"""Semantic baseline: the full Swan Valley compilation is sent to Gemini.

Question: are at least five bears visible anywhere in that video?

Run:
    ./run examples/animals_five_bears_map.py
"""

from mmds import Input, Map, Record

clips = Input("data/swan_valley_full.jsonl")

output = Map(
    clips,
    [
        "Watch this video.\n",
        Record["video"],
        "\nAre at least five distinct bears visible anywhere in this video? "
        "Answer true only when at least five bears are visible, otherwise false.",
    ],
    schema={"at_least_five_bears": "boolean"},
    name="at_least_five_bears",
)
