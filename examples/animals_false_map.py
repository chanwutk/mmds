"""Semantic baseline: the full Swan Valley compilation is sent to Gemini.

Question: is a sedan clearly visible anywhere in that wildlife video?
The expected answer is false.

Run:
    ./run examples/animals_false_map.py
"""

from mmds import Input, Map, Record

clips = Input("data/swan_valley_full.jsonl")

output = Map(
    clips,
    [
        "Watch this video.\n",
        Record["video"],
        "\nIs a sedan clearly visible anywhere in this video? "
        "Answer true only when a sedan is visible, otherwise false.",
    ],
    schema={"sedan_present": "boolean"},
    name="sedan_present",
)
