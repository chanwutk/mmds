"""Semantic baseline: the full Swan Valley compilation is sent to Gemini.

The input is the 19:35 YouTube video, not the short windows in data/animals.jsonl.
Question: is a dog visible anywhere in that video?

Run:
    ./run examples/animals_dog_map.py
"""

from mmds import Input, Map, Record

clips = Input("data/swan_valley_full.jsonl")

output = Map(
    clips,
    [
        "Watch this video.\n",
        Record["video"],
        "\nIs a dog clearly visible anywhere in this video? "
        "Answer true only when a dog is visible, otherwise false.",
    ],
    schema={"dog_present": "boolean"},
    name="dog_present",
)
