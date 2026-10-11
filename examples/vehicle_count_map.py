"""Semantic baseline: Gemini counts the vehicles in 5 seconds of I24V highway2.

Question: how many vehicles pass through this clip? The input is
data/i24v_highway2_5s.jsonl, one Gemini call over the 0-5s window.

Needs data/i24v_traffic/videos/highway2.mp4 (see data/i24v_traffic/README.md).

Run:
    ./run examples/vehicle_count_map.py
"""

from mmds import Input, Map, Record

clips = Input("data/i24v_highway2_5s.jsonl")

output = Map(
    clips,
    [
        "Watch this highway traffic clip.\n",
        Record["video"],
        "\nHow many distinct vehicles appear in this clip, in all lanes and "
        "both directions? Count each vehicle once, including cars, pickups, "
        "and trucks, even if it is only visible briefly. Answer 0 if there "
        "are none.",
    ],
    schema={"vehicle_count": "integer"},
    name="vehicle_count",
)
