"""Semantic Gemini baseline for the I24V cross-camera vehicle join.

LLM Reduce–Unnest counterpart to ``examples/join_cross_camera_vehicle.py``.
Both pipelines default to the same highway2/3 manifest and emit rows with:

- ``vehicle_id``
- ``attributes`` ``{class, color, subtype}``
- ``timeline`` ``[{camera_id, entered, exited}, ...]``
- ``match_score``

Run (requires Gemini + local highway2/3 clips):
  ./run examples/semantic_join_cross_camera_vehicle.py
"""

from __future__ import annotations

from mmds import ForEach, Input, Map, Record, Reduce, Unnest
from mmds.model import DatasetExpr
from udfs.trajectory_ops import promote_vehicle_trajectory_row

DEFAULT_MANIFEST = "data/i24v_traffic_highway2_highway3_5s.jsonl"

_VEHICLE_TRAJECTORY_SCHEMA = {
    "vehicles": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "vehicle_id": {"type": "string"},
                "attributes": {
                    "type": "object",
                    "properties": {
                        "class": {"type": "string"},
                        "color": {"type": "string"},
                        "subtype": {"type": "string"},
                    },
                    "required": ["class", "color", "subtype"],
                },
                "timeline": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "camera_id": {"type": "string"},
                            "entered": {"type": "number"},
                            "exited": {"type": "number"},
                        },
                        "required": ["camera_id", "entered", "exited"],
                    },
                },
                "match_score": {"type": "number"},
            },
            "required": ["vehicle_id", "attributes", "timeline", "match_score"],
        },
    },
}


def build_query(feeds_jsonl: str = DEFAULT_MANIFEST) -> DatasetExpr:
    """Build the prompt-backed Reduce–Unnest cross-camera stitch."""
    feeds = Input(feeds_jsonl)
    stitched = Reduce(
        feeds,
        "_all",
        [
            "You are a traffic analyst. Watch the synchronized adjacent-camera "
            "clips and stitch each vehicle that appears across both feeds into "
            "one continuous trajectory. Match by vehicle class, color, and "
            "subtype. Colors must be one of: white, silver, gray, black, beige, "
            "yellow, red, blue, green, brown. Truck subtypes must be one of: "
            "tractor trailer, flatbed, box truck, pickup. Car subtypes must be "
            "one of: coupe, suv, sedan. For each vehicle, report camera-local "
            "entered/exited times in seconds from the start of each clip, plus "
            "a match_score in [0, 1].\n",
            ForEach(
                [
                    "Camera ",
                    Record["camera_id"],
                    " (",
                    Record["title"],
                    "):\n",
                    Record["video"],
                    "\n",
                ]
            ),
        ],
        schema=_VEHICLE_TRAJECTORY_SCHEMA,
        name="stitch_cross_camera_vehicles",
    )
    unnested = Unnest(stitched, "vehicles", name="one_vehicle_per_row")
    return Map(
        unnested,
        promote_vehicle_trajectory_row,
        replace=True,
        name="promote_vehicle_trajectory",
    )


output = build_query()
