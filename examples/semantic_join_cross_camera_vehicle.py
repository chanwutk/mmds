"""Semantic cross-camera vehicle join — I24V highway vehicle timeline (requires Gemini).

LLM baseline to the UDF pipeline in ``join_cross_camera_vehicle.py``: watch both
synchronized feeds and stitch vehicles that appear on adjacent cameras into
continuous trajectory records.

Input: first 5 seconds of adjacent ``highway2.mp4`` and ``highway3.mp4`` feeds
from ``data/i24v_traffic/videos/`` (same as ``join_cross_camera_vehicle.py``).

Pipeline:
1. **Input** — two synchronized camera feeds.
2. **Reduce** — one prompt over both clips (``ForEach`` expands each feed's
   ``camera_id``, ``title``, and ``video`` into the prompt).
3. **Unnest** — one output row per stitched vehicle trajectory.

Run:
  uv run python examples/run_expr.py examples/semantic_join_cross_camera_vehicle.py
Output::

[
  {
    "vehicles": {
      "vehicle_id": "v1",
      "attributes": {
        "class": "car",
        "color": "white",
        "subtype": "sedan"
      },
      "timeline": [
        {
          "camera_id": "cam-i24v-highway2",
          "entered": 0.0,
          "exited": 3.4
        },
        {
          "camera_id": "cam-i24v-highway3",
          "entered": 3.7,
          "exited": 5.0
        }
      ],
      "match_score": 0.95
    }
  },
  {
    "vehicles": {
      "vehicle_id": "v2",
      "attributes": {
        "class": "truck",
        "color": "brown",
        "subtype": "semi"
      },
      "timeline": [
        {
          "camera_id": "cam-i24v-highway2",
          "entered": 1.2,
          "exited": 5.0
        },
        {
          "camera_id": "cam-i24v-highway3",
          "entered": 0.0,
          "exited": 4.1
        }
      ],
      "match_score": 0.98
    }
  },
  {
    "vehicles": {
      "vehicle_id": "v3",
      "attributes": {
        "class": "car",
        "color": "black",
        "subtype": "suv"
      },
      "timeline": [
        {
          "camera_id": "cam-i24v-highway2",
          "entered": 0.0,
          "exited": 2.2
        },
        {
          "camera_id": "cam-i24v-highway3",
          "entered": 0.0,
          "exited": 3.2
        }
      ],
      "match_score": 0.92
    }
  },
  {
    "vehicles": {
      "vehicle_id": "v4",
      "attributes": {
        "class": "car",
        "color": "gray",
        "subtype": "sedan"
      },
      "timeline": [
        {
          "camera_id": "cam-i24v-highway2",
          "entered": 0.0,
          "exited": 3.8
        },
        {
          "camera_id": "cam-i24v-highway3",
          "entered": 0.0,
          "exited": 4.8
        }
      ],
      "match_score": 0.94
    }
  }
]
"""

from mmds import ForEach, Input, Record, Reduce, Unnest

feeds = Input("data/i24v_traffic_highway2_highway3_5s.jsonl")

stitched = Reduce(
    feeds,
    "_all",
    [
        "You are traffic analyst. Watch the two clips and stitch each of the same "
        "vehicles, found by same vehicle class, color, and subtype across adjacent "
        "highway cameras. Colors should be chosen from the following list: white, "
        "silver, gray, black, beige, yellow, red, blue, green, brown. "
        "Subtypes of trucks should be chosen from the following list: tractor trailer,"
        "flatbed truck, box truck, and pickup. Subtypes of cars should be chosen from "
        "the following list: coupe, suv, hatchback, and sedan. Build a continuous "
        "trajectory record of each vehicle that appears in both camera footages with "
        "specific timestamps down to the milisecond (from start/end frames).\n",
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
    schema={
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
                "required": ["vehicle_id", "attributes", "timeline"],
            },
        },
    },
)

output = Unnest(stitched, "vehicles")
