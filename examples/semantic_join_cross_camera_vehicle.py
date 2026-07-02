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
        "class": "truck",
        "color": "brown",
        "subtype": "semi-truck"
      },
      "timeline": [
        {
          "camera_id": "cam-i24v-highway2",
          "entered": 0,
          "exited": 5
        },
        {
          "camera_id": "cam-i24v-highway3",
          "entered": 0,
          "exited": 5
        }
      ],
      "match_score": 0.95
    }
  },
  {
    "vehicles": {
      "vehicle_id": "v2",
      "attributes": {
        "class": "car",
        "color": "silver",
        "subtype": "sedan"
      },
      "timeline": [
        {
          "camera_id": "cam-i24v-highway2",
          "entered": 0,
          "exited": 5
        },
        {
          "camera_id": "cam-i24v-highway3",
          "entered": 0,
          "exited": 5
        }
      ],
      "match_score": 0.92
    }
  },
  {
    "vehicles": {
      "vehicle_id": "v3",
      "attributes": {
        "class": "car",
        "color": "white",
        "subtype": "sedan"
      },
      "timeline": [
        {
          "camera_id": "cam-i24v-highway2",
          "entered": 0,
          "exited": 5
        },
        {
          "camera_id": "cam-i24v-highway3",
          "entered": 0,
          "exited": 5
        }
      ],
      "match_score": 0.9
    }
  },
  {
    "vehicles": {
      "vehicle_id": "v4",
      "attributes": {
        "class": "pickup",
        "color": "white",
        "subtype": "pickup truck"
      },
      "timeline": [
        {
          "camera_id": "cam-i24v-highway2",
          "entered": 2,
          "exited": 5
        },
        {
          "camera_id": "cam-i24v-highway3",
          "entered": 2,
          "exited": 5
        }
      ],
      "match_score": 0.98
    }
  },
  {
    "vehicles": {
      "vehicle_id": "v5",
      "attributes": {
        "class": "car",
        "color": "black",
        "subtype": "sedan"
      },
      "timeline": [
        {
          "camera_id": "cam-i24v-highway2",
          "entered": 0,
          "exited": 5
        },
        {
          "camera_id": "cam-i24v-highway3",
          "entered": 0,
          "exited": 5
        }
      ],
      "match_score": 0.88
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
        "vehicles across adjacent highway cameras and build a continuous trajectory "
        "record of each vehicle that appears in both camera footages.\n",
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
