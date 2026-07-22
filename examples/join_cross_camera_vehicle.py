"""Cross-camera vehicle join — I24V highway vehicle timeline (work in progress).

Input: first 5 seconds of adjacent ``highway2.mp4`` and ``highway3.mp4`` feeds
from ``data/i24v_traffic/videos/``.

Pipeline (so far):
1. **Input** — two synchronized camera feeds.
2. **Detect** — YOLOE inference for ``sedan``, ``suv``, and ``truck``.
3. **Map** — class-wise per-frame NMS, then color/subtype prediction.
   Outputs ``frame_detections`` — one record per box with ``frame_id``,
   ``camera_id``, ``bbox``, ``confidence``, ``vehicle_class``, ``color``, and
   ``subtype``.
4. **Map** — StrongSORT-shaped tracking per camera feed.
   Output: ``track_summaries`` — one summary dict per track.
5. **Unnest** + **Map** — explode summaries and promote track fields to the
   row top level for joining.
6. **Join** — one-to-one hash join on ``(vehicle_class, color, subtype)``,
   filtered by ``same_vehicle`` and scored by mean track confidence. For 
   each left track only checks the tracks of the same partition in the right once
   resulting in a O(∑|Bi|^2) complexity, where Bi is the set of tracks in the i-th
   partition of the right, instead of a O(T^2) complexity of a nested loop join.
   Output: pairs of track summary records from two different camera feeds—one "left" and one "right"
   Each row is a dict with: {"left": <track_summary_dict>, "right": <track_summary_dict>, "match_score": <float>}
7. **Map** — ``join_match_to_trajectory`` exports vehicle trajectory records.
   Output: vehicle trajectory records with the shape:
   {
   "vehicle_id": "join_unique_<hash>",
   "attributes": {"class": "...", "color": "...", "subtype": "..."},
   "timeline": [
   {"camera_id": "...", "entered": <epoch_sec>, "exited": <epoch_sec>},
   ...
   ],
   "match_score": <float>  # when present on the join row
   }

In SQL, it would be similar to:
```sql
SELECT *
FROM tracks AS a
JOIN tracks AS b
ON a.camera_id != b.camera_id
...
AND a.vehicle_class = b.vehicle_class
AND a.color = b.color
AND a.subtype = b.subtype;
```

Run:
  uv run python examples/run_expr.py examples/join_cross_camera_vehicle.py
[
  {
    "vehicle_id": "join_unique_df442575",
    "attributes": {
      "class": "sedan",
      "color": "blue",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 3.670333,
        "exited": 4.938267
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 3.9039,
        "exited": 4.938267
      }
    ],
    "match_score": 0.9107320735240723
  },
  {
    "vehicle_id": "join_unique_b7398975",
    "attributes": {
      "class": "suv",
      "color": "gray",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.0,
        "exited": 0.767433
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 0.0,
        "exited": 3.9039
      }
    ],
    "match_score": 0.8830857152455046
  },
  {
    "vehicle_id": "join_unique_de9a7406",
    "attributes": {
      "class": "suv",
      "color": "gray",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.166833,
        "exited": 3.370033
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 1.034367,
        "exited": 4.938267
      }
    ],
    "match_score": 0.857392891585414
  },
  {
    "vehicle_id": "join_unique_c8af37e6",
    "attributes": {
      "class": "suv",
      "color": "gray",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.0,
        "exited": 1.9019
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 0.0,
        "exited": 4.938267
      }
    ],
    "match_score": 0.854265808173901
  },
  {
    "vehicle_id": "join_unique_3ac60960",
    "attributes": {
      "class": "truck",
      "color": "gray",
      "subtype": "pickup"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.0,
        "exited": 0.3003
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 0.0,
        "exited": 0.133467
      }
    ],
    "match_score": 0.8485770863825516
  },
  {
    "vehicle_id": "join_unique_309e0cb9",
    "attributes": {
      "class": "suv",
      "color": "gray",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.0,
        "exited": 0.567233
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 0.0,
        "exited": 0.333667
      }
    ],
    "match_score": 0.8106446811683293
  },
  {
    "vehicle_id": "join_unique_3b314f83",
    "attributes": {
      "class": "truck",
      "color": "silver",
      "subtype": "pickup"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 4.371033,
        "exited": 4.938267
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 4.170833,
        "exited": 4.938267
      }
    ],
    "match_score": 0.8029128891312
  },
  {
    "vehicle_id": "join_unique_8550930d",
    "attributes": {
      "class": "sedan",
      "color": "silver",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.0,
        "exited": 1.8018
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 0.0,
        "exited": 4.938267
      }
    ],
    "match_score": 0.735242618498014
  },
  {
    "vehicle_id": "join_unique_0d2a20f6",
    "attributes": {
      "class": "suv",
      "color": "blue",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 1.968633,
        "exited": 4.938267
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 3.269933,
        "exited": 4.938267
      }
    ],
    "match_score": 0.7334414300027923
  },
  {
    "vehicle_id": "join_unique_036e1113",
    "attributes": {
      "class": "suv",
      "color": "blue",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.0,
        "exited": 3.370033
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 0.834167,
        "exited": 4.938267
      }
    ],
    "match_score": 0.6950238039666103
  },
  {
    "vehicle_id": "join_unique_8aebf622",
    "attributes": {
      "class": "sedan",
      "color": "white",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 4.637967,
        "exited": 4.938267
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 2.8028,
        "exited": 4.938267
      }
    ],
    "match_score": 0.6883508573419033
  },
  {
    "vehicle_id": "join_unique_311de449",
    "attributes": {
      "class": "sedan",
      "color": "white",
      "subtype": "hatchback"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.0,
        "exited": 0.9009
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 2.168833,
        "exited": 4.938267
      }
    ],
    "match_score": 0.6414128231195493
  }
]
"""

from mmds import Detect, Input, Join, Map, Unnest
from udfs.detection_ops import build_vehicle_frame_detections, nms_vehicle_detections
from udfs.join_ops import same_vehicle
from udfs.reid_ops import appearance_match_score, attach_track_embedding
from udfs.tracking_ops import (
    project_track_summary_row,
    promote_track_summary_row,
    strongsort_track_frame_detections,
)
from udfs.trajectory_ops import join_match_to_trajectory

feeds = Input("data/i24v_traffic_highway2_highway3_5s.jsonl")

detected = Detect(
    feeds,
    "video",
    ["sedan", "suv", "truck"],
    output_field="detections",
    frame_stride=1,  # run inference on every frame for tracking
    # High-mounted I-24 traffic camera: vehicles are small in a 1920x1080
    # frame, so raise the inference resolution and lower the confidence floor
    # to recover boxes the 640px/conf=0.25 defaults would miss.
    conf=0.1,
    imgsz=1280,
)

nmsed = Map(detected, nms_vehicle_detections)

framed = Map(nmsed, build_vehicle_frame_detections)

tracked = Map(framed, strongsort_track_frame_detections)

# Unnest track_summaries to one row per track
unnested = Unnest(tracked, "track_summaries")

# Flatten track_summaries to the row top level for joining
promoted = Map(unnested, promote_track_summary_row, replace=True)

# Attach a per-track appearance embedding (from the representative crop) for
# re-identification across cameras.
embedded = Map(promoted, attach_track_embedding, replace=True)

# Keep only fields needed for cross-camera join and trajectory export
track_rows = Map(embedded, project_track_summary_row, replace=True)

# Self-join across camera feeds. Candidate pairs are pruned by same_vehicle
# (different cameras, corridor order, travel-time / direction / speed) and
# scored by appearance re-ID: the exact (class, color, subtype) hash key is
# deliberately dropped — cross-camera attribute labels disagree, so an exact key
# silently excludes true matches. Appearance-embedding cosine similarity is the
# match signal instead.
matches = Join(
    track_rows,
    track_rows,
    same_vehicle,
    one_to_one=True,
    score=appearance_match_score,
    min_score=0.4,
    left_key=("camera_id", "track_id"),
    right_key=("camera_id", "track_id"),
)

# Extract important fields for trajectory export (i.e., vehicle_id, attributes, timeline, match_score)
output = Map(matches, join_match_to_trajectory, replace=True)
