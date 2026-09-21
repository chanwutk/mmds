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
    "vehicle_id": "join_unique_074f20c7",
    "attributes": {
      "class": "sedan",
      "color": "silver",
      "subtype": "sedan"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 3.9039,
        "exited": 4.938267
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 4.004,
        "exited": 4.938267
      }
    ],
    "match_score": 0.9111949592939887
  },
  {
    "vehicle_id": "join_unique_918572af",
    "attributes": {
      "class": "sedan",
      "color": "white",
      "subtype": "sedan"
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
        "exited": 2.168833
      }
    ],
    "match_score": 0.9091881328974148
  },
  {
    "vehicle_id": "join_unique_de055e30",
    "attributes": {
      "class": "sedan",
      "color": "silver",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.0,
        "exited": 1.534867
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 0.166833,
        "exited": 1.7017
      }
    ],
    "match_score": 0.9054222791578828
  },
  {
    "vehicle_id": "join_unique_212bfca7",
    "attributes": {
      "class": "truck",
      "color": "silver",
      "subtype": "tractor trailer"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 3.370033,
        "exited": 4.9049
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 3.336667,
        "exited": 4.938267
      }
    ],
    "match_score": 0.8935604166881397
  },
  {
    "vehicle_id": "join_unique_f8784a15",
    "attributes": {
      "class": "sedan",
      "color": "black",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 3.3033,
        "exited": 4.938267
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 3.4034,
        "exited": 4.938267
      }
    ],
    "match_score": 0.8621495394387985
  },
  {
    "vehicle_id": "join_unique_c321b9bd",
    "attributes": {
      "class": "suv",
      "color": "blue",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.2002,
        "exited": 3.2032
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 1.2012,
        "exited": 4.3043
      }
    ],
    "match_score": 0.8580926744224155
  },
  {
    "vehicle_id": "join_unique_a47bdb7c",
    "attributes": {
      "class": "sedan",
      "color": "black",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.033367,
        "exited": 3.2032
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 0.867533,
        "exited": 3.670333
      }
    ],
    "match_score": 0.8542719867675412
  },
  {
    "vehicle_id": "join_unique_redpickup26",
    "attributes": {
      "class": "truck",
      "color": "red",
      "subtype": "flatbed"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.867533,
        "exited": 0.967633
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 0.867533,
        "exited": 0.967633
      }
    ],
    "match_score": 0.8485770863825516
  },
  {
    "vehicle_id": "join_unique_afc0460c",
    "attributes": {
      "class": "sedan",
      "color": "white",
      "subtype": "sedan"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.734067,
        "exited": 3.069733
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 0.767433,
        "exited": 2.669333
      }
    ],
    "match_score": 0.8449244530356607
  },
  {
    "vehicle_id": "join_unique_21f13705",
    "attributes": {
      "class": "sedan",
      "color": "gray",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.0,
        "exited": 0.4004
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 0.0,
        "exited": 0.233567
      }
    ],
    "match_score": 0.8373454628992413
  },
  {
    "vehicle_id": "join_unique_309e0cb9",
    "attributes": {
      "class": "sedan",
      "color": "black",
      "subtype": "sedan"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.0,
        "exited": 1.735067
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 0.0,
        "exited": 2.469133
      }
    ],
    "match_score": 0.8317100914639672
  },
  {
    "vehicle_id": "join_unique_984e4ae4",
    "attributes": {
      "class": "truck",
      "color": "white",
      "subtype": "pickup"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 2.769433,
        "exited": 4.8048
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 3.470133,
        "exited": 4.938267
      }
    ],
    "match_score": 0.8184057283186698
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
    "vehicle_id": "join_unique_3767633e",
    "attributes": {
      "class": "sedan",
      "color": "white",
      "subtype": "sedan"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 3.3033,
        "exited": 4.6046
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 2.369033,
        "exited": 4.4044
      }
    ],
    "match_score": 0.7833206389990407
  }
]
"""

from mmds import Detect, Input, Join, Map, Unnest
from udfs.detection_ops import build_vehicle_frame_detections, nms_vehicle_detections
from udfs.join_ops import same_vehicle
from udfs.reid_ops import appearance_match_score, attach_track_summary_embeddings
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

# Read representative frames once per camera and embed all track crops in
# bounded batches before exploding the summaries into individual rows.
embedded = Map(tracked, attach_track_summary_embeddings, replace=True)

# Unnest track_summaries to one row per track
unnested = Unnest(embedded, "track_summaries")

# Flatten track_summaries to the row top level for joining
promoted = Map(unnested, promote_track_summary_row, replace=True)

# Keep only fields needed for cross-camera join and trajectory export
track_rows = Map(promoted, project_track_summary_row, replace=True)

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
