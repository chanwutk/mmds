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
    "vehicle_id": "join_unique_90f58dfb",
    "attributes": {
      "class": "sedan",
      "color": "gray",
      "subtype": "hatchback"
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
    "match_score": 0.4564186195553891
  },
  ...
  {
    "vehicle_id": "join_unique_4c322784",
    "attributes": {
      "class": "suv",
      "color": "gray",
      "subtype": "suv"
    },
    "timeline": [
      {
        "camera_id": "cam-i24v-highway2",
        "entered": 0.0,
        "exited": 0.266933
      },
      {
        "camera_id": "cam-i24v-highway3",
        "entered": 4.037367,
        "exited": 4.1041
      }
    ],
    "match_score": 0.35023628175258636
  }
]

"""

from mmds import Detect, Input, Join, Map, Unnest
from udfs.detection_ops import build_vehicle_frame_detections, nms_vehicle_detections
from udfs.join_ops import same_vehicle, vehicle_match_score
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
)

nmsed = Map(detected, nms_vehicle_detections)

framed = Map(nmsed, build_vehicle_frame_detections)

tracked = Map(framed, strongsort_track_frame_detections)

# Unnest track_summaries to one row per track
unnested = Unnest(tracked, "track_summaries")

# Flatten track_summaries to the row top level for joining
promoted = Map(unnested, promote_track_summary_row, replace=True)

# Keep only fields needed for cross-camera join and trajectory export
track_rows = Map(promoted, project_track_summary_row, replace=True)

# Currently, self-join on different camera feeds. In the future, we can join
# to implement an asymmetric cross-camera vehicle join.
# However, with many cameras, self-join on one flat track_rows scales
# better and stays symmetric (single pipeline and predicate handles different cameras)
matches = Join(
    track_rows,
    track_rows,
    same_vehicle,
    on=("vehicle_class", "color", "subtype"),
    one_to_one=True,
    score=vehicle_match_score,
    min_score=0.35,
    left_key=("camera_id", "track_id"),
    right_key=("camera_id", "track_id"),
)

# Extract important fields for trajectory export (i.e., vehicle_id, attributes, timeline, match_score)
output = Map(matches, join_match_to_trajectory, replace=True)
