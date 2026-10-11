"""Exact vehicle count for the first 5 seconds of I24V highway2, without Gemini.

This is ``detect_count_map`` applied to examples/vehicle_count_map.py with
``frame_stride=1`` and ``min_track_boxes=3``. The input is
data/i24v_highway2_5s.jsonl: one fixed highway camera with no cuts, where each
vehicle crosses the view once, which is the case one track = one vehicle fits.

YOLOE runs on every frame with the single class ``vehicle``. Two settings
matter. A wider stride breaks tracks, because a highway car moves more than
its own length in a few frames and its boxes stop overlapping. Separate
``car`` and ``truck`` classes split a vehicle whenever its label flickers. A
code Map then counts the tracks with at least 3 boxes, dropping short
fragments. A bare function is used so this file can be imported as Python. A
rendered rewrite calls ``map_detection_track_count("vehicle_count", "3")``.

Needs data/i24v_traffic/videos/highway2.mp4 (see data/i24v_traffic/README.md).

Run:
    ./run examples/vehicle_count.py
"""

from mmds import Detect, Input, Map
from udfs.detection_ops import map_vehicle_count

clips = Input("data/i24v_highway2_5s.jsonl")

detected = Detect(
    clips,
    "video",
    ["vehicle"],
    output_field="detections",
    frame_stride=1,
    name="detect_vehicles",
)

output = Map(
    detected,
    map_vehicle_count,
    name="vehicle_count",
)
