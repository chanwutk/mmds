"""Sedan presence on a wildlife video, answered by detection.

This is the ``detect_presence_map`` rewrite of examples/animals_false_map.py.
The Swan Valley compilation is trail-camera wildlife, so a sedan is not
expected. YOLOE samples every tenth frame and stops only if it finds a sedan
track. No boxes drops the row, which the evaluator scores as
``sedan_present=false``. Gemini is not called.

``map_sedan_present`` is a bare function so this file can be imported as Python.

Run:
    ./run examples/animals_false_detect_presence.py
"""

from mmds import Detect, Filter, Input, Map
from udfs.detection_ops import keep_rows_with_detections, map_sedan_present

clips = Input("data/swan_valley_full.jsonl")

detected = Detect(
    clips,
    "video",
    ["sedan"],
    output_field="detections",
    frame_stride=10,
    stop_after_n=1,
    name="detect_sedans",
)

kept = Filter(
    detected,
    keep_rows_with_detections,
    name="keep_sedan_detections",
)

output = Map(
    kept,
    map_sedan_present,
    name="sedan_present",
)
