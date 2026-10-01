"""Bear presence answered by detection, not by a prompt Map.

This is the ``detect_presence_map`` rewrite of examples/animals_bear_map.py.
YOLOE stops at the first bear track. Rows with a box are kept and a code Map
sets ``bear_present`` to true. Rows with no box are dropped. Gemini is not called.

``map_bear_present`` is a bare function so this file can be imported as Python.
A rendered rewrite uses ``map_detection_presence("bear_present")``, which
``parse_query`` accepts and which would run immediately if this file called it.

Run:
    ./run examples/animals_bear_detect_presence.py
"""

from mmds import Detect, Filter, Input, Map
from udfs.detection_ops import keep_rows_with_detections, map_bear_present

clips = Input("data/swan_valley_full.jsonl")

detected = Detect(
    clips,
    "video",
    ["bear"],
    output_field="detections",
    stop_after_n=1,
    name="detect_bears",
)

kept = Filter(
    detected,
    keep_rows_with_detections,
    name="keep_bear_detections",
)

output = Map(
    kept,
    map_bear_present,
    name="bear_present",
)
