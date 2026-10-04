"""Dog presence answered by detection, not by a prompt Map.

This is the ``detect_presence_map`` rewrite of examples/animals_bear_map.py.
YOLOE stops at the first dog track. Rows with a box are kept and a code Map
sets ``dog_present`` to true. Rows with no box are dropped. Gemini is not called.

fps set to 10 to reduce the number of frames to process.

``map_dog_present`` is a bare function so this file can be imported as Python.
A rendered rewrite uses ``map_detection_presence("dog_present")``, which
``parse_query`` accepts and which would run immediately if this file called it.

Run:
    ./run examples/animals_dog_detect_presence.py
"""

from mmds import Detect, Filter, Input, Map
from udfs.detection_ops import keep_rows_with_detections, map_dog_present

clips = Input("data/swan_valley_full.jsonl")

detected = Detect(
    clips,
    "video",
    ["dog"],
    output_field="detections",
    frame_stride=10,
    stop_after_n=1,
    name="detect_dogs",
)

kept = Filter(
    detected,
    keep_rows_with_detections,
    name="keep_dog_detections",
)

output = Map(
    kept,
    map_dog_present,
    name="dog_present",
)
