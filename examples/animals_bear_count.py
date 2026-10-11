"""Exact bear count for the full Swan Valley compilation, without Gemini.

This is ``detect_count_map`` applied to examples/animals_bear_count_map.py.
The input is the full 19:35 video in data/swan_valley_full.jsonl. YOLOE scans
the whole clip (no ``stop_after_n``), the tracker gives each bear one
``track_id`` across frames, and a code Map sets ``bear_count`` to the number
of distinct bear tracks.

``frame_stride=10`` samples every tenth frame, within the tracker's 30-frame
gap, so overlapping boxes on consecutive samples stay one track. Known error
sources: a bear that leaves the frame for longer than the gap, or moves far
enough between samples that its boxes stop overlapping, is counted again. A
false-positive box counts as a bear. A bare function is used so this file can
be imported as Python. A rendered rewrite calls
``map_detection_track_count("bear_count")``.

Run:
    ./run examples/animals_bear_count.py
"""

from mmds import Detect, Input, Map
from udfs.detection_ops import map_bear_count

videos = Input("data/swan_valley_full.jsonl")

detected = Detect(
    videos,
    "video",
    ["bear"],
    output_field="detections",
    frame_stride=10,
    name="detect_bears",
)

output = Map(
    detected,
    map_bear_count,
    name="bear_count",
)
