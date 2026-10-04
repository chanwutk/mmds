"""At least five bears, answered by detection tracks instead of a prompt Map.

This is ``detect_presence_map`` with ``min_tracks=5`` for
examples/animals_five_bears_map.py. YOLOE stops once five bear tracks exist.
Rows that reach five tracks are kept and a code Map sets
``at_least_five_bears`` to true. Fewer than five tracks drops the row.
Gemini is not called.

``frame_stride=10`` samples every tenth frame. Consecutive samples are still
close enough to stay on one track when the boxes overlap. A bare function is
used so this file can be imported as Python. A rendered rewrite calls
``map_at_least_n_tracks("at_least_five_bears", "5")``.

Run:
    ./run examples/animals_five_bears_detect_presence.py
"""

from mmds import Detect, Filter, Input, Map
from udfs.detection_ops import keep_at_least_five_bear_tracks, map_at_least_five_bears

clips = Input("data/swan_valley_full.jsonl")

detected = Detect(
    clips,
    "video",
    ["bear"],
    output_field="detections",
    frame_stride=10,
    stop_after_n=5,
    name="detect_bears",
)

kept = Filter(
    detected,
    keep_at_least_five_bear_tracks,
    name="keep_five_bear_tracks",
)

output = Map(
    kept,
    map_at_least_five_bears,
    name="at_least_five_bears",
)
