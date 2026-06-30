"""Local Detect → Filter pipeline (no API key).

After frame-level detection, keep only clips where a bear was found.
Analogous to filtering a library down to clips that match a visual query.

uv run python examples/run_expr.py examples/detect_filter_bears.py
[
  {
    "video": {
      "type": "VideoView",
      "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8",
      "start": 0,
      "end": 19
    },
    "title": "2021 Swan Valley Wildlife Trail Camera Compilation",
    "detections": [
      {
        "type": "bear",
        "bboxes": [
          {
            "frame_idx": 3,
            "bbox": [
              90.51139831542969,
              193.19683837890625,
              159.41290283203125,
              255.416748046875
            ],
            "confidence": 0.4024558663368225
          },

…

          {
            "frame_idx": 567,
            "bbox": [
              277.41973876953125,
              245.86468505859375,
              365.99224853515625,
              350.52203369140625
            ],
            "confidence": 0.3946681320667267
          }
        ]
      }
    ]
  }
]
"""

from mmds import Detect, Filter, Input
from udfs.detection_ops import keep_rows_with_class

input_data = Input("data/animals-small.jsonl")

detected = Detect(
    input_data,
    "video",
    ["bear"],
    output_field="detections"
)

output = Filter(detected, keep_rows_with_class)
