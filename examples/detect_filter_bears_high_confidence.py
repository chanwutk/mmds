"""Local Detect → Map(prune) → Filter pipeline (no API key).

Keeps only bear boxes whose YOLOE confidence is at or above the configured
threshold (see ``udfs.detection_ops.HIGH_CONFIDENCE_THRESHOLD``).

Run:
  PYTHONPATH=src:. ./.venv/bin/python examples/run_detect.py examples/detect_filter_bears_high_confidence.py
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
            "frame_idx": 37,
            "bbox": [
              96.29179382324219,
              199.113525390625,
              160.9265594482422,
              253.9493408203125
            ],
            "confidence": 0.5068932175636292
          },

          ...

          {
            "frame_idx": 375,
            "bbox": [
              379.106689453125,
              188.4339141845703,
              640.0,
              350.54522705078125
            ],
            "confidence": 0.5090929865837097
          }
        ]
      }
    ]
  }
]
"""

from mmds import Detect, Filter, Input, Map
from udfs.detection_ops import (
    keep_rows_with_high_confidence_class,
    prune_class_detections_to_high_confidence,
)

input_data = Input("data/animals-small.jsonl")

detected = Detect(
    input_data,
    "video",
    ["bear"],
    output_field="detections"
)

# Rewrites "detections" so that only high-confidence bear boxes are kept.
pruned = Map(detected, prune_class_detections_to_high_confidence)

# Keeps only rows with at least one high-confidence bear box.
output = Filter(pruned, keep_rows_with_high_confidence_class)
