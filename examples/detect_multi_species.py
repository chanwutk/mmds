"""Local object discovery with YOLOE (no API key).

Searches what is in the video using open-vocabulary detection instead of an LLM.
Runs on a single short clip and checks for multiple species individually.

Run:
  PYTHONPATH=src:. ./.venv/bin/python examples/run_detect.py examples/detect_multi_species.py
"""

from mmds import Detect, Input, Unnest

input_data = Input("data/animals-small.jsonl")

detected = Detect(
    input_data,
    "video",
    ["bear", "deer", "bird"],
    output_field="detections"
)

output = Unnest(detected, "detections")
