from mmds import Detect, Input, Unnest

# Example input rows for the "clips" dataset:
#
# rows:
# { "video": {"type": "VideoView", "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8", "start": 0, "end": 19}, "title": "2021 Swan Valley Wildlife Trail Camera Compilation"}
# { "video": {"type": "VideoView", "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8", "start": 20, "end": 34}, "title": "2021 Swan Valley Wildlife Trail Camera Compilation"}
# { "video": {"type": "VideoView", "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8", "start": 36, "end": 58}, "title": "2021 Swan Valley Wildlife Trail Camera Compilation"}
# { "video": {"type": "VideoView", "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8", "start": 61, "end": 78}, "title": "2021 Swan Valley Wildlife Trail Camera Compilation"}
# { "video": {"type": "VideoView", "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8", "start": 81, "end": 100}, "title": "2021 Swan Valley Wildlife Trail Camera Compilation"}
# { "video": {"type": "VideoView", "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8", "start": 102, "end": 108}, "title": "2021 Swan Valley Wildlife Trail Camera Compilation"}

input_data = Input("data/animals-small.jsonl")

detected = Detect(
    input_data,
    "video",
    ["bear"],
    output_field="detections",
)

output = Unnest(detected, "detections")
