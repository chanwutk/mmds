from mmds import Input, Map, Record, Unnest

# Example input rows for the "clips" dataset:
#
# rows:
# { "video": {"type": "VideoView", "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8", "start": 0, "end": 19}, "title": "2021 Swan Valley Wildlife Trail Camera Compilation"}
# { "video": {"type": "VideoView", "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8", "start": 20, "end": 34}, "title": "2021 Swan Valley Wildlife Trail Camera Compilation"}
# { "video": {"type": "VideoView", "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8", "start": 36, "end": 58}, "title": "2021 Swan Valley Wildlife Trail Camera Compilation"}
# { "video": {"type": "VideoView", "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8", "start": 61, "end": 78}, "title": "2021 Swan Valley Wildlife Trail Camera Compilation"}
# { "video": {"type": "VideoView", "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8", "start": 81, "end": 100}, "title": "2021 Swan Valley Wildlife Trail Camera Compilation"}
# { "video": {"type": "VideoView", "source": "https://www.youtube.com/watch?v=s5iU3nLOvi8", "start": 102, "end": 108}, "title": "2021 Swan Valley Wildlife Trail Camera Compilation"}

input_data = Input("data/animals.jsonl")

mapped = Map(
    input_data,
    [
        "Watch this video clip.\n",
        "What kind of animals do you see in the video clip?"
        "video clips: ",
        Record["video"],
    ],
    schema={"animal_types": {"type": "array", "items": {"type": "string"}}},
)

output = Unnest(mapped, "animal_types")
