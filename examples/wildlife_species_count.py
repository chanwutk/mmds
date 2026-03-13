from mmds import ForEach, Input, Map, Record, Reduce, Unnest

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

unnested = Unnest(mapped, "animal_types")

output = Reduce(
    unnested,
    "animal_types",
    [
        "The following video clips all contain the same species of animal:\n",
        ForEach(["- species: ", Record["animal_types"], "\n"]),
        "How many clips contain this species? Reply with just the count.",
    ],
    schema={"count": "integer"},
)
