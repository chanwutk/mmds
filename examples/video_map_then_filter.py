from mmds import Filter, Input, Map, Record

# Example input rows for the "clips" dataset:
#
# clips_rows = [
#     {
#         "clip_id": "clip-001",
#         "title": "Golden retriever runs through a park",
#         "video": {"type": "Video", "path": "/data/clips/dog-park.mp4"},
#     },
#     {
#         "clip_id": "clip-002",
#         "title": "Empty office lobby security footage",
#         "video": {"type": "Video", "uri": "https://youtube.com/watch?v=example"},
#     },
# ]

clips = Input("data/clips.jsonl")

mapped = Map(
    clips,
    [
        "Watch this video clip.\n",
        "Video content: ",
        Record["video"],
        "\nReturn a short summary and whether a dog is clearly visible.",
    ],
    schema={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "contains_dog": {"type": "boolean"},
        },
        "required": ["summary", "contains_dog"],
    },
)

output = Filter(
    mapped,
    [
        "Keep this row only when the clip clearly contains a dog.\n",
        "Model summary: ",
        Record["summary"],
        "\ncontains_dog flag: ",
        Record["contains_dog"],
    ],
)
