"""TwelveLabs-style Search & Discover (semantic, requires Gemini).

Inspired by natural-language search over a video library:
https://www.twelvelabs.io/

Run:
  PYTHONPATH=src:. ./.venv/bin/python examples/run_expr.py examples/twelvelabs_search_and_discover.py
[
  {
    "video": {
      "type": "Video",
      "uri": "https://www.youtube.com/watch?v=YLslsZuEaNE"
    },
    "title": "A dog video",
    "has_animal": true,
    "primary_subjects": [
      "puppies",
      "dogs",
      "pets"
    ],
    "search_summary": "Two cute white puppies playing with a soft toy in a backyard and drinking milk from a bowl."
  }
]
"""

from mmds import Filter, Input, Map, Record

# data/clips.jsonl rows:
# { "title": "A dog video", "video": {"type": "Video", "uri": "https://www.youtube.com/watch?v=YLslsZuEaNE"} }
# { "title": "What are skills?", "video": {"type": "Video", "uri": "https://www.youtube.com/watch?v=bjdBVZa66oU"} }
# { "title": "Players Try to Get in a Room before Elimination | Squid Game: Season 2 | Netflix", "video": {"type": "Video", "uri": "https://youtu.be/s3UxNVqpnec?si=-AholtxVdVERgDNN"}}
# { "title": "MY NEIGHBOR TOTORO | Official English Trailer", "video": {"type": "Video", "uri": "https://youtu.be/HaLISMAGdOE?si=s68AY7BPqzO8lOPx"}}

clips = Input("data/clips.jsonl")

mapped = Map(
    clips,
    [
        "You are indexing a short-form video library for semantic search.\n",
        "Watch the clip and extract what a user might search for later.\n",
        "Video: ",
        Record["video"],
        "\nTitle hint: ",
        Record["title"],
    ],
    schema={
        "search_summary": "string",
        "primary_subjects": {"type": "array", "items": {"type": "string"}},
        "has_animal": "boolean"
    }
)

output = Filter(
    mapped,
    [
        "Keep this row only if the clip is a good match for the query: "
        "'dog or puppy outdoors'.\n",
        "search_summary: ",
        Record["search_summary"],
        "\nprimary_subjects: ",
        Record["primary_subjects"],
        "\nhas_animal: ",
        Record["has_animal"]
    ]
)
