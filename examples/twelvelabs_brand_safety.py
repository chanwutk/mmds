"""TwelveLabs-style compliance / brand-safety screening (semantic, requires Gemini).

Inspired by policy and brand-safety review at scale:
https://playground.twelvelabs.io/analyze?task_id=6a1f22b8df3693601774cd79

Third video is from Squid Game Season 2, which is expected to be filtered out due to violence.

Run:
  PYTHONPATH=src:. ./.venv/bin/python examples/run_expr.py examples/twelvelabs_brand_safety.py
[
  {
    "video": {
      "type": "Video",
      "uri": "https://www.youtube.com/watch?v=YLslsZuEaNE"
    },
    "title": "A dog video",
    "brand_safe": true,
    "review_notes": "The video displays two puppies playing on the grass. There is no violent, explicit, or inappropriate content.",
    "risk_level": "low"
  },
  {
    "video": {
      "type": "Video",
      "uri": "https://www.youtube.com/watch?v=bjdBVZa66oU"
    },
    "title": "What are skills?",
    "brand_safe": true,
    "review_notes": "The video is an instructional guide about a coding tool feature called 'Skills'. It contains no violence, hate speech, or explicit content. The content is professional, educational, and safe for all audiences.",
    "risk_level": "low"
  },
  {
    "video": {
      "type": "Video",
      "uri": "https://youtu.be/HaLISMAGdOE?si=s68AY7BPqzO8lOPx"
    },
    "title": "MY NEIGHBOR TOTORO | Official English Trailer",
    "brand_safe": true,
    "review_notes": "The video is a trailer for a G-rated family movie. It contains no violence, explicit content, or other policy-violating materials.",
    "risk_level": "none"
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
        "Review this clip for brand-safety and policy risk before ad placement.\n",
        "Video: ",
        Record["video"],
        "\nTitle: ",
        Record["title"],
        "\nFlag violence, explicit content, hate symbols, or other high-risk material."
    ],
    schema={
        "risk_level": "string",
        "brand_safe": "boolean",
        "review_notes": "string"
    }
)

output = Filter(
    mapped,
    [
        "Keep only clips that are brand_safe according to the prior analysis.\n",
        "risk_level: ",
        Record["risk_level"],
        "\nbrand_safe: ",
        Record["brand_safe"],
        "\nreview_notes: ",
        Record["review_notes"]
    ]
)
