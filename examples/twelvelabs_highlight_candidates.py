"""TwelveLabs-style highlight discovery (semantic Map → Filter, requires Gemini).

Inspired by finding highlight-worthy moments from long-form footage:
https://playground.twelvelabs.io/analyze?task_id=6a1f156e5ae0f4f8a5c0a0f0

Uses New York Knicks NBA game highlight videos from data/nba_knicks.jsonl.
Last video is a Spurs vs Thunder game, which is expected to be filtered out.

Run:
  PYTHONPATH=src:. ./.venv/bin/python examples/run_expr.py examples/twelvelabs_highlight_candidates.py
[
  {
    "video": {
      "type": "Video",
      "uri": "https://www.youtube.com/watch?v=xUml0jWLdv8"
    },
    "title": "New York Knicks vs Philadelphia 76ers Full Game 3 Highlights | East Semifinals | 2026 NBA Play-Off",
    "highlight_reason": "This is a comprehensive full-game highlights reel for a crucial Knicks playoff victory
    in the Eastern Conference Semifinals, featuring multiple impactful plays including high-scoring sequences, key
    defensive plays, and significant contributions from standout Knicks players.",
    "highlight_score": 9.5,
    "recommended_for_reel": true
  },
  {
    "video": {
      "type": "Video",
      "uri": "https://www.youtube.com/watch?v=sQISGYHIV1A"
    },
    "title": "SPURS at KNICKS | FULL GAME HIGHLIGHTS | March 1, 2026",
    "highlight_reason": "The video provides a comprehensive recap of a New York Knicks game, showcasing key
    highlights like dunks, deep threes, and impactful plays from several Knicks players, making it highly suitable
    for a fan highlight reel.",
    "highlight_score": 9.5,
    "recommended_for_reel": true
  },
  {
    "video": {
      "type": "Video",
      "uri": "https://www.youtube.com/watch?v=k530Y0_oWss"
    },
    "title": "New York Knicks vs Cleveland Cavaliers Full Game 3 Highlights | 2026 NBA East Finals",
    "highlight_reason": "The video provides a comprehensive overview of the Knicks vs. Cavaliers Game 3,
    highlighting multiple key plays including strong finishes, perimeter shooting, defensive stops, and solid
    passing, making it an excellent candidate for a fan highlight reel.",
    "highlight_score": 9.5,
    "recommended_for_reel": true
  }
]
"""

from mmds import Filter, Input, Map, Record

clips = Input("data/nba_knicks.jsonl")

mapped = Map(
    clips,
    [
        "Score this New York Knicks game highlight video for a fan highlight reel.\n",
        "Prefer videos with exciting Knicks plays: dunks, deep threes, blocks, "
        "assists, clutch sequences, and standout performances by Knicks players.\n",
        "Video: ",
        Record["video"],
        "\nGame title: ",
        Record["title"]
    ],
    schema={
        "highlight_score": "number",
        "highlight_reason": "string",
        "recommended_for_reel": "boolean"
    }
)

output = Filter(
    mapped,
    [
        "Keep only Knicks highlight videos recommended_for_reel with highlight_score "
        "at least 7 (on a 1-10 scale).\n",
        "highlight_score: ",
        Record["highlight_score"],
        "\nrecommended_for_reel: ",
        Record["recommended_for_reel"],
        "\nhighlight_reason: ",
        Record["highlight_reason"]
    ]
)
