"""TwelveLabs-style library insights (semantic Map → Reduce, requires Gemini).

Inspired TwelveLabs example by analyzing many clips to surface specific patterns
and get summary insights:
https://playground.twelvelabs.io/analyze?task_id=6a1f156e5ae0f4f8a5c0a0f0

Run:
  PYTHONPATH=src:. ./.venv/bin/python examples/run_expr.py examples/twelvelabs_content_insights.py
[
  {
    "species_labels": "grizzly bear",
    "clip_count": 2,
    "insight": "The bear appears in both clips, showing different types of behavior: playing in the
    water in the first clip and standing on its hind legs to explore or mark a tree in the second.
    This suggests that the species is active and engages in various activities, ranging from playful
    interactions with water to more investigative or communicative behaviors like marking territory."
  },
  {
    "species_labels": "cougar",
    "clip_count": 2,
    "insight": "The footage captures two distinct scenarios involving mountain lions in a forest
    setting: one shows an adult solitary cougar marking its territory near a tree, while the other
    features a mother with two cubs emerging from a den. These interactions highlight different
    facets of mountain lion behavior, demonstrating both territorial marking and maternal care,
    emphasizing that these animals are frequently observed in varying life stages and social
    structures within their natural habitat."
  },
  {
    "species_labels": "wolf",
    "clip_count": 1,
    "insight": "The single clip captures a lone wolf moving slowly and cautiously through the tall
    green grass. It stops frequently, appearing to scan its surroundings and scent the air, suggesting
    a behavior of careful, alert observation as it navigates the terrain."
  },
  {
    "species_labels": "white-tailed deer",
    "clip_count": 1,
    "insight": "The buck appears in a single segment, moving cautiously through a snow-covered forest,
    highlighting its vigilance and adaptation to harsh winter conditions by navigating dense woodland
    environments."
  }
]
"""

from mmds import ForEach, Input, Map, Record, Reduce, Unnest

input_data = Input("data/animals.jsonl")

mapped = Map(
    input_data,
    [
        "List animal species clearly visible in this trail-camera clip.\n",
        "Video: ",
        Record["video"],
    ],
    schema={"species_labels": {"type": "array", "items": {"type": "string"}}}
)

unnested = Unnest(mapped, "species_labels")

output = Reduce(
    unnested,
    "species_labels",
    [
        "These clips were tagged with the same species label.\n",
        ForEach(
            [
                "- clip window ",
                Record["video"],
                " from ",
                Record["title"],
                "\n",
            ]
        ),
        "Write a one-paragraph insight about how often this species appears "
        "across the grouped clips and patterns in the behavior of the species.",
    ],
    schema={"insight": "string", "clip_count": "integer"}
)
