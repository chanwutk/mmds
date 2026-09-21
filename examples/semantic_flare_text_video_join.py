"""Semantic FLARE text-video join — UCA Abuse001/002/003 temporal grounding (requires Gemini).

Fully-semantic counterpart to ``flare_text_video_join.py``. Instead of joining
ground-truth captions to their ground-truth timestamps, Gemini watches each
video and *grounds* every provided caption to a ``[start, end]`` window on the
video's own timeline. Same task, same output shape — so the two examples can be
compared directly (GT timestamps vs. Gemini-predicted timestamps for the same
sentences).

Input is ``data/uca_grounding.jsonl``: one row per video carrying the whole
video plus its ordered caption ``sentences`` (pre-merged from
``data/uca_gallery.jsonl`` and ``data/uca_captions.jsonl``, since a single
``Map`` reads one row at a time and needs both the media and the sentences
together).

Pipeline:
1. **Input** — one row per video (video + ordered caption sentences).
2. **Map** — one prompt per video: watch the clip and return, index-aligned to
   the input captions, each caption's ``[start, end]`` window; echo the
   sentences unchanged. Default merge keeps ``video_id``/``duration``/``title``/
   ``video`` from the row, so each output row matches ``flare_text_video_join.py``
   and ``data/uca/annotation_excerpt.json``.

Run:
  uv run python examples/run_expr.py examples/semantic_flare_text_video_join.py
Output::

[
  {
    "video_id": "Abuse001_x264",
    "duration": 91.0,
    "sentences": [
      "A woman with short hair, slightly fat, wearing a white top and black pants stood in front of the table, picked up a book from the table, and opened it to read",
      "A man wearing a white shirt and black pants entered the house and walked towards the short-haired and fat woman in front who was reading a book.",
      ...
    ],
    "title": "UCA Abuse001",
    "video": {
      "type": "Video",
      "path": "data/uca/videos/Abuse001_x264.mp4"
    },
    "timestamps": [
      [
        0.0,
        5.1
      ],
      [
        6.8,
        8.6
      ],
      ...
    ]
  },
  ...
]
"""

from mmds import Input, Map, Record

videos = Input("data/uca_grounding.jsonl")

output = Map(
    videos,
    [
        "You are a precise video temporal-grounding annotator. Watch the "
        "surveillance clip and localize each provided caption to the [start, end] "
        "window (in seconds on the video's own timeline) when that event occurs. "
        "The captions are chronological and describe consecutive events.\n",
        "Video duration (seconds): ",
        Record["duration"],
        "\n",
        "Video: ",
        Record["video"],
        "\n",
        "Captions (JSON array, in order):\n",
        Record["sentences"],
        "\n",
        "Return a 'timestamps' array with one [start, end] pair per caption, "
        "index-aligned to the input order, with 0 <= start <= end <= duration. "
        "Return a 'sentences' array echoing the input captions unchanged and in "
        "the same order.",
    ],
    schema={
        "timestamps": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "sentences": {"type": "array", "items": {"type": "string"}},
    },
)
