"""UCA caption and video join on Abuse001/002/003 (no Gemini).

Ground truth is ``data/uca/annotation_excerpt.json`` (Abuse001/002/003 slice of
UCFCrime_Train). Captions are loaded from ``data/uca_captions.jsonl``.
- Notes: maybe join at specific time? and then run comparison operator to check for eqaulity

Pipeline materializes the oracle grouped **by video** (same shape as each
excerpt entry: ``duration``, ``timestamps``, ``sentences``), since each caption
belongs to exactly one video:

1. **Map** — tag gallery rows with a stable source id.
2. **Join** — captions ⋈ videos on ``video_id``.
3. **Map** — one clip row per caption (VideoView at timestamps).
4. **Reduce** — group clips by ``video_id`` into per clipped sequential captions.

Run:
  uv run python examples/run_expr.py examples/flare_text_video_join.py
[
  {
    "video_id": "Abuse001_x264",
    "duration": 91.0,
    "timestamps": [
      [
        0.0,
        5.3
      ],
      [
        7.0,
        8.5
      ],
      [
        7.2,
        8.5
      ],
      [
        8.2,
        8.9
      ],
      [
        8.9,
        11.2
      ],
      [
        8.9,
        11.2
      ],
      [
        11.3,
        13.3
      ],
      [
        15.2,
        18.9
      ],
      [
        19.7,
        25.4
      ]
    ],
    "sentences": [
      "A woman with short hair, slightly fat, wearing a white top and black pants stood in front of the table, picked up a book from the table, and opened it to read",
      "A man wearing a white shirt and black pants entered the house and walked towards the short-haired and fat woman in front who was reading a book.",
      "A man wearing a black shirt and black pants entered the house and walked towards the short-haired and fat woman in front who was reading a book.",
      "A man wearing a white shirt and black pants approached a short-haired, fat woman wearing a white shirt and black pants. When the woman was notpaying attention, the man suddenly pulled out a piece of red cloth from the left side of the woman's body, then turned and ran away.",
      "A man in black clothes approached a short-haired, fat woman wearinga white top and black pants. When the woman turned back to the right, he punched the woman in the head and ran out. .",
      "The woman fell to the ground in pain, and the book in her hand fellto the ground when she fell. At the same time, she knocked the red wooden table in front of her crookedly. There were three things on the table.",
      "A woman with short hair and a fat figure wearing a white top and black pants fell to the ground, raised her right hand to touch her forehead, and hit her head on the wooden table leg.",
      "A woman with short hair, slightly fat, wearing a white top and black pants fell to the ground. She touched her forehead with her right hand and looked towards the direction of entering the house. She retracted her left leg, supported her left hand on the ground, and held her forehead with her right hand. Upper body.",
      "A woman with short hair, slightly fat, wearing a white top and black pants is sitting on the ground, leaning on the ground with her left hand,her right hand dropped from her head and placed on the knee of her right leg, her right leg is bent, she turns her head and kicks with her left leg Kneel down on the ground, stand up with your right hand on the ground."
    ],
    "title": "UCA Abuse001",
    "video": {
      "type": "Video",
      "path": "data/uca/videos/Abuse001_x264.mp4"
    }
  },
  ...
]
"""

from mmds import Input, Join, Map, Reduce
from udfs.retrieval_ops import (
    format_gt_caption_clip,
    group_gt_clips_by_video,
    tag_gallery_video,
)

gallery = Input("data/uca_gallery.jsonl")
captions = Input("data/uca_captions.jsonl")

tagged = Map(gallery, tag_gallery_video)

matches = Join(
    captions,
    tagged,
    on=("video_id",),
)

clips = Map(matches, format_gt_caption_clip, replace=True)

output = Reduce(clips, "video_id", group_gt_clips_by_video)
