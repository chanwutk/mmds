"""Find the most exciting clip per NBA highlight video (Split → Map → Reduce, requires Gemini).

Uses Warriors and Lakers NBA game highlight videos from ``data/nba_warriors.jsonl``.

Pipeline:
1. **Split** — slice each video into 3-minute chunks (``chunk_sec=180``).
2. **Map** — one LLM call **per chunk** scores excitement (one video part per request).
3. **Reduce** — one LLM call per source video reads the text scores and picks the best chunk
   (no video parts — only numeric scores and reasons from step 2).

Run:
  uv run python examples/run_expr.py examples/split_highlight_videos.py
"""

from mmds import ForEach, Input, Map, Record, Reduce, Split

clips = Input("data/nba_warriors.jsonl")

chunks = Split(
    clips,
    "video",
    chunk_sec=30.0,
    doc_id_key="video_id",
)

scored = Map(
    chunks,
    [
        "Score this NBA highlight clip for fan excitement on a 1–10 scale.\n",
        "Prefer dunks, deep threes, blocks, assists, crowd reactions, and clutch plays.\n",
        "Clip ",
        Record["split_video_chunk_num"],
        " (",
        Record["split_video_chunk_start"],
        "s–",
        Record["split_video_chunk_end"],
        "s)\n",
        "Game: ",
        Record["title"],
        "\n",
        "Video: ",
        Record["video"],
        "\n",
    ],
    schema={
        "excitement_score": "number",
        "excitement_reason": "string",
    },
)

output = Reduce(
    scored,
    "split_video_id",
    [
        "These are excitement scores for consecutive clips from one NBA highlight video.\n",
        "Pick the single best clip — do not invent scores; use the values below.\n",
        ForEach(
            [
                "Clip ",
                Record["split_video_chunk_num"],
                " (",
                Record["split_video_chunk_start"],
                "s–",
                Record["split_video_chunk_end"],
                "s): score ",
                Record["excitement_score"],
                " — ",
                Record["excitement_reason"],
                "\n",
            ]
        ),
        "Return the winning clip number, its time window, score, and reason.",
    ],
    schema={
        "best_clip_num": "integer",
        "best_clip_start": "number",
        "best_clip_end": "number",
        "excitement_score": "number",
        "excitement_reason": "string",
    },
)
