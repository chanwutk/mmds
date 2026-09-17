# Examples

Runnable MMDS query examples, plus small drivers for executing them. See the
top-level [GET_START.md](../GET_START.md) for setup and [README.md](../README.md) for an overview.

## Queries

| File | Operators | Needs |
|------|-----------|-------|
| [`wildlife_species.py`](wildlife_species.py) | `Map` (video prompt) → `Unnest` | Gemini API key |
| [`wildlife_species_count.py`](wildlife_species_count.py) | `Map` → `Unnest` → `Reduce` + `ForEach` | Gemini API key |
| [`video_map_then_filter.py`](video_map_then_filter.py) | `Map` (video prompt) → `Filter` | Gemini API key |
| [`twelvelabs_search_and_discover.py`](twelvelabs_search_and_discover.py) | `Map` (video prompt) → `Filter` | Gemini API key; `data/clips.jsonl` |
| [`twelvelabs_brand_safety.py`](twelvelabs_brand_safety.py) | `Map` (video prompt) → `Filter` | Gemini API key; `data/clips.jsonl` |
| [`twelvelabs_highlight_candidates.py`](twelvelabs_highlight_candidates.py) | `Map` (video prompt) → `Filter` | Gemini API key; `data/nba_knicks.jsonl` |
| [`twelvelabs_map_reduce.py`](twelvelabs_map_reduce.py) | `Map` → `Unnest` → `Reduce` + `ForEach` | Gemini API key; `data/animals.jsonl` |
| [`wildlife_detection.py`](wildlife_detection.py) | `Detect` (local YOLOE) → `Unnest` | no API key; downloads weights + video |
| [`detect_multi_species.py`](detect_multi_species.py) | `Detect` → `Unnest` | no API key; downloads weights + video |
| [`detect_filter_bears.py`](detect_filter_bears.py) | `Detect` → `Filter` (UDF) | no API key; downloads weights + video |
| [`detect_filter_bears_high_confidence.py`](detect_filter_bears_high_confidence.py) | `Detect` → `Map` (UDF prune) → `Filter` (UDF) | no API key; downloads weights + video |
| [`lecture_event_localization_video_only.py`](lecture_event_localization_video_only.py) | full-video `Map` → `Reduce` → `Unnest` | Gemini API key; `data/lectures.jsonl` |
| [`lecture_event_localization_transcript_only.py`](lecture_event_localization_transcript_only.py) | transcript `Map` → `Reduce` → `Unnest` | Gemini API key; `data/lectures.jsonl` |
| [`lecture_event_localization.py`](lecture_event_localization.py) | transcript `Map` → `Unnest` → `Window` → `Coalesce` → video `Map` → UDF `Map` → `Reduce` → `Unnest` | Gemini API key; `data/lectures.jsonl` |
| [`semantic_join_cross_camera_vehicle.py`](semantic_join_cross_camera_vehicle.py) | `Reduce` + `ForEach` → `Unnest` → promote `Map` | Gemini API key; local I24V highway2/3 clips |
| [`join_cross_camera_vehicle.py`](join_cross_camera_vehicle.py) | `Detect` → tracking/re-ID `Map`s → `Unnest` → one-to-one `Join` → trajectory `Map` | local I24V highway2/3 clips; YOLOE and re-ID model weights |

## Drivers

- [`run_expr.py`](run_expr.py) — imports a query module and executes its `output`
  expression through `GeminiPromptExecutor`. This is what the top-level
  [`./run`](../run) script wraps, e.g. `./run examples/wildlife_species.py`.
  Prompt-backed examples require a Gemini key; `Detect` examples do not call
  Gemini even when run through this driver.
- [`run_text.py`](run_text.py) — reads a query file as **DSL text**, parses it with
  `parse_query(...)`, then executes the resulting program. Use this to exercise the
  text → plan parsing path: `PYTHONPATH=src:. ./.venv/bin/python examples/run_text.py <query_file>`.
- [`run_detect.py`](run_detect.py) — same as `run_expr.py` but calls `execute(...)`
  without constructing `GeminiPromptExecutor`. Optional; `run_expr.py` is enough
  for local `Detect` pipelines.

## Lecture event localization data

[`data/lectures.jsonl`](../data/lectures.jsonl) is the single self-contained
lecture dataset. Each of its three rows includes video metadata, timestamped
transcript cues, a localization query, and evaluation annotations. The local MP4
paths in those rows must exist before a video-backed query can run.

All three query variants consume the same `data/lectures.jsonl` rows:

```bash
./run examples/lecture_event_localization_video_only.py
./run examples/lecture_event_localization_transcript_only.py
./run examples/lecture_event_localization.py
```

`Window` and `Coalesce` can be rendered and parsed through the restricted DSL,
so the two-stage plan can also be inspected as normalized query text.

## Cross-camera vehicle data

The cross-camera example defaults to the adjacent highway2/3 manifest at
[`data/i24v_traffic_highway2_highway3_5s.jsonl`](../data/i24v_traffic_highway2_highway3_5s.jsonl).
The licensed MP4 inputs are intentionally not committed. Follow
[`data/i24v_traffic/README.md`](../data/i24v_traffic/README.md) to install them
locally before running:

```bash
./run examples/semantic_join_cross_camera_vehicle.py
./run examples/join_cross_camera_vehicle.py
```

The semantic query is the Gemini Reduce–Unnest baseline. The Detect–Track–Join
query is the rewritten UDF plan. Both emit the same final row fields:
`vehicle_id`, `attributes`, `timeline`, and `match_score`.

## Notes

- Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) before running any prompt-backed example.
  `Detect` examples need no key.
- Video fields are ordinary record fields. The Gemini executor treats values whose `type`
  is case-insensitively `"video"`/`"videoview"` as video parts; the canonical shapes are
  `{"type": "Video", ...}` and `{"type": "VideoView", ...}`. A public `https://` `source`
  (e.g. a YouTube URL) is passed straight to Gemini; the local `Detect` path downloads it
  via `yt-dlp` instead.
- Prompt-backed `Map` and `Reduce` use the concise schema form, e.g. `schema={"count": "integer"}`.
- `Input(...)` takes a `.json` or `.jsonl` file path directly — no data catalog required.
- `Detect` filter/prune examples use UDFs from [`udfs/detection_ops.py`](../udfs/detection_ops.py).
- YOLOE may download `yoloe-11s-seg.pt` and `mobileclip_blt.ts` into the repo root on first
  run; they are gitignored and should not be committed.
