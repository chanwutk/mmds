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
| [`split_highlight_videos.py`](split_highlight_videos.py) | `Split` → `Map` → `Reduce` + `ForEach` | Gemini API key; `data/nba_warriors.jsonl` |
| [`flare_text_video_join.py`](flare_text_video_join.py) | `Map` → `Join` → `Map` → `Reduce` | Local UCA Abuse001/002/003 + GT captions; output grouped by video like `annotation_excerpt.json` (no Gemini) |
| [`semantic_flare_text_video_join.py`](semantic_flare_text_video_join.py) | `Map` (video + captions prompt) | Gemini API key; `data/uca_grounding.jsonl` (per-video fixture); Gemini temporal-grounding baseline for `flare_text_video_join.py` — predicts timestamps for the same captions, same output shape |
| [`join_cross_camera_vehicle.py`](join_cross_camera_vehicle.py) | `Detect` → `Map` → `Unnest` → `Map` (promote/embed/project) → `Join` → `Map` | Local I24V 5s highway2/3; motion-aware tracking + appearance re-ID one-to-one join trajectories (no Gemini; ResNet50 weights auto-download, histogram fallback) |
| [`semantic_join_cross_camera_vehicle.py`](semantic_join_cross_camera_vehicle.py) | `Reduce` + `ForEach` → `Unnest` | Same I24V clips; Gemini soft-reference stitch |
| [`hybrid_join_cross_camera_vehicle.py`](hybrid_join_cross_camera_vehicle.py) | per branch: `Filter` → `Detect` → `Map` → `Unnest` → `Map` (crop) → `Map` (Gemini crop label) then cross-camera `Join` → `Map` | Local I24V 5s highway2/3 + Gemini API key; per-track crop labeling, cheap cross-camera join |
| [`compare_cross_camera_joins.py`](compare_cross_camera_joins.py) | driver (not a DSL query) | Soft-reference P/R/F1 + wall-time/prompt-call/token cost proxies for UDF vs semantic |
| [`eval_cross_camera_join.py`](eval_cross_camera_join.py) | driver (not a DSL query) | Scores the UDF join against `data/i24v_..._ground_truth.json`: P/R/F1, attribute agreement, per-match FP/FN diagnostics, and wall time/tokens; `--compare-semantic` adds a side-by-side UDF vs semantic table |
| [`twelvelabs_map_reduce.py`](twelvelabs_map_reduce.py) | `Map` → `Unnest` → `Reduce` + `ForEach` | Gemini API key; `data/animals.jsonl` |
| [`wildlife_detection.py`](wildlife_detection.py) | `Detect` (local YOLOE) → `Unnest` | no API key; downloads weights + video |
| [`detect_multi_species.py`](detect_multi_species.py) | `Detect` → `Unnest` | no API key; downloads weights + video |
| [`detect_filter_bears.py`](detect_filter_bears.py) | `Detect` → `Filter` (UDF) | no API key; downloads weights + video |
| [`detect_filter_bears_high_confidence.py`](detect_filter_bears_high_confidence.py) | `Detect` → `Map` (UDF prune) → `Filter` (UDF) | no API key; downloads weights + video |

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
