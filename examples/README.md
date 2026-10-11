# Examples

Runnable MMDS query examples, plus small drivers for executing them. See the
top-level [GET_START.md](../GET_START.md) for setup and [README.md](../README.md) for an overview.

## Queries

| File | Operators | Needs |
|------|-----------|-------|
| [`wildlife_species.py`](wildlife_species.py) | `Map` (video prompt) → `Unnest` | Gemini API key |
| [`animals_bear_map.py`](animals_bear_map.py) | `Map` (video prompt: is a bear visible?) | Gemini API key; full Swan Valley video in `data/swan_valley_full.jsonl` |
| [`animals_bear_detect_presence.py`](animals_bear_detect_presence.py) | `Detect` (`stop_after_n=1`) → `Filter` → code `Map` (`bear_present`) | YOLOE weights; `data/swan_valley_full.jsonl` |
| [`animals_five_bears_map.py`](animals_five_bears_map.py) | `Map` (video prompt: at least five bears?) | Gemini API key; `data/swan_valley_full.jsonl` |
| [`animals_five_bears_detect_presence.py`](animals_five_bears_detect_presence.py) | `Detect` (`stop_after_n=5`, `frame_stride=10`) → `Filter` → code `Map` (`at_least_five_bears`) | YOLOE weights; `data/swan_valley_full.jsonl` |
| [`animals_false_map.py`](animals_false_map.py) | `Map` (video prompt: is a sedan visible?) | Gemini API key; `data/swan_valley_full.jsonl` |
| [`animals_false_detect_presence.py`](animals_false_detect_presence.py) | `Detect` (`stop_after_n=1`, `frame_stride=10`) → `Filter` → code `Map` (`sedan_present`) | YOLOE weights; `data/swan_valley_full.jsonl` |
| [`animals_bear_count_map.py`](animals_bear_count_map.py) | `Map` (video prompt: how many bears?) | Gemini API key; `data/swan_valley_full.jsonl` |
| [`animals_bear_count.py`](animals_bear_count.py) | `Detect` (full scan, `frame_stride=10`) → code `Map` (`bear_count` = distinct tracks) | YOLOE weights; `data/swan_valley_full.jsonl` |
| [`vehicle_count_map.py`](vehicle_count_map.py) | `Map` (video prompt: how many vehicles?) | Gemini API key; `data/i24v_highway2_5s.jsonl` + local highway2 MP4 |
| [`vehicle_count.py`](vehicle_count.py) | `Detect` (`vehicle`, every frame) → code `Map` (`vehicle_count` = tracks with ≥3 boxes) | YOLOE weights; `data/i24v_highway2_5s.jsonl` + local highway2 MP4 |
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
| [`lecture_event_localization_video_only.py`](lecture_event_localization_video_only.py) | full-video `Map` → `Unnest` | Gemini API key; `data/lectures.jsonl` |
| [`lecture_event_localization_transcript_only.py`](lecture_event_localization_transcript_only.py) | transcript `Map` → `Unnest` | Gemini API key; `data/lectures.jsonl` |
| [`lecture_event_localization.py`](lecture_event_localization.py) | transcript `Map` → `Unnest` → `Window` → `Coalesce` → video `Map` → UDF `Map` → `Reduce` → `Unnest` | Gemini API key; `data/lectures.jsonl` |
| [`join_cross_camera_vehicle.py`](join_cross_camera_vehicle.py) | `Detect` → UDF `Map`s (NMS, frame detections, tracking, embeddings) → `Unnest` → UDF `Map`s (track rows) → `Join` (self-join, `one_to_one`) → UDF `Map` | no API key; downloads YOLOE + ResNet weights; local I24V MP4s (see [`data/i24v_traffic/README.md`](../data/i24v_traffic/README.md)) |
| [`semantic_join_cross_camera_vehicle.py`](semantic_join_cross_camera_vehicle.py) | `Reduce` (video prompt) + `ForEach` → `Unnest` → UDF `Map` | Gemini API key; local I24V MP4s (see [`data/i24v_traffic/README.md`](../data/i24v_traffic/README.md)) |

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
- [`eval_animals_bear.py`](eval_animals_bear.py) — scores
  [`animals_bear_detect_presence.py`](animals_bear_detect_presence.py), the
  `detect_presence_map` rewrite, against
  [`data/animals_bear_ground_truth.json`](../data/animals_bear_ground_truth.json).
  YOLOE stops at the first bear track and a code Map sets `bear_present`;
  there is no prompt call. Pass `--compare-semantic` to also score
  [`animals_bear_map.py`](animals_bear_map.py) and print precision, recall, F1,
  and token cost side by side. A clip with no detection is dropped and counted
  as `bear_present=false`.
- [`eval_animals_five_bears.py`](eval_animals_five_bears.py) — same comparison
  for at least five bear tracks:
  [`animals_five_bears_detect_presence.py`](animals_five_bears_detect_presence.py)
  against
  [`animals_five_bears_map.py`](animals_five_bears_map.py) and
  [`data/animals_five_bears_ground_truth.json`](../data/animals_five_bears_ground_truth.json).
- [`eval_animals_false.py`](eval_animals_false.py) — scores a negative presence
  question (is a sedan visible?) using
  [`animals_false_detect_presence.py`](animals_false_detect_presence.py)
  against
  [`animals_false_map.py`](animals_false_map.py) and
  [`data/animals_false_ground_truth.json`](../data/animals_false_ground_truth.json).
  The label is false. A clip with no sedan track is dropped and counted as
  `sedan_present=false`.
- [`eval_animals_bear_count.py`](eval_animals_bear_count.py) — scores the
  exact bear count from [`animals_bear_count.py`](animals_bear_count.py), the
  `detect_count_map` rewrite, against
  [`data/animals_bear_count_ground_truth.json`](../data/animals_bear_count_ground_truth.json)
  (21 bears in the full video). Pass `--compare-semantic` to also score
  [`animals_bear_count_map.py`](animals_bear_count_map.py) and print the
  predicted count, signed error, MAE, relative error, and token cost side by
  side.
- [`eval_vehicle_count.py`](eval_vehicle_count.py) — the same count scoring
  for [`vehicle_count.py`](vehicle_count.py) against
  [`vehicle_count_map.py`](vehicle_count_map.py) and
  [`data/vehicle_count_ground_truth.json`](../data/vehicle_count_ground_truth.json)
  (20 vehicles in the first 5 s of I24V highway2). It is a wrapper around
  `eval_animals_bear_count.py`.

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

Only the transcript-gated query needs a `Reduce`: several selected video
windows can produce separate event arrays for one lecture. Its
`collect_sorted_events` UDF flattens and sorts those arrays; it deliberately
does not merge or deduplicate event intervals.

The two-stage example is built in Python (a `build_query` function, a shared
schema constant, and keyword arguments to `Window`), so it cannot run through
`examples/run_text.py`.

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
