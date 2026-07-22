# MMDS Design

`DESIGN.md` is a living document. Any change that affects the DSL surface, logical plan model, execution semantics, optimizer behavior, executor integrations, or UDF contract must update this file in the same change.

## Overview

MMDS is a Python-first DSL for semantic data workflows. A query is written as ordinary Python assignments:

```python
from mmds import Input, Map, Reduce, Record, ForEach

docs = Input("data/docs.jsonl")
mapped = Map(
    docs,
    ["Summarize ", Record["title"], " from ", Record["video"]],
    schema={"summary": "string"},
)
output = Reduce(
    mapped,
    "_all",
    ["Summaries:\n", ForEach(["- ", Record["summary"], "\n"])],
    schema={"report": "string"},
)
```

The system currently supports three representations of the same query:

1. Python query text in a restricted DSL subset.
2. An immutable logical operator tree built from `DatasetExpr`.
3. An executable local runtime plan over `Iterable[dict[str, Any]]`.

The design goal is to keep those representations close enough that:

- Python text can be parsed into a plan.
- Plans can be rendered back to normalized Python text.
- The same plan can be rewritten by symbolic or LLM-backed optimizers.
- Queries can execute locally for development and tests.

## Current Scope

The current implementation supports:

- operators: `Input`, `Map`, `Filter`, `Reduce`, `Unnest`, `Split`, `Detect`, `Join`
- public video utility: `VideoView(video, start, end)` for seek-based clip-range iteration
- file-backed `Input(...)` roots over `.json` and `.jsonl`
- prompt-backed semantics as either:
  - a plain string
  - a structured prompt list made of strings, `Record[...]`, and Reduce-level `ForEach([...])`
- function-backed semantics as imported UDFs from `udfs.*`
- prompt-backed `Map` and `Reduce` with concise output field schemas
- prompt-backed `Filter` returning a bare JSON boolean
- source parsing for straight-line top-level assignments only
- plan rendering back to normalized Python code
- local execution with an injected prompt executor
- Gemini-backed prompt execution with executor-side video translation
- UDF discovery from `.py` and `.pyi`
- a conservative rule optimizer and a validation-heavy LLM optimizer scaffold
- `Detect` operator for frame-level YOLOE object detection on video fields

The current implementation intentionally does not support:

- inline lambdas
- nested functions or callables outside `udfs.*`
- loops, conditionals, comprehensions, classes, or arbitrary Python control flow in query files
- joins, sorts, projections, or cost-based optimization beyond hash join
- automatic `.py` implementation synthesis from `.pyi`
- nested `ForEach(...)`
- provider-specific media syntax in the DSL
- data catalogs or named dataset registries

## Architecture

### Public DSL

The main entrypoints are exported from [src/mmds/__init__.py](/Users/chanwutk/Documents/mmds/src/mmds/__init__.py):

- `Input(path)`
- `Map(data, spec, *, schema=None, name=None)`
- `Filter(data, spec, *, name=None)`
- `Reduce(data, group_by, reducer, *, schema=None, name=None)`
- `Unnest(data, field, *, keep_empty=False, name=None)`
- `Split(data, video_field, *, chunk_sec=30, doc_id_key="camera_id", output_prefix="split_video", name=None)`
- `Join(left, right, predicate?, *, on=..., one_to_one=..., score=..., min_score=..., left_key=..., right_key=..., name=None)`
- `Detect(data, video_field, classes, *, model="yoloe-11s-seg.pt", output_field="detections", frame_stride=1, conf=None, imgsz=None, name=None)`
- `VideoView(video, start, end)`
- `Record[...]`
- `ForEach([...])`
- `execute(plan_or_query, prompt_executor=None)`
- `GeminiPromptExecutor(...)`
- `load_query(source)` / `parse_query(source)`
- `render_query(plan_or_query)`
- `optimize(plan)` / `canonicalize(plan)`

### Logical Plan Model

The core model lives in [src/mmds/model.py](/Users/chanwutk/Documents/mmds/src/mmds/model.py).

- `DatasetExpr` is the immutable logical operator node.
- `PromptSpec` stores prompt-backed semantics as prompt parts plus optional output schema.
- `RecordPath` represents `Record["field"]...` references.
- `ForEachPrompt` represents repeated prompt expansion over grouped records.
- `ResolvedPrompt` is the execution-time prompt after all `Record[...]` references are resolved against data.
- `UdfSpec` stores a stable import path for a UDF.
- `SplitSpec` stores fixed-duration video chunking parameters: `video_field`, `chunk_sec`, `doc_id_key`, `output_prefix`.
- `DetectSpec` stores the parameters for a `Detect` node: `video_field`, `classes`, `model`, `output_field`, `frame_stride`, `conf`, `imgsz`.
- `JoinSpec` stores hash-join keys, optional predicate/score UDFs, and one-to-one matching options.
- `Assignment` and `QueryProgram` represent a parsed query file.
- `MMDSValidationError` is the shared validation failure type.

`DatasetExpr` uses a unary tree shape today:

- `Input` has no source.
- `Map`, `Filter`, `Reduce`, `Unnest`, `Split`, and `Detect` each have one `source`.
- `Join` has `source` (left) and `right_source` (right).

That shape is sufficient for the first operator set and keeps rendering and execution simple. If future operators introduce multiple inputs, `DatasetExpr` will need a general child list instead of a single `source`.

### Prompt Expression Model

Prompt expressions are now structured.

Allowed prompt parts:

- string literals
- `Record["field"]` and nested references like `Record["video"]["uri"]`
- `ForEach([...])`, but only at the top level of a `Reduce` prompt

Semantics:

- In `Map` and `Filter`, `Record[...]` refers to the current input row.
- In `Reduce`, row-level field access must be inside `ForEach([...])`.
- `ForEach([...])` expands its body once per grouped row in order.
- Nested `ForEach(...)` is not supported.

This model makes prompt construction explicit and gives executors enough structure to preserve non-text field values, including media.

### Output Schema Model

Prompt-backed `Map` and `Reduce` use a concise record-schema form:

- `schema={"summary": "string"}`
- `schema={"summary": "string", "contains_dog": "boolean"}`
- `schema={"items": {"type": "array", "items": {"type": "string"}}}`

Design rules:

- the output row shape is always an object
- every declared output field is always required
- string values are shorthand for `{"type": ...}`
- field values may also be full JSON-schema fragments when needed

The executor expands this concise field map into a full object-shaped JSON Schema before sending it to an LLM provider.
The parser still accepts the old verbose object wrapper form for compatibility, but it normalizes and renders it back in the concise form.

### Parser

The parser lives in [src/mmds/parser.py](/Users/chanwutk/Documents/mmds/src/mmds/parser.py).

It accepts only:

- an optional module docstring
- `from mmds import ...`
- `from udfs... import ...`
- top-level assignments where the right-hand side is a direct DSL operator call

Parser rules:

- sources must reference a previously assigned variable
- semantic specs must be:
  - a prompt string
  - a prompt-part list
  - an imported UDF name
- UDF import aliasing is rejected
- only absolute imports are allowed
- `Input(...)` must be a string literal ending in `.json` or `.jsonl`
- `Reduce.group_by` must be a string or list/tuple of strings
- `Unnest.keep_empty` must be a literal boolean
- `Map` and `Reduce` prompt specs must include `schema={...}` as an output field map
- `Filter` does not accept `schema=`
- `Reduce` prompt lists may only use `Record[...]` inside `ForEach([...])`

The canonical output variable is the last assignment in the file.

### Renderer

The renderer lives in [src/mmds/render.py](/Users/chanwutk/Documents/mmds/src/mmds/render.py).

It has two jobs:

- render a parsed `QueryProgram` back to normalized Python
- synthesize a normalized `QueryProgram` from a runtime-built `DatasetExpr`

Normalization behavior:

- always emits `from mmds import Input, Map, Filter, Reduce, Unnest, Record, ForEach`
- emits grouped `from udfs... import ...` lines for referenced UDFs
- uses double-quoted string literals
- renders structured prompt lists explicitly
- renders schemas as normalized Python literals
- assigns synthesized names like `source_docs`, `step_1`, `output` when rendering from a bare plan

This is semantic round-tripping, not source-fidelity round-tripping. Comments, whitespace, and original local variable names are not preserved unless they naturally match the normalized output.

### Execution

Local execution lives in the [src/mmds/execution/](/Users/chanwutk/Documents/mmds/src/mmds/execution/) package (entrypoint in [execution/__init__.py](/Users/chanwutk/Documents/mmds/src/mmds/execution/__init__.py); per-operator logic under [execution/ops/](/Users/chanwutk/Documents/mmds/src/mmds/execution/ops/)).

Execution input model:

- each `Input(...)` points directly to a `.json` or `.jsonl` file
- `.json` inputs must contain a top-level list of row objects
- `.jsonl` inputs must contain one JSON object per non-empty line
- each loaded row is copied into a mutable `dict`

Operator semantics:

- `Input(path)`: reads rows from the referenced `.json` or `.jsonl` file
- `Map`: applies prompt/UDF to one row and merges returned fields into that row
- `Filter`: applies prompt/UDF to one row and keeps rows whose result is truthy
- `Reduce`: groups rows by the configured fields, calls the reducer once per group, and merges returned aggregate fields with the group key fields
- `Unnest`: expands one field; lists and tuples explode into multiple rows, scalars pass through unchanged, and missing/empty values produce no row unless `keep_empty=True`
- `Split`: reads the `VideoView` (or string path plus row `duration_sec`) at `video_field`, slices the clip into contiguous `chunk_sec` intervals, and fans out one output row per chunk; each chunk row adds `{output_prefix}_id`, `{output_prefix}_chunk_num`, `{output_prefix}_chunk_start`, `{output_prefix}_chunk_end`, and narrows `video_field` to the chunk's absolute `start`/`end`
- `Detect`: reads the video pointed to by `video_field`; if the value is a `VideoView`-shaped dict with `start`/`end`, wraps the source in a `VideoView`, runs YOLOE detection on the selected frames (every frame by default, or every `frame_stride`-th frame when `frame_stride > 1`), and merges a detection list with absolute source-video `frame_idx` values into `output_field`
- `Join`: executes the left (`source`) and right (`right_source`) inputs and applies the join. As a **self-join optimization**, when `source is right_source` (the exact same plan node feeds both sides, as in the cross-camera vehicle join) the shared subtree is executed **once** and the materialized rows are reused for both sides, so an expensive upstream pipeline (e.g. `Detect` → tracking → appearance embedding) is not evaluated twice. The join helpers only read rows and copy them via `dict(...)` before emitting, so sharing row objects across both sides is safe. Distinct left/right nodes are still executed independently.

Prompt execution flow:

1. Resolve `Record[...]` references against the current row or grouped rows.
2. Expand `ForEach([...])` over grouped rows for `Reduce`.
3. Produce a `ResolvedPrompt`.
4. Delegate to `PromptExecutor.execute(op_type, prompt_spec, resolved_prompt, payload, context)`.

`StaticPromptExecutor` exists for deterministic tests and local development.

Relative path handling:

- when executing a parsed query file, relative `Input(...)` paths resolve from that query file’s directory
- when executing a runtime-built plan, relative `Input(...)` paths resolve from the current working directory

### Media Handling And Gemini Execution

Gemini execution lives in [src/mmds/execution/llm/gemini.py](/Users/chanwutk/Documents/mmds/src/mmds/execution/llm/gemini.py).

Design rule:

- the DSL does not introduce a `Video(...)` wrapper
- media stays in row fields as regular data values
- provider-specific translation happens inside executors

Current media detection rules:

- any resolved prompt value whose `type` is case-insensitively equal to `"video"` or `"videoview"` is treated as video input
- any resolved prompt value whose `type` is case-insensitively equal to `"image"` is treated as image input
- the canonical documented forms are `{"type": "Video", ...}`, `{"type": "VideoView", ...}`, and `{"type": "image", ...}`, but the executor also accepts lowercase/case variants because external data may not preserve that capitalization

Supported video payload forms:

- `{"type": "Video", "uri": "..."}`
- `{"type": "Video", "path": "/local/file.mp4"}`
- `{"type": "Video", "source": "https://..."}` or `{"type": "Video", "source": "file:///local/file.mp4"}`
- `{"type": "Video", "bytes": b"...", "mime_type": "video/mp4"}`
- `{"type": "VideoView", "source": "...", "start": 10, "end": 20}`
- `{"type": "VideoView", "source": "...", "start": 10, "end": 20, "fps": 1.0}`

Optional metadata keys:

- `start_offset`
- `end_offset`
- `fps`

Supported image payload forms (translated to a Gemini image part with no video metadata):

- `{"type": "image", "bytes": b"...", "mime_type": "image/jpeg"}` (inline; `mime_type` required)
- `{"type": "image", "path": "/local/file.jpg"}` (uploaded via the Files API)
- `{"type": "image", "uri": "https://..."}`
- `{"type": "image", "source": "https://..."}` or `{"type": "image", "source": "file:///local/file.jpg"}`

Images are the cheap alternative to sending video: a single cropped still per object
(e.g. a per-track vehicle crop) can be labeled by Gemini for a tiny fraction of the
token cost of the underlying clip.

`VideoView` translation rule:

- `start` and `end` are numeric seconds and are translated to Gemini `VideoMetadata.start_offset` and `VideoMetadata.end_offset` by appending `"s"`
- `fps` is passed directly to Gemini `VideoMetadata.fps`
- a payload may use either `start`/`end` or `start_offset`/`end_offset`, but not both for the same boundary

Gemini executor behavior:

- uploads local files through Gemini’s Files API when needed
- waits for uploaded files to become active
- converts structured prompts into Gemini content parts
- preserves `Video` and `VideoView` fields as video parts instead of stringifying them
- preserves `image` fields as image parts (inline `Blob` for `bytes`, otherwise `FileData`), reusing the shared `_upload_file` / `_resolve_source_media` helpers
- expands concise MMDS output schemas into object-shaped JSON Schema and requests JSON output using Gemini structured output config
- uses a bare boolean schema for `Filter`
- uses the operator-provided output field schema for `Map` and `Reduce`

This design follows Gemini’s official support for video inputs and structured JSON output. Sources used to align the design and implementation:

- [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Gemini SDK downloads](https://ai.google.dev/gemini-api/docs/downloads)

### Detect Operator

The `Detect` operator lives in [src/mmds/execution/ops/detect.py](src/mmds/execution/ops/detect.py) and uses `mmds.utilities.video.open_video` plus `VideoView` to read frames.

Design rules:

- `Detect` carries a `DetectSpec` (not a `PromptSpec`); it does not use a `PromptExecutor`
- the `video_field` value in a row may be a plain string (path or URL) or a `{"source"|"path"|"uri": ...}` dict — the same dict forms used by Gemini video payloads
- when the dict contains `"start"` and `"end"` fields (seconds), the operator creates a `VideoView` that restricts processing to that clip range using `cv2` frame seeking — this avoids iterating the entire video for `VideoView` payloads
- `VideoView` is a general-purpose class in `mmds.utilities.video` that wraps a `Video` and uses `CAP_PROP_POS_FRAMES` to seek directly to the start frame; both `Video` and `VideoView` share the same iterable interface
- detection records store absolute `frame_idx` values from the underlying source video, even when the operator is iterating a `VideoView`
- `open_video` handles local paths, `file://` URLs, and `http/https` downloads with a disk cache under `~/.cache/mmds/videos/`; platform URLs (e.g. YouTube) are downloaded via `yt-dlp`
- the runtime probes whether CUDA is genuinely usable for YOLOE inference and falls back to CPU when the installed PyTorch/CUDA stack cannot execute the required kernels
- YOLOE model instances are cached by name in a module-level dict; a `threading.Lock` serialises `set_classes` + `predict` calls so the operator is safe to run from a thread pool
- `frame_stride` (default `1`) controls how densely frames are sampled for inference: only frames whose index (relative to the start of the iterated `Video`/`VideoView`) is a multiple of `frame_stride` are sent through the model; skipped frames are still decoded (iteration is unchanged) but contribute no detections, so this trades temporal density for lower model inference cost — appropriate when detections are expected to stay consistent across a few consecutive frames
- `conf` (default `None`) and `imgsz` (default `None`) are optional per-`predict` overrides forwarded to the underlying model. When a value is `None` it is **not** passed, so the model's own default applies (Ultralytics: `conf≈0.25`, `imgsz=640`); only explicitly-set values are forwarded. Lower `conf` (e.g. `0.1`) recovers low-confidence small objects the default threshold discards; raise `imgsz` (e.g. `1280`) so small objects survive downscaling — both matter for high-mounted traffic footage where vehicles are small in the frame. `conf` is validated to `[0.0, 1.0]`; `imgsz` to a positive `int`
- **class-name coupling caveat**: `classes` are open-vocabulary prompt strings, and detection records carry whatever class name the model returns. Downstream UDFs that filter by a fixed vocabulary (e.g. `udfs/detection_ops.py` `_VEHICLE_CLASSES = {"sedan", "suv", "truck"}`) silently drop records whose `type` is not in that set — so `Detect` `classes` must match the downstream vocabulary exactly (use `"sedan"`, not `"car"`, for the vehicle UDFs)
- the `output_field` (default `"detections"`) is merged into the row, containing a list of per-class detection records:

```json
[
  {"type": "dog", "bboxes": [{"frame_idx": 0, "bbox": [x1, y1, x2, y2], "confidence": 0.9}]},
  ...
]
```

- the OpenCV/NumPy stack (and, at run time, `torch`/`ultralytics`) is imported **lazily**: `mmds.execution` imports `.ops.detect` only when a `detect` node actually executes, and `mmds.VideoView` is a lazy export via module `__getattr__`. This keeps `import mmds` and the prompt/UDF execution paths usable without the heavy CV/ML dependencies installed.
- `Detect` is **not** parsed from or rendered back to DSL text (it is for internal/programmatic use only)

### Join predicates (library)

Cross-camera relational filters live in [src/mmds/join/predicates.py](src/mmds/join/predicates.py).
The UDF wrapper [udfs/join_ops.py](udfs/join_ops.py) exposes `same_vehicle(left, right)` and
`vehicle_match_score(left, right)` for the flat self-join, plus `tracks_temporally_overlap(left, right)`
and `temporal_iou_score(left, right)` for the two-branch cross-camera hybrid join.

`temporal_overlap` returns `True` when the two tracks' `[start_time, end_time]` intervals overlap
(with a small `_MAX_SYNC_OVERLAP_SEC` tolerance for synchronized feeds); `temporal_iou` returns the
intersection-over-union of those intervals in seconds and is used as the greedy one-to-one match score.

Text↔clip retrieval helpers live in [src/mmds/join/text_retrieval.py](src/mmds/join/text_retrieval.py).
The UDF wrapper [udfs/retrieval_ops.py](udfs/retrieval_ops.py) exposes
`tag_gallery_video`, `format_gt_caption_clip`, and `group_gt_clips_by_video`.
The UCA example treats `data/uca/annotation_excerpt.json` as ground truth: Join
captions to gallery videos on `video_id`, Map each caption to a clip, then Reduce
by `video_id` into annotation_excerpt-style rows (`duration` / `timestamps` /
`sentences`). Gallery/caption inputs are assumed scoped to Abuse001/002/003.

Soft-reference trajectory evaluation lives in
[src/mmds/join/eval_trajectories.py](src/mmds/join/eval_trajectories.py).
It compares UDF join trajectories (e.g. `examples/join_cross_camera_vehicle.py`)
to an LLM baseline (e.g. `examples/semantic_join_cross_camera_vehicle.py`) via
greedy one-to-one matching on per-camera timeline IoU plus soft attribute
agreement (`class` / `color` / `subtype`, with class aliases such as `car`→`sedan`
and soft color families `blue≈gray≈silver≈black` and `red≈brown`). If colors do
not soft-match, the soft attribute score is forced to `0.0` (exact agreement is
unchanged).
Per shared camera, interval IoU is gated: the camera contributes `0.0` unless
`|Δentered| ≤ max_endpoint_delta_sec` **or** `|Δexited| ≤ max_endpoint_delta_sec`
(default `1.0` seconds; pass `None` / CLI `--max-endpoint-delta -1` to disable).
Missing / extra cameras still contribute `0.0`. The semantic output is a **soft
reference**, not oracle ground truth. The driver
[examples/compare_cross_camera_joins.py](examples/compare_cross_camera_joins.py)
reports precision / recall / F1 and cost proxies (wall time + prompt call count +
Gemini token counts). `GeminiPromptExecutor` accumulates
`response.usage_metadata` across calls (`prompt_tokens` / `candidates_tokens` /
`total_tokens`, exposed via `usage_snapshot()` and cleared by `reset_usage()`);
`CostReport` carries those token counts. Dollar pricing is still not wired.

For evaluation against **hand-labeled ground truth** (rather than the LLM soft
reference), [examples/eval_cross_camera_join.py](examples/eval_cross_camera_join.py)
runs the UDF join and scores its trajectories against
`data/i24v_traffic_highway2_highway3_5s_ground_truth.json` with the same
`evaluate_trajectories` (including the endpoint gate above; override with
`--max-endpoint-delta`), printing precision / recall / F1, attribute agreement,
per-match true-positive / false-positive / false-negative detail (including
a note when a missed GT vehicle is single-camera and thus impossible for a
cross-camera join to recover), and the run's wall time / token usage. With
`--compare-semantic` (or `--semantic-json` to load a saved dump) it also scores
the semantic (Gemini) join against the **same** ground truth and prints a
side-by-side table of accuracy and cost (wall time, prompt calls, tokens). Use it
to record a baseline before join-quality changes and to re-measure after.

`same_vehicle` returns `True` only when **all** of the following pass:

- `different_cameras` — `left.camera_id != right.camera_id`
- `canonical_corridor_pair` — when both ids end in `highway<N>`, requires `N_left < N_right`
  so self-joins emit each unordered track pair once (upstream camera on the left), eliminates duplication of entries
- `travel_time_compatible` — track intervals overlap, or upstream `end_time` precedes
  downstream `start_time` within `120 s` (with `2 s` overlap tolerance)
- `direction_compatible` — upstream `exit_direction` and downstream `entry_direction` align
  within `120°` on the compass; both tracks must be moving
- `speed_compatible` — `avg_speed` values within `50%` relative difference

### Join operator

`Join(left, right, predicate?, on=..., one_to_one=..., score=..., left_key=..., right_key=...)`
pairs rows from two sources. When `on=` names equi-join keys, the executor uses a hash join
implemented in [src/mmds/join/hash_join.py](src/mmds/join/hash_join.py): it builds a hash index
on the right input and only compares rows inside the same bucket.

For cross-camera vehicle matching, the intended hash key is:

```python
hash_key = (vehicle_class, color, subtype)
```

(`VEHICLE_APPEARANCE_KEYS` in `hash_join.py`.)

When `one_to_one=True`, candidate pairs inside each bucket are filtered by the optional
`predicate`, scored with `score(left, right)`, sorted by descending score, and greedily matched
so each `left_key` / `right_key` identity appears at most once. Output rows are
`{"left": ..., "right": ..., "match_score": ...}`.

`Join` and `Split` are parsed from and rendered back to DSL text. `Detect` remains programmatic-only.

#### Hybrid cross-camera vehicle join

[examples/hybrid_join_cross_camera_vehicle.py](examples/hybrid_join_cross_camera_vehicle.py) is a
cost-optimized middle ground between the local UDF join and the full-video semantic join. It keeps
detection, tracking, and joining local, and spends only a tiny amount of Gemini: one cropped still per
track (not video) to label constrained-enum appearance attributes.

- The two feeds are split into distinct branches with `Filter` (`udfs/camera_ops.py`
  `keep_highway2` / `keep_highway3`), so this is a true **cross-camera** join rather than a flat
  self-join — the `different_cameras`, `canonical_corridor_pair`, `direction_compatible`, and
  `speed_compatible` checks are all dropped.
- NMS is **class-agnostic**: `udfs/detection_ops.py` `nms_vehicle_detections` pools all vehicle
  boxes in a frame across classes before suppression, so when YOLOE labels the same physical
  vehicle with two overlapping boxes (e.g. `sedan` and `suv`), only the highest-confidence one
  survives (its class is retained) instead of both — which would otherwise double-count the vehicle.
- Tracking is **class-agnostic**: `strongsort_track_frame_detections` associates boxes across
  frames regardless of the per-frame YOLOE class (which flickers for one vehicle), then resolves
  each track's class/color/subtype by majority vote.
- Association is **motion-aware**: `_assign_track_ids` forecasts each active track forward with a
  constant-velocity model (smoothed/EMA center velocity) and links on the **better of the
  predicted-box and last-box IoU**. The prediction only *adds* reach — bridging moderate motion and
  short detection gaps a pure last-box-IoU tracker would drop — and can never lose a last-box match,
  so it does not increase fragmentation. On the I24V 5s clips it modestly reduces the track count
  (≈54→52). Two richer heuristics were evaluated against the ground-truth eval and **rejected**
  because both *increased* fragmentation: a center-distance fallback gate (greedy mis-assignment
  churn) and a class-consistency gate (split a vehicle whenever its label flickered to `truck`).
  A robust further reduction needs global per-frame (Hungarian) assignment + a Kalman filter — i.e. a
  real StrongSORT backend — not more heuristics on the greedy matcher.
- **Vehicle color** is predicted per box by `udfs/detection_ops.py` `predict_vehicle_attributes`.
  It prefers a learned **MobileNetV3-small** classifier (`udfs/vehicle_color_model.py`) trained on
  the 8-color *Vehicle Color Recognition* dataset (Chen et al.) via
  `scripts/train_vehicle_color_model.py`; its labels are remapped onto the project color vocabulary
  (`cyan`→`blue`, others identity). The trained checkpoint is an **uncommitted artifact** located via
  `MMDS_VEHICLE_COLOR_WEIGHTS` (default `models/vehicle_color_mobilenet.pt`); `torch`/`torchvision`
  are imported lazily and the model is cached behind a lock (thread-safe under the executor pool).
  When the checkpoint is absent or inference is unavailable, `predict_color_from_crop` returns `None`
  and the code **falls back** to an HSV heuristic (`classify_vehicle_color` over
  `_dominant_rgb_from_crop`) — so the pipeline runs with or without trained weights. The heuristic
  samples the **central body region** of the box (avoiding background/road at the edges) and takes the
  per-channel **median** (robust to glare/shadow), then classifies in HSV: low-saturation crops are
  white/silver/gray/black by brightness, saturated crops are named by hue. This replaced an earlier
  whole-crop **mean**-RGB nearest-color heuristic that biased everything toward neutral grays (averaging
  the whole box desaturates the color). Output names come from `VEHICLE_COLOR_VOCAB`. The learned model
  remains the higher-accuracy path.
- **Fragment filtering**: the greedy IoU tracker (a stub for real StrongSORT) has no motion model,
  so fast-moving highway vehicles fragment into many short tracks. `strongsort_track_frame_detections`
  drops these directly — it keeps only tracks with `>= min_track_frames` observed detections and mean
  `confidence >= min_track_confidence` (defaults `_MIN_TRACK_FRAMES=5`, `_MIN_TRACK_CONFIDENCE=0.15`;
  both overridable per call). Length is the primary signal because the detector runs at a low `conf`
  floor, so a real track's mean confidence is legitimately modest. The same thresholds back the
  `is_substantial_track` `Filter` predicate (both call the shared `_track_is_substantial` helper), used
  on the promoted-row path before the paid crop + Gemini steps.
- `_summarize_track` records `rep_frame_id` and `rep_bbox` (the track's highest-confidence
  detection); `udfs/detection_ops.py` `crop_from_track_row` slices that box out of that frame (shared
  by the crop and re-ID ops). `udfs/crop_ops.py` `attach_track_crop` JPEG-encodes it into an
  `{"type": "image", "bytes": ...}` prompt descriptor.
- **Appearance re-ID** (`udfs/reid_ops.py`, `udfs/vehicle_reid_model.py`): `attach_track_embedding`
  embeds each track's representative crop into an appearance vector, and the cross-camera `Join` in
  [examples/join_cross_camera_vehicle.py](examples/join_cross_camera_vehicle.py) **drops the exact
  `(class, color, subtype)` hash key** and scores candidate pairs by embedding **cosine similarity**
  (`appearance_match_score`), still pruned by the `same_vehicle` predicate and matched `one_to_one`.
  Exact categorical keys silently exclude true matches because cross-camera class/color labels
  disagree; appearance similarity fixed that — on the 9-vehicle ground truth this lifted recall
  0.44→1.0 and F1 0.57→0.86. The embedding is a frozen ImageNet ResNet50 penultimate feature (real
  learned appearance, no training) with a coarse RGB-histogram fallback when `torch`/weights are
  unavailable; `appearance_match_score` falls back to mean track confidence when an embedding is
  absent, so the join always runs. `torch`/`torchvision` load lazily; the backbone is cached behind
  a lock (thread-safe under the executor pool).
- A prompt-backed `Map` sends the crop to Gemini with a constrained enum schema for
  `class` / `color` / `subtype`, so both cameras map to identical hash-join keys.
- `udfs/attribute_ops.py` `project_labeled_track` promotes Gemini's `class` to `vehicle_class` and
  keeps the identity/timing fields; `wrap_vehicle_record` wraps each exported trajectory as
  `{"vehicles": {...}}`.
- The join uses `on=(vehicle_class, color, subtype)` hash keys, `tracks_temporally_overlap` as the
  predicate, and `temporal_iou_score` as the one-to-one match score.

### UDF Contract

UDF discovery lives in [src/mmds/udf_catalog.py](/Users/chanwutk/Documents/mmds/src/mmds/udf_catalog.py).

Current rules:

- executable UDFs must live under `udfs.*`
- query files import UDFs directly and pass the function object into DSL calls
- lambdas and nested functions are rejected
- `.py` files are treated as implemented UDFs
- `.pyi` files are treated as declared-only UDFs for future synthesis

`discover_udfs()` returns a `UdfCatalog` of `UdfEntry` records that capture:

- module + function name
- whether an implementation exists
- source path(s)
- stub signature and docstring when available

The current system does not generate `.py` from `.pyi`; it only records the contract.

### Optimizers

#### Rule Optimizer

The rule optimizer lives in [src/mmds/optimizers/rewriter/rule.py](/Users/chanwutk/Documents/mmds/src/mmds/optimizers/rewriter/rule.py).

Current behavior is intentionally conservative:

- recursively rebuild the tree
- structurally deduplicate equivalent nodes through memoization

It does not yet reorder operators, fold operators, infer safety, or reason about prompt/UDF semantics.

#### LLM Optimizer

The LLM rewrite scaffold lives in [src/mmds/optimizers/rewriter/agent.py](/Users/chanwutk/Documents/mmds/src/mmds/optimizers/rewriter/agent.py).

Flow:

1. Parse the original query.
2. Build a constrained rewrite prompt.
3. Ask an `LLMClient` for rewritten code.
4. Extract Python from a fenced or raw response.
5. Parse the rewritten code.
6. Reject rewrites that change `Input(...)` file paths.
7. Return normalized rendered Python.

The LLM optimizer is currently a controlled interface, not a production optimizer. It is designed to make later provider integration safe by validating every rewrite against the same parser used elsewhere.

## Invariants

The following invariants are part of the current design and should not change silently:

- the supported query language is a strict Python subset
- prompts and UDFs are the only valid semantic specs
- prompt specs are structured data, not opaque runtime callables
- UDFs must come from `udfs.*`
- operator trees are immutable
- the last assignment is the output unless a future explicit sink is added
- input roots are direct file paths, not catalog identifiers
- rendered queries are normalized, not source-exact
- prompt-backed execution always requires an injected executor
- `.pyi` discovery does not imply executability
- provider-specific media handling belongs in executors, not DSL syntax
- `Reduce` row access must go through `ForEach([...])`
- importing `mmds` must not require the optional computer-vision stack (OpenCV/NumPy/`torch`/`ultralytics`); `Detect` and `VideoView` load those dependencies lazily on use

## Validation and Tests

Tests live under [tests/](/Users/chanwutk/Documents/mmds/tests/).

The current suite covers:

- parse/render/parse equivalence for structured prompts
- file-backed execution for `.json` and `.jsonl`
- rendering from runtime-built plans
- execution for UDF-backed and prompt-backed queries
- `Record[...]` resolution and `ForEach([...])` expansion
- `Unnest` behavior on scalar, empty, and missing values
- `Split` behavior on `VideoView` clips, tail chunks, and missing bounds
- `Detect` behavior, including `VideoView` clip slicing and absolute-frame detection indices
- video utility behavior for direct downloads, platform downloads via `yt-dlp`, and `VideoView` iteration
- parser validation for unsupported Python and invalid prompt forms
- optimizer result preservation and LLM rewrite validation
- Gemini prompt compilation for video URI and uploaded local file inputs
- UDF catalog discovery for `.py` and `.pyi`

Primary verification command:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m unittest discover -s tests -t .
```

## Near-Term Extension Points

Likely next areas of change:

- more operators beyond the current unary set
- explicit group-key prompt helpers for `Reduce`
- richer multimodal field translation beyond video
- more aggressive rule rewrites once semantic safety rules exist
- `.pyi`-driven UDF synthesis
- provider-backed prompt execution and LLM optimization beyond Gemini

Any of those changes should update this document alongside the code.
