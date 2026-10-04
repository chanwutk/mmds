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

- operators: `Input`, `Map`, `Filter`, `Reduce`, `Unnest`, `Join`, `Detect`,
  `Window`, `Coalesce`, `VideoMap`, and `VideoMapEach`
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
- `Window` and `Coalesce` operators for constructing padded
  source-time video views and merging overlapping candidate intervals
- logical `VideoMap` and `VideoMapEach` operators that lower candidate intervals
  into non-materialized video-view execution plans

The current implementation intentionally does not support:

- inline lambdas
- nested functions or callables outside `udfs.*`
- loops, conditionals, comprehensions, classes, or arbitrary Python control flow in query files
- sorts, projections, or cost-based optimization
- automatic `.py` implementation synthesis from `.pyi`
- nested `ForEach(...)`
- provider-specific media syntax in the DSL
- data catalogs or named dataset registries

## Architecture

### Public DSL

The main entrypoints are exported from [src/mmds/__init__.py](/Users/chanwutk/Documents/mmds/src/mmds/__init__.py):

- `Input(path)`
- `Map(data, spec, *, schema=None, replace=False, name=None)`
- `Filter(data, spec, *, name=None)`
- `Reduce(data, group_by, reducer, *, schema=None, name=None)`
- `Unnest(data, field, *, keep_empty=False, name=None)`
- `Join(left, right, predicate=None, *, on=None, one_to_one=False, score=None, min_score=None, left_key=None, right_key=None, name=None)`
- `Detect(data, video_field, classes, *, model="yoloe-11s-seg.pt", output_field="detections", frame_stride=1, conf=None, imgsz=None, stop_after_n=None, name=None)`
- `Window(data, video_field, candidate_field, output_field, padding_time, name=None)`
- `Coalesce(data, group_by, field, *, name=None)`
- `VideoMap(data, spec, *, video_field, views_field, group_by, schema, padding_time=0, clip_field="clip", name=None)`
- `VideoMapEach(data, spec, *, video_field, views_field, group_by, schema=None, padding_time=0, clip_field="clip", name=None)`
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
- `FieldPredicateSpec` stores a top-level field name for a non-LLM `Filter`
  that keeps rows where that field is truthy (`Filter(..., Record["flag"])`).
- `JoinSpec` stores equi-join keys, an optional binary predicate UDF, and
  optional score, threshold, and side-identity keys for greedy one-to-one
  matching.
- `DetectSpec` stores the parameters for a `Detect` node: `video_field`,
  `classes`, `model`, `output_field`, `frame_stride`, `conf`, `imgsz`, and
  `stop_after_n`.
- `WindowSpec` stores the parameters for a `Window` node: `video_field`,
  `candidate_field`, `output_field`, and `padding_time`.
- `VideoMapSpec` stores the candidate-view field, source-video field, grouping
  fields, semantic map spec, padding, and clip field.
- `Assignment` and `QueryProgram` represent a parsed query file.
- `MMDSValidationError` is the shared validation failure type.

`DatasetExpr` supports unary and binary nodes:

- `Input` has no source.
- `Map`, `Filter`, `Reduce`, `Unnest`, `Detect`, `Window`, `Coalesce`,
  `VideoMap`, and `VideoMapEach` each have one `source`.
- `Join` has a left `source` and `right_source`. Its `children()` and
  `walk_postorder()` methods traverse both sources and skip repeated visits
  by object identity, so equal-but-distinct branches stay distinct.

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
  - an imported UDF name, or a call of that name with positional string
    literal arguments (`map_detection_presence("bear_present")`). Those
    arguments are stored on `UdfSpec.args` and passed to the callable after
    the row. A bare name is the no-argument form; `udf()` is rejected so the
    rendered text stays stable
  - for `Filter` only: a bare top-level `Record["field"]` field predicate
- UDF import aliasing is rejected
- only absolute imports are allowed
- `Input(...)` must be a string literal ending in `.json` or `.jsonl`
- `Reduce.group_by` must be a string or list/tuple of strings
- `Unnest.keep_empty` must be a literal boolean
- `Map.replace` and `Join.one_to_one` must be literal booleans
- `Map` and `Reduce` prompt specs must include `schema={...}` as an output field map
- `Filter` does not accept `schema=`
- `Filter` field predicates must be a single top-level `Record["field"]`
- `Reduce` prompt lists may only use `Record[...]` inside `ForEach([...])`
- `Join` sources must reference prior assignments; predicates and score
  functions must be imported `udfs.*` names; keys and identity fields must be
  string literals or literal lists/tuples of strings
- operators and `Record`/`ForEach` helpers must be imported from `mmds`
  before they are used; unused imports remain allowed
- `Detect` requires literal `video_field` and class-list arguments and accepts
  literal `model`, `output_field`, `frame_stride`, `conf`, `imgsz`,
  `stop_after_n`, and `name`
  keyword arguments; parsing constructs a validated `DetectSpec`
- `Window` requires literal `video_field`, `candidate_field`, `output_field`,
  and `padding_time` arguments and accepts a literal `name`
- `Coalesce` requires a literal string or string-list/tuple `group_by`, a
  non-empty literal interval field, and an optional literal `name`
- `VideoMap` and `VideoMapEach` require literal `video_field`, `views_field`,
  and `group_by` keyword arguments
- prompt-backed video maps require `schema=...`; joint `VideoMap` is
  prompt-only, while `VideoMapEach` may also use an imported UDF

The set of names accepted in source (and imported by the renderer) comes
from `OPERATOR_DEFINITIONS` in `src/mmds/operator_catalog.py`, plus `Record`
and `ForEach`. `VideoMap` and `VideoMapEach` are the logical interface to the
`Window` → `Coalesce` → `Map`/`Reduce` pipeline; the physical `Detect`,
`Window`, and `Coalesce` operators are also source-visible.

The canonical output variable is the last assignment in the file.

### Renderer

The renderer lives in [src/mmds/render.py](/Users/chanwutk/Documents/mmds/src/mmds/render.py).

It has two jobs:

- render a parsed `QueryProgram` back to normalized Python
- synthesize a normalized `QueryProgram` from a runtime-built `DatasetExpr`

Normalization behavior:

- emits a `from mmds import ...` line with only the operators (from the
  operator catalog) and prompt helpers the plan uses
- emits grouped `from udfs... import ...` lines for referenced UDFs
- uses double-quoted string literals
- renders structured prompt lists explicitly
- renders schemas as normalized Python literals
- renders `Detect`, `Window`, and `Coalesce` with literal validated arguments,
  including non-default extended `Detect` options
- assigns synthesized names like `source_docs`, `step_1`, `output` when rendering from a bare plan

This is semantic round-tripping, not source-fidelity round-tripping. Comments, whitespace, and original local variable names are not preserved unless they naturally match the normalized output.

`Join`, `Detect`, `Window`, `Coalesce`, `VideoMap`, `VideoMapEach`, and
`Map(replace=True)` are rendered and parse back to equivalent plans.

### Execution

Local execution lives in the [src/mmds/execution/](/Users/chanwutk/Documents/mmds/src/mmds/execution/) package (entrypoint in [execution/__init__.py](/Users/chanwutk/Documents/mmds/src/mmds/execution/__init__.py); per-operator logic under [execution/ops/](/Users/chanwutk/Documents/mmds/src/mmds/execution/ops/)).

Execution input model:

- each `Input(...)` points directly to a `.json` or `.jsonl` file
- `.json` inputs must contain a top-level list of row objects
- `.jsonl` inputs must contain one JSON object per non-empty line
- each loaded row is copied into a mutable `dict`

Operator semantics:

- `Input(path)`: reads rows from the referenced `.json` or `.jsonl` file
- `Map`: applies prompt/UDF to one row and merges returned fields into that
  row, or returns only the produced mapping when `replace=True`
- `Filter`: applies prompt/UDF/field-predicate to one row and keeps rows whose
  result is truthy; `Filter(..., Record["field"])` is a non-LLM predicate
- `Reduce`: groups rows by the configured fields, calls the reducer once per group, and merges returned aggregate fields with the group key fields
- `Unnest`: expands one field; lists and tuples explode into multiple rows, scalars pass through unchanged, and missing/empty values produce no row unless `keep_empty=True`
- `Join`: emits `{"left": ..., "right": ...}` pairs. Equi-join keys use a
  right-side hash index; predicate-only joins use quadratic pair matching.
  One-to-one joins score candidates, apply an optional minimum score, and
  greedily retain the highest-scoring pairs with unique left/right identity
  keys. A self-join whose sources are the same plan object executes and
  materializes that shared upstream subtree exactly once.
- `Detect`: reads the video pointed to by `video_field`; if the value is a `VideoView`-shaped dict with `start`/`end`, wraps the source in a `VideoView`, runs YOLOE detection on sampled frames according to `frame_stride`, forwards optional `conf` and `imgsz` inference parameters, and merges a detection list with absolute source-video `frame_idx` values into `output_field`. `stop_after_n` ends the scan once that many distinct tracks of the requested classes exist; overlapping boxes of the same class within 30 source frames stay one track, and `None` scans every sampled frame
- `Window`: reads one `{start, end}` candidate interval per row, applies
  symmetric padding, clamps the start to zero, and stores a non-materialized
  `VideoView` descriptor in `output_field` while preserving the input row
- `Coalesce`: groups rows by `group_by`, sorts the mappings in `field` by
  `start`, merges overlapping or touching intervals, and emits one row per
  merged interval containing only the grouping fields and `field`. Only
  intervals whose non-time fields are equal (compared by value, e.g. the same
  video `path`) merge, so views of different videos in one group stay separate
- `VideoMap`: logically applies one prompt to all selected video views in each
  group; execution lowers it to `Unnest -> Window -> Coalesce -> Reduce`
- `VideoMapEach`: logically applies the same map independently to every
  selected view; execution lowers it to `Unnest -> Window -> Coalesce -> Map`

Video-map lowering never materializes clip files. `Window` creates
`VideoView` dictionaries that retain the original video source and absolute
`start`/`end` seconds. `Coalesce` removes redundant overlap, and every
resulting view is processed. Any future sampling or limiting policy should be
an explicit approximate rewrite rather than part of the logical operators.

`group_by` is also the boundary for row fields preserved through coalescing.
For prompt-backed video maps, every referenced non-video field must therefore
appear in `group_by`; construction fails early when it does not. The video,
candidate, and clip fields cannot be grouping keys. The prompt must reference the complete
`Record[video_field]` value directly so lowering can replace it with each
generated `VideoView`. A UDF-backed `VideoMapEach` receives only the grouping
fields and generated `clip_field`, matching the row shape emitted by
`Coalesce`.

Prompt execution flow:

1. Resolve `Record[...]` references against the current row or grouped rows.
2. Expand `ForEach([...])` over grouped rows for `Reduce`.
3. Produce a `ResolvedPrompt`.
4. Delegate to `PromptExecutor.execute(op_type, prompt_spec, resolved_prompt, payload, context)`.

`StaticPromptExecutor` exists for deterministic tests and local development.

Windowed video queries use two deterministic UDFs from
`udfs.temporal_ops`. `rebase_clip_events` converts `clip_events` from offsets
relative to the beginning of `clip` into source-time `events` by adding
`clip.start`. `collect_sorted_events` is used by
`Reduce(..., "source_id", ...)` to flatten and sort source-time event
collections. It does not merge or deduplicate events; interval merging remains
the responsibility of `Coalesce`. `reconcile_events` remains as a compatibility
wrapper for the former name. Keeping `clip_events` and `events` as separate
fields makes their coordinate systems explicit.

Relative path handling:

- when executing a parsed query file, relative `Input(...)` paths resolve from that query file’s directory
- when executing a runtime-built plan, relative `Input(...)` paths resolve from the current working directory

### Media Handling And Gemini Execution

Gemini execution lives in [src/mmds/execution/llm/gemini.py](/Users/chanwutk/Documents/mmds/src/mmds/execution/llm/gemini.py).

Design rule:

- the DSL does not introduce a `Video(...)` wrapper
- media stays in row fields as regular data values
- provider-specific translation happens inside executors

Current video detection rule:

- any resolved prompt value whose `type` is case-insensitively equal to `"video"` or `"videoview"` is treated as video input
- the canonical documented forms are `{"type": "Video", ...}` and `{"type": "VideoView", ...}`, but the executor also accepts lowercase variants because external data may not preserve that capitalization

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

`VideoView` translation rule:

- `start` and `end` are numeric seconds and are translated to Gemini `VideoMetadata.start_offset` and `VideoMetadata.end_offset` by appending `"s"`
- `fps` is passed directly to Gemini `VideoMetadata.fps`
- a payload may use either `start`/`end` or `start_offset`/`end_offset`, but not both for the same boundary

Gemini executor behavior:

- uploads local files through Gemini’s Files API when needed
- waits for uploaded files to become active
- converts structured prompts into Gemini content parts
- preserves `Video` and `VideoView` fields as video parts instead of stringifying them
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
- the `output_field` (default `"detections"`) is merged into the row, containing a list of per-class detection records:

```json
[
  {"type": "dog", "bboxes": [{"frame_idx": 0, "track_id": 1, "bbox": [x1, y1, x2, y2], "confidence": 0.9}]},
  ...
]
```

- the OpenCV/NumPy stack (and, at run time, `torch`/`ultralytics`) is imported **lazily**: `mmds.execution` imports `.ops.detect` only when a `detect` node actually executes, and `mmds.VideoView` is a lazy export via module `__getattr__`. This keeps `import mmds` and the prompt/UDF execution paths usable without the heavy CV/ML dependencies installed.
- `Detect` is parsed from and rendered back to DSL text (including `frame_stride`, `conf`, `imgsz`, and `stop_after_n`) so rewrite directives can insert it while preserving render/parse round trips
- `stop_after_n` counts tracks, not raw boxes. A box joins the same-class track with the highest intersection-over-union of at least 0.3 whose last box is at most 30 source frames earlier and has not already been matched in this frame. Boxes that match none of those tracks open a new track, and overlapping unmatched boxes in the same frame share that new track. Each box stores that track's integer `track_id`. The scan returns the boxes seen so far and does not read later frames. Fewer than `n` tracks still reads the video to the end
- Detect writes `_mmds_video_fps` when the opened video reports a positive fps so
  detection-window rewrites can convert `frame_idx` values to source time without
  re-opening the file in the interval UDF
- `udfs.detection_ops.detections_to_candidate_views` maps Detect boxes to
  `_mmds_candidate_views` intervals `[frame_idx/fps, (frame_idx+1)/fps)`
- `udfs.detection_ops.keep_rows_with_detections` keeps a row when any class
  entry has a non-empty `bboxes` list
- `udfs.detection_ops.map_detection_presence(row, flag_field)` returns
  `{flag_field: true}` when that keep check passes, otherwise
  `{flag_field: false}`. Rewrite text binds the field as
  `map_detection_presence("bear_present")`. `map_bear_present(row)` is the
  same check with `flag_field` fixed to `"bear_present"`, so an example file
  imported as Python can pass the function itself
- `udfs.detection_ops.keep_rows_with_at_least_n_tracks(row, min_tracks)` and
  `map_at_least_n_tracks(row, flag_field, min_tracks)` use distinct `track_id`
  values. `min_tracks` is a decimal integer string. `map_at_least_five_bears`
  and `keep_at_least_five_bear_tracks` fix that threshold at five for the
  importable example

### Vehicle Case-Study Helpers

The cross-camera vehicle case study adds reusable helpers plus UDF modules
built on `Detect` output. None of them are DSL operators; queries use them as
ordinary `Map`, `Filter`, and `Join` UDFs.

Shared utilities:

- [src/mmds/utilities/media.py](src/mmds/utilities/media.py):
  `resolve_video_source` extracts a path/URL from a video field value (a string,
  or a dict with a `source`, `path`, or `uri` string key; otherwise
  `MMDSValidationError`). `resolve_local_media_path` resolves a local path and
  requires it to stay under a given media root.
- `read_frames_at_indices(video_path, frame_indices)` in
  [src/mmds/utilities/video.py](src/mmds/utilities/video.py) decodes the
  requested frames with a single `cv2.VideoCapture`, in ascending order. It
  ignores duplicate, negative, and non-integer indices and omits frames that
  fail to decode, so the result may be missing requested frames (or be empty if
  the file cannot be opened).
- [src/mmds/case_studies/](src/mmds/case_studies/) holds the case-study logic,
  lazily exported from `mmds.case_studies`. `predicates` has the cross-camera
  pair checks (`different_cameras`, `canonical_corridor_pair`,
  `travel_time_compatible`, `direction_compatible`, `speed_compatible`, combined
  as `same_vehicle`), `temporal_overlap`/`temporal_iou`, and the fallback
  `vehicle_match_score`. Camera order comes from a `highway<N>` suffix on
  `camera_id`. `trajectory` converts a Join match into the
  `vehicle_id`/`attributes`/`timeline`/`match_score` trajectory record.

UDF modules under [udfs/](udfs/) (`join_ops` and `trajectory_ops` are thin
wrappers over `mmds.case_studies`):

- `detection_ops.build_vehicle_frame_detections` flattens vehicle `Detect`
  output into per-frame records (`frame_id`, `camera_id`, `bbox`, `confidence`,
  `vehicle_class`, `color`, `subtype`). Color comes from the learned classifier
  in `vehicle_color_model` when weights are available and falls back to an HSV
  heuristic otherwise; subtype comes from box geometry. Frames for all detected
  boxes are decoded up front, so memory grows with the number of detected
  frames.
- `tracking_ops.strongsort_track_frame_detections` runs a greedy IoU tracker
  (with velocity prediction and gap bridging) over those records and emits
  `track_summaries` with ISO-8601 `start_time`/`end_time`, attributes, average
  image-plane speed, and entry/exit directions. Timestamps are
  `recorded_at + frame_id / fps` when the row has `recorded_at`, otherwise
  epoch + `frame_id / fps`; `frame_id` is the absolute source-video frame index
  that `Detect` reports, including for `VideoView` rows.
  `is_substantial_track` filters short or low-confidence tracks.
- `reid_ops.attach_track_summary_embeddings` embeds one representative crop per
  track through `vehicle_reid_model` (a frozen torchvision ResNet50, falling back
  to an RGB histogram when torch or its weights are unavailable).
  `appearance_match_score` is the Join `score=` UDF: cosine similarity of the two
  embeddings, or `vehicle_match_score` when either side has none.
- `vehicle_color_model` loads MobileNetV3-small color weights from
  `MMDS_VEHICLE_COLOR_WEIGHTS` (or the default `models/` path); train them with
  [scripts/train_vehicle_color_model.py](scripts/train_vehicle_color_model.py).
  Missing weights or ML dependencies make it return `None` rather than raise.

These helpers degrade instead of failing: an unreadable video yields no crops,
so colors fall back to the heuristic and embeddings are empty. Callers that need
a hard failure must check inputs themselves.

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

### Cross-Camera Vehicle Join

The I24V case study has a semantic baseline and a UDF rewrite that share the
same default manifest and final row schema
(`vehicle_id`, `attributes`, `timeline`, `match_score`):

- [`examples/semantic_join_cross_camera_vehicle.py`](/Users/chanwutk/Documents/mmds/examples/semantic_join_cross_camera_vehicle.py)
  is the Gemini Reduce–Unnest baseline. It stitches both camera feeds in one
  prompt, unnests `vehicles`, then promotes each item to the top-level
  trajectory schema.
- [`examples/join_cross_camera_vehicle.py`](/Users/chanwutk/Documents/mmds/examples/join_cross_camera_vehicle.py)
  is the rewritten Detect–Track–Join plan. It detects vehicles, normalizes
  detections, tracks and embeds per-camera trajectories, unnests one row per
  track, and self-joins those rows before mapping each match to the trajectory
  output schema.

The self-join uses `same_vehicle` to enforce camera ordering and source-time
compatibility, `appearance_match_score` to score candidate associations, and
`(camera_id, track_id)` as each side's identity. It intentionally has no exact
hash key: detector-derived class and color labels can disagree between cameras,
so exact attribute blocking would silently remove valid candidates. The
tradeoff is quadratic candidate generation and up to quadratic candidate
memory before greedy score ordering. This is acceptable for the documented
two-camera five-second example, but larger feeds need a measured blocking or
approximate-neighbor strategy.

Input track intervals remain in absolute source time on the rewrite path. The
semantic baseline reports clip-relative entered/exited times in the same
timeline field shape. Controlled rewrite-contract tests start from the
semantic baseline text and accept a static Detect–Track–Join target; they do
not claim that a live model invents the rewrite.

### Optimizers

#### Rule Optimizer

The rule optimizer lives in [src/mmds/optimizers/rewriter/rule.py](/Users/chanwutk/Documents/mmds/src/mmds/optimizers/rewriter/rule.py).

Current behavior is intentionally conservative:

- recursively rebuild both unary and binary sources
- structurally deduplicate equivalent nodes through memoization
- preserve shared source identity, including self-join sources

It does not yet reorder operators, fold operators, infer safety, or reason about prompt/UDF semantics.

#### Typed Directive Rewriter Core

The deterministic directive core lives in
[src/mmds/optimizers/rewriter/core.py](/Users/chanwutk/Documents/mmds/src/mmds/optimizers/rewriter/core.py).
It is deliberately independent of model selection and dataset profiling.

`DatasetExpr` remains the only operator-tree representation:

- `NodePath` is an immutable address from the output node, such as
  `output.source`.
- `PlanEntry` pairs one address with the existing `DatasetExpr` at that
  location. It contains no field or schema analysis.
- `PlanIndex` traverses the plan, resolves addresses, and replaces a subtree
  by rebuilding only its ancestors. It never mutates or copies the full plan.
  Addresses follow `source` edges only, so `PlanIndex.build` raises
  `MMDSRewriteError` for any plan containing a multi-input operator (`Join`);
  the rewriter does not support those plans yet.
- `RewriteMatch` records that a directive can be applied at one path.
- `RewriteDirective` separates applicability (`find_matches`) from a
  deterministic structural transformation (`apply`).

Directive parameters are strict Pydantic models owned by each directive.
`apply_rewrite()` accepts a previously offered match, validates its parameters,
calls the directive, validates structural invariants, and returns a normalized
`QueryProgram`. It rejects matches that the directive did not offer.

Structural validation guarantees that rewrites preserve reachable input paths,
rejects conflicts when both plans declare a final output schema, and requires
the resulting plan to round-trip through normalized MMDS Python. A UDF-backed
output has no declared schema to compare today. Validation does not claim to
prove semantic equivalence; that responsibility belongs to directive
preconditions and later evaluation.

Model-based selection is a separate orchestration layer over the deterministic
core:

```text
QueryProgram -> PlanIndex -> directive matches
                                 |
                                 v
                       model call 1: select
                                 |
                         validate offered option
                                 |
                                 v
                    model call 2: parameters
                                 |
                       validate typed parameters
                                 |
                                 v
                    deterministic apply_rewrite
                                 |
                                 v
                       rewritten QueryProgram
```

#### Implemented Video Rewrite Directives

The first directive library lives in
`src/mmds/optimizers/rewriter/directives/`. Directives are deterministic plan
transformations; they do not call a model to select themselves or invent their
parameters.

- `ModalitySubstitution` rewrites a direct `Record[video_field]` reference in
  one prompt-backed `Map` to `Record[transcript_field]` and uses a generated
  replacement instruction.
- `PromptFieldPruning` removes one or more unused top-level `Record[...]`
  references from a prompt-backed `Map` and replaces the instruction. It
  requires every listed drop field to be directly referenced, leaves at least
  one remaining `Record` reference, and rejects nested drop targets.
- `BooleanMapCodeFilter` replaces a prompt-backed `Filter` with a Map that
  materializes a boolean keep flag plus a non-LLM `Filter(..., Record[flag])`
  field predicate. When the Filter already sits on a prompt Map, `map_schema`
  must preserve that Map's declared schema and the rewritten Map prompt reads
  that Map's input fields; otherwise the inserted Map reads the Filter's input
  fields. `map_schema` only declares outputs, never prompt inputs.
- `DetectPresenceMap` replaces a prompt `Map` whose schema is exactly one
  boolean field. `find_matches` offers only those Maps. `apply` requires
  `flag_field` to be that field, `video_field` to be a direct top-level
  `Record` reference in the prompt, and `classes` to be non-empty unique
  YOLOE names. `model` defaults to `yoloe-11s-seg.pt`. `min_tracks` defaults
  to `1`. The prompt is removed. The rewritten plan is `Detect`
  (`stop_after_n=min_tracks`, `output_field="detections"`) then a keep Filter
  and a code Map. `min_tracks=1` uses `Filter(keep_rows_with_detections)` and
  `Map(map_detection_presence(flag_field))`: one track keeps the row. A larger
  `min_tracks` uses `Filter(keep_rows_with_at_least_n_tracks(str(n)))` and
  `Map(map_at_least_n_tracks(flag_field, str(n)))`, counting distinct
  `track_id` values rather than boxes. The scan stops once that many tracks
  exist. Fewer than `n` reads the video to the end and drops the row, which
  scores as false when a missing row is false. This does not answer an exact
  count. Presence is
  [`examples/animals_bear_detect_presence.py`](examples/animals_bear_detect_presence.py)
  against [`examples/animals_bear_map.py`](examples/animals_bear_map.py), scored
  by [`examples/eval_animals_bear.py`](examples/eval_animals_bear.py) on
  `data/swan_valley_full.jsonl`. At least five bears is
  [`examples/animals_five_bears_detect_presence.py`](examples/animals_five_bears_detect_presence.py)
  against [`examples/animals_five_bears_map.py`](examples/animals_five_bears_map.py),
  scored by [`examples/eval_animals_five_bears.py`](examples/eval_animals_five_bears.py)
  with [`data/animals_five_bears_ground_truth.json`](data/animals_five_bears_ground_truth.json).
  Those examples import bare UDF wrappers so Python execution does not call
  the bound UDF at import time.
- `DetectedFrameWindowBeforeMap` inserts `Detect`, a UDF Map that converts
  absolute `frame_idx` hits into `_mmds_candidate_views` one-frame intervals,
  then a joint `VideoMap` over those views. Window padding + lowering's
  Coalesce merge nearby hits; empty detection lists yield no views so the
  semantic Map/Reduce is not called. Requires ctor `identity_fields`. Optional
  `min_confidence` becomes Detect `conf`. The original Map prompt and schema
  are preserved on the VideoMap. It derives grouping fields and guards
  downstream field reads exactly like the temporal directives (shared helpers
  in `temporal.py`; see below).
- `JointTemporalPushdown` replaces a video `Map` with a transcript candidate
  `Map` followed by logical `VideoMap`. The final prompt sees all coalesced
  candidate views for a group and runs once.
- `PerViewTemporalPushdown` replaces an event-localization `Map` with a
  transcript candidate `Map`, logical `VideoMapEach`, deterministic timestamp
  rebasing, and `Reduce(reconcile_events)`. `reconcile_events` only collects
  and sorts source-time events; it does not merge or deduplicate them.

Temporal directives require explicit `identity_fields`, such as
`lecture_id`, as the stable grouping key for each source row. They preserve
non-video fields read by the original prompt and downstream grouping keys
(except fields the matched `Map` itself produces). Because the rewritten stage
emits only those grouping fields plus the `Map`'s output fields, a match is
offered only when every field read downstream survives: consumers are checked
from the matched `Map` toward the output, stopping after the first
row-rebuilding operator (`Reduce`, `Coalesce`, `VideoMap`, `VideoMapEach`). A
downstream UDF makes the needed fields unknowable, so no match is offered;
`apply` re-checks once the model has chosen the video field. Candidate intervals use source-video time, while a
per-view verifier returns clip-relative time that is subsequently rebased.
The internal candidate field is reserved as `_mmds_candidate_views`.

Generated `rewritten_prompt`/`video_prompt` parameters replace the original
literal instruction instead of merely prepending to it. This prevents stale
phrases such as "complete video" or "absolute time" from contradicting a
transcript-only, field-pruned, or per-view rewrite. Structured `Record[...]`
references are rebuilt deterministically by the directive.

Directive matching is intentionally broader than parameter validation:
`find_matches()` offers prompt-backed `Map` locations, then `apply_rewrite()`
validates the chosen fields before changing the plan.

#### Automatic Rewrite Flow

`build_rewrite_context()` creates compact, ephemeral model input. It summarizes
the complete reachable plan, including prompt templates and output schemas,
and profiles at most eight JSON/JSONL rows to expose field types and semantic
roles. Dataset row values are never included. Unreferenced fields whose names
look like ground truth, labels, or annotations are excluded to avoid benchmark
leakage. Missing input files are reported as unavailable; malformed existing
files fail explicitly.

`rewrite_once()` applies at most one rewrite:

1. Build one structural `PlanIndex` and enumerate directive matches.
2. Model call 1 sees the complete plan summary, value-free dataset fields, and
   offered directive/path options. It returns one ephemeral option ID or null.
3. Validate that the ID was offered before making another call.
4. Model call 2 sees only the selected node, dataset fields, directive metadata,
   and its compact Pydantic parameter contract.
5. Pass the returned object through `apply_rewrite()`, which validates the
   parameters and deterministically builds the new plan.

`GeminiRewriteModel` is the concrete JSON-response adapter. Tests use a small
sequence model so both prompts, response validation, and call ordering remain
deterministic. Selection and parameter prompts are also available through
debug logging.

This initial engine intentionally has no budgets, multi-plan search, ranking,
fingerprints, projection cleanup, metrics, or repeated rewrite chains. Those
are independent research extensions rather than prerequisites for one safe,
inspectable rewrite.

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
- directive rewriting never mutates the original operator tree
- rewrite models or policies select offered matches; directive code owns plan construction
- the last assignment is the output unless a future explicit sink is added
- input roots are direct file paths, not catalog identifiers
- rendered queries are normalized, not source-exact
- prompt-backed execution always requires an injected executor
- `.pyi` discovery does not imply executability
- provider-specific media handling belongs in executors, not DSL syntax
- `Reduce` row access must go through `ForEach([...])`
- logical video-map plans lower deterministically before execution without
  mutating the original immutable plan
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
- logical video-map construction, validation, parse/render round trips,
  lowering, joint/per-view execution, coalescing, and empty candidates
- typed rewrite paths, structural indexing, immutable subtree replacement,
  directive parameter validation, and rewrite structural invariants
- deterministic modality-substitution, prompt-field pruning, boolean-map code
  filter, detect-presence-map, detected-frame-window-before-map, and
  joint/per-view temporal-pushdown directives, including plan-shape and
  end-to-end execution tests
- `Filter(..., Record["field"])` field-predicate parse/render/execute round trips
- `Detect(...)` parse/render round trips, including optional `conf` and `stop_after_n`
- value-free rewrite context, sequential model selection/parameter calls,
  response validation, null selection, and the Gemini adapter
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
