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

- operators: `Input`, `Map`, `Filter`, `Reduce`, `Unnest`, `Resolve`, `View`, `Detect`
- public video utility: `VideoView(video, start, end)` for seek-based clip-range iteration
- file-backed `Input(...)` roots over `.json` and `.jsonl`
- prompt-backed semantics as either:
  - a plain string
  - a structured prompt list made of strings, `Record[...]`, and Reduce-level `ForEach([...])`
- function-backed semantics as imported UDFs from `udfs.*`
- serializable built-in deterministic functions, currently `PadInterval` and
  `ReconcileIntervals`
- prompt-backed `Map` and `Reduce` with concise output field schemas
- prompt-backed `Filter` returning a bare JSON boolean
- source parsing for straight-line top-level assignments only
- plan rendering back to normalized Python code
- local execution with an injected prompt executor
- explicit `ExecutionContext` resources, bounded provider concurrency, and
  execution statistics
- Gemini-backed prompt execution with executor-side video translation
- UDF discovery from `.py` and `.pyi`
- a conservative rule optimizer and a validation-heavy LLM optimizer scaffold
- `Detect` operator for frame-level YOLOE object detection on video fields

The current implementation intentionally does not support:

- inline lambdas
- nested functions or callables outside `udfs.*`
- loops, conditionals, comprehensions, classes, or arbitrary Python control flow in query files
- joins, sorts, projections, or cost-based optimization
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
- `Resolve(data, group_by, start_field, end_field, *, merge_touching=True, name=None)`
- `View(data, video_field, start_field, end_field, *, output_field="view", name=None)`
- `PadInterval(...)` and `ReconcileIntervals(...)` as deterministic `Map`/`Reduce` specs
- `Detect(data, video_field, classes, *, model="yoloe-11s-seg.pt", output_field="detections", name=None)`
- `VideoView(video, start, end)`
- `Record[...]`
- `ForEach([...])`
- `execute(plan_or_query, prompt_executor=None, *, context=None)`
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
- `PadIntervalSpec` and `ReconcileIntervalsSpec` store core deterministic
  temporal functions without routing them through experiment UDF modules.
- `ResolveSpec` stores interval grouping and overlap-coalescing semantics.
- `ViewSpec` identifies the video, source-boundary, and output attributes for
  standalone clip materialization.
- `DetectSpec` stores the parameters for a `Detect` node: `video_field`, `classes`, `model`, `output_field`.
- `Assignment` and `QueryProgram` represent a parsed query file.
- `MMDSValidationError` is the shared validation failure type.

`DatasetExpr` uses a unary tree shape today:

- `Input` has no source.
- `Map`, `Filter`, `Reduce`, `Unnest`, `Resolve`, `View`, and `Detect` each have one `source`.

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
- `Resolve` requires explicit grouping, start, and end fields
- `View` always materializes a standalone, zero-origin clip; it has no lazy mode
- built-in temporal function arguments must be literal and are validated while parsing
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

- always emits the base operator imports and adds `Resolve`, `View`, and built-in
  temporal functions when the plan uses them
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
- `Resolve`: blocks per execution input, groups by explicit keys, sorts intervals,
  and coalesces overlapping (and, by default, touching) ranges only within each group
- `View`: accurately seeks, re-encodes, validates, and atomically promotes one
  standalone MP4/H.264/AAC clip per source-time interval; the output retains
  its source identifier and source boundaries
- `Detect`: reads the video pointed to by `video_field`; if the value is a `VideoView`-shaped dict with `start`/`end`, wraps the source in a `VideoView`, runs YOLOE detection on the selected frames, and merges a detection list with absolute source-video `frame_idx` values into `output_field`

`ExecutionContext` owns resources and policies for one execution. Its
`max_workers` default is one, its optional workspace is required by `View`, and
its optional materializer can be replaced by a deterministic fake in tests.
Materialized views are content-addressed by the source fingerprint, interval,
and encoding contract and are reused only within that execution context. The
runtime records per-operator row counts and processing time, materialized-view
bytes and reuse, and Gemini token usage when the provider reports it. Pricing
and evaluation metrics remain outside the core runtime.

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

- `start` and `end` are finite, non-negative numeric seconds and are translated to Gemini `VideoMetadata.start_offset` and `VideoMetadata.end_offset`
- numeric offsets are rendered as canonical protobuf durations with at most nine fractional digits; trailing zeroes are removed so binary-float artifacts cannot produce invalid duration strings
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
  {"type": "dog", "bboxes": [{"frame_idx": 0, "bbox": [x1, y1, x2, y2], "confidence": 0.9}]},
  ...
]
```

- the OpenCV/NumPy stack (and, at run time, `torch`/`ultralytics`) is imported **lazily**: `mmds.execution` imports `.ops.detect` only when a `detect` node actually executes, and `mmds.VideoView` is a lazy export via module `__getattr__`. This keeps `import mmds` and the prompt/UDF execution paths usable without the heavy CV/ML dependencies installed.
- `Detect` is **not** parsed from or rendered back to DSL text (it is for internal/programmatic use only)

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

#### Cross-Modal Rewrites

Explicit paper-aligned rewrites live in
`src/mmds/optimizers/cross_modal.py`. `ModalitySubstitution` (O1) replaces one
named prompt-backed `Map` with a schema-compatible transcript prompt.
`CrossModalTemporalPushdown` (O2) replaces one named full-video semantic `Map`
with:

`Map(candidate prompt) -> Unnest -> Map(PadInterval) -> Resolve -> View ->`
`Map(original video prompt over clip) -> Reduce(ReconcileIntervals)`.

The O2 candidate function is a query-conditioned, high-recall MLLM prompt that
returns source-video intervals directly. The rewrite verifies that the
candidate prompt reads the transcript field, keeps the original video
instructions and output schema unchanged, and changes only the video attribute
from the source video to the materialized clip. Reconciliation translates
clip-relative events to source time, deduplicates overlapping predictions, and
restores the configured output fields. The rule constructs an alternative
query; it does not estimate quality or automatically choose among alternatives.

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
- prompts, stable imported UDFs, and declared core deterministic functions are
  the only valid semantic specs
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
- `View` means physical standalone-clip materialization; lazy `VideoView` values
  are not an alternative execution mode for this operator
- `Resolve` never coalesces intervals across its explicit grouping keys
- importing `mmds` must not require the optional computer-vision stack (OpenCV/NumPy/`torch`/`ultralytics`); `Detect` and `VideoView` load those dependencies lazily on use

## SoccerNet Goal-Pushdown Experiment

The active paper experiment under `scripts/soccernet/` is
`goal_pushdown_3_games_v4_materialized`. It is an immutable successor to v3 and
preplans three physical strategies over the same three English-commentary games:

- naive full-half video analysis
- transcript-only final goal detection, with no video verification
- per-half high-recall transcript candidate generation, physical clip
  materialization, and short-video localization

Naive and transcript-to-video use exactly one video-localization prompt and one
`goals` output schema. The only intended model-input difference is media extent:
the naive plan supplies a complete half through `Record["video"]`, while the
optimized plan supplies an ordinary standalone `Video` through
`Record["candidate_video"]`. The active v4 plan contains no `VideoView`. Both
model inputs therefore have a zero-origin local timeline. The full-video
normalizer converts clip-minute/clip-second components directly to half time.
The materialized-clip normalizer validates the standalone-media context and adds
the frozen candidate-window start exactly once, exposing normalized rows under
`transcript_video_goals`.

The physical workflow is explicit:

`prepare -> run-naive -> run-transcript-only -> run-candidates ->`
`materialize-clips -> run-transcript-video -> evaluate -> compare`

Prediction preparation writes a label-isolated `input.jsonl`, a
`prediction_config.json`, rendered plans, source hashes, and a content-addressed
prediction-prefix completion marker. Ground truth and tolerance/deduplication
settings live separately in `ground_truth.json` and `evaluation_config.json`.
None of the prediction, candidate, materialization, or video-localization stages
opens either evaluator file. Only `evaluate` loads labels, after validating all
prediction completion markers. This keeps the same 5/10/30/60-second matching
tolerances and 5-second prediction deduplication policy across all three modes.

Candidate windows are materialized serially as standalone MP4/H.264/AAC clips
using the same shared encoding contract as the lecture paper evaluation:
`libx264`, CRF 18, `veryfast`, `yuv420p`, AAC 128 kbps, and reset video/audio
timestamps. Each source hash is checked once per validation stage. Each output
clip is independently checked for both streams, zero start time, requested
duration within 0.25 seconds, size, SHA-256, and frozen `ffprobe` metadata.
Checkpoint entries make interrupted encoding resumable; a complete manifest can
recover a missing completion marker; corrupt or unmanifested clips fail closed.
The model input contains the standalone clip path and hash plus non-model
candidate/source provenance, never an offset-bearing media view.

Every stage content-addresses its outputs and dependencies. Model calls and
local encodes are serialized. Stage reports include API usage and media-upload
bytes/time. Transcript-to-video query-time totals include candidate generation,
local encoding, clip upload, and clip localization; one-time Whisper
transcription remains separate. Cache hits, failed API/upload attempts, or
failed clip encodes, or reused checkpoint clips invalidate headline cold-start latency without
invalidating accuracy or durable usage records.

The transcript-only method is frozen during initial preparation and makes one
text-only call per match half. The model must copy the exact start time of an
existing transcript segment for every final goal prediction. A deterministic
UDF preserves but marks out-of-duration or non-segment-start timestamps invalid;
evaluator labels are never used to repair them. This method isolates whether
short-video localization improves over transcript reasoning alone.

The completed `goal_pushdown_3_games_v3` experiment remains immutable and is the
historical `VideoView` implementation. Its optional 1B signal ablation remains a
historical v3 artifact and is deliberately excluded from the active v4
three-method paper comparison. V2 also remains immutable; its video prompts
differed and its transcript-only method was post-hoc. The historical 1B method
uses an LLM-generated generic include/exclude lexicon plus deterministic
token-exact matching; no signal is manually tuned to evaluation data.

End-to-end latency is comparable only for a clean cold-start execution. The
comparison validates every constituent stage and reports combined latency and
latency reduction only when there are no cache hits, failed API/upload/encoding
attempts, or reused materialized clips. Otherwise, the raw sum of observed stage times and
cumulative API time remain available as diagnostics, but the comparable
end-to-end value is `null`. Accuracy, token usage, upload accounting, and
estimated cost remain available from immutable artifacts.

## Cross-Modal Lecture Pushdown Experiment

The paper experiment under `scripts/lectures/` is a frozen three-by-three pilot:
three MIT 8.03SC lecture videos, three cross-modal event queries, and therefore
nine lecture-query pairs. Source selection and queries live in
`scripts/lectures/catalog.py`; the catalog contains no expected matches or
ground-truth intervals.

The three primary evaluated physical plans use the same query text and frozen labels:

- **naive:** `Input -> Map(shared audiovisual verifier) -> Map(normalize intervals) -> Unnest`
- **transcript only:** `Input -> Map(select Whisper segment ranges) -> Map(ground ranges) -> Unnest`
- **optimized:** `Input -> Map(select candidate segment ranges) -> Map(pad/merge windows)`, then local standalone-clip materialization followed by `Input(materialized clips) -> Map(shared audiovisual verifier) -> Map(normalize intervals) -> Filter -> Unnest`

The naive and optimized plans share one verifier prompt and one output schema.
Only the supplied media extent changes: the naive plan receives the full
lecture, while the optimized plan receives a physically materialized standalone
candidate clip whose timeline starts at zero. Both explicitly pass `fps=1.0`;
audio remains present in the video input.
Candidate ranges receive 30 seconds of context on each side, are clamped to the
physical lecture duration, and are union-merged before verification. The
primary benchmark makes one verifier call per merged window and serializes all
provider calls (`max_in_flight=1`).

Clip materialization is an explicit, content-addressed physical stage. FFmpeg
accurately seeks and re-encodes each merged range as MP4/H.264 (`libx264`, CRF
18, `veryfast`, `yuv420p`) with AAC 128 kbps audio. Video and audio timestamps
are reset to zero. Every clip must contain both streams, have a duration within
0.25 seconds of its requested range, and match its frozen size, SHA-256, and
`ffprobe` metadata before verification. A checkpoint makes interrupted local
materialization resumable; reused clips make cold-start latency invalid. The
materialized model input contains no source-video start/end offset. The original
absolute candidate window remains evaluator-inaccessible row metadata used only
after inference to translate clip-relative results back to lecture time.

An earlier `VideoView` offset implementation is retained under `runs/optimized`
as a diagnostic artifact. Gemini returned a mixture of clip-relative and
source-relative timestamps for those offset views, violating the frozen
clip-relative contract. The corrected primary optimized stage is stored
separately under `runs/optimized_materialized`; the diagnostic run is never
silently rewritten and is excluded from primary accuracy and efficiency
comparisons.

The additive v2 method separates expensive audiovisual verification from
localization instead of treating a nonempty interval list as an implicit
Boolean. It preserves every v1 plan and artifact. Its physical plans are:

- **v2 naive verification:** `Input(full lectures) -> Map(binary audiovisual verifier) -> Map(strict Boolean validation)`
- **v2 naive localization:** `Input(binary decisions) -> Filter(event_present) -> Map(complete-episode localizer) -> Map(normalize intervals) -> Unnest`
- **v2 optimized verification:** reuse the frozen high-recall transcript
  candidates and standalone clips, then `Input(materialized clips) -> Map(binary audiovisual verifier) -> Map(strict Boolean validation)`
- **v2 optimized localization:** `Input(binary candidate decisions) -> Filter(event_present) -> Map(complete-episode localizer) -> Map(normalize intervals) -> Unnest`

Naive and optimized use identical binary prompts/schemas and identical
complete-episode prompts/schemas; only their supplied media extent differs.
The binary contract contains exactly a Boolean `event_present`, bounded
confidence, and nonempty evidence. It never coerces strings or numbers through
Python truthiness. A positive decision must pass through the explicit `Filter`
before localization, and the localizer must return at least one complete
contiguous episode. For an experiment, that episode starts with physical
performance rather than verbal setup and ends when the demonstrated outcome and
its immediate activity conclude. Isolated sub-actions within one continuing
experiment are not separate episodes.

V2 reports pair-level binary classification separately from interval
localization. Multiple candidate-window decisions for one lecture-query pair
are combined with logical OR for binary precision, recall, F1, and accuracy;
pairs with no transcript candidates predict absence. Temporal localization
retains the frozen v1 one-to-one tIoU evaluator and human labels, which are never
modified using v1 or v2 outputs. `v2_config.json` content-addresses both new
contracts and rendered plans plus the reused candidate and clip prefix.
Versioned stages live under `runs/v2/`, and `v2_comparison.json` never replaces
the v1 comparison.

The reused prefix is an accounting reuse, not a zero-cost cache claim. V2 totals
include the original cold-start transcript-candidate and clip-materialization
times, tokens, and cost. Binary verification runs on every available input;
episode localization runs only for accepted inputs. Calls remain serialized.
Each VLM stage records usage separately, and application cache hits, failed
attempts, or reused clips continue to invalidate cold-start latency.

The additive v3 method makes transcript segment boundaries the timestamp
coordinate system throughout the optimized plan. It preserves v1/v2 plans and
artifacts and content-binds the completed v1 naive result instead of rerunning
that baseline. Its physical stages are:

- **transcript episode proposals:** `Input -> Map(select complete episode
  segment ranges) -> Map(validate, ground, and pad each proposal)`
- **proposal clip materialization:** accurately re-encode one standalone clip
  per valid proposal, retaining distinct hypotheses even when their padded
  contexts overlap
- **transcript-grounded audiovisual refinement:** `Input(materialized proposal
  clips plus aligned transcript excerpts) -> Map(refine to zero or one allowed
  segment range) -> Map(ground segment IDs) -> Filter(nonempty) -> Unnest`

The transcript proposal is the timestamp hypothesis rather than a disposable
search gate. It must describe one complete continuing physical or audible
episode and copy inclusive existing Whisper segment IDs. A continuing
multi-action or multi-minute experiment remains one proposal. Every proposal is
expanded by the existing 30-second context padding for audiovisual inspection,
but that padding never becomes the predicted interval.

The refinement prompt receives the padded zero-origin video clip, the exact
proposal, the clip's source-time extent, and an aligned timestamped transcript
excerpt. Its structured response contains an `episode_refinements` array with a
maximum of one element. An empty array rejects the proposal; one element must
copy its start and end IDs from the supplied excerpt. It cannot emit free-form
minute/second timestamps. Deterministic validation rejects unknown IDs, IDs
outside the allowed excerpt, reversed boundaries, or grounded intervals outside
the supplied clip. Invalid nonempty outputs are retained as false positives and
are never repaired from labels.

V3 is explicitly marked as a post-hoc method-development diagnostic because its
contracts were designed after inspecting v1/v2 failures on the three-lecture
pilot. Its results are not held-out paper evidence. The prompt, padding, and
boundary rules must be frozen on separate development lectures before a final
held-out evaluation. V3 nevertheless uses the same frozen human intervals and
one-to-one tIoU evaluator so its diagnostic numbers remain directly auditable.
Its comparison includes transcript-proposal cost, proposal-clip materialization,
and audiovisual-refinement cost, while keeping one-time Whisper materialization
separate.

The additive v4 method retains the frozen v3 high-recall transcript proposals
and standalone clips but replaces broad complete-episode refinement with
mandatory-predicate evidence grounding. It does not reuse or count the v3
refinement stage. Its new physical stages are:

- **query condition compilation:** one text-only call per distinct query maps
  the query to the smallest complete set of mandatory directly observable
  conditions; a deterministic UDF assigns stable `condition_0`, `condition_1`,
  ... identifiers
- **predicate grounding:** each proposal clip, aligned transcript excerpt, and
  compiled condition list are supplied to one audiovisual call; the model
  returns segment-ID evidence only for conditions directly established in that
  clip
- **deterministic conjunction and interval derivation:** accept only when the
  returned condition-ID set exactly covers the required set, then derive the
  final interval as the smallest source-time span covering all condition
  evidence ranges

The query compiler receives only generic query text—never transcripts, videos,
lecture identities, model outputs, or labels. It must preserve explicit objects,
actions, modalities, measurements, temporal relations, and causal relations
without weakening them to related topics. Each compiled condition is separately
auditable before any VLM grounding call.

The grounding response contains a `condition_evidence` array. Each item must
copy a required condition ID and inclusive start/end IDs from the aligned
Whisper excerpt. Missing any required condition rejects the proposal. Unknown or
duplicate condition IDs, invented or out-of-context segment IDs, and invalid
ranges produce an invalid retained prediction rather than a silent rejection.
The VLM never emits a final interval or free-form timestamp; final boundaries
are a deterministic aggregate over the per-condition evidence.

V4 includes the original cold-start transcript-proposal and clip-materialization
costs plus query-compilation and predicate-grounding costs. Like v3, it is
explicitly post-hoc method development on the three-lecture pilot and requires
separate development lectures plus a once-only held-out evaluation before its
numbers can support a paper claim.

The additive v5 method preserves every v1-v4 artifact and addresses a boundary
coupling exposed by inspection of the v4 query decomposition. It reuses the
same frozen v3 transcript proposals and standalone clips, but compiles each
mandatory condition with one of two explicit roles:

- **gate:** required evidence for accepting a proposal, but evidence for the
  qualification does not define or widen the target interval
- **anchor:** required evidence whose direct occurrence defines the target
  interval's temporal boundaries

The v5 query compiler receives only the generic query text. It must produce the
smallest nonredundant set of conditions, preserve every explicit requirement,
avoid adding simultaneity, causality, ordering, or actor identity that the query
does not state, and emit at least one anchor. A deterministic UDF validates the
roles and assigns stable condition IDs. The grounding VLM still returns only
per-condition inclusive Whisper segment-ID ranges; it never emits a final
interval or free-form timestamp.

All gate and anchor conditions remain conjunctive: missing any condition rejects
the proposal. After exact condition-set validation, deterministic code derives
the final interval from anchor evidence only. Gate evidence can occur elsewhere
inside the supplied padded clip without expanding the prediction. Unknown,
duplicate, invented, or out-of-context segment IDs retain the same fail-closed
behavior as v4.

V5 comparison includes the original v3 proposal/materialization costs plus the
v5 gate/anchor compiler and role-aware grounding costs. It excludes v3 and v4
grounding costs because those stages do not contribute to the v5 result. The
gate/anchor distinction was designed after inspecting v4 output on this pilot,
so v5 is also explicitly post-hoc method development and is not held-out paper
evidence.

Whisper `small` English transcripts are materialized once per source video.
Segments have consecutive integer IDs and finite, chronologically ordered
boundaries. Adjacent Whisper segments may overlap when both their starts and
ends remain nondecreasing; their original boundaries are preserved rather than
silently clamped or merged. Backtracking, reversed, and non-finite boundaries
still fail closed.
The sole physical-duration adjustment is explicit and versioned: when only the
final segment overruns the `ffprobe` duration by at most 0.1 seconds, its
normalized end is clamped to the physical duration while the raw end and reason
are retained in both the raw checkpoint and normalized adjustment metadata.
Internal overruns and larger final overruns fail closed.
Transcript-backed model outputs must copy inclusive existing segment IDs. Video
model outputs use clip-relative start and end boundaries represented as elapsed
minutes plus seconds. Full-lecture naive input and every optimized standalone
clip therefore share an unambiguous zero-origin timeline. Deterministic UDFs
translate both contracts into absolute lecture intervals. Unknown IDs, reversed
ranges, non-finite values, intervals outside the supplied clip, and other
contract violations fail closed. Invalid final predictions are retained as
false positives and are never repaired from evaluator labels.

Ground truth is a separate two-step human workflow. `init-review` creates all
nine pending decisions; `freeze-review` requires each positive or negative pair
to be explicitly completed by a named reviewer. `prepare` then creates
prediction inputs with a recursive evaluator-key isolation check. Source media,
transcripts, input rows, prompts, schemas, and ground truth are content-addressed
in the frozen configuration. Downloading is staged and resumable, and validates
both audio and video streams before promotion. Every run stage writes a final
content-hashed completion marker; downstream work rejects changed or partial
artifacts.

Whisper output is atomically checkpointed before normalization in a
`*.whisper.raw.json` artifact bound to the source ID, source SHA-256, model, and
requested language. Normalized transcripts record the checkpoint hash. If
normalization or a later write fails, a retry reuses the checkpoint without a
second model invocation; mismatched or corrupted checkpoints fail closed. The
normalization contract is versioned, so stale normalized artifacts can be
rebuilt from matching raw checkpoints without loading the Whisper model.

Evaluation is interval-based and identical across methods. It uses temporal IoU
thresholds 0.1, 0.3, and 0.5, with 0.3 primary. Prediction deduplication uses
tIoU 0.8 and retains the highest-confidence valid interval; invalid predictions
are never deduplicated away. Per-pair matching maximizes one-to-one match count
and then total tIoU. Metrics include precision, recall, F1, and exact-pair
accuracy over all nine pairs, so false positives on reviewed negatives count.
Candidate evaluation separately reports union duration, selectivity, duration
reduction, at-least-50% truth-event coverage, and full truth-event coverage.

API attempts, elapsed time, modality token counts, and estimated costs use the
pricing snapshot in the catalog. Audio tokens use their separate input rate when
the provider returns modality details; missing breakdowns produce a visible
cost warning. Result caching exists only for recovery. A stage with cache hits
or failed attempts is invalid for cold-start latency. Comparison keeps query
time separate from one-time Whisper materialization and also reports the full
materialized workload. Optimized query-time latency includes transcript
candidate generation, local clip materialization, and standalone-clip video
verification. Local materialization has no API tokens or USD estimate, but its
wall-clock time, output bytes, clip count, and requested/output durations are
reported explicitly.

The guarded command sequence and manual verification checkpoints are documented
in `scripts/lectures/README.md`. External work requires an explicit `--execute`.

## Frozen Four-Lecture Paper Evaluation

`scripts/lecture_paper_eval/` is an isolated successor to the exploratory
three-lecture v1-v5 pilot. It never mutates or reads pilot run artifacts and
uses its own default root, `data/lecture_paper_eval/`. Its frozen experiment ID
is `cross_modal_lecture_4x4_v1`.

The benchmark is the Cartesian product of MIT 8.03SC Lectures 3, 7, 9, and 15
and four exact event queries: tone-induced glass shattering, fire-induced
resonant sound, hand-driven traveling waves on a long spring, and speaker-driven
Chladni formation. The matrix therefore contains 16 unique lecture-query pairs.
Four pairs are positive and twelve are explicit negatives. The four positive
pairs contain seven human intervals because the Lecture 15 Chladni event has four
distinct repetitions. Lecture 11 is absent from the catalog, inputs, labels,
and expected source/transcript manifests.

Human labels live only in `scripts/lecture_paper_eval/annotations.py` and the
frozen evaluator artifact. The seven annotation-v2 intervals, in seconds, are Lecture 3
`[4427,4432]`, Lecture 7 `[2793,2805]`, Lecture 9 `[1643,1654]`, and Lecture 15
`[4195,4225]`, `[4289,4300]`, `[4338,4360]`, and `[4375,4395]`. A positive label is the smallest
continuous interval satisfying the complete query, excluding setup,
discussion, completed static results, and reactions. The benchmark records
positive-pair, negative-pair, and event counts separately so repeated events
cannot be confused with independent query pairs.

Annotation version 1 mistakenly omitted the leading one-hour component from
three Lecture 15 timestamps and omitted the separately confirmed first
formation. A prepared version-1 experiment is corrected by an evaluator-only,
pre-evaluation amendment. The original `ground_truth.json`, evaluation config,
prepare completion, label-free prediction prefix, and completed predictions stay
byte-for-byte unchanged. Versioned annotation-v2 truth/config sidecars and a
content-addressed amendment record store both revisions' hashes, the human
correction reason, and hashes of every prediction artifact present at amendment
time. Fresh preparations write version 2 directly. The amendment is refused for
a non-exact version-1 base, a partial/duplicate amendment, or any experiment
whose evaluation has started.

The evaluated methods are exactly:

- **naive:** full zero-origin lecture video plus query to an audiovisual
  timestamp localizer
- **transcript only:** complete timestamped Whisper transcript plus query to a
  segment-ID localizer
- **transcript then video:** complete transcript plus query to a high-recall
  candidate selector; deterministic 30-second padding, clamping, and union
  merging; standalone zero-origin clip materialization; then the same
  audiovisual timestamp localizer used by naive

Naive and transcript-then-video share one prompt, one structured schema, one
Gemini model, one `fps=1.0` setting, and one strict normalization contract. The
optimized VLM receives only the query and standalone video clip. It does not
receive transcript text, transcript timestamps, the source-video offset, or
labels. Absolute optimized timestamps are obtained exactly once by adding the
frozen candidate-window start in deterministic code after inference.

The transcript filter is explicitly high recall. It has no confidence
threshold, top-k, candidate-count cap, or retained-duration budget. Exact
duplicate segment ranges are audited and deduplicated; valid padded ranges that
overlap or touch are union-merged. Unknown, negative, or reversed segment IDs
remain audited invalid candidates and are never materialized. Candidate reports
include raw, valid, invalid, duplicate, and merged counts; union duration and
selectivity; and any, 50-percent, full, and mean truth-event coverage.

Provider structural failures fail the stage. An empty structured list is a
valid negative prediction. A structurally valid but reversed or out-of-video
interval is retained as an invalid prediction and therefore becomes a false
positive during evaluation. Unknown or reversed transcript ranges are likewise
retained with null absolute boundaries. Confidence is never clamped, missing
evidence is never synthesized, invalid predictions are never repaired, and no
label-derived suppression is allowed.

Preparation writes separate prediction and evaluation configurations. The
prediction-prefix completion marker contains only the source manifest,
transcripts, label-free 16-row input, prompts, schemas, and rendered plans.
Prediction commands validate only this marker and cannot open ground-truth
files, the annotation module, or evaluation configuration. A separate evaluator
validation resolves and hashes the active label revision only after all three
prediction artifacts are complete. Changing label bytes leaves prediction
dependencies unchanged but makes evaluation validation fail.

The immutable workflow is `prepare -> run-naive -> run-transcript-only ->
run-candidates -> materialize-clips -> run-transcript-video -> evaluate ->
compare`. Every stage records content hashes for inputs, plans, prompts,
schemas, outputs, and direct dependencies. Completion markers reject missing,
partial, or modified artifacts. Result caches and clip checkpoints support
recovery, but a cache hit, failed provider attempt, failed media upload, or
reused materialized clip invalidates headline cold-start latency. Such runs
remain auditable and valid for accuracy, token, and cost analysis.

Provider calls are serialized with `max_in_flight=1`. A media path is uploaded
once per stage-local executor and may be reused for other queries in that same
stage; provider file URIs are not persisted across stages or runs. Reports
separate unique upload count, upload attempts/failures/reuse, bytes, media
upload/processing-wait time, generation time, and end-to-end time. Query-time
optimized totals include transcript filtering, local clip materialization, and
candidate-video inference. One-time Whisper `small` transcription remains a
separate materialized-workload metric.

All methods use one-to-one interval matching at tIoU 0.1, 0.3, and 0.5 with 0.3
primary and confidence-prioritized deduplication at tIoU 0.8. Reports include
event precision, recall, F1, exact-pair accuracy, per-match tIoU, signed and
absolute boundary errors, and separate pair-presence diagnostics. Invalid
predictions count as false positives at every threshold.

This frozen evaluation is cleaner than the v1-v5 artifact chain but is not a
fully unseen held-out corpus: Lectures 3 and 9 were inspected during earlier
method development. Results may support the concrete systems comparison but
must not be described as a once-only held-out generalization result without an
additional untouched corpus.

## Verified Three-Pair Lecture Evaluation

The `scripts/lecture_paper_eval/` engine also exposes an explicit
`verified-3pair` profile. Experiment profiles are immutable label-free workload
definitions: each profile owns a distinct experiment name, default data root,
source list, query list, and exact input pair list. The original
`frozen-4x4` profile remains the default, retains its Cartesian 16-row payload
shape, and continues to validate the completed four-by-four artifacts without
changing their catalog or completion hashes.

The verified profile is `cross_modal_lecture_verified_3pair_v1` under
`data/lecture_verified_eval/`. It contains exactly three matched positive rows,
not a Cartesian product:

- Lecture 3 with the tone-induced glass-shattering query and interval
  `[4427,4432]`;
- Lecture 9 with the fire-induced resonant-sound query and interval
  `[1643,1654]`;
- Lecture 20 with the successful-large-soap-bubble query and intervals
  `[435,441]` and `[460,466]`.

The bubble query requires a large intact film to span the separated sticks and
remain visible until rupture. Partial failed attempts at `[444,446]` and
`[479,480]` are explicitly outside the query and evaluator. The profile therefore
has three positive pairs, zero negative pairs, and four events. It measures
conditional temporal localization and execution efficiency; it cannot support
negative-rejection or collection-level specificity claims. Results must report
exact event counts alongside percentages because the sample is small and
Lecture 20 contributes two of four events.

All prediction contracts, plans, model settings, one-frame-per-second video
sampling, transcript-candidate padding, clip materialization, usage accounting,
and tIoU thresholds are shared with the frozen four-by-four profile. Evaluator
annotations remain in the label-owning annotation module. Prediction validation
receives the profile but never imports annotation timestamps or evaluator
configuration. The profile-specific source manifest, input rows, transcripts,
plans, configurations, and completions are content-addressed under the new root.
The CLI selects it with `--profile verified-3pair`; omitting `--profile` preserves
the original four-by-four behavior.

Lectures 3 and 9 were inspected in earlier development, so this verified subset
is a high-confidence case study rather than a once-only held-out generalization
benchmark. Lecture 20 was selected and annotated before running any of the three
methods on it.

## Validation and Tests

Tests live under [tests/](/Users/chanwutk/Documents/mmds/tests/).

The current suite covers:

- parse/render/parse equivalence and validation for `View`, `Resolve`,
  `PadInterval`, and `ReconcileIntervals`
- deterministic padding, source-boundary clamping, per-key overlap resolution,
  clip-to-source translation, and temporal deduplication edge cases
- materialized `View` execution with a fake materializer, execution-local reuse,
  explicit failure modes, and an ffmpeg integration test when ffmpeg is available
- O1 schema preservation and O2 plan structure, prompt/schema control, and a
  provider-free end-to-end lecture rewrite execution
- parse/render/parse equivalence for structured prompts
- file-backed execution for `.json` and `.jsonl`
- rendering from runtime-built plans
- execution for UDF-backed and prompt-backed queries
- `Record[...]` resolution and `ForEach([...])` expansion
- `Unnest` behavior on scalar, empty, and missing values
- `Detect` behavior, including `VideoView` clip slicing and absolute-frame detection indices
- video utility behavior for direct downloads, platform downloads via `yt-dlp`, and `VideoView` iteration
- parser validation for unsupported Python and invalid prompt forms
- optimizer result preservation and LLM rewrite validation
- Gemini prompt compilation for video URI and uploaded local file inputs
- UDF catalog discovery for `.py` and `.pyi`
- SoccerNet v4 prompt/schema control, strict local-to-half timestamps,
  label-isolated prediction stages, standalone-clip provenance and corruption
  checks, checkpoint/completion recovery, cold-start accounting, predecessor
  immutability, and a hermetic prepare-through-compare workflow
- lecture interval-contract edge cases, one-to-one tIoU matching, reviewed negatives, and candidate coverage
- standalone lecture-clip materialization, duration/hash/provenance validation, corruption rejection, and checkpoint recovery
- staged download and transcription recovery, content-bound manifests, label isolation, stage-completion hashing, and a deterministic offline three-by-three end-to-end experiment
- strict lecture binary-verification and complete-episode contracts, conditional localization, separate binary/tIoU evaluation, versioned prefix provenance, and offline v2 stage accounting
- transcript-owned complete-episode proposals, unmerged padded proposal clips, allowed-segment audiovisual boundary refinement, v3 artifact isolation, and offline v3 accounting
- generic mandatory-condition compilation, exact condition-set conjunction, per-condition Whisper evidence grounding, deterministic minimal-span derivation, v4 prefix provenance, and offline v4 accounting
- nonredundant gate/anchor compilation, mandatory gate acceptance, anchor-only deterministic timestamp derivation, v5 prefix provenance, and offline v5 accounting
- the isolated four-by-four lecture definition, exact human intervals, twelve explicit negative pairs, shared naive/optimized video contract, strict malformed-output behavior, adversarial label-access denial, upload accounting, immutable stage hashes, and a deterministic offline eight-stage paper-eval workflow
- explicit profile selection, unchanged four-by-four defaults, the matched
  three-pair/four-event verified definition, and a deterministic offline
  end-to-end verified-profile workflow

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
