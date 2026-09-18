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

- operators: `Input`, `Map`, `Filter`, `Reduce`, `Unnest`, `Detect`, `Window`,
  `Coalesce`, `DropFields`, `VideoMap`, and `VideoMapEach`
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
- a conservative rule optimizer, the legacy whole-query LLM rewrite scaffold,
  and a typed directive rewriter over immutable plans
- `Detect` operator for frame-level YOLOE object detection on video fields
- programmatic `Window` and `Coalesce` operators for constructing padded
  source-time video views and merging overlapping candidate intervals
- bounded one-hop rewrite search with deterministic plan fingerprints,
  structured rejection records, and baseline retention
- optional measured execution for prompt calls, prompt latency, submitted video
  seconds, input rows, output rows, and end-to-end latency

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
- `Detect(data, video_field, classes, *, model="yoloe-11s-seg.pt", output_field="detections", name=None)`
- `Window(data, video_field, candidate_field, output_field, padding_time, name=None)`
- `Coalesce(data, group_by, field, *, name=None)`
- `DropFields(data, fields, *, name=None)`
- `VideoMap(data, spec, *, video_field, views_field, group_by, schema=None, ...)`
- `VideoMapEach(data, spec, *, video_field, views_field, group_by, schema=None, ...)`
- `VideoView(video, start, end)`
- `Record[...]`
- `ForEach([...])`
- `execute(plan_or_query, prompt_executor=None)`
- `execute_measured(plan_or_query, prompt_executor=None)`
- `GeminiPromptExecutor(...)`
- `load_query(source)` / `parse_query(source)`
- `render_query(plan_or_query)`
- `optimize(plan)` / `canonicalize(plan)`
- `rewrite_once(program, *, agent, directives)`
- `search_rewrites(program, *, agent, directives, max_candidates=8)`

### Logical Plan Model

The core model lives in [src/mmds/model.py](/Users/chanwutk/Documents/mmds/src/mmds/model.py).

- `DatasetExpr` is the immutable logical operator node.
- `PromptSpec` stores prompt-backed semantics as prompt parts plus optional output schema.
- `RecordPath` represents `Record["field"]...` references.
- `ForEachPrompt` represents repeated prompt expansion over grouped records.
- `ResolvedPrompt` is the execution-time prompt after all `Record[...]` references are resolved against data.
- `UdfSpec` stores a stable import path for a UDF.
- `DetectSpec` stores the parameters for a `Detect` node: `video_field`, `classes`, `model`, `output_field`.
- `WindowSpec` stores the parameters for a `Window` node: `video_field`,
  `candidate_field`, `output_field`, and `padding_time`.
- `DropFieldsSpec` stores the non-empty, unique fields removed by a
  `DropFields` node.
- `VideoMapSpec` stores a video field, candidate-view field, grouping fields,
  semantic map specification, padding, and hard view-count/duration budgets.
- `ViewBudgetSpec` is an internal physical specification introduced by video
  lowering.
- `Assignment` and `QueryProgram` represent a parsed query file.
- `MMDSValidationError` is the shared validation failure type.

`DatasetExpr` uses a unary tree shape today:

- `Input` has no source.
- `Map`, `Filter`, `Reduce`, `Unnest`, `Detect`, `Window`, `Coalesce`,
  `DropFields`, `VideoMap`, and `VideoMapEach`
  each have one `source`.

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

`Window`, `Coalesce`, `DropFields`, `VideoMap`, and `VideoMapEach` are accepted
by the restricted parser. `Detect` and the internal `ViewBudget` physical node
remain programmatic-only.

The canonical output variable is the last assignment in the file.

### Renderer

The renderer lives in [src/mmds/render.py](/Users/chanwutk/Documents/mmds/src/mmds/render.py).

It has two jobs:

- render a parsed `QueryProgram` back to normalized Python
- synthesize a normalized `QueryProgram` from a runtime-built `DatasetExpr`

Normalization behavior:

- emits the source-visible operators and prompt helpers actually used by the plan
- emits grouped `from udfs... import ...` lines for referenced UDFs
- uses double-quoted string literals
- renders structured prompt lists explicitly
- renders schemas as normalized Python literals
- assigns synthesized names like `source_docs`, `step_1`, `output` when rendering from a bare plan

This is semantic round-tripping, not source-fidelity round-tripping. Comments, whitespace, and original local variable names are not preserved unless they naturally match the normalized output.

`Window`, `Coalesce`, `DropFields`, `VideoMap`, and `VideoMapEach` plans render
to normalized Python. `Detect` and internal physical-only nodes do not.

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
- `Detect`: reads the video pointed to by `video_field`; if the value is a `VideoView`-shaped dict with `start`/`end`, wraps the source in a `VideoView`, runs YOLOE detection on the selected frames, and merges a detection list with absolute source-video `frame_idx` values into `output_field`
- `Window`: reads one `{start, end}` candidate interval per row, applies
  symmetric padding, clamps the start to zero, and stores a non-materialized
  `VideoView` descriptor in `output_field` while preserving the input row
- `Coalesce`: groups rows by `group_by`, sorts the mappings in `field` by
  `start`, merges overlapping or touching intervals, and emits one row per
  merged interval containing only the grouping fields and `field`
- `DropFields`: copies a row and explicitly removes the configured fields;
  missing fields are validation errors
- `VideoMap`: logical operator lowered to candidate expansion, padding,
  coalescing, budgeting, and one grouped prompt call over the selected views
- `VideoMapEach`: logical operator lowered through the same view planner and a
  prompt call for each selected view

`VideoMap` and `VideoMapEach` never materialize video files. Lowering constructs
`VideoView` values over the original source. It applies `max_views` and
`max_total_video_seconds` per group after coalescing; a final view is clipped if
needed to respect the duration budget.

`execute_measured` wraps the supplied prompt executor without changing operator
semantics. It returns rows plus an `ExecutionMetrics` snapshot containing
end-to-end time, prompt calls/time, submitted `VideoView` seconds, input rows,
and output rows. Provider token or monetary cost is not inferred when the
provider does not expose it.

Prompt execution flow:

1. Resolve `Record[...]` references against the current row or grouped rows.
2. Expand `ForEach([...])` over grouped rows for `Reduce`.
3. Produce a `ResolvedPrompt`.
4. Delegate to `PromptExecutor.execute(op_type, prompt_spec, resolved_prompt, payload, context)`.

`StaticPromptExecutor` exists for deterministic tests and local development.

Windowed video queries use two deterministic UDFs from
`udfs.temporal_ops`. `rebase_clip_events` converts `clip_events` from offsets
relative to the beginning of `clip` into source-time `events` by adding
`clip.start`. `reconcile_events` is used by a source-grouped `Reduce` to flatten
and sort source-time event collections. Interval merging remains the
responsibility of `Coalesce`; reconciliation only restores one result
collection per source. Keeping `clip_events` and `events` as separate fields
makes their coordinate systems explicit.

The lecture event localization example applies that contract as an explicit
O2-style plan:

1. A transcript-backed `Map` returns broad source-time candidate intervals.
2. `Unnest` creates one row per candidate.
3. `Window` pads candidates into lazy `VideoView` values.
4. `Coalesce` merges overlapping views for the same lecture and query.
5. A video-backed `Map` returns clip-relative verified event intervals.
6. A UDF-backed `Map` rebases those intervals to source time.
7. `Reduce` flattens and sorts results per lecture, then `Unnest` emits events.

Three lecture query variants read the same generic `data/lectures.jsonl` input:

- video-only localization over each complete lecture
- transcript-only localization over timestamped cues
- transcript-gated localization followed by windowed video verification

Each variant reduces and unnests its predictions into the same
`{lecture_id, events}` output shape so evaluation can compare plans directly.
The dataset is self-contained: each lecture row carries its timestamped
`{start, end, text}` transcript cues inline, so query execution performs no
transcript acquisition or preprocessing.

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

#### Directive Rewriter

The directive rewriter lives under
`src/mmds/optimizers/rewriter/`. It does not accept model-generated Python.
Instead, a `RewriteAgent` proposes a registered directive, one of the exact
matches offered by the system, and JSON-compatible parameters. Strict frozen
Pydantic models validate the parameters before deterministic directive code
changes the plan.

`program.output_expr` is the canonical rewrite root and the only operator-tree
representation. `NodePath` is an address through immutable `source` edges, not
a second node representation. `PlanEntry` binds an address to the existing
`DatasetExpr` and its derived field effects. A request-scoped `PlanIndex` owns
those entries and is shared by directive matching, lookup, and immutable
replacement. Replacing a node rebuilds only the path back to the output,
leaving the original plan unchanged. Rewritten assignments are regenerated as
normalized Python.

The initial directives are:

- `modality_substitution`: replace a video field reference with a transcript
  field reference while preserving the map prompt and schema
- `projection_before_map`: insert a text projection map, retarget the original
  map to the reserved intermediate field, and remove that field with
  `DropFields`
- `temporal_pushdown_joint`: create transcript candidates and use `VideoMap`
  to answer once over bounded lazy views
- `temporal_pushdown_per_view`: create transcript candidates, use
  `VideoMapEach`, rebase clip-relative events, and reconcile source-time events

Temporal agents provide only semantic parameters: the video, transcript, and
query fields plus the candidate-generation instruction. Grouping is derived
deterministically from ancestor grouping operators, configured identity fields,
and non-video fields used by the matched prompt. Padding, view-count limits,
and duration limits are experiment settings configured on the directive rather
than model-generated plan structure.

Before model selection, the engine builds one ephemeral `RewriteContext` from
the existing `DatasetExpr` plan and its file-backed inputs. The query portion is
a flat path-addressed summary containing operator kinds, prompt templates,
field effects, grouping, UDF names, and output schemas. The dataset portion
profiles field shapes and semantic roles from at most 32 JSON/JSONL rows while
never including row values. Evaluation-looking fields such as ground truth,
labels, and annotations are excluded unless the original query reads them.
Missing input files are reported as unavailable context rather than blocking a
structurally valid rewrite; existing malformed inputs fail explicitly.

`RewriteContext` is model input only, not another plan representation. The
system supplies a fixed rewrite policy that requests lower-cost or lower-latency
alternatives while preserving query semantics, outputs, inputs, modalities,
and temporal coordinates. Typed directive rewriting therefore does not require
a caller-authored, workload-specific objective.

The full context is not serialized into both model calls. Selection receives a
compact plan outline, detailed task templates only for matched nodes, relevant
field roles, and directive choices grouped by node. Instantiation receives only
the selected nodes, compatible fields, and simplified parameter contracts.
Strict Pydantic validation remains local and uses the complete schemas.

Expected rewrite failures use `MMDSRewriteError`. `rewrite_once` surfaces them;
`search_rewrites` records them as `RewriteRejection` values while allowing
other candidates to continue. Unexpected exceptions are never swallowed.

The first search implementation is intentionally bounded to one rewrite hop.
It always retains the original plan, applies an explicit candidate limit, and
deduplicates rewritten plans using normalized-code fingerprints. A
`StaticRewriteAgent` supports deterministic tests and replay.
`ModelRewriteAgent` uses progressive disclosure: one model call selects from
applicable directive/match pairs using the compact selection context, and a
second call sees selected-node context plus parameter contracts only for those
selections.
Prompts and raw responses are stored in `RewriteTrace`.
The legacy whole-query LLM rewriter remains available separately for
compatibility.

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
- model output for directive rewriting is untrusted structured data and cannot
  directly replace Python source
- fields created only by rewrites use the reserved `_mmds_` prefix and are
  removed explicitly when they are no longer part of the observable result
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
- `Detect` behavior, including `VideoView` clip slicing and absolute-frame detection indices
- video utility behavior for direct downloads, platform downloads via `yt-dlp`, and `VideoView` iteration
- parser validation for unsupported Python and invalid prompt forms
- optimizer result preservation and LLM rewrite validation
- immutable node-path lookup and replacement
- automatic plan summarization, dataset shape profiling, semantic field roles,
  evaluation-field exclusion, and missing/malformed input handling
- typed directive parameters, applicability, structural output, and rejection paths
- baseline-preserving, deduplicated, bounded one-hop search
- two-call model selection and typed parameter instantiation, including invalid
  and unoffered response rejection
- `DropFields` execution and source round trips
- logical video lowering, view budgets, joint execution, per-view timestamp
  rebasing, and reconciliation
- execution metric collection without live provider calls
- Gemini prompt compilation for video URI and uploaded local file inputs
- UDF catalog discovery for `.py` and `.pyi`

Primary verification command:

```bash
uv run --locked python -m unittest discover -s tests -t .
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
