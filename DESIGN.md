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
    schema={"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
)
output = Reduce(
    mapped,
    "_all",
    ["Summaries:\n", ForEach(["- ", Record["summary"], "\n"])],
    schema={"type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"]},
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

- operators: `Input`, `Map`, `Filter`, `Reduce`, `Unnest`
- file-backed `Input(...)` roots over `.json` and `.jsonl`
- prompt-backed semantics as either:
  - a plain string
  - a structured prompt list made of strings, `Record[...]`, and Reduce-level `ForEach([...])`
- function-backed semantics as imported UDFs from `udfs.*`
- prompt-backed `Map` and `Reduce` with explicit JSON-schema output contracts
- prompt-backed `Filter` returning a bare JSON boolean
- source parsing for straight-line top-level assignments only
- plan rendering back to normalized Python code
- local execution with an injected prompt executor
- Gemini-backed prompt execution with executor-side video translation
- UDF discovery from `.py` and `.pyi`
- a conservative rule optimizer and a validation-heavy LLM optimizer scaffold

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
- `Assignment` and `QueryProgram` represent a parsed query file.
- `MMDSValidationError` is the shared validation failure type.

`DatasetExpr` uses a unary tree shape today:

- `Input` has no source.
- `Map`, `Filter`, `Reduce`, and `Unnest` each have one `source`.

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
- `Map` and `Reduce` prompt specs must include `schema={...}`
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

Local execution lives in [src/mmds/execution.py](/Users/chanwutk/Documents/mmds/src/mmds/execution.py).

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

Gemini execution lives in [src/mmds/gemini_executor.py](/Users/chanwutk/Documents/mmds/src/mmds/gemini_executor.py).

Design rule:

- the DSL does not introduce a `Video(...)` wrapper
- media stays in row fields as regular data values
- provider-specific translation happens inside executors

Current video detection rule:

- any resolved prompt value shaped like `{"type": "Video", ...}` is treated as video input

Supported video payload forms:

- `{"type": "Video", "uri": "..."}`
- `{"type": "Video", "path": "/local/file.mp4"}`
- `{"type": "Video", "bytes": b"...", "mime_type": "video/mp4"}`

Optional metadata keys:

- `start_offset`
- `end_offset`
- `fps`

Gemini executor behavior:

- uploads local files through Gemini’s Files API when needed
- waits for uploaded files to become active
- converts structured prompts into Gemini content parts
- preserves video fields as video parts instead of stringifying them
- requests JSON output using Gemini structured output config
- uses a bare boolean schema for `Filter`
- uses the operator-provided schema for `Map` and `Reduce`

This design follows Gemini’s official support for video inputs and structured JSON output. Sources used to align the design and implementation:

- [Gemini video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Gemini SDK downloads](https://ai.google.dev/gemini-api/docs/downloads)

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

The rule optimizer lives in [src/mmds/rule_optimizer.py](/Users/chanwutk/Documents/mmds/src/mmds/rule_optimizer.py).

Current behavior is intentionally conservative:

- recursively rebuild the tree
- structurally deduplicate equivalent nodes through memoization

It does not yet reorder operators, fold operators, infer safety, or reason about prompt/UDF semantics.

#### LLM Optimizer

The LLM rewrite scaffold lives in [src/mmds/llm_optimizer.py](/Users/chanwutk/Documents/mmds/src/mmds/llm_optimizer.py).

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

## Validation and Tests

Tests live in [tests/test_mmds.py](/Users/chanwutk/Documents/mmds/tests/test_mmds.py).

The current suite covers:

- parse/render/parse equivalence for structured prompts
- file-backed execution for `.json` and `.jsonl`
- rendering from runtime-built plans
- execution for UDF-backed and prompt-backed queries
- `Record[...]` resolution and `ForEach([...])` expansion
- `Unnest` behavior on scalar, empty, and missing values
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
