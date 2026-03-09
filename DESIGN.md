# MMDS Design

`DESIGN.md` is a living document. Any change that affects the DSL surface, logical plan model, execution semantics, optimizer behavior, or UDF contract must update this file in the same change.

## Overview

MMDS is a Python-first DSL for semantic data workflows. A query is written as ordinary Python assignments:

```python
from mmds import Input, Map, Filter, Reduce, Unnest
from udfs.test_ops import add_bucket, summarize_group

docs = Input("docs")
mapped = Map(docs, add_bucket)
filtered = Filter(mapped, "keep rows with useful content")
expanded = Unnest(filtered, "tags", keep_empty=True)
output = Reduce(expanded, ["bucket"], summarize_group)
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
- semantic specs as either prompt strings or imported UDF functions from `udfs.*`
- source parsing for straight-line top-level assignments only
- plan rendering back to normalized Python code
- local execution with an injected prompt executor
- UDF discovery from `.py` and `.pyi`
- a conservative rule optimizer and a validation-heavy LLM optimizer scaffold

The current implementation intentionally does not support:

- inline lambdas
- nested functions or callables outside `udfs.*`
- loops, conditionals, comprehensions, classes, or arbitrary Python control flow in query files
- joins, sorts, projections, or cost-based optimization
- automatic `.py` implementation synthesis from `.pyi`
- provider-specific LLM integration

## Architecture

### Public DSL

The main entrypoints are exported from [src/mmds/__init__.py](/Users/chanwutk/Documents/mmds/src/mmds/__init__.py):

- `Input(name)`
- `Map(data, spec, *, name=None)`
- `Filter(data, spec, *, name=None)`
- `Reduce(data, group_by, reducer, *, name=None)`
- `Unnest(data, field, *, keep_empty=False, name=None)`
- `execute(plan_or_query, inputs, prompt_executor=None)`
- `load_query(source)` / `parse_query(source)`
- `render_query(plan_or_query)`
- `optimize(plan)` / `canonicalize(plan)`

### Logical Plan Model

The core model lives in [src/mmds/model.py](/Users/chanwutk/Documents/mmds/src/mmds/model.py).

- `DatasetExpr` is the immutable logical operator node.
- `PromptSpec` stores prompt-backed semantics.
- `UdfSpec` stores a stable import path for a UDF.
- `Assignment` and `QueryProgram` represent a parsed query file.
- `MMDSValidationError` is the shared validation failure type.

`DatasetExpr` uses a unary tree shape today:

- `Input` has no source.
- `Map`, `Filter`, `Reduce`, and `Unnest` each have one `source`.

That shape is sufficient for the first operator set and keeps rendering and execution simple. If future operators introduce multiple inputs, `DatasetExpr` will need a general child list instead of a single `source`.

### Parser

The parser lives in [src/mmds/parser.py](/Users/chanwutk/Documents/mmds/src/mmds/parser.py).

It accepts only:

- an optional module docstring
- `from mmds import ...`
- `from udfs... import ...`
- top-level assignments where the right-hand side is a direct DSL operator call

Parser rules:

- sources must reference a previously assigned variable
- semantic specs must be a string literal or an imported UDF name
- UDF import aliasing is rejected
- only absolute imports are allowed
- `Reduce.group_by` must be a string or list/tuple of strings
- `Unnest.keep_empty` must be a literal boolean

The canonical output variable is the last assignment in the file.

### Renderer

The renderer lives in [src/mmds/render.py](/Users/chanwutk/Documents/mmds/src/mmds/render.py).

It has two jobs:

- render a parsed `QueryProgram` back to normalized Python
- synthesize a normalized `QueryProgram` from a runtime-built `DatasetExpr`

Normalization behavior:

- always emits `from mmds import Input, Map, Filter, Reduce, Unnest`
- emits grouped `from udfs... import ...` lines for referenced UDFs
- uses double-quoted string literals
- assigns synthesized names like `source_docs`, `step_1`, `output` when rendering from a bare plan

This is semantic round-tripping, not source-fidelity round-tripping. Comments, whitespace, and original local variable names are not preserved unless they naturally match the normalized output.

### Execution

Local execution lives in [src/mmds/execution.py](/Users/chanwutk/Documents/mmds/src/mmds/execution.py).

Execution input model:

- each dataset is provided as `Mapping[str, Iterable[Mapping[str, Any]]]`
- each row is copied into a mutable `dict`

Operator semantics:

- `Input(name)`: reads `inputs[name]`
- `Map`: applies prompt/UDF to one row and merges returned fields into that row
- `Filter`: applies prompt/UDF to one row and keeps rows whose result is truthy
- `Reduce`: groups rows by the configured fields, calls the reducer once per group, and merges returned aggregate fields with the group key fields
- `Unnest`: expands one field; lists and tuples explode into multiple rows, scalars pass through unchanged, and missing/empty values produce no row unless `keep_empty=True`

Prompt execution is delegated through `PromptExecutor.execute(op_type, spec_text, payload, context)`.

`StaticPromptExecutor` exists for deterministic tests and local development.

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
6. Reject rewrites that change `Input(...)` roots.
7. Return normalized rendered Python.

The LLM optimizer is currently a controlled interface, not a production optimizer. It is designed to make later provider integration safe by validating every rewrite against the same parser used elsewhere.

## Invariants

The following invariants are part of the current design and should not change silently:

- the supported query language is a strict Python subset
- prompts and UDFs are the only valid semantic specs
- UDFs must come from `udfs.*`
- operator trees are immutable
- the last assignment is the output unless a future explicit sink is added
- rendered queries are normalized, not source-exact
- prompt-backed execution always requires an injected executor
- `.pyi` discovery does not imply executability

## Validation and Tests

Tests live in [tests/test_mmds.py](/Users/chanwutk/Documents/mmds/tests/test_mmds.py).

The current suite covers:

- parse/render/parse equivalence
- rendering from runtime-built plans
- execution for UDF-backed and prompt-backed queries
- `Unnest` behavior on scalar, empty, and missing values
- parser validation for unsupported Python and non-UDF callables
- optimizer result preservation and LLM rewrite validation
- UDF catalog discovery for `.py` and `.pyi`

Primary verification command:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m unittest discover -s tests -t .
```

## Near-Term Extension Points

Likely next areas of change:

- more operators beyond the current unary set
- richer multimodal value types instead of plain dict rows
- explicit query sinks or named outputs
- more aggressive rule rewrites once semantic safety rules exist
- `.pyi`-driven UDF synthesis
- provider-backed prompt execution and LLM optimization

Any of those changes should update this document alongside the code.
