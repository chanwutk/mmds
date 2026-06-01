# MMDS — Multi-modal Data Systems

> A Python DSL and execution engine for **semantic, multi-modal (video + text) data workflows**, with operators backed by large multi-modal models (e.g. Gemini) or ordinary Python functions.

MMDS lets you express a pipeline over videos and documents as a few lines of restricted
Python. Each operator is driven either by an **LLM prompt** or by an **imported Python
function (UDF)**, and the same query exists simultaneously in three equivalent forms —
editable Python text, an immutable logical plan, and a locally executable runtime. That
round-trippable design is what makes MMDS a research substrate for **semantic query
rewriting** (see [Research context](#research-context)).

```python
# examples/wildlife_species.py — watch each clip, ask what animals appear, one row per species
from mmds import Input, Map, Record, Unnest

input_data = Input("data/animals.jsonl")

mapped = Map(
    input_data,
    [
        "Watch this video clip.\n",
        "What kind of animals do you see in the video clip?"
        "video clips: ",
        Record["video"],
    ],
    schema={"animal_types": {"type": "array", "items": {"type": "string"}}},
)

output = Unnest(mapped, "animal_types")
```

---

## Quick start

> **Full setup is in [GET_START.md](GET_START.md).** This is the 60-second version.
> Supported platforms: **Linux, Apple-Silicon macOS, Windows**. Intel macOS is **not**
> supported (a transitive `torch` dependency ships no Intel-mac wheel — see
> [Project status](#project-status)).

```bash
# 1. Install uv (https://docs.astral.sh/uv/), then build the environment:
uv sync

# 2. Smoke-test the install (no API key, no network):
PYTHONPATH=src:. ./.venv/bin/python -m unittest discover -s tests -t .

# 3. Run your first real query (needs a free Gemini API key):
export GEMINI_API_KEY="…"        # https://aistudio.google.com/apikey
./run examples/wildlife_species.py
```

---

## Examples

All code below is taken verbatim from the [`examples/`](examples/) directory.

### Aggregate with Reduce + ForEach

[`examples/wildlife_species_count.py`](examples/wildlife_species_count.py) extends the
example above — the same `Map`, then group by species and ask the model to count the clips
per group. Inside a `Reduce`, per-row fields are reached through `ForEach([...])`:

```python
unnested = Unnest(mapped, "animal_types")

output = Reduce(
    unnested,
    "animal_types",
    [
        "The following video clips all contain the same species of animal:\n",
        ForEach(["- species: ", Record["animal_types"], "\n"]),
        "How many clips contain this species? Reply with just the count.",
    ],
    schema={"count": "integer"},
)
```

More examples: a [Map → Filter video pipeline](examples/video_map_then_filter.py) and
local [YOLOE object detection](examples/wildlife_detection.py) (no API key required).

---

## Operators

A query is a sequence of top-level assignments; the **last assignment is the output**.

| Operator | Signature | What it does |
|----------|-----------|--------------|
| `Input`  | `Input(path)` | Load rows from a `.json` (top-level list) or `.jsonl` file. |
| `Map`    | `Map(data, spec, *, schema=…)` | Run the prompt/UDF once per row; **merge** returned fields into the row. |
| `Filter` | `Filter(data, spec)` | Keep rows where the prompt/UDF result is truthy. |
| `Reduce` | `Reduce(data, group_by, reducer, *, schema=…)` | Group rows, run the reducer once per group, merge the aggregate with the group key. |
| `Unnest` | `Unnest(data, field, *, keep_empty=False)` | Explode a list/tuple field into one row per item. |
| `Detect` | `Detect(data, video_field, classes, …)` | Frame-level YOLOE object detection on a video field (programmatic-only; not parsed/rendered as DSL text). |

Helpers: `Record["field"]["nested"]` references a row field inside a prompt; `ForEach([...])`
repeats a prompt fragment once per grouped row (only at the top level of a `Reduce`);
`VideoView(video, start, end)` is a seek-based clip-range view used by `Detect`.

---

## How semantics are attached

Every semantic operator is driven by exactly one of two things — never arbitrary inline code:

- **Prompt-backed.** A string, or a list mixing strings, `Record[...]` references, and
  (in `Reduce`) `ForEach([...])`. Prompt-backed `Map` and `Reduce` **require** `schema=`
  (a concise output-field map such as `{"summary": "string", "score": "number"}`);
  `Filter` returns a bare boolean. Prompt execution is delegated to an injected
  **`PromptExecutor`** — `GeminiPromptExecutor` for real runs, `StaticPromptExecutor` for
  deterministic tests — so the core never hard-codes a provider.

  ```python
  from mmds import execute, GeminiPromptExecutor
  rows = execute(output, prompt_executor=GeminiPromptExecutor())
  ```

- **UDF-backed.** A plain function imported from the `udfs.*` package and passed directly:

  ```python
  from mmds import Input, Map
  from udfs.test_ops import add_bucket
  output = Map(Input("data/rows.json"), add_bucket)   # no schema=, no executor needed
  ```

  Inline lambdas and nested functions are **rejected on purpose** — UDFs must have a stable
  import path so plans stay serializable, renderable, and analyzable.

---

## Running queries & tests

```bash
# Run a query module's `output` expression through Gemini (./run wraps this):
./run examples/wildlife_species.py
PYTHONPATH=src:. ./.venv/bin/python examples/run_expr.py examples/wildlife_species.py

# Parse + run a query written as DSL *text* (restricted-Python source):
PYTHONPATH=src:. ./.venv/bin/python examples/run_text.py path/to/query.py

# Run the full test suite (the project's primary verification command):
PYTHONPATH=src:. ./.venv/bin/python -m unittest discover -s tests -t .
```

Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for any prompt-backed run; `Detect` and
UDF-only queries need no key. The Gemini examples pass their public YouTube `source` URL
straight to Gemini (which ingests it server-side — no local download). The local `Detect`
path instead downloads videos via `yt-dlp` and caches them under `~/.cache/mmds/videos/`.

---

## Architecture

The central idea: **one query, three equivalent forms**, with clean converters between them.

```
        write / edit
   ┌──────────────────────┐
   │   Python DSL text     │   restricted Python: imports + top-level assignments
   └──────────┬───────────┘
   parse_query │   ▲ render_query  (normalized, not source-exact)
              ▼   │
   ┌──────────────────────┐
   │     QueryProgram      │   assignment sequence + chosen output variable
   └──────────┬───────────┘
              │ .output_expr
              ▼
   ┌──────────────────────┐   program_from_plan
   │    DatasetExpr plan    │◀──────────────────── built directly in Python
   │   (immutable, unary)   │──▶ optimize / canonicalize  (rule + LLM rewriters)
   └──────────┬───────────┘
   execute(plan, prompt_executor)
              ▼
   ┌──────────────────────┐
   │    rows: list[dict]    │   local interpreter over Iterable[dict]
   └──────────────────────┘
```

Components (all under [`src/mmds/`](src/mmds/)):

- **`dsl.py`** — the public constructors (`Input`, `Map`, …) that build plan nodes.
- **`model.py`** — immutable `DatasetExpr` nodes, `PromptSpec`/`UdfSpec`, `Record`/`ForEach`.
- **`parser.py`** — turns restricted-Python text into a validated `QueryProgram`; the
  language boundary (unsupported syntax fails explicitly).
- **`render.py`** — renders a plan/program back to *normalized* Python.
- **`execution/`** — the local interpreter; `execution/llm/gemini.py` is the Gemini executor;
  `execution/ops/` holds per-operator logic including `Detect`.
- **`optimizers/rewriter/`** — `rule.py` (conservative structural canonicalization) and
  `agent.py` (validation-heavy LLM rewrite scaffold).
- **`udf_catalog.py`** — discovers UDFs from `udfs/*.py` (implemented) and `*.pyi` (declared-only).

**[DESIGN.md](DESIGN.md) is the authoritative architecture document** — read it before
making changes. **[AGENTS.md](AGENTS.md)** records the contribution rules and invariants.

---

## Research context

MMDS is the implementation substrate for ongoing research on **query rewrite for
multi-modal data systems** (see [`proposal.tex`](proposal.tex)). The motivating problem:
extracting insight from unstructured video + text increasingly means combining modalities,
and large multi-modal models can do it but are costly and unreliable to author by hand.

The project explores a new class of optimization — **semantic rewrite** — where a query is
rewritten to *change its semantics while preserving the original intent* (for example,
rewriting a query over video into an equivalent query over the text associated with that
video). This is why the codebase invests so heavily in a small, analyzable DSL with
clean text ↔ plan round-tripping and pluggable rewriters: those are the levers a rule
engine or LLM agent needs to rewrite queries safely. The data model extends
[DocETL](https://github.com/ucbepic/docetl) with multi-modal types (`Video`, `VideoView`).

---

## Repository layout

```
src/mmds/        core DSL, plan model, parser, renderer, execution, optimizers
udfs/            user-defined functions (.py = implemented, .pyi = declared-only)
examples/        runnable query examples + run_expr.py / run_text.py drivers
data/            small JSON/JSONL fixtures used by the examples
tests/           unit tests (hermetic — mock cv2/yt-dlp/YOLOE)
DESIGN.md        authoritative architecture document
AGENTS.md        contribution rules and invariants
GET_START.md     setup + first-run onboarding guide
proposal.tex     research proposal (semantic query rewrite)
```

---

## Project status

Early-stage **research prototype** — APIs and semantics may change.

- **Supported platforms:** Linux (x86_64 / aarch64), Apple-Silicon macOS, Windows.
  **Intel macOS is unsupported** because `ultralytics` pulls in a `torch` version with no
  Intel-mac wheel; `uv sync` will fail there.
- **Implemented:** `Input`/`Map`/`Filter`/`Reduce`/`Unnest`/`Detect`, structured prompts,
  UDFs, local execution, Gemini prompt execution, conservative rule + LLM rewriters,
  `.py`/`.pyi` UDF discovery.
- **Not yet supported:** inline lambdas; loops/conditionals/comprehensions/classes in
  queries; joins/sorts/projections; cost-based optimization; `.pyi` → `.py` synthesis;
  nested `ForEach`; provider-specific media syntax in the DSL. (See DESIGN.md for the full list.)

---

## Learn more

- **[GET_START.md](GET_START.md)** — set up the project and run your first example.
- **[DESIGN.md](DESIGN.md)** — architecture, operator semantics, invariants (source of truth).
- **[AGENTS.md](AGENTS.md)** — required workflow and architectural constraints for contributors.
