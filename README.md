# Multi-modal Data Systems

This repository now includes a first MMDS DSL implementation with:

- `Input`, `Map`, `Filter`, `Reduce`, and `Unnest`
- structured prompt expressions with `Record[...]` and `ForEach([...])`
- immutable operator-tree construction
- query parsing from restricted Python source
- query regeneration back to normalized Python
- local execution over `Iterable[dict]`
- `rule_optimizer`, `llm_optimizer`, and a Gemini prompt executor
- `udfs/` discovery for implemented `.py` UDFs and declared-only `.pyi` stubs

## Query shape

Queries use straight-line Python assignments:

```python
from mmds import Input, Map, Filter, Reduce, Unnest, Record, ForEach
from udfs.test_ops import add_bucket, summarize_group

docs = Input("docs")
mapped = Map(docs, add_bucket)
filtered = Filter(mapped, ["keep rows for ", Record["title"]])
expanded = Unnest(filtered, "tags", keep_empty=True)
output = Reduce(
    expanded,
    ["bucket"],
    ["summaries:\n", ForEach(["- ", Record["summary"], "\n"])],
    schema={"type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"]},
)
```

## Runtime model

- Prompt-based operators require an injected prompt executor.
- Prompt-based `Map` and `Reduce` require `schema=...` so the executor can request structured JSON output.
- `Filter` prompt execution expects a bare JSON boolean.
- Prompt lists can contain strings, `Record[...]` field references, and top-level `ForEach([...])` in `Reduce`.
- Function-based operators must use imported functions from `./udfs`.
- Inline lambdas are intentionally unsupported.
- Video fields are normal row fields. The Gemini executor treats any prompt value shaped like `{"type": "Video", ...}` as video input.

## Validation

Run the unit suite with:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m unittest discover -s tests -t .
```
