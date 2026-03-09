# Multi-modal Data Systems

This repository now includes a first MMDS DSL implementation with:

- `Input`, `Map`, `Filter`, `Reduce`, and `Unnest`
- immutable operator-tree construction
- query parsing from restricted Python source
- query regeneration back to normalized Python
- local execution over `Iterable[dict]`
- `rule_optimizer` and `llm_optimizer` scaffolding
- `udfs/` discovery for implemented `.py` UDFs and declared-only `.pyi` stubs

## Query shape

Queries use straight-line Python assignments:

```python
from mmds import Input, Map, Filter, Reduce, Unnest
from udfs.test_ops import add_bucket, summarize_group

docs = Input("docs")
mapped = Map(docs, add_bucket)
filtered = Filter(mapped, "keep large rows")
expanded = Unnest(filtered, "tags", keep_empty=True)
output = Reduce(expanded, ["bucket"], summarize_group)
```

## Runtime model

- Prompt-based operators require an injected prompt executor.
- Function-based operators must use imported functions from `./udfs`.
- Inline lambdas are intentionally unsupported.

## Validation

Run the unit suite with:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m unittest discover -s tests -t .
```
