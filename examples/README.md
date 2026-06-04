# Examples

This directory contains MMDS query examples.

## Running examples

Run any example with `examples/run_expr.py` (or `./run` from the repo root):

```bash
uv run python examples/run_expr.py examples/detect_filter_bears.py
./run examples/video_map_then_filter.py
```

`run_expr.py` loads the query module, calls `execute(module.output, prompt_executor=GeminiPromptExecutor())`, and prints JSON.

- **Local `Detect` examples** (YOLOE): no API key required. Gemini is not used unless the plan includes prompt-backed `Map` / `Filter` / `Reduce`.
- **Prompt-backed examples** (TwelveLabs-style, wildlife Gemini): need a configured Gemini API key.

`examples/run_detect.py` is optional—the same Detect pipelines work through `run_expr.py` without calling Gemini.

## Example queries

- `video_map_then_filter.py`: map over video rows with a structured prompt, then filter the mapped rows with a second prompt.
- `detect_filter_bears.py`: local YOLOE `Detect` → `Filter` (any bear box).
- `detect_filter_bears_high_confidence.py`: `Detect` → `Map` (prune) → `Filter` (bear boxes with confidence >= 0.5 only).
- `detect_multi_species.py`: `Detect` → `Unnest` (bear, deer, bird).

## Notes

Video fields are regular record fields. The Gemini executor treats values whose `type` is case-insensitively equal to `"video"` as video parts. The canonical shape is still `{"type": "Video", ...}`.
Prompt-backed `Map` and `Reduce` examples use the concise schema form, for example `schema={"summary": "string"}`.

`Input(...)` takes a `.json` or `.jsonl` file path directly, so queries do not need a separate data catalog.
