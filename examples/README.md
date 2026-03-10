# Examples

This directory contains MMDS query examples.

- `video_map_then_filter.py`: map over video rows with a structured prompt, then filter the mapped rows with a second prompt.

Video fields are regular record fields. The Gemini executor treats values whose `type` is case-insensitively equal to `"video"` as video parts. The canonical shape is still `{"type": "Video", ...}`.
Prompt-backed `Map` and `Reduce` examples use the concise schema form, for example `schema={"summary": "string"}`.

`Input(...)` takes a `.json` or `.jsonl` file path directly, so queries do not need a separate data catalog.
