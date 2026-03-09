# Examples

This directory contains MMDS query examples.

- `video_map_then_filter.py`: map over video rows with a structured prompt, then filter the mapped rows with a second prompt.

Video fields are regular record fields. The Gemini executor treats values shaped like `{"type": "Video", ...}` as video parts.

`Input(...)` takes a `.json` or `.jsonl` file path directly, so queries do not need a separate data catalog.
