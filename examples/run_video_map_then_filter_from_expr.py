from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
QUERY_PATH = ROOT / "examples" / "video_map_then_filter.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import GeminiPromptExecutor, execute  # noqa: E402


def main() -> None:
    os.chdir(ROOT)
    spec = importlib.util.spec_from_file_location("video_map_then_filter", QUERY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load query module from {str(QUERY_PATH)!r}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = execute(module.output, prompt_executor=GeminiPromptExecutor())
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
