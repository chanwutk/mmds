from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
QUERY_PATH = ROOT / "examples" / "video_map_then_filter.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import GeminiPromptExecutor, execute, parse_query  # noqa: E402


def main() -> None:
    os.chdir(ROOT)
    query_text = QUERY_PATH.read_text(encoding="utf-8")
    program = parse_query(query_text)
    rows = execute(program, prompt_executor=GeminiPromptExecutor())
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
