from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import GeminiPromptExecutor, execute, parse_query  # noqa: E402
from examples._media_preflight import require_local_media  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <query_file>", file=sys.stderr)
        sys.exit(1)
    query_path = Path(sys.argv[1])
    os.chdir(ROOT)
    query_text = query_path.read_text(encoding="utf-8")
    program = parse_query(query_text)
    require_local_media(program.output_expr, ROOT)
    rows = execute(program, prompt_executor=GeminiPromptExecutor())
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
