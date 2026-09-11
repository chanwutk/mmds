"""Run the paper-aligned lecture localization baseline, O1, or O2 query."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mmds import ExecutionContext, GeminiPromptExecutor, execute

from examples.lecture_event_localization import (
    build_transcript_only_query,
    build_transcript_to_video_query,
    build_video_only_query,
)


BUILDERS = {
    "video-only": build_video_only_query,
    "transcript-only": build_transcript_only_query,
    "transcript-to-video": build_transcript_to_video_query,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Lecture/query .json or .jsonl")
    parser.add_argument("--mode", choices=tuple(BUILDERS), required=True)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("output/lecture_event_localization"),
        help="Execution-local directory used by materialized View",
    )
    parser.add_argument("--model", help="Optional Gemini model override")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    plan = BUILDERS[args.mode](args.input)
    context = ExecutionContext(
        workspace=args.workspace,
        max_workers=args.max_workers,
    )
    executor = (
        GeminiPromptExecutor(model=args.model)
        if args.model
        else GeminiPromptExecutor()
    )
    rows = execute(plan, prompt_executor=executor, context=context)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "rows": rows,
                "operator_stats": [asdict(value) for value in context.stats.operators],
                "view_stats": [asdict(value) for value in context.stats.views],
                "provider_stats": [
                    asdict(value) for value in context.stats.provider_calls
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
