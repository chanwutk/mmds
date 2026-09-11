"""CLI for immutable cross-modal lecture paper-evaluation profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.experiments.common import ExperimentDataError

from .catalog import PROFILES, catalog_payload, get_profile
from .workflow import (
    amend_ground_truth,
    compare,
    download_sources,
    evaluate,
    materialize_clips,
    prepare,
    run_candidates,
    run_naive,
    run_transcript_only,
    run_transcript_video,
    status,
    transcribe_sources,
)


EXECUTION_COMMANDS = {
    "download",
    "transcribe",
    "run-naive",
    "run-transcript-only",
    "run-candidates",
    "materialize-clips",
    "run-transcript-video",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="frozen-4x4",
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "catalog",
        "status",
        "prepare",
        "amend-ground-truth",
        "evaluate",
        "compare",
        *sorted(EXECUTION_COMMANDS),
    ):
        subparser = subparsers.add_parser(command)
        if command in EXECUTION_COMMANDS:
            subparser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = str(args.command)
    profile = get_profile(str(args.profile))
    root = args.root or profile.default_root
    if command in EXECUTION_COMMANDS and not args.execute:
        _print(
            {
                "command": command,
                "profile": profile.profile_id,
                "experiment": profile.experiment_name,
                "preview_only": True,
                "message": (
                    "No download, transcription, model call, or media encoding was run. "
                    "Add --execute only after reviewing the frozen code, catalog, and status."
                ),
            }
        )
        return 0
    try:
        if command == "catalog":
            result = catalog_payload(profile)
        elif command == "status":
            result = status(root, profile=profile)
        elif command == "download":
            result = download_sources(root, profile=profile)
        elif command == "transcribe":
            result = transcribe_sources(root, profile=profile)
        elif command == "prepare":
            result = prepare(root, profile=profile)
        elif command == "amend-ground-truth":
            result = amend_ground_truth(root, profile=profile)
        elif command == "run-naive":
            result = run_naive(root, profile=profile, env_file=args.env_file)
        elif command == "run-transcript-only":
            result = run_transcript_only(
                root, profile=profile, env_file=args.env_file
            )
        elif command == "run-candidates":
            result = run_candidates(
                root, profile=profile, env_file=args.env_file
            )
        elif command == "materialize-clips":
            result = materialize_clips(root, profile=profile)
        elif command == "run-transcript-video":
            result = run_transcript_video(
                root, profile=profile, env_file=args.env_file
            )
        elif command == "evaluate":
            result = evaluate(root, profile=profile)
        elif command == "compare":
            result = compare(root, profile=profile)
        else:  # pragma: no cover - argparse enforces the choices
            raise AssertionError(command)
    except (ExperimentDataError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print(result)
    return 0


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
