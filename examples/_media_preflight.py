"""Fail early when an example's local media has not been installed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _records(path: Path) -> list[Any]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def _local_media_paths(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        media_type = str(value.get("type", "")).lower()
        path = value.get("path")
        if media_type in {"image", "video", "videoview"} and isinstance(path, str):
            paths.add(path)
        for child in value.values():
            paths.update(_local_media_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.update(_local_media_paths(child))
    return paths


def require_local_media(output: Any, root: Path) -> None:
    """Check local media referenced by the plan's committed input manifests."""
    missing: set[str] = set()
    for node in output.walk_postorder():
        if node.kind != "input" or node.input_path is None:
            continue
        manifest = Path(node.input_path)
        if not manifest.is_absolute():
            manifest = root / manifest
        if not manifest.is_file() or manifest.suffix not in {".json", ".jsonl"}:
            continue
        for record in _records(manifest):
            for media_path in _local_media_paths(record):
                candidate = Path(media_path)
                if not candidate.is_absolute():
                    candidate = root / candidate
                if not candidate.is_file():
                    missing.add(media_path)

    if missing:
        formatted = "\n".join(f"  - {path}" for path in sorted(missing))
        raise SystemExit(
            "Required local media is missing:\n"
            f"{formatted}\n"
            "See data/README.md for dataset acquisition and setup instructions."
        )
