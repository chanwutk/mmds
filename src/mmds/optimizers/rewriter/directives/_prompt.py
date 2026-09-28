from __future__ import annotations

from collections.abc import Iterable

from ....model import ForEachPrompt, PromptPart, PromptSpec, RecordPath
from ..core import PlanEntry, PlanIndex


def prompt_map_entries(index: PlanIndex) -> tuple[PlanEntry, ...]:
    """Return prompt-backed Map nodes in output-to-input order."""

    return tuple(
        entry
        for entry in index.entries
        if entry.node.kind == "map" and isinstance(entry.node.spec, PromptSpec)
    )


def record_paths(parts: Iterable[PromptPart]) -> tuple[RecordPath, ...]:
    """Collect record references from a structured prompt."""

    paths: list[RecordPath] = []
    for part in parts:
        if isinstance(part, RecordPath):
            paths.append(part)
        elif isinstance(part, ForEachPrompt):
            paths.extend(record_paths(part.parts))
    return tuple(paths)


def format_record_paths(paths: Iterable[RecordPath]) -> str:
    rendered = sorted({".".join(path.path) for path in paths if path.path})
    return ", ".join(rendered) if rendered else "no record fields"


def prompt_from_fields(
    instruction: str,
    paths: Iterable[RecordPath],
) -> tuple[PromptPart, ...]:
    """Build a structured prompt from one instruction and named row fields."""

    parts: list[PromptPart] = [instruction]
    seen: set[RecordPath] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        parts.extend((f"\n{'.'.join(path.path)}:\n", path))
    return tuple(parts)
