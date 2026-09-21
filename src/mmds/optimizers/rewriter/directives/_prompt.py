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


def replace_record_path(
    parts: Iterable[PromptPart],
    old: RecordPath,
    new: RecordPath,
) -> tuple[PromptPart, ...]:
    """Replace one exact record reference throughout a structured prompt."""

    replaced: list[PromptPart] = []
    for part in parts:
        if part == old:
            replaced.append(new)
        elif isinstance(part, ForEachPrompt):
            replaced.append(
                ForEachPrompt(parts=replace_record_path(part.parts, old, new))
            )
        else:
            replaced.append(part)
    return tuple(replaced)


def format_record_paths(paths: Iterable[RecordPath]) -> str:
    rendered = sorted({".".join(path.path) for path in paths if path.path})
    return ", ".join(rendered) if rendered else "no record fields"
