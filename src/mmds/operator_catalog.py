from __future__ import annotations

from dataclasses import dataclass

from .model import OperatorKind


@dataclass(frozen=True)
class OperatorDefinition:
    name: str
    kind: OperatorKind
    source_visible: bool = True


OPERATOR_DEFINITIONS: tuple[OperatorDefinition, ...] = (
    OperatorDefinition("Input", "input"),
    OperatorDefinition("Map", "map"),
    OperatorDefinition("Filter", "filter"),
    OperatorDefinition("Reduce", "reduce"),
    OperatorDefinition("Unnest", "unnest"),
    OperatorDefinition("VideoMap", "video_map"),
    OperatorDefinition("VideoMapEach", "video_map_each"),
    OperatorDefinition("Detect", "detect", source_visible=False),
    OperatorDefinition("Window", "window", source_visible=False),
    OperatorDefinition("Coalesce", "coalesce", source_visible=False),
)

OPERATOR_BY_KIND = {
    definition.kind: definition for definition in OPERATOR_DEFINITIONS
}
SOURCE_OPERATOR_NAMES = frozenset(
    definition.name
    for definition in OPERATOR_DEFINITIONS
    if definition.source_visible
)


def operator_name(kind: OperatorKind) -> str:
    return OPERATOR_BY_KIND[kind].name
