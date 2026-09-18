from __future__ import annotations

from dataclasses import dataclass, replace
from time import monotonic
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from pydantic import BaseModel

from ...model import (
    DatasetExpr,
    DropFieldsSpec,
    ForEachPrompt,
    MMDSValidationError,
    PromptPart,
    PromptSpec,
    QueryProgram,
    RecordPath,
    VideoMapSpec,
    WindowSpec,
)

if TYPE_CHECKING:
    from .context import RewriteContext


class MMDSRewriteError(MMDSValidationError):
    """Expected failure while selecting, applying, or validating a rewrite."""


@dataclass(frozen=True)
class NodePath:
    steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(step != "source" for step in self.steps):
            raise MMDSRewriteError(
                "NodePath currently supports only unary 'source' steps."
            )

    def source(self) -> NodePath:
        return NodePath((*self.steps, "source"))

    def __str__(self) -> str:
        if not self.steps:
            return "output"
        return "output." + ".".join(self.steps)


@dataclass(frozen=True)
class DirectiveMetadata:
    name: str
    description: str
    when_to_use: str

    def __post_init__(self) -> None:
        if not self.name or not self.description or not self.when_to_use:
            raise MMDSRewriteError(
                "Directive metadata fields must all be non-empty."
            )


@dataclass(frozen=True)
class FieldEffects:
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()
    drops: frozenset[str] = frozenset()
    unknown: bool = False


@dataclass(frozen=True)
class PlanEntry:
    path: NodePath
    expr: DatasetExpr
    field_effects: FieldEffects


@dataclass(frozen=True)
class PlanIndex:
    root: DatasetExpr
    entries: tuple[PlanEntry, ...]

    @classmethod
    def build(cls, root: DatasetExpr) -> PlanIndex:
        entries: list[PlanEntry] = []

        def visit(node: DatasetExpr, path: NodePath) -> None:
            entries.append(
                PlanEntry(
                    path=path,
                    expr=node,
                    field_effects=field_effects(node),
                )
            )
            if node.source is not None:
                visit(node.source, path.source())

        visit(root, NodePath())
        return cls(root=root, entries=tuple(entries))

    def node_at(self, path: NodePath) -> DatasetExpr:
        for entry in self.entries:
            if entry.path == path:
                return entry.expr
        raise MMDSRewriteError(f"Node path {str(path)!r} does not exist.")

    def replace(self, path: NodePath, replacement: DatasetExpr) -> DatasetExpr:
        self.node_at(path)

        def rebuild(node: DatasetExpr, steps: tuple[str, ...]) -> DatasetExpr:
            if not steps:
                return replacement
            if node.source is None:
                raise MMDSRewriteError(
                    f"Node path {str(path)!r} does not exist."
                )
            return replace(node, source=rebuild(node.source, steps[1:]))

        return rebuild(self.root, path.steps)

    def known_fields(self) -> frozenset[str]:
        fields: set[str] = set()
        for entry in self.entries:
            effects = entry.field_effects
            fields.update(effects.reads)
            fields.update(effects.writes)
            fields.update(effects.drops)
        return frozenset(fields)


@dataclass(frozen=True)
class RewriteMatch:
    id: str
    path: NodePath
    summary: str


@dataclass(frozen=True)
class RewriteOption:
    directive: str
    match_id: str
    path: str
    summary: str
    description: str
    when_to_use: str
    params_schema: Mapping[str, Any]


@dataclass(frozen=True)
class RewriteSelection:
    directive: str
    match_id: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class RewriteTrace:
    directive: str
    match_id: str
    path: str
    params: Mapping[str, Any]
    original_fingerprint: str
    rewritten_fingerprint: str
    rewrite_seconds: float
    agent_details: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RewriteResult:
    original: QueryProgram
    program: QueryProgram
    trace: RewriteTrace


@dataclass(frozen=True)
class RewriteCandidate:
    program: QueryProgram
    fingerprint: str
    trace: RewriteTrace | None = None


@dataclass(frozen=True)
class RewriteRejection:
    directive: str
    match_id: str | None
    reason: str


@dataclass(frozen=True)
class RewriteSearchResult:
    candidates: tuple[RewriteCandidate, ...]
    rejections: tuple[RewriteRejection, ...]
    truncated: bool = False


class RewriteDirective(Protocol):
    metadata: DirectiveMetadata
    params_type: type[BaseModel]

    def find_matches(self, index: PlanIndex) -> tuple[RewriteMatch, ...]: ...

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr: ...


class RewriteAgent(Protocol):
    def propose(
        self,
        options: tuple[RewriteOption, ...],
        *,
        context: RewriteContext,
        max_candidates: int,
    ) -> tuple[RewriteSelection, ...]: ...


class StaticRewriteAgent:
    def __init__(
        self,
        *,
        directive: str | None = None,
        match: str | None = None,
        params: Mapping[str, Any] | None = None,
        selections: tuple[RewriteSelection, ...] | None = None,
    ) -> None:
        if selections is not None:
            if directive is not None or match is not None or params is not None:
                raise TypeError(
                    "StaticRewriteAgent accepts either selections= or one directive/match/params selection."
                )
            self._selections = selections
            self.last_trace: Mapping[str, Any] = {"kind": "static"}
            return
        if directive is None or match is None or params is None:
            raise TypeError(
                "StaticRewriteAgent requires directive, match, and params."
            )
        self._selections = (
            RewriteSelection(
                directive=directive,
                match_id=match,
                params=dict(params),
            ),
        )
        self.last_trace = {"kind": "static"}

    def propose(
        self,
        options: tuple[RewriteOption, ...],
        *,
        context: RewriteContext,
        max_candidates: int,
    ) -> tuple[RewriteSelection, ...]:
        del options, context, max_candidates
        # A static agent represents an exact recorded proposal. Return it in
        # full so the engine can test and report candidate truncation itself.
        return self._selections


def make_match(
    metadata: DirectiveMetadata,
    path: NodePath,
    summary: str,
) -> RewriteMatch:
    return RewriteMatch(
        id=f"{metadata.name}:{str(path)}",
        path=path,
        summary=summary,
    )


def prompt_map_nodes(index: PlanIndex) -> tuple[PlanEntry, ...]:
    return tuple(
        entry
        for entry in index.entries
        if entry.expr.kind == "map"
        and isinstance(entry.expr.spec, PromptSpec)
    )


def record_paths(parts: tuple[PromptPart, ...]) -> tuple[RecordPath, ...]:
    paths: list[RecordPath] = []
    for part in parts:
        if isinstance(part, RecordPath):
            paths.append(part)
        elif isinstance(part, ForEachPrompt):
            paths.extend(record_paths(part.parts))
    return tuple(paths)


def replace_record_path(
    parts: tuple[PromptPart, ...],
    old: RecordPath,
    new: RecordPath,
) -> tuple[PromptPart, ...]:
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


def field_effects(node: DatasetExpr) -> FieldEffects:
    spec = node.spec
    if isinstance(spec, PromptSpec):
        reads = frozenset(
            path.path[0] for path in record_paths(spec.parts) if path.path
        )
        writes = frozenset(spec.output_schema or {}) if node.kind != "filter" else frozenset()
        return FieldEffects(reads=reads, writes=writes)
    if node.kind in {"map", "filter", "reduce"}:
        return FieldEffects(unknown=True)
    if isinstance(spec, WindowSpec):
        return FieldEffects(
            reads=frozenset({spec.video_field, spec.candidate_field}),
            writes=frozenset({spec.output_field}),
        )
    if node.kind == "coalesce" and node.field is not None:
        return FieldEffects(
            reads=frozenset({*node.group_by, node.field}),
            writes=frozenset({*node.group_by, node.field}),
        )
    if node.kind == "unnest" and node.field is not None:
        return FieldEffects(
            reads=frozenset({node.field}),
            writes=frozenset({node.field}),
        )
    if isinstance(spec, DropFieldsSpec):
        return FieldEffects(drops=frozenset(spec.fields))
    if isinstance(spec, VideoMapSpec):
        nested = (
            field_effects(DatasetExpr(kind="map", source=node.source, spec=spec.map_spec))
            if node.source is not None
            else FieldEffects(unknown=True)
        )
        return FieldEffects(
            reads=frozenset(
                {*nested.reads, spec.video_field, spec.views_field, *spec.group_by}
            ),
            writes=nested.writes,
            unknown=nested.unknown,
        )
    return FieldEffects()


def timed_agent_proposal(
    agent: RewriteAgent,
    options: tuple[RewriteOption, ...],
    *,
    context: RewriteContext,
    max_candidates: int,
) -> tuple[tuple[RewriteSelection, ...], float]:
    started = monotonic()
    selections = agent.propose(
        options,
        context=context,
        max_candidates=max_candidates,
    )
    return selections, monotonic() - started
