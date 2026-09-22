from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol

from pydantic import BaseModel, ValidationError

from ...model import DatasetExpr, QueryProgram
from ...render import program_from_plan
from .errors import MMDSRewriteError
from .validation import validate_rewrite_structure


@dataclass(frozen=True)
class NodePath:
    """Address of a node, expressed as child-edge steps from the output."""

    steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple) or any(
            step != "source" for step in self.steps
        ):
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
class PlanEntry:
    """One existing plan node paired with its address."""

    path: NodePath
    node: DatasetExpr


@dataclass(frozen=True)
class PlanIndex:
    """Temporary structural index over an immutable operator plan."""

    root: DatasetExpr
    entries: tuple[PlanEntry, ...]

    @classmethod
    def build(cls, root: DatasetExpr) -> PlanIndex:
        if not isinstance(root, DatasetExpr):
            raise TypeError("PlanIndex.build() expects a DatasetExpr.")

        entries: list[PlanEntry] = []

        def visit(node: DatasetExpr, path: NodePath) -> None:
            entries.append(PlanEntry(path=path, node=node))
            if node.source is not None:
                visit(node.source, path.source())

        visit(root, NodePath())
        return cls(root=root, entries=tuple(entries))

    def node_at(self, path: NodePath) -> DatasetExpr:
        for entry in self.entries:
            if entry.path == path:
                return entry.node
        raise MMDSRewriteError(f"Node path {str(path)!r} does not exist.")

    def replace(self, path: NodePath, replacement: DatasetExpr) -> DatasetExpr:
        """Return a new root with one subtree replaced; leave this plan unchanged."""

        if not isinstance(replacement, DatasetExpr):
            raise TypeError("Plan replacements must be DatasetExpr instances.")
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


@dataclass(frozen=True)
class DirectiveMetadata:
    name: str
    description: str
    when_to_use: str

    def __post_init__(self) -> None:
        values = (self.name, self.description, self.when_to_use)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise MMDSRewriteError(
                "Directive metadata fields must all be non-empty strings."
            )


@dataclass(frozen=True)
class RewriteMatch:
    """Evidence that one directive can be applied at one plan location."""

    path: NodePath
    summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise MMDSRewriteError("Rewrite match summaries must be non-empty strings.")


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


def apply_rewrite(
    program: QueryProgram,
    *,
    directive: RewriteDirective,
    match: RewriteMatch,
    params: Mapping[str, Any],
) -> QueryProgram:
    """Validate and deterministically apply one previously matched directive."""

    index = PlanIndex.build(program.output_expr)
    if match not in directive.find_matches(index):
        raise MMDSRewriteError(
            f"Directive {directive.metadata.name!r} did not offer the selected match."
        )

    try:
        validated_params = directive.params_type.model_validate(dict(params))
    except (TypeError, ValueError, ValidationError) as exc:
        raise MMDSRewriteError(
            f"Invalid parameters for directive {directive.metadata.name!r}: {exc}"
        ) from exc

    rewritten = directive.apply(index, match, validated_params)
    if not isinstance(rewritten, DatasetExpr):
        raise MMDSRewriteError(
            f"Directive {directive.metadata.name!r} did not return a DatasetExpr."
        )
    validate_rewrite_structure(program, rewritten)

    normalized = program_from_plan(rewritten)
    return replace(normalized, path=program.path)
