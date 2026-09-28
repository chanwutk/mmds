from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....model import DatasetExpr, PromptSpec, RecordPath
from ..core import (
    DirectiveMetadata,
    PlanIndex,
    RewriteMatch,
)
from ..errors import MMDSRewriteError
from ._prompt import (
    format_record_paths,
    prompt_from_fields,
    prompt_map_entries,
    record_paths,
)


class PromptFieldPruningParams(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    drop_fields: list[str] = Field(
        min_length=1,
        description=(
            "Top-level prompt Record fields to remove because they are unused "
            "or unnecessarily expensive for the task."
        ),
    )
    rewritten_prompt: str = Field(
        min_length=1,
        description=(
            "Complete replacement instruction that preserves the original task "
            "without referring to the dropped fields."
        ),
    )

    @model_validator(mode="after")
    def validate_fields(self) -> PromptFieldPruningParams:
        if not self.rewritten_prompt.strip():
            raise ValueError("rewritten_prompt cannot be blank.")
        if any(not isinstance(field, str) or not field.strip() for field in self.drop_fields):
            raise ValueError("drop_fields must contain non-empty strings.")
        normalized = tuple(field.strip() for field in self.drop_fields)
        if len(set(normalized)) != len(normalized):
            raise ValueError("drop_fields must be unique.")
        return self


class PromptFieldPruning:
    """Remove unused top-level Record references from one Map prompt."""

    metadata = DirectiveMetadata(
        name="prompt_field_pruning",
        description=(
            "Remove unused or unnecessarily expensive Record field references "
            "from a Map prompt."
        ),
        when_to_use=(
            "Use when the original prompt over-includes modalities or fields "
            "that are not required for the task."
        ),
    )
    params_type = PromptFieldPruningParams

    def find_matches(self, index: PlanIndex) -> tuple[RewriteMatch, ...]:
        return tuple(
            RewriteMatch(
                path=entry.path,
                summary=(
                    "Prompt-backed Map reading "
                    + format_record_paths(record_paths(entry.node.spec.parts))
                ),
            )
            for entry in prompt_map_entries(index)
            if isinstance(entry.node.spec, PromptSpec)
        )

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr:
        if not isinstance(params, PromptFieldPruningParams):
            raise MMDSRewriteError(
                "PromptFieldPruning received invalid parameters."
            )

        node = index.node_at(match.path)
        if node.kind != "map" or not isinstance(node.spec, PromptSpec):
            raise MMDSRewriteError(
                "PromptFieldPruning requires a prompt-backed Map."
            )

        paths = record_paths(node.spec.parts)
        drop = {field.strip() for field in params.drop_fields}
        missing = sorted(
            field
            for field in drop
            if RecordPath((field,)) not in paths
        )
        if missing:
            raise MMDSRewriteError(
                "Matched Map does not directly reference drop field(s) "
                + ", ".join(repr(field) for field in missing)
                + "."
            )

        nested = tuple(
            path
            for path in paths
            if len(path.path) != 1 and path.path and path.path[0] in drop
        )
        if nested:
            raise MMDSRewriteError(
                "Prompt field pruning currently supports only top-level drop "
                f"fields; found {format_record_paths(nested)}."
            )

        kept = tuple(
            path
            for path in paths
            if not path.path or path.path[0] not in drop
        )
        if not kept:
            raise MMDSRewriteError(
                "Prompt field pruning must leave at least one Record reference."
            )
        if kept == paths:
            raise MMDSRewriteError(
                "Prompt field pruning did not remove any Record references."
            )

        replacement = replace(
            node,
            spec=replace(
                node.spec,
                parts=prompt_from_fields(params.rewritten_prompt, kept),
            ),
        )
        return index.replace(match.path, replacement)
