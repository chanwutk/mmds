from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ....model import (
    DatasetExpr,
    DropFieldsSpec,
    PromptSpec,
    RecordPath,
)
from ..directive import (
    DirectiveMetadata,
    MMDSRewriteError,
    PlanIndex,
    RewriteMatch,
    make_match,
    prompt_map_nodes,
    record_paths,
    replace_record_path,
)


class ProjectionBeforeMapParams(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    source_field: str = Field(min_length=1)
    intermediate_field: str = "_mmds_projection"
    projection_prompt: str = Field(min_length=1)
    context_fields: tuple[str, ...] = ()

    @field_validator("intermediate_field")
    @classmethod
    def _reserved_intermediate_field(cls, value: str) -> str:
        if not value.startswith("_mmds_"):
            raise ValueError("intermediate_field must use the reserved '_mmds_' prefix")
        return value

    @field_validator("context_fields")
    @classmethod
    def _valid_context_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not field for field in value):
            raise ValueError("context_fields must contain non-empty names")
        if len(set(value)) != len(value):
            raise ValueError("context_fields must be unique")
        return value


class ProjectionBeforeMap:
    metadata = DirectiveMetadata(
        name="projection_before_map",
        description="Insert a Map that creates a smaller textual projection before another Map.",
        when_to_use="Use when a Map can answer from a shorter extract or summary of one large field.",
    )
    params_type = ProjectionBeforeMapParams

    def find_matches(self, index: PlanIndex) -> tuple[RewriteMatch, ...]:
        return tuple(
            make_match(
                self.metadata,
                entry.path,
                "Prompt-backed Map with fields "
                + ", ".join(sorted(entry.field_effects.reads)),
            )
            for entry in prompt_map_nodes(index)
        )

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr:
        if not isinstance(params, ProjectionBeforeMapParams):
            raise MMDSRewriteError("ProjectionBeforeMap received invalid parameters.")
        node = index.node_at(match.path)
        if node.kind != "map" or not isinstance(node.spec, PromptSpec):
            raise MMDSRewriteError(
                "ProjectionBeforeMap requires a prompt-backed Map."
            )
        if node.source is None:
            raise MMDSRewriteError("ProjectionBeforeMap requires a Map source.")

        source_reference = RecordPath((params.source_field,))
        if source_reference not in record_paths(node.spec.parts):
            raise MMDSRewriteError(
                f"Matched Map does not reference source field {params.source_field!r}."
            )
        if params.intermediate_field in index.known_fields():
            raise MMDSRewriteError(
                f"Temporary field {params.intermediate_field!r} collides with an existing field."
            )

        projection_parts: list[str | RecordPath] = [
            params.projection_prompt,
            f"\n{params.source_field}:\n",
            source_reference,
        ]
        for field in params.context_fields:
            projection_parts.extend((f"\n{field}:\n", RecordPath((field,))))

        preparation = DatasetExpr(
            kind="map",
            source=node.source,
            spec=PromptSpec(
                parts=tuple(projection_parts),
                output_schema={params.intermediate_field: "string"},
            ),
            name="rewrite_projection",
        )
        intermediate_reference = RecordPath((params.intermediate_field,))
        consumer = replace(
            node,
            source=preparation,
            spec=replace(
                node.spec,
                parts=replace_record_path(
                    node.spec.parts,
                    source_reference,
                    intermediate_reference,
                ),
            ),
        )
        cleanup = DatasetExpr(
            kind="drop_fields",
            source=consumer,
            spec=DropFieldsSpec(fields=(params.intermediate_field,)),
            name="drop_rewrite_projection",
        )
        return index.replace(match.path, cleanup)
