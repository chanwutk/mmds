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
    prompt_map_entries,
    record_paths,
    replace_record_path,
)


class ModalitySubstitutionParams(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    video_field: str = Field(min_length=1)
    transcript_field: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_fields(self) -> ModalitySubstitutionParams:
        if not self.video_field.strip() or not self.transcript_field.strip():
            raise ValueError("Modality field names cannot be blank.")
        if self.video_field == self.transcript_field:
            raise ValueError(
                "Modality substitution requires distinct video and transcript fields."
            )
        return self


class ModalitySubstitution:
    """Replace a direct video reference in one Map prompt with transcript."""

    metadata = DirectiveMetadata(
        name="modality_substitution",
        description=(
            "Replace a Map's video field reference with a transcript field reference."
        ),
        when_to_use=(
            "Use when spoken or transcribed content is sufficient to answer the query."
        ),
    )
    params_type = ModalitySubstitutionParams

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
        if not isinstance(params, ModalitySubstitutionParams):
            raise MMDSRewriteError(
                "ModalitySubstitution received invalid parameters."
            )

        node = index.node_at(match.path)
        if node.kind != "map" or not isinstance(node.spec, PromptSpec):
            raise MMDSRewriteError(
                "ModalitySubstitution requires a prompt-backed Map."
            )

        old = RecordPath((params.video_field,))
        paths = record_paths(node.spec.parts)
        if old not in paths:
            raise MMDSRewriteError(
                f"Matched Map does not directly reference video field "
                f"{params.video_field!r}."
            )

        replacement = replace(
            node,
            spec=replace(
                node.spec,
                parts=replace_record_path(
                    node.spec.parts,
                    old,
                    RecordPath((params.transcript_field,)),
                ),
            ),
        )
        return index.replace(match.path, replacement)
