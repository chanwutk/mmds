from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field

from ....model import DatasetExpr, PromptSpec, RecordPath
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


class ModalitySubstitutionParams(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    video_field: str = Field(min_length=1)
    transcript_field: str = Field(min_length=1)


class ModalitySubstitution:
    metadata = DirectiveMetadata(
        name="modality_substitution",
        description="Replace a video field reference with a transcript field reference.",
        when_to_use="Use when the answer depends only on spoken or transcribed content.",
    )
    params_type = ModalitySubstitutionParams

    def find_matches(self, index: PlanIndex) -> tuple[RewriteMatch, ...]:
        return tuple(
            make_match(
                self.metadata,
                entry.path,
                "Prompt-backed Map reading "
                + ", ".join(
                    sorted(
                        {
                            path.path[0]
                            for path in record_paths(entry.expr.spec.parts)
                            if path.path
                        }
                    )
                ),
            )
            for entry in prompt_map_nodes(index)
        )

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr:
        if not isinstance(params, ModalitySubstitutionParams):
            raise MMDSRewriteError("ModalitySubstitution received invalid parameters.")
        node = index.node_at(match.path)
        if node.kind != "map" or not isinstance(node.spec, PromptSpec):
            raise MMDSRewriteError(
                "ModalitySubstitution requires a prompt-backed Map."
            )
        old = RecordPath((params.video_field,))
        new = RecordPath((params.transcript_field,))
        if old not in record_paths(node.spec.parts):
            raise MMDSRewriteError(
                f"Matched Map does not reference video field {params.video_field!r}."
            )
        if old == new:
            raise MMDSRewriteError(
                "ModalitySubstitution requires distinct video and transcript fields."
            )
        replacement = replace(
            node,
            spec=replace(
                node.spec,
                parts=replace_record_path(node.spec.parts, old, new),
            ),
        )
        return index.replace(match.path, replacement)
