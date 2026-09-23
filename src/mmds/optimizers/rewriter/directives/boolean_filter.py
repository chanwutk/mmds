from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....model import (
    DatasetExpr,
    FieldPredicateSpec,
    PromptSpec,
    RecordPath,
    normalize_output_schema,
)
from ..core import (
    DirectiveMetadata,
    PlanIndex,
    RewriteMatch,
)
from ..errors import MMDSRewriteError
from ._prompt import (
    format_record_paths,
    prompt_from_fields,
    record_paths,
)


def _schema_declares_boolean(schema: Mapping[str, Any] | None, field: str) -> bool:
    if schema is None or field not in schema:
        return False
    value = schema[field]
    if value == "boolean":
        return True
    return isinstance(value, Mapping) and value.get("type") == "boolean"


class BooleanMapCodeFilterParams(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    flag_field: str = Field(
        min_length=1,
        description=(
            "Boolean Map output field used as the keep/drop predicate for the "
            "rewritten code Filter."
        ),
    )
    rewritten_map_prompt: str = Field(
        min_length=1,
        description=(
            "Complete replacement Map instruction that materializes flag_field "
            "(and any other retained Map outputs) without relying on a second "
            "LLM Filter."
        ),
    )
    map_schema: dict[str, Any] = Field(
        description=(
            "Full Map output schema. It must include flag_field as a boolean "
            "and preserve any other fields the rewritten plan must still emit."
        ),
    )

    @model_validator(mode="after")
    def validate_fields(self) -> BooleanMapCodeFilterParams:
        if not self.flag_field.strip():
            raise ValueError("flag_field cannot be blank.")
        if not self.rewritten_map_prompt.strip():
            raise ValueError("rewritten_map_prompt cannot be blank.")
        try:
            schema = normalize_output_schema(self.map_schema)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid map_schema: {exc}") from exc
        if schema is None or not schema:
            raise ValueError("map_schema cannot be empty.")
        flag = self.flag_field.strip()
        if not _schema_declares_boolean(schema, flag):
            raise ValueError(f"map_schema must declare {flag!r} as boolean.")
        object.__setattr__(self, "flag_field", flag)
        object.__setattr__(self, "map_schema", dict(schema))
        return self


class BooleanMapCodeFilter:
    """Replace a prompt Filter with Map(boolean flag) + Filter(Record[flag])."""

    metadata = DirectiveMetadata(
        name="boolean_map_code_filter",
        description=(
            "Materialize a boolean keep flag in a Map, then filter with a "
            "non-LLM Record[field] predicate."
        ),
        when_to_use=(
            "Use when a prompt-backed Filter only needs a boolean decision that "
            "a Map can emit, avoiding a second LLM call per row."
        ),
    )
    params_type = BooleanMapCodeFilterParams

    def find_matches(self, index: PlanIndex) -> tuple[RewriteMatch, ...]:
        matches: list[RewriteMatch] = []
        for entry in index.entries:
            node = entry.node
            if node.kind != "filter" or not isinstance(node.spec, PromptSpec):
                continue
            paths = record_paths(node.spec.parts)
            summary = (
                "Prompt-backed Filter reading " + format_record_paths(paths)
                if paths
                else "Prompt-backed Filter"
            )
            matches.append(RewriteMatch(path=entry.path, summary=summary))
        return tuple(matches)

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr:
        if not isinstance(params, BooleanMapCodeFilterParams):
            raise MMDSRewriteError(
                "BooleanMapCodeFilter received invalid parameters."
            )

        node = index.node_at(match.path)
        if node.kind != "filter" or not isinstance(node.spec, PromptSpec):
            raise MMDSRewriteError(
                "BooleanMapCodeFilter requires a prompt-backed Filter."
            )
        if node.source is None:
            raise MMDSRewriteError(
                "BooleanMapCodeFilter requires a Filter source."
            )

        schema = normalize_output_schema(params.map_schema)
        assert schema is not None
        map_paths = tuple(RecordPath((field,)) for field in schema)
        map_spec = PromptSpec(
            parts=prompt_from_fields(params.rewritten_map_prompt, map_paths),
            output_schema=schema,
        )

        source = node.source
        if source.kind == "map" and isinstance(source.spec, PromptSpec):
            if source.spec.output_schema != schema:
                raise MMDSRewriteError(
                    "BooleanMapCodeFilter map_schema must preserve the upstream "
                    "Map output schema when rewriting Map -> Filter."
                )
            mapped = replace(source, spec=map_spec)
        else:
            mapped = DatasetExpr(
                kind="map",
                source=source,
                spec=map_spec,
                name="rewrite_materialize_keep_flag",
            )

        replacement = replace(
            node,
            source=mapped,
            spec=FieldPredicateSpec(field=params.flag_field),
        )
        return index.replace(match.path, replacement)
