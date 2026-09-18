from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...model import (
    DatasetExpr,
    DropFieldsSpec,
    ForEachPrompt,
    PromptPart,
    PromptSpec,
    QueryProgram,
    RecordPath,
    UdfSpec,
    VideoMapSpec,
    WindowSpec,
)
from .directive import MMDSRewriteError, PlanIndex


DEFAULT_REWRITE_POLICY = (
    "Generate semantically valid alternative plans that may reduce execution "
    "cost or latency. Preserve the query meaning, final output schema, input "
    "datasets, required modalities, and temporal coordinate systems. Use only "
    "query-visible dataset fields; evaluation-only fields are excluded."
)

_EVALUATION_FIELD_MARKERS = (
    "annotation",
    "expected",
    "gold",
    "ground_truth",
    "groundtruth",
    "label",
)
_PROFILE_ROWS = 32
_PROFILE_LIST_ITEMS = 8
_PROFILE_DEPTH = 3


@dataclass(frozen=True)
class RewriteContext:
    """Ephemeral query and dataset metadata supplied to the rewrite model."""

    policy: str
    query_plan: tuple[Mapping[str, Any], ...]
    datasets: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "query_plan": [dict(node) for node in self.query_plan],
            "datasets": [dict(dataset) for dataset in self.datasets],
        }


def build_rewrite_context(
    program: QueryProgram,
    *,
    index: PlanIndex | None = None,
) -> RewriteContext:
    plan_index = index or PlanIndex.build(program.output_expr)
    referenced_fields = frozenset(
        {
            *plan_index.known_fields(),
            *(
                field
                for entry in plan_index.entries
                for field in entry.expr.group_by
            ),
        }
    )
    return RewriteContext(
        policy=DEFAULT_REWRITE_POLICY,
        query_plan=tuple(
            _summarize_node(
                entry.expr,
                path=str(entry.path),
                reads=entry.field_effects.reads,
                writes=entry.field_effects.writes,
                drops=entry.field_effects.drops,
                effects_unknown=entry.field_effects.unknown,
            )
            for entry in plan_index.entries
        ),
        datasets=tuple(
            _profile_dataset(
                entry.expr.input_path,
                program_path=program.path,
                referenced_fields=referenced_fields,
            )
            for entry in plan_index.entries
            if entry.expr.kind == "input"
        ),
    )


def _summarize_node(
    node: DatasetExpr,
    *,
    path: str,
    reads: frozenset[str],
    writes: frozenset[str],
    drops: frozenset[str],
    effects_unknown: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": path,
        "kind": node.kind,
    }
    if node.name is not None:
        summary["name"] = node.name
    if reads:
        summary["reads"] = sorted(reads)
    if writes:
        summary["writes"] = sorted(writes)
    if drops:
        summary["drops"] = sorted(drops)
    if effects_unknown:
        summary["field_effects_unknown"] = True
    if node.kind == "input":
        summary["input_path"] = node.input_path
    if node.group_by:
        summary["group_by"] = list(node.group_by)
    if node.field is not None:
        summary["field"] = node.field
    if node.kind == "unnest":
        summary["keep_empty"] = node.keep_empty

    spec = node.spec
    if isinstance(spec, PromptSpec):
        summary["prompt"] = [_summarize_prompt_part(part) for part in spec.parts]
        if spec.output_schema is not None:
            summary["output_schema"] = spec.output_schema
    elif isinstance(spec, UdfSpec):
        summary["udf"] = f"{spec.module}.{spec.name}"
    elif isinstance(spec, WindowSpec):
        summary["window"] = {
            "video_field": spec.video_field,
            "candidate_field": spec.candidate_field,
            "output_field": spec.output_field,
            "padding_time": spec.padding_time,
        }
    elif isinstance(spec, DropFieldsSpec):
        summary["fields"] = list(spec.fields)
    elif isinstance(spec, VideoMapSpec):
        summary["video_map"] = {
            "video_field": spec.video_field,
            "views_field": spec.views_field,
            "group_by": list(spec.group_by),
            "map_spec": _summarize_map_spec(spec.map_spec),
        }
    return summary


def _summarize_map_spec(spec: PromptSpec | UdfSpec) -> dict[str, Any]:
    if isinstance(spec, UdfSpec):
        return {"udf": f"{spec.module}.{spec.name}"}
    result: dict[str, Any] = {
        "prompt": [_summarize_prompt_part(part) for part in spec.parts]
    }
    if spec.output_schema is not None:
        result["output_schema"] = spec.output_schema
    return result


def _summarize_prompt_part(part: PromptPart) -> Any:
    if isinstance(part, str):
        return part
    if isinstance(part, RecordPath):
        return {"record": list(part.path)}
    if isinstance(part, ForEachPrompt):
        return {
            "for_each": [_summarize_prompt_part(nested) for nested in part.parts]
        }
    raise MMDSRewriteError(f"Unsupported prompt part {type(part).__name__!r}.")


def _profile_dataset(
    input_path: str | None,
    *,
    program_path: str | None,
    referenced_fields: frozenset[str],
) -> dict[str, Any]:
    if input_path is None:
        raise MMDSRewriteError("Cannot profile an Input node without a path.")
    resolved = _resolve_input_path(input_path, program_path=program_path)
    if not resolved.exists():
        return {
            "input_path": input_path,
            "available": False,
            "fields": {},
        }

    rows = _read_profile_rows(resolved)
    fields: dict[str, dict[str, Any]] = {}
    excluded_fields: set[str] = set()
    for row in rows:
        for name, value in row.items():
            if _looks_evaluation_only(name) and name not in referenced_fields:
                excluded_fields.add(name)
                continue
            shape = _shape(value, depth=0)
            previous = fields.get(name)
            fields[name] = shape if previous is None else _merge_shapes(previous, shape)

    profiled_fields = {
        name: {
            **shape,
            "role": _field_role(name, shape, referenced=name in referenced_fields),
        }
        for name, shape in sorted(fields.items())
    }
    return {
        "input_path": input_path,
        "available": True,
        "rows_profiled": len(rows),
        "fields": profiled_fields,
        "evaluation_fields_excluded": len(excluded_fields),
    }


def _resolve_input_path(input_path: str, *, program_path: str | None) -> Path:
    path = Path(input_path)
    if path.is_absolute():
        return path
    base = Path(program_path).resolve().parent if program_path else Path.cwd()
    return base / path


def _read_profile_rows(path: Path) -> list[Mapping[str, Any]]:
    try:
        if path.suffix == ".jsonl":
            rows: list[Mapping[str, Any]] = []
            with path.open(encoding="utf-8") as source:
                for line_number, raw_line in enumerate(source, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise MMDSRewriteError(
                            f"Cannot profile invalid JSON on line {line_number} of {str(path)!r}."
                        ) from exc
                    rows.append(_require_row(value, path))
                    if len(rows) >= _PROFILE_ROWS:
                        break
            return rows
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                raise MMDSRewriteError(
                    f"Cannot profile {str(path)!r}: JSON input must contain a list."
                )
            return [_require_row(row, path) for row in value[:_PROFILE_ROWS]]
    except OSError as exc:
        raise MMDSRewriteError(f"Cannot read input file {str(path)!r}.") from exc
    except json.JSONDecodeError as exc:
        raise MMDSRewriteError(f"Cannot profile invalid JSON file {str(path)!r}.") from exc
    raise MMDSRewriteError(
        f"Cannot profile {str(path)!r}: inputs must use .json or .jsonl."
    )


def _require_row(value: Any, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MMDSRewriteError(
            f"Cannot profile {str(path)!r}: every input row must be an object."
        )
    return value


def _shape(value: Any, *, depth: int) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, (int, float)):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, Mapping):
        media_type = value.get("type")
        if isinstance(media_type, str) and media_type.lower() in {
            "video",
            "videoview",
        }:
            return {"type": "video"}
        if depth >= _PROFILE_DEPTH:
            return {"type": "object"}
        return {
            "type": "object",
            "properties": {
                str(name): _shape(nested, depth=depth + 1)
                for name, nested in value.items()
            },
        }
    if isinstance(value, list):
        items: dict[str, Any] | None = None
        for nested in value[:_PROFILE_LIST_ITEMS]:
            nested_shape = _shape(nested, depth=depth + 1)
            items = nested_shape if items is None else _merge_shapes(items, nested_shape)
        return {
            "type": "array",
            "items": items or {"type": "unknown"},
        }
    return {"type": type(value).__name__}


def _merge_shapes(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if left == right:
        return left
    if "types" in left or "types" in right:
        return {
            "types": sorted(
                {
                    *_shape_type_names(left),
                    *_shape_type_names(right),
                }
            )
        }
    left_type = left.get("type")
    right_type = right.get("type")
    if left_type != right_type:
        return {"types": sorted({str(left_type), str(right_type)})}
    if left_type == "object":
        properties = dict(left.get("properties", {}))
        for name, shape in right.get("properties", {}).items():
            properties[name] = (
                shape
                if name not in properties
                else _merge_shapes(properties[name], shape)
            )
        return {"type": "object", "properties": properties}
    if left_type == "array":
        return {
            "type": "array",
            "items": _merge_shapes(left["items"], right["items"]),
        }
    return {"type": left_type}


def _shape_type_names(shape: Mapping[str, Any]) -> set[str]:
    types = shape.get("types")
    if isinstance(types, list):
        return {str(value) for value in types}
    return {str(shape.get("type", "unknown"))}


def _field_role(
    name: str,
    shape: Mapping[str, Any],
    *,
    referenced: bool,
) -> str:
    if shape.get("type") == "video":
        return "video"
    if _is_timestamped_text(shape):
        return "timestamped_transcript"
    lowered = name.lower()
    text_modality_names = {"caption", "captions", "subtitle", "subtitles", "transcript"}
    if (
        lowered in text_modality_names
        or lowered.endswith(("_captions", "_subtitles", "_transcript"))
    ) and shape.get("type") in {"string", "array"}:
        return "transcript"
    if referenced and shape.get("type") == "string" and any(
        marker in lowered for marker in ("query", "question", "prompt")
    ):
        return "query"
    if lowered == "id" or lowered.endswith("_id"):
        return "identifier"
    return "data"


def _is_timestamped_text(shape: Mapping[str, Any]) -> bool:
    if shape.get("type") != "array":
        return False
    items = shape.get("items")
    if not isinstance(items, Mapping) or items.get("type") != "object":
        return False
    properties = items.get("properties")
    return isinstance(properties, Mapping) and {"start", "end", "text"}.issubset(
        properties
    )


def _looks_evaluation_only(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _EVALUATION_FIELD_MARKERS)
