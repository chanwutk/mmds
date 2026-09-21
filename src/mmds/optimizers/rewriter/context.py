from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ...model import (
    DatasetExpr,
    DetectSpec,
    ForEachPrompt,
    PromptPart,
    PromptSpec,
    QueryProgram,
    RecordPath,
    UdfSpec,
    VideoMapSpec,
    WindowSpec,
)
from .core import PlanIndex
from .errors import MMDSRewriteError


_PROFILE_ROWS = 8
_EVALUATION_MARKERS = (
    "annotation",
    "expected",
    "gold",
    "ground_truth",
    "groundtruth",
    "label",
)


def build_rewrite_context(
    program: QueryProgram,
    *,
    index: PlanIndex | None = None,
) -> dict[str, Any]:
    """Return compact, value-free plan and dataset information for a model."""

    if not isinstance(program, QueryProgram):
        raise TypeError("build_rewrite_context() expects a QueryProgram.")
    plan_index = index or PlanIndex.build(program.output_expr)
    if plan_index.root != program.output_expr:
        raise MMDSRewriteError(
            "Rewrite context index must describe the program output plan."
        )

    referenced = _referenced_fields(plan_index)
    return {
        "plan": [
            _summarize_node(entry.node, path=str(entry.path))
            for entry in plan_index.entries
        ],
        "datasets": [
            _profile_input(
                entry.node.input_path,
                program_path=program.path,
                referenced_fields=referenced,
            )
            for entry in plan_index.entries
            if entry.node.kind == "input"
        ],
    }


def _summarize_node(node: DatasetExpr, *, path: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": path, "kind": node.kind}
    if node.name is not None:
        summary["name"] = node.name
    if node.kind == "input":
        summary["input_path"] = node.input_path
    if node.group_by:
        summary["group_by"] = list(node.group_by)
    if node.field is not None:
        summary["field"] = node.field

    spec = node.spec
    if isinstance(spec, PromptSpec):
        summary["prompt"] = [_prompt_part(part) for part in spec.parts]
        summary["output_schema"] = spec.output_schema
    elif isinstance(spec, UdfSpec):
        summary["udf"] = f"{spec.module}.{spec.name}"
    elif isinstance(spec, DetectSpec):
        summary["detect"] = {
            "video_field": spec.video_field,
            "classes": list(spec.classes),
            "output_field": spec.output_field,
        }
    elif isinstance(spec, WindowSpec):
        summary["window"] = {
            "video_field": spec.video_field,
            "candidate_field": spec.candidate_field,
            "output_field": spec.output_field,
            "padding_time": spec.padding_time,
        }
    elif isinstance(spec, VideoMapSpec):
        summary["video_map"] = {
            "video_field": spec.video_field,
            "views_field": spec.views_field,
            "group_by": list(spec.group_by),
            "padding_time": spec.padding_time,
            "map_spec": _semantic_spec(spec.map_spec),
        }
    return summary


def _semantic_spec(spec: PromptSpec | UdfSpec) -> dict[str, Any]:
    if isinstance(spec, UdfSpec):
        return {"udf": f"{spec.module}.{spec.name}"}
    return {
        "prompt": [_prompt_part(part) for part in spec.parts],
        "output_schema": spec.output_schema,
    }


def _prompt_part(part: PromptPart) -> Any:
    if isinstance(part, str):
        return part
    if isinstance(part, RecordPath):
        return {"record": ".".join(part.path)}
    if isinstance(part, ForEachPrompt):
        return {"for_each": [_prompt_part(child) for child in part.parts]}
    raise MMDSRewriteError(f"Unsupported prompt part {type(part).__name__!r}.")


def _referenced_fields(index: PlanIndex) -> frozenset[str]:
    fields: set[str] = set()
    for entry in index.entries:
        node = entry.node
        fields.update(node.group_by)
        if node.field is not None:
            fields.add(node.field)
        spec = node.spec
        if isinstance(spec, PromptSpec):
            fields.update(_prompt_fields(spec))
        elif isinstance(spec, DetectSpec):
            fields.add(spec.video_field)
        elif isinstance(spec, WindowSpec):
            fields.update(
                (spec.video_field, spec.candidate_field, spec.output_field)
            )
        elif isinstance(spec, VideoMapSpec):
            fields.update(
                (spec.video_field, spec.views_field, spec.clip_field, *spec.group_by)
            )
            if isinstance(spec.map_spec, PromptSpec):
                fields.update(_prompt_fields(spec.map_spec))
    return frozenset(fields)


def _prompt_fields(spec: PromptSpec) -> set[str]:
    fields: set[str] = set()

    def visit(parts: tuple[PromptPart, ...]) -> None:
        for part in parts:
            if isinstance(part, RecordPath) and part.path:
                fields.add(part.path[0])
            elif isinstance(part, ForEachPrompt):
                visit(part.parts)

    visit(spec.parts)
    return fields


def _profile_input(
    input_path: str | None,
    *,
    program_path: str | None,
    referenced_fields: frozenset[str],
) -> dict[str, Any]:
    if input_path is None:
        raise MMDSRewriteError("Cannot profile an Input node without a path.")
    resolved = _resolve_input(input_path, program_path=program_path)
    if not resolved.exists():
        return {"input_path": input_path, "available": False, "fields": {}}

    rows = _read_rows(resolved)
    signatures: dict[str, set[str]] = {}
    roles: dict[str, set[str]] = {}
    excluded_names: set[str] = set()
    for row in rows:
        for name, value in row.items():
            if _evaluation_only(name) and name not in referenced_fields:
                excluded_names.add(name)
                continue
            signature, detected_role = _value_signature(value)
            signatures.setdefault(name, set()).add(signature)
            if detected_role is not None:
                roles.setdefault(name, set()).add(detected_role)
    fields = {
        name: {
            "type": " | ".join(sorted(field_types)),
            "role": _field_role(name, roles.get(name, set())),
        }
        for name, field_types in sorted(signatures.items())
    }
    return {
        "input_path": input_path,
        "available": True,
        "rows_profiled": len(rows),
        "fields": fields,
        "evaluation_fields_excluded": len(excluded_names),
    }


def _resolve_input(input_path: str, *, program_path: str | None) -> Path:
    path = Path(input_path)
    if path.is_absolute():
        return path
    base = Path(program_path).resolve().parent if program_path else Path.cwd()
    return base / path


def _read_rows(path: Path) -> list[Mapping[str, Any]]:
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
                            f"Cannot profile invalid JSON on line {line_number} "
                            f"of {str(path)!r}."
                        ) from exc
                    rows.append(_require_row(value, path))
                    if len(rows) == _PROFILE_ROWS:
                        break
            return rows
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                raise MMDSRewriteError(
                    f"Cannot profile {str(path)!r}: JSON must contain a list."
                )
            return [_require_row(row, path) for row in value[:_PROFILE_ROWS]]
    except OSError as exc:
        raise MMDSRewriteError(f"Cannot read input file {str(path)!r}.") from exc
    except json.JSONDecodeError as exc:
        raise MMDSRewriteError(
            f"Cannot profile invalid JSON file {str(path)!r}."
        ) from exc
    raise MMDSRewriteError(
        f"Cannot profile {str(path)!r}: expected .json or .jsonl."
    )


def _require_row(value: Any, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MMDSRewriteError(
            f"Cannot profile {str(path)!r}: every row must be an object."
        )
    return value


def _value_signature(value: Any) -> tuple[str, str | None]:
    if value is None:
        return "null", None
    if isinstance(value, bool):
        return "boolean", None
    if isinstance(value, (int, float)):
        return "number", None
    if isinstance(value, str):
        return "string", None
    if isinstance(value, Mapping):
        media_type = value.get("type")
        if isinstance(media_type, str) and media_type.lower() in {
            "video",
            "videoview",
        }:
            return "video", "video"
        return "object", None
    if isinstance(value, list):
        for item in value[:4]:
            if isinstance(item, Mapping) and {"start", "end", "text"}.issubset(item):
                return "array<object>", "timestamped_transcript"
        return "array", None
    return type(value).__name__, None


def _field_role(
    name: str,
    detected_roles: set[str],
) -> str:
    if "video" in detected_roles:
        return "video"
    if "timestamped_transcript" in detected_roles:
        return "timestamped_transcript"
    lowered = name.lower()
    if lowered == "video" or lowered.endswith("_video"):
        return "video"
    if "transcript" in lowered or "caption" in lowered or "subtitle" in lowered:
        return "transcript"
    if any(marker in lowered for marker in ("query", "question", "prompt")):
        return "query"
    if lowered == "id" or lowered.endswith("_id"):
        return "identifier"
    return "data"


def _evaluation_only(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _EVALUATION_MARKERS)
