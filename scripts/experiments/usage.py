"""Provider-call aggregation with an explicit, frozen pricing snapshot."""

from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from .common import ExperimentDataError


def _usage_count(record: Mapping[str, Any], field: str) -> int:
    usage = record.get("usage_metadata") or {}
    if not isinstance(usage, Mapping):
        return 0
    value = usage.get(field)
    if value is None:
        camel = field.split("_")[0] + "".join(
            word.capitalize() for word in field.split("_")[1:]
        )
        value = usage.get(camel)
    return int(value) if isinstance(value, Real) and not isinstance(value, bool) else 0


def _modality_input_tokens(record: Mapping[str, Any]) -> dict[str, int]:
    usage = record.get("usage_metadata") or {}
    if not isinstance(usage, Mapping):
        return {}
    details = usage.get("prompt_tokens_details") or usage.get("promptTokensDetails") or []
    if not isinstance(details, list):
        return {}
    result: dict[str, int] = {}
    for detail in details:
        if not isinstance(detail, Mapping):
            continue
        modality = detail.get("modality")
        count = detail.get("token_count", detail.get("tokenCount"))
        if not isinstance(modality, str) or not isinstance(count, Real):
            continue
        # google-genai may serialize the value as either "AUDIO" or
        # "Modality.AUDIO" depending on the SDK/Pydantic version.
        key = modality.rsplit(".", maxsplit=1)[-1].casefold()
        result[key] = result.get(key, 0) + int(count)
    return result


def aggregate_api_usage(
    path: Path,
    *,
    input_usd_per_million_tokens: float,
    audio_input_usd_per_million_tokens: float,
    output_usd_per_million_tokens: float,
    pricing_source: str,
    error_type: type[Exception] = ExperimentDataError,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if path.is_file():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise error_type(
                    f"Invalid API usage JSON on line {line_number} of {path}"
                ) from exc
            if not isinstance(record, dict):
                raise error_type(
                    f"API usage line {line_number} of {path} must be an object"
                )
            records.append(record)

    successful = [record for record in records if record.get("status") == "ok"]
    prompt_tokens = sum(_usage_count(record, "prompt_token_count") for record in successful)
    candidate_tokens = sum(
        _usage_count(record, "candidates_token_count") for record in successful
    )
    thought_tokens = sum(
        _usage_count(record, "thoughts_token_count") for record in successful
    )
    modality_tokens: dict[str, int] = {}
    breakdown_complete = True
    for record in successful:
        breakdown = _modality_input_tokens(record)
        if not breakdown or sum(breakdown.values()) != _usage_count(
            record, "prompt_token_count"
        ):
            breakdown_complete = False
        for modality, count in breakdown.items():
            modality_tokens[modality] = modality_tokens.get(modality, 0) + count

    audio_tokens = modality_tokens.get("audio", 0) if breakdown_complete else 0
    standard_input_tokens = prompt_tokens - audio_tokens
    estimated_input_cost = (
        standard_input_tokens * input_usd_per_million_tokens
        + audio_tokens * audio_input_usd_per_million_tokens
    ) / 1_000_000
    estimated_output_cost = (
        (candidate_tokens + thought_tokens) * output_usd_per_million_tokens
    ) / 1_000_000
    warning = None
    if successful and not breakdown_complete:
        estimated_input_cost = (
            prompt_tokens * input_usd_per_million_tokens / 1_000_000
        )
        warning = (
            "Some responses lack modality token details; aggregate input tokens were "
            "priced at the non-audio rate. Raw usage records are authoritative."
        )

    return {
        "api_call_count": len(records),
        "successful_api_call_count": len(successful),
        "failed_api_call_count": len(records) - len(successful),
        "api_elapsed_seconds": sum(float(record["elapsed_seconds"]) for record in records),
        "prompt_token_count": prompt_tokens,
        "candidate_token_count": candidate_tokens,
        "thought_token_count": thought_tokens,
        "total_token_count": sum(
            _usage_count(record, "total_token_count") for record in successful
        ),
        "input_tokens_by_modality": modality_tokens,
        "modality_breakdown_complete": breakdown_complete if successful else True,
        "estimated_input_cost_usd": estimated_input_cost,
        "estimated_output_cost_usd": estimated_output_cost,
        "estimated_cost_usd": estimated_input_cost + estimated_output_cost,
        "cost_assumptions": {
            "input_usd_per_million_tokens": input_usd_per_million_tokens,
            "audio_input_usd_per_million_tokens": audio_input_usd_per_million_tokens,
            "output_usd_per_million_tokens": output_usd_per_million_tokens,
            "pricing_source": pricing_source,
            "warning": warning,
        },
    }
