from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from ...model import MMDSValidationError
from ...parser import parse_query
from ...render import render_query

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class StaticLLMClient:
    response: str

    def generate(self, prompt: str) -> str:
        return self.response


def rewrite(query_text: str, llm_client: LLMClient, objective: str | None = None) -> str:
    original = parse_query(query_text)
    prompt = build_rewrite_prompt(query_text, objective=objective)
    logger.debug("Sending rewrite prompt to LLM:\n%s", prompt)
    raw_response = llm_client.generate(prompt)
    rewritten_text = _extract_python(raw_response)
    rewritten = parse_query(rewritten_text)
    if rewritten.input_paths() != original.input_paths():
        raise MMDSValidationError("LLM rewrites must preserve the same Input(...) file paths.")
    return render_query(rewritten)


def build_rewrite_prompt(query_text: str, objective: str | None = None) -> str:
    objective_line = f"Optimization objective: {objective}.\n" if objective else ""
    return (
        "Rewrite the following MMDS query.\n"
        f"{objective_line}"
        "Constraints:\n"
        "- Output only Python code.\n"
        "- Stay within the straight-line MMDS DSL subset.\n"
        "- Preserve the same Input(...) file paths.\n"
        "- Use only Input, Map, Filter, Reduce, Unnest, Record, and ForEach.\n"
        '- Prompt-backed Map/Reduce operators must include schema=... using the concise field-map form, for example schema={"summary": "string"}.\n\n'
        f"{query_text.strip()}\n"
    )


def _extract_python(response: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return response.strip()
