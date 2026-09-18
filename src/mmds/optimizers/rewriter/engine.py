from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from time import monotonic
from typing import Any, Mapping

from pydantic import ValidationError

from ...model import DatasetExpr, QueryProgram
from ...render import program_from_plan, render_query
from .directive import (
    MMDSRewriteError,
    PlanIndex,
    RewriteAgent,
    RewriteCandidate,
    RewriteDirective,
    RewriteMatch,
    RewriteOption,
    RewriteRejection,
    RewriteResult,
    RewriteSearchResult,
    RewriteSelection,
    RewriteTrace,
    timed_agent_proposal,
)
from .context import build_rewrite_context
from .validation import validate_rewrite


def rewrite_once(
    program: QueryProgram,
    *,
    agent: RewriteAgent,
    directives: tuple[RewriteDirective, ...] | list[RewriteDirective],
) -> RewriteResult:
    index, catalog, matches, options = _prepare(
        program.output_expr,
        directives,
    )
    context = build_rewrite_context(program, index=index)
    selections, agent_seconds = timed_agent_proposal(
        agent,
        options,
        context=context,
        max_candidates=1,
    )
    if not selections:
        raise MMDSRewriteError("The rewrite agent did not select a directive.")
    return _apply_selection(
        program,
        selections[0],
        index=index,
        catalog=catalog,
        matches=matches,
        agent_seconds=agent_seconds,
        agent_details=getattr(agent, "last_trace", None),
    )


def search_rewrites(
    program: QueryProgram,
    *,
    agent: RewriteAgent,
    directives: tuple[RewriteDirective, ...] | list[RewriteDirective],
    max_depth: int = 1,
    max_candidates: int = 8,
) -> RewriteSearchResult:
    if max_depth != 1:
        raise MMDSRewriteError("The initial search engine supports only max_depth=1.")
    if (
        not isinstance(max_candidates, int)
        or isinstance(max_candidates, bool)
        or max_candidates <= 0
    ):
        raise MMDSRewriteError("max_candidates must be a positive integer.")

    index, catalog, matches, options = _prepare(
        program.output_expr,
        directives,
    )
    context = build_rewrite_context(program, index=index)
    selections, agent_seconds = timed_agent_proposal(
        agent,
        options,
        context=context,
        max_candidates=max_candidates,
    )
    truncated = len(selections) > max_candidates
    selections = selections[:max_candidates]

    baseline_fingerprint = plan_fingerprint(program.output_expr)
    candidates: list[RewriteCandidate] = [
        RewriteCandidate(
            program=program,
            fingerprint=baseline_fingerprint,
            trace=None,
        )
    ]
    fingerprints = {baseline_fingerprint}
    rejections: list[RewriteRejection] = []

    for selection in selections:
        try:
            result = _apply_selection(
                program,
                selection,
                index=index,
                catalog=catalog,
                matches=matches,
                agent_seconds=agent_seconds,
                agent_details=getattr(agent, "last_trace", None),
            )
        except MMDSRewriteError as exc:
            rejections.append(
                RewriteRejection(
                    directive=selection.directive,
                    match_id=selection.match_id,
                    reason=str(exc),
                )
            )
            continue
        fingerprint = result.trace.rewritten_fingerprint
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        candidates.append(
            RewriteCandidate(
                program=result.program,
                fingerprint=fingerprint,
                trace=result.trace,
            )
        )

    return RewriteSearchResult(
        candidates=tuple(candidates),
        rejections=tuple(rejections),
        truncated=truncated,
    )


def plan_fingerprint(plan: DatasetExpr) -> str:
    normalized = render_query(program_from_plan(plan)).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _prepare(
    plan: DatasetExpr,
    directives: tuple[RewriteDirective, ...] | list[RewriteDirective],
) -> tuple[
    PlanIndex,
    dict[str, RewriteDirective],
    dict[tuple[str, str], RewriteMatch],
    tuple[RewriteOption, ...],
]:
    catalog: dict[str, RewriteDirective] = {}
    for directive in directives:
        name = directive.metadata.name
        if name in catalog:
            raise MMDSRewriteError(f"Duplicate rewrite directive name {name!r}.")
        catalog[name] = directive

    index = PlanIndex.build(plan)
    matches: dict[tuple[str, str], RewriteMatch] = {}
    options: list[RewriteOption] = []
    for directive in directives:
        for match in directive.find_matches(index):
            key = (directive.metadata.name, match.id)
            if key in matches:
                raise MMDSRewriteError(
                    f"Directive {directive.metadata.name!r} produced duplicate match ID {match.id!r}."
                )
            matches[key] = match
            options.append(
                RewriteOption(
                    directive=directive.metadata.name,
                    match_id=match.id,
                    path=str(match.path),
                    summary=match.summary,
                    description=directive.metadata.description,
                    when_to_use=directive.metadata.when_to_use,
                    params_schema=directive.params_type.model_json_schema(),
                )
            )
    return index, catalog, matches, tuple(options)


def _apply_selection(
    program: QueryProgram,
    selection: RewriteSelection,
    *,
    index: PlanIndex,
    catalog: dict[str, RewriteDirective],
    matches: dict[tuple[str, str], RewriteMatch],
    agent_seconds: float,
    agent_details: Mapping[str, Any] | None,
) -> RewriteResult:
    started = monotonic()
    directive = catalog.get(selection.directive)
    if directive is None:
        raise MMDSRewriteError(
            f"Rewrite agent selected unknown directive {selection.directive!r}."
        )
    match = matches.get((selection.directive, selection.match_id))
    if match is None:
        raise MMDSRewriteError(
            f"Rewrite agent selected unoffered match {selection.match_id!r} for directive {selection.directive!r}."
        )

    try:
        raw_json = json.dumps(dict(selection.params))
        params = directive.params_type.model_validate_json(raw_json)
    except (TypeError, ValueError, ValidationError) as exc:
        raise MMDSRewriteError(
            f"Invalid parameters for directive {selection.directive!r}: {exc}"
        ) from exc

    rewritten = directive.apply(
        index,
        match,
        params,
    )
    validate_rewrite(program, rewritten)

    normalized = program_from_plan(rewritten)
    normalized = replace(normalized, path=program.path)
    original_fingerprint = plan_fingerprint(program.output_expr)
    rewritten_fingerprint = plan_fingerprint(rewritten)
    trace = RewriteTrace(
        directive=selection.directive,
        match_id=selection.match_id,
        path=str(match.path),
        params=params.model_dump(mode="json"),
        original_fingerprint=original_fingerprint,
        rewritten_fingerprint=rewritten_fingerprint,
        rewrite_seconds=agent_seconds + (monotonic() - started),
        agent_details=agent_details,
    )
    return RewriteResult(original=program, program=normalized, trace=trace)
