from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ...model import QueryProgram
from .agent import LLMClient
from .context import build_rewrite_context
from .core import NodePath, PlanIndex, RewriteDirective, apply_rewrite
from .selector import (
    enumerate_options,
    instantiate_parameters,
    select_option,
)


@dataclass(frozen=True)
class RewriteResult:
    program: QueryProgram
    directive: str | None
    path: NodePath | None
    parameters: Mapping[str, Any]

    @property
    def applied(self) -> bool:
        return self.directive is not None


def rewrite_once(
    program: QueryProgram,
    *,
    directives: Sequence[RewriteDirective],
    model: LLMClient,
) -> RewriteResult:
    """Use two model calls to choose, instantiate, and apply at most one rewrite."""

    index = PlanIndex.build(program.output_expr)
    options = enumerate_options(directives, index=index)
    if not options:
        return RewriteResult(
            program=program,
            directive=None,
            path=None,
            parameters={},
        )

    context = build_rewrite_context(program, index=index)
    selected = select_option(model, context=context, options=options)
    if selected is None:
        return RewriteResult(
            program=program,
            directive=None,
            path=None,
            parameters={},
        )

    parameters = instantiate_parameters(
        model,
        context=context,
        option=selected,
    )
    rewritten = apply_rewrite(
        program,
        directive=selected.directive,
        match=selected.match,
        params=parameters,
    )
    return RewriteResult(
        program=rewritten,
        directive=selected.directive.metadata.name,
        path=selected.match.path,
        parameters=dict(parameters),
    )
