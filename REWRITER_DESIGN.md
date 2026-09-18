# MMDS Video Query Rewriter

`DESIGN.md` is authoritative. This document is the short guide to the typed
directive rewriter.

## Goal

Users write ordinary MMDS queries. The rewriter produces inspectable MMDS plan
alternatives that can be evaluated for accuracy, cost, and latency.

```text
Map(video -> events)

may become

Map(transcript -> candidate intervals)
-> VideoMapEach(candidate views -> local events)
-> rebase timestamps
-> reconcile events
```

The model never returns Python or a replacement plan. It selects a registered
directive and supplies strictly typed semantic parameters. Python directive
code performs the structural transformation.

## Lifecycle

```text
Python DSL
   -> immutable QueryProgram / DatasetExpr
   -> PlanIndex finds directive matches
   -> RewriteContext summarizes the query and input field shapes
   -> model call 1 selects directive + match
   -> model call 2 supplies semantic parameters
   -> Pydantic validation
   -> deterministic directive.apply(...)
   -> global plan validation
   -> normalized Python / execution / measurement
```

`DatasetExpr` is the only operator-tree representation:

- `NodePath` is an address such as `output.source.source`.
- `PlanEntry` associates an address with its existing node and field effects.
- `PlanIndex` supports matching, lookup, and immutable subtree replacement.
- `RewriteMatch` identifies one valid directive application location.

Replacing a node rebuilds only its ancestors. The original plan is unchanged.

## Automatic model context

The caller does not write a workload-specific optimization objective. The
system uses a fixed policy: propose semantically valid alternatives that may
reduce cost or latency while preserving query meaning, schemas, inputs,
modalities, and temporal coordinates.

`RewriteContext` is generated from the query and file-backed inputs. It records:

- the operator chain, names, grouping, prompts, UDFs, and output schemas;
- input field shapes inferred from at most 32 JSON/JSONL rows; and
- semantic roles such as video, query, identifier, and timestamped transcript.

Row values are never included. Evaluation-looking fields such as ground truth,
labels, and annotations are excluded unless the original query reads them.

Each model call receives only the projection it needs:

1. Selection receives a compact plan outline, detailed task templates for
   matched nodes, relevant field roles, and directive choices grouped by node.
2. Instantiation receives only selected nodes, compatible fields, and compact
   parameter contracts.

Prompts and raw model responses are retained in `RewriteTrace`.

## Directive contract

```python
class RewriteDirective(Protocol):
    metadata: DirectiveMetadata
    params_type: type[BaseModel]

    def find_matches(self, index: PlanIndex) -> tuple[RewriteMatch, ...]: ...

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr: ...
```

Parameter models are strict, frozen, and reject unknown fields. Model-generated
parameters express semantics; plan structure and physical settings remain
deterministic.

## Implemented directives

| Directive | Transformation | Intended use |
|---|---|---|
| `modality_substitution` | `Map(video)` -> `Map(transcript)` | Spoken content is sufficient |
| `projection_before_map` | `Map(x)` -> projection `Map` -> original `Map` -> `DropFields` | A shorter representation may improve cost or focus |
| `temporal_pushdown_joint` | transcript candidates -> `VideoMap` | One answer across selected views |
| `temporal_pushdown_per_view` | transcript candidates -> `VideoMapEach` -> rebase -> reconcile | Event localization or exhaustive retrieval |

Temporal model parameters are limited to:

```python
video_field: str
transcript_field: str
query_field: str
candidate_prompt: str
```

Grouping is derived from the plan. Temporary field names, padding, view limits,
and video-duration budgets are system or experiment configuration.

## Logical video operators

- `VideoMap` answers once over a bounded set of lazy video views.
- `VideoMapEach` applies the same semantic map independently to every view.

Lowering turns them into executable primitives:

```text
Unnest -> Window -> Coalesce -> ViewBudget -> Map/Reduce
```

Views keep the original video source plus absolute `start` and `end`; no clip
file is materialized. Per-view localization adds deterministic timestamp
rebasing and reconciliation.

## API

```python
result = rewrite_once(
    program,
    agent=model_agent,
    directives=[
        ModalitySubstitution(),
        ProjectionBeforeMap(),
        JointTemporalPushdown(padding_seconds=10),
        PerViewTemporalPushdown(padding_seconds=10),
    ],
)

print(render_query(result.program))
print(result.trace)
```

`StaticRewriteAgent` selects exact directives and parameters for deterministic
tests and experiment replay. `search_rewrites(..., max_candidates=N)` retains
the original plan, applies bounded one-hop proposals, records expected
rejections, and deduplicates normalized plans.

## Safety and validation

Every candidate must:

- preserve all `Input(...)` paths;
- preserve the final output schema;
- use registered operators and executable UDFs;
- pass directive parameter validation; and
- render and parse as normalized MMDS Python.

The model can select only offered directive/match pairs. Expected applicability
or validation failures use `MMDSRewriteError`; programming errors are not
swallowed.

## Measurement

`execute_measured(...)` reports end-to-end time, prompt calls and latency,
submitted `VideoView` seconds, input rows, and output rows. Accuracy and
provider-specific monetary cost remain experiment-level metrics.

The original plan is always the baseline when searching rewrite candidates.

## Implementation map

```text
src/mmds/optimizers/rewriter/
    context.py         automatic plan and dataset context
    directive.py       contracts, NodePath, PlanIndex, static agent
    engine.py          rewrite_once and bounded search
    model_agent.py     compact two-call model interaction
    validation.py      global rewrite invariants
    directives/        deterministic transformations

src/mmds/optimizers/lowering.py
    VideoMap / VideoMapEach lowering
```

Run the complete suite with:

```bash
uv run --locked python -m unittest discover -s tests -t .
```
