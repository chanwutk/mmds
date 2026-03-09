Review this plan thoroughly before making any code changes. For every issue or recommendation, explain the concrete tradeoffs, give me an opinionated recommendation, and ask for my input before assuming a direction.
My engineering preferences (use these to guide your recommendations):
* DRY is important-flag repetition aggressively.
* Well-tested code is non-negotiable, I'd rather have too many tests than too few.
* I want code that's "engineered enough"--not under-engineered (fragile, hacky) and not over-engineered (premature abstraction, unnecessary complexity).
* l err on the side of handling more edge cases, not fewer; thoughtfulness > speed.
* Bias toward explicit over cleverness.

1. Architecture review
Evaluate:
  * Overall system design and component boundaries.
  * Dependency graph and coupling concerns.
  * Data flow patterns and potential bottlenecks.
  * Scaling characteristics and single points of failure.
  * Security architecture (auth, data access, API boundaries)
2. Code quality review
Evaluate:
  * Code organization and module structure.
  * DRY violations–be aggressive here.
  * Error handling patterns and missing edge cases (call these out explicitly).
  * Technical debt hotspots.
  * Areas that are over-engineered or under-engineered relative to my preferences.
3. Test review
Evaluate:
  * Test coverage gaps (unit, integration, e2e). test quay and assertion strength
  * Missing edge case coverage–be thorough.
  * Untested failure modes and error paths.
4. Performance review
Evaluate:
  * N+1 queries and database access patterns.
  * Memory-usage concerns.
  * Caching opportunities.
  * Slow or high-complexity code paths.

For each issue you find
For every specific issue (bug, smell, design concern, or risk):
* Describe the problem concretely, with file and line references.
* Presence-s options, including do nothing where that's reasonable.
* For each option, specify: implementation effort, risk, impact on other code, and maintenance burden.
* Give me your recommended option and why, mapped to my preferences above.
* Then explicitly ask whether i agree or want to choose a different direction before proceeding
Workflow and interaction
* Do not assume my priorities on timeline or scale.
* After each section, pause and ask for my feedback before moving on.

BEFORE YOU START:
Ask it I want one of two options:
1/ BIG CHANGE: Work through this interactively, one section at a time (Architecture → Code Quality → Tests → Performance) with at most 4 top issues in each section.
SMALL CHANGE: work through interactively oNe question per review section

FOR EACH STAGE OF REVIEW: output the explanation and pros and cons of each stage's questions AND your opinionated recommendation and why, and then use AskUserQuestion. Also NUMBER issues and then give LETTERS for options and when using AskUserQuestion make sure each option clearly labels the issue NUMBER and option LETTER so the user doesn't get confused. Make the recommended option always the 1st option.


# AGENTS.md

## Purpose

This repository implements a Python DSL for multimodal data workflows. Before making changes, read [DESIGN.md](/Users/chanwutk/Documents/mmds/DESIGN.md). It is the authoritative architecture document for the current codebase.

## Required Workflow

- Keep `DESIGN.md` in sync with the code.
- If you change public APIs, operator semantics, plan structure, parsing rules, rendering rules, optimizer behavior, or the UDF contract, update `DESIGN.md` in the same change.
- If you add behavior without updating `DESIGN.md`, the change is incomplete.
- Add or update tests for every behavior change.

## Current Architectural Constraints

- The query language is a restricted Python subset: imports plus top-level assignments only.
- Supported operators are `Input`, `Map`, `Filter`, `Reduce`, and `Unnest`.
- `Input(...)` takes a `.json` or `.jsonl` file path directly; do not reintroduce a catalog abstraction unless explicitly requested.
- Prompt-backed operators support strings and structured prompt lists using `Record[...]` and `ForEach([...])`.
- Prompt-backed `Map` and `Reduce` require `schema=...`.
- Inline lambdas and nested functions are not supported.
- Plans are represented by immutable `DatasetExpr` nodes.
- Query regeneration targets normalized Python, not source-exact reproduction.
- `.pyi` files define planned UDF contracts only; they are not executable implementations.
- Media types should stay DSL-agnostic. Provider-specific translation belongs in executors, not in query syntax.

## Implementation Notes

- Put core runtime and planning logic under `src/mmds/`.
- Keep user-defined function examples and test fixtures under `udfs/`.
- Prefer extending the existing parser/renderer/plan model instead of adding ad hoc side paths.
- Preserve validation-heavy behavior. Unsupported syntax should fail explicitly.

## Verification

Run at least:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m unittest discover -s tests -t .
```

If you change semantics, parsing, or optimizers, expand tests accordingly.
