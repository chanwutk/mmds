• The design is centered on one idea: the same MMDS query should exist in three equivalent forms.

  1. Python DSL text that users write.
  2. An immutable logical plan tree.
  3. A locally executable runtime plan.

  That is the core contract described in DESIGN.md. The user-facing API is small and lives in src/mmds/init.py: Input, Map,
  Filter, Reduce, and Unnest.

  A typical query starts as straight-line Python:

  from mmds import Input, Map, Filter, Reduce, Unnest
  from udfs.test_ops import add_bucket, summarize_group

  docs = Input("docs")
  mapped = Map(docs, add_bucket)
  filtered = Filter(mapped, "keep rows with useful content")
  expanded = Unnest(filtered, "tags", keep_empty=True)
  output = Reduce(expanded, ["bucket"], summarize_group)

  From there, the system can go in two directions. If you execute that code directly, the DSL constructors build an
  immutable operator tree using DatasetExpr in src/mmds/model.py. If you load the query as text, the parser in src/mmds/
  parser.py reads a restricted Python subset and produces a QueryProgram, which is basically the assignment sequence plus
  the output variable. The important design choice is that parsed source text is the canonical path for rewrite fidelity,
  while runtime construction is still valid for execution and planning.

  The plan model is intentionally simple right now. Every operator is unary, so each DatasetExpr has at most one source.
  Semantic behavior is attached as either a PromptSpec or a UdfSpec. That means the system does not treat arbitrary Python
  as query semantics. It only allows prompt strings or imported functions from udfs.*. Inline lambdas are rejected on
  purpose. That keeps plans serializable, renderable, and analyzable.

  Execution lives in src/mmds/execution.py. The runtime assumes datasets are Iterable[dict]. Map merges returned fields
  into each row, Filter keeps rows with truthy results, Reduce groups rows and merges aggregate output with the group keys,
  and Unnest explodes one field. Prompt-backed operators do not directly call an LLM; instead they require an injected
  PromptExecutor. That separation is deliberate. It keeps the logical design independent from provider details and makes
  tests deterministic.

  The renderer in src/mmds/render.py turns plans back into normalized Python. “Normalized” matters here: it preserves
  meaning, not original formatting. It will regenerate imports, stable operator calls, and synthesized variable names like
  step_1 if needed. So this is semantic round-tripping, not exact source preservation.

  There are two optimizer layers by design. The rule optimizer in src/mmds/rule_optimizer.py works on the operator tree and
  is conservative right now: mostly canonical rebuild and deduplication. The LLM optimizer in src/mmds/llm_optimizer.py
  works on Python query text. It asks an LLMClient for rewritten code, then reparses and validates the result. The key
  safety rule today is that rewritten queries must preserve the same Input(...) roots.

  The UDF story is split cleanly. Implemented functions live in .py files under udfs/. Declared-only future functions can
  live in .pyi files. Discovery is handled by src/mmds/udf_catalog.py. That gives you a stable contract for future code
  generation without pretending those .pyi-only functions are executable today.

  So the design priorities are:

  - keep the DSL small and analyzable
  - keep plans immutable
  - separate source rewriting from plan rewriting
  - allow semantic operators through prompts or UDFs
  - reject unsupported Python early instead of being permissive

  If useful, I can next walk through one specific layer in detail:

  - parser and AST restrictions
  - execution semantics operator by operator
  - optimizer design and likely next extensions




























› Please do one by one.


• 1. Parser And AST Restrictions

  This layer is in src/mmds/parser.py. Its job is to turn Python query text into a validated QueryProgram from src/mmds/
  model.py.

  The key design decision is: MMDS does not accept “general Python.” It accepts a small Python-shaped DSL. That constraint is
  what makes plan reconstruction, rule rewriting, and LLM rewriting reliable.

  The parser currently allows only:

  - an optional module docstring
  - from mmds import ...
  - from udfs... import ...
  - top-level assignments like x = Map(...)

  Everything else is rejected. So for, if, comprehensions, classes, helper functions, arbitrary expressions, and nested
  control flow are all outside the language for now.

  The pipeline is:

  - load_query(...) accepts either a file path or raw source text.
  - parse_query(...) builds a Python AST with ast.parse(...).
  - It walks the module body in order.
  - Imports are validated first.
  - Each assignment is converted into an Assignment(target, expr).
  - The last assignment becomes the query output.

  There are a few important rules baked in:

  - Imports from mmds must be direct and unaliased.
  - Imports from udfs.* must also be direct and unaliased.
  - Only previously assigned variables can be used as operator inputs.
  - Semantic specs must be either:
      - a string literal, which becomes PromptSpec
      - an imported UDF name, which becomes UdfSpec
  - Inline lambdas are rejected.
  - Reduce.group_by must be a string or a literal list/tuple of strings.
  - Unnest.keep_empty must be a literal True or False.

  That means this is valid:

  docs = Input("docs")
  mapped = Map(docs, "extract title")
  output = Filter(mapped, keep_large)

  and this is invalid:

  output = Map(Input("docs"), lambda row: row)

  because the parser and model intentionally do not allow inline callables.

  Why this matters:

  - the parser can always rebuild a clean operator tree
  - the renderer can turn that tree back into normalized Python
  - the LLM optimizer can be forced to stay inside a checkable subset
  - unsupported syntax fails early instead of creating ambiguous semantics

  So the parser is not just a loader. It is the language boundary for the whole system.

  Next I can walk through 2. execution semantics operator by operator.


› please do


• 2. Execution Semantics Operator By Operator

  This layer lives in src/mmds/execution.py. Its job is to take either a DatasetExpr plan or a parsed QueryProgram and run it
  over in-memory datasets.

  execute(...) expects:

  - a plan or query
  - inputs: Mapping[str, Iterable[Mapping[str, Any]]]
  - an optional prompt executor

  The runtime model is intentionally simple: each dataset is a sequence of dict-like rows, and the executor recursively
  evaluates the operator tree from the leaves upward.

  The first important rule is that execution copies input rows into plain dicts. That means MMDS does not mutate the caller’s
  original row objects.

  The operators work like this:

  - Input("docs")
      - Reads inputs["docs"].
      - Fails if that dataset name is missing.
      - Produces a list of copied rows.
  - Map(data, spec)
      - Runs the semantic spec once per row.
      - The result must be mapping-like.
      - The returned fields are merged into the original row copy.
      - So Map is row-preserving: one input row becomes one output row.
  - Filter(data, spec)
      - Runs the semantic spec once per row.
      - Keeps the row if the result is truthy.
      - The spec can return bool, but the runtime only requires truthiness.
  - Reduce(data, group_by, reducer)
      - Groups rows by the configured key fields.
      - Calls the reducer once per group, with the whole list of rows for that group.
      - The reducer must return a mapping-like aggregate row.
      - The runtime merges aggregate fields with the group key fields.
      - Special case: group_by="_all" creates one global group and emits one aggregate row without key fields.
  - Unnest(data, field, keep_empty=False)
      - Looks at one field in each row.
      - If the value is a list or tuple, it emits one output row per item.
      - If the value is a scalar, it passes the row through unchanged.
      - If the value is None, missing, or an empty list:
          - it emits nothing by default
          - it emits one row with field=None when keep_empty=True

  Semantic execution is delegated through _execute_spec(...):

  - If the spec is a PromptSpec, MMDS requires a PromptExecutor.
  - It calls PromptExecutor.execute(op_type, spec_text, payload, context).
  - If the spec is a UdfSpec, MMDS imports the function from udfs.* and calls it directly.

  That split is important. It means the logical plan does not know or care how prompts are actually served. Local tests can
  use StaticPromptExecutor, while a future system can plug in a real LLM-backed executor later.

  There are also a few enforced runtime contracts:

  - Map must return a mapping
  - Reduce must return a mapping
  - prompt-backed execution fails fast if no prompt executor is provided
  - UDFs must resolve back to the expected udfs.* symbol

  So the execution layer is deliberately not “smart.” It is a deterministic interpreter for the logical plan, with strict
  contracts that make semantic operators testable and composable.

  Next is 3. optimizer design and why it is split into rule_optimizer and llm_optimizer.


› Please do.


• 3. Optimizer Design And Why It Is Split

  The optimizer is split because MMDS has two different rewrite targets:

  - Python query text
  - logical operator trees

  Those are related, but they are not the same problem.

  The code reflects that split:

  - src/mmds/rule_optimizer.py
  - src/mmds/llm_optimizer.py

  ### Rule Optimizer

  rule_optimizer works on DatasetExpr plans, not source text.

  That means it is the right place for rewrites that are:

  - structural
  - deterministic
  - explainable
  - independent of Python formatting or variable names

  Today it is intentionally conservative. optimize(plan) recursively rebuilds the tree and memoizes equivalent nodes, so the
  current effect is mostly canonicalization and structural deduplication.

  Why start small:

  - prompt semantics are opaque
  - UDF semantics are also opaque unless we add annotations later
  - aggressive rewrites would be unsafe without semantic guarantees

  So for now, rule-based optimization is limited to transformations that do not require understanding what a prompt or UDF
  “means.”

  Longer term, this is where things like these would belong:

  - removing redundant operators
  - pushing filters earlier when proven safe
  - fusing compatible operators
  - normalizing equivalent plan shapes
  - adding cost-based or heuristic planning

  ### LLM Optimizer

  llm_optimizer works on Python query text.

  That is a different use case: instead of transforming the already-built tree, it asks an LLM to rewrite the user’s program
  itself. The output is still Python DSL text, not a plan object.

  The flow in src/mmds/llm_optimizer.py is:

  1. Parse the original query.
  2. Build a constrained rewrite prompt.
  3. Ask an LLMClient for rewritten code.
  4. Extract Python from the response.
  5. Parse the rewritten code again.
  6. Validate that required invariants still hold.
  7. Return normalized rendered Python.

  The important design point is that the LLM is never trusted directly. Its output must pass back through the same parser and
  validation rules as handwritten queries.

  Right now the main safety check is that the rewritten query must preserve the same Input(...) roots. That prevents the
  optimizer from silently changing what datasets the query reads.

  ### Why Both Are Needed

  If MMDS had only a rule optimizer, it would be limited to rewrites we can prove structurally. That is safe, but narrow.

  If MMDS had only an LLM optimizer, rewrites would be flexible, but much harder to trust.

  The split gives you both:

  - rule_optimizer for safe symbolic rewrites on plans
  - llm_optimizer for broader source-level rewrites under validation

  That also matches the project’s architecture:

  - source text is important because queries must be regenerable and LLM-rewritable
  - plans are important because DB-style optimization should happen over operator trees

  So the split is not accidental. It comes directly from the requirement that MMDS supports both Python-form rewriting and
  operator-tree rewriting.

  ### Current State vs Future State

  Current state:

  - rule_optimizer is mostly canonicalization
  - llm_optimizer is mostly interface + validation scaffold

  Future state likely:

  - richer rule library in rule_optimizer
  - semantic safety metadata for prompts/UDFs
  - real LLM provider integration in llm_optimizer
  - objective-aware rewriting and plan comparison

  The key invariant is that both optimizers must stay inside the MMDS language boundary defined by src/mmds/parser.py.

  Next I can do 4. the UDF contract and why .py and .pyi are both part of the design.


› Please do.


• 4. The UDF Contract And Why .py And .pyi Both Matter

  The UDF design exists to solve two different needs at once:

  - users need executable custom logic today
  - the system needs a way to represent intended-but-not-yet-implemented logic for future synthesis

  That contract is split across:

  - src/mmds/model.py
  - src/mmds/dsl.py
  - src/mmds/parser.py
  - src/mmds/udf_catalog.py

  ### Executable UDFs

  Executable UDFs live in udfs/*.py, for example udfs/test_ops.py.

  A query imports them directly:

  from udfs.test_ops import add_bucket

  docs = Input("docs")
  output = Map(docs, add_bucket)

  The system immediately converts that callable into a stable UdfSpec(module, name). That is important: the logical plan does
  not store an arbitrary Python function object. It stores an importable identity like:

  - module: udfs.test_ops
  - name: add_bucket

  That makes the plan:

  - serializable
  - renderable back to Python
  - comparable across runs
  - usable by optimizers

  At execution time, MMDS resolves that UdfSpec back to the actual function and calls it.

  ### Why Lambdas Are Rejected

  Inline lambdas and nested functions are deliberately blocked.

  Reason: they are bad plan artifacts.

  A lambda does not give the system a stable import path, and it is hard to:

  - round-trip back to source
  - compare structurally
  - validate consistently
  - ship across optimizer boundaries

  So the design forces user logic into named functions under udfs.*. That is stricter than plain Python, but much better for a
  DSL.

  ### What .pyi Is For

  Declared-only UDFs live in udfs/*.pyi, for example udfs/planned_only.pyi.

  A .pyi file lets the user specify:

  - function name
  - signature
  - docstring / intended behavior

  without implementing the body yet.

  That matters because your intended roadmap includes: the user may define only the interface and description, and the system
  will eventually write the function.

  So .pyi is not just type metadata here. It is a design-time contract for future synthesis.

  ### Why Keep Both .py And .pyi

  They represent different states of the same concept.

  - .py: implemented, executable behavior
  - .pyi: planned, non-executable behavior contract

  Keeping both gives MMDS a clean lifecycle:

  1. User declares what a UDF should do in .pyi.
  2. Future synthesis machinery can generate .py.
  3. Once implemented, queries can execute against the real function.

  This is why src/mmds/udf_catalog.py discovers both. discover_udfs() builds a catalog that records whether each UDF is
  implemented and, if not, what its declared signature/docstring is.

  ### Important Current Limitation

  Right now, .pyi entries are discoverable but not executable.

  That means:

  - they can be cataloged
  - they can inform future tooling
  - they cannot yet be imported and run in a query

  So the current implementation supports the contract boundary, not the synthesis workflow itself.

  ### Why This Design Is Good

  It keeps user logic structured enough for a database-like system.

  Instead of “any Python callable,” MMDS gets:

  - stable function identities
  - analyzable query plans
  - future code-generation hooks
  - a clean separation between declared intent and executable implementation

  That is the real reason .py and .pyi are both part of the design: one is for execution, the other is for planned semantics.

  Next I can do 5. rendering and round-tripping: how MMDS turns plans back into Python and what fidelity is intentionally
  lost.


› Please do.


• 5. Rendering And Round-Tripping

  This part lives mainly in src/mmds/render.py, with the parser side in src/mmds/parser.py.

  The purpose is simple: a query should not become trapped inside an internal plan representation. MMDS must be able to take a
  plan and regenerate Python DSL code.

  That matters for two reasons:

  - the query may be rewritten by rule_optimizer and then shown back to the user as Python
  - the query may be rewritten by llm_optimizer, reparsed, and normalized into a stable form

  ### Two Round-Trip Paths

  There are really two related round-trip flows.

  First:

  - Python source
  - parser
  - QueryProgram
  - renderer
  - normalized Python source

  Second:

  - runtime-built DatasetExpr
  - program_from_plan(...)
  - QueryProgram
  - renderer
  - normalized Python source

  So the renderer is the bridge from internal representation back to the user-facing DSL.

  ### What The Renderer Preserves

  The renderer preserves semantics, not original formatting.

  It will regenerate:

  - the core from mmds import ... import
  - from udfs... import ... imports for referenced UDFs
  - assignment order
  - operator structure
  - prompt strings
  - UDF references by importable symbol name
  - normalized operator calls

  So if the plan means “Map docs with annotate, then Filter the result,” the rendered code will still mean exactly that.

  ### What It Does Not Preserve

  It does not try to preserve exact original source fidelity.

  That means it may change:

  - whitespace
  - quote style
  - import grouping
  - variable names when rendering from a bare plan
  - comments
  - formatting details

  For example, if a plan was built at runtime rather than parsed from source, MMDS has to synthesize names. That is why
  program_from_plan(...) may generate names like:

  - source_docs
  - step_1
  - output

  This is intentional. The system is aiming for stable semantic regeneration, not source-code archaeology.

  ### Why Normalization Matters

  Normalization is useful because it gives the system a canonical textual form.

  That helps with:

  - comparing rewritten queries
  - testing parse/render/parse equivalence
  - feeding stable code to downstream tools
  - avoiding meaningless diffs caused only by formatting

  So normalization is not a cosmetic choice. It is part of making the DSL mechanically reliable.

  ### How Rendering Works

  At a high level:

  - render_query(...) accepts either a QueryProgram or a DatasetExpr
  - if given a raw plan, it first calls program_from_plan(...)
  - it collects UDF imports from the plan
  - it emits import lines
  - it emits assignments in plan order
  - each operator is rendered in normalized call syntax

  Examples:

  - Input("docs")
  - Map(source_docs, annotate)
  - Filter(step_1, "keep labels")
  - Reduce(step_2, ["bucket"], summarize_group)
  - Unnest(step_3, "tags", keep_empty=True)

  ### Design Tradeoff

  The key tradeoff is this:

  - exact source preservation would require storing much more AST/source detail
  - normalized regeneration is simpler, safer, and enough for optimizer workflows

  Right now MMDS chooses the second option.

  That choice fits the current system because the main requirement is:
  “reconstruct the query as Python,” not “reconstruct the exact same file byte for byte.”

  ### Why This Matters For Future Optimizers

  This rendering layer is what makes the whole optimization architecture coherent.

  Without it:

  - rule-based plan rewrites could not be shown back as Python
  - runtime-built queries could not be exported cleanly
  - LLM rewrites would not have a canonical final form

  So rendering is not just output formatting. It is one of the core interoperability layers of the system.

  Next I can do 6. the main invariants of the whole design and what must stay true as the project grows.


› Please do.


• 6. Main Invariants Of The Whole Design

  These are the design rules that hold the system together. If any of them change, DESIGN.md should change with the code.

  ### 1. MMDS Is A Restricted Python DSL, Not General Python

  The system looks like Python on purpose, but it is not trying to support arbitrary Python programs.

  Current allowed shape:

  - imports
  - top-level assignments
  - direct DSL operator calls

  This invariant is what makes the parser, renderer, and both optimizers tractable. If MMDS starts accepting arbitrary control
  flow, then source rewriting and plan reconstruction become much harder and much less reliable.

  ### 2. The Same Query Must Exist In Three Equivalent Forms

  A query must be representable as:

  - Python DSL text
  - a QueryProgram
  - a DatasetExpr plan

  This is one of the deepest invariants in the design. The whole architecture assumes those forms can convert cleanly enough
  to support parsing, execution, plan rewriting, and regeneration.

  ### 3. Plans Are Immutable

  DatasetExpr is frozen in src/mmds/model.py. That means a plan node is not mutated after creation.

  This matters because immutable plans are easier to:

  - compare
  - memoize
  - rewrite safely
  - share across optimizer passes

  If later optimization becomes more complex, this invariant becomes even more important.

  ### 4. Semantic Specs Are Only Prompts Or UDFs

  Every semantic operator is driven by either:

  - PromptSpec
  - UdfSpec

  Not arbitrary code.

  That keeps operator semantics explicit and serializable. It is what lets MMDS store behavior in a plan without embedding
  opaque runtime state.

  ### 5. Executable UDFs Must Come From udfs.*

  This is a deliberate boundary. It ensures UDFs have:

  - stable module names
  - stable symbol names
  - a place for discovery
  - a clean path for future synthesis workflows

  Without this invariant, UDFs would be much harder to track and regenerate.

  ### 6. Inline Lambdas And Nested Functions Are Not Supported

  This is really a corollary of the UDF rule, but important enough to call out on its own.

  The system is rejecting these not because Python cannot run them, but because MMDS needs stable, analyzable semantics.

  ### 7. Rendered Queries Are Normalized, Not Source-Exact

  When MMDS regenerates Python, it preserves meaning rather than exact formatting.

  This invariant keeps the renderer simple and keeps textual output canonical. If exact source fidelity ever becomes a
  requirement, that would be a major architectural change.

  ### 8. Prompt Execution Is Always Pluggable

  The core runtime does not directly depend on an LLM provider. Prompt-backed execution goes through PromptExecutor.

  That separation is important because it keeps:

  - testing deterministic
  - the runtime provider-agnostic
  - the logical plan independent of external service details

  ### 9. .pyi Discovery Does Not Mean Executability

  Stub-only UDFs are part of the contract layer, not the runtime layer.

  This keeps future synthesis support from leaking confusing behavior into current execution.

  ### 10. Optimizers Must Stay Inside The DSL Boundary

  Both optimizers are constrained by the same parser-defined language.

  - rule_optimizer must output valid plans
  - llm_optimizer must output valid DSL Python that reparses cleanly

  This is the key safety property of the rewrite system.

  ### Why These Invariants Matter

  If you strip it down, the project is trying to behave like a semantic data system, not just a Python library. These
  invariants are what make that possible.

  They are what let MMDS be:

  - analyzable
  - rewritable
  - renderable
  - testable
  - extensible

  without collapsing into “just run arbitrary Python and hope for the best.”

  If you want, the next useful step is 7. likely next design moves and what parts of the current design will need to change
  first.