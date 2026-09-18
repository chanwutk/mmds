from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import (  # noqa: E402
    MMDSRewriteError,
    Map,
    Input,
    ModalitySubstitution,
    ProjectionBeforeMap,
    Record,
    RewriteSelection,
    StaticPromptExecutor,
    StaticRewriteAgent,
    execute,
    parse_query,
    rewrite_once,
    search_rewrites,
)
from mmds.model import DatasetExpr, PromptSpec  # noqa: E402
from mmds.optimizers.rewriter.directive import (  # noqa: E402
    DirectiveMetadata,
    PlanIndex,
    RewriteMatch,
    make_match,
)


def _program(path: str = "data.jsonl"):
    return parse_query(
        f'''
from mmds import Input, Map, Record
rows = Input("{path}")
output = Map(
    rows,
    ["Answer from ", Record["video"], " for ", Record["question"]],
    schema={{"answer": "string"}},
)
'''
    )


def _nodes_by_name(plan: DatasetExpr) -> dict[str, DatasetExpr]:
    return {
        node.name: node
        for node in plan.walk_postorder()
        if node.name is not None
    }


class DirectiveEngineTests(unittest.TestCase):
    def test_modality_substitution_changes_only_record_reference(self) -> None:
        program = _program()
        agent = StaticRewriteAgent(
            directive="modality_substitution",
            match="modality_substitution:output",
            params={"video_field": "video", "transcript_field": "transcript"},
        )

        result = rewrite_once(
            program,
            agent=agent,
            directives=[ModalitySubstitution()],
        )

        spec = result.program.output_expr.spec
        self.assertIsInstance(spec, PromptSpec)
        self.assertIn(Record["transcript"], spec.parts)
        self.assertNotIn(Record["video"], spec.parts)
        self.assertEqual(spec.output_schema, {"answer": "string"})
        self.assertEqual(program.output_expr.spec.parts[1], Record["video"])

    def test_projection_executes_and_removes_intermediate_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "video": "long input",
                        "question": "What happened?",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            program = _program(str(path))
            agent = StaticRewriteAgent(
                directive="projection_before_map",
                match="projection_before_map:output",
                params={
                    "source_field": "video",
                    "intermediate_field": "_mmds_projection",
                    "projection_prompt": "Extract relevant evidence.",
                    "context_fields": ["question"],
                },
            )
            result = rewrite_once(
                program,
                agent=agent,
                directives=[ProjectionBeforeMap()],
            )
            names = _nodes_by_name(result.program.output_expr)
            projection_spec = names["rewrite_projection"].spec
            consumer = result.program.output_expr.source
            self.assertIsInstance(projection_spec, PromptSpec)
            self.assertIsInstance(consumer.spec, PromptSpec)

            seen: dict[str, object] = {}

            def project(resolved, payload, context):
                seen["projection_payload"] = payload
                return {"_mmds_projection": "relevant evidence"}

            def answer(resolved, payload, context):
                seen["answer_payload"] = payload
                return {"answer": "done"}

            executor = StaticPromptExecutor(
                {
                    ("map", projection_spec.cache_key()): project,
                    ("map", consumer.spec.cache_key()): answer,
                }
            )
            rows = execute(result.program, prompt_executor=executor)

        self.assertEqual(rows[0]["answer"], "done")
        self.assertNotIn("_mmds_projection", rows[0])
        self.assertEqual(
            seen["answer_payload"]["_mmds_projection"],  # type: ignore[index]
            "relevant evidence",
        )

    def test_invalid_selection_and_parameters_raise(self) -> None:
        cases = (
            StaticRewriteAgent(
                directive="unknown",
                match="unknown:output",
                params={},
            ),
            StaticRewriteAgent(
                directive="modality_substitution",
                match="not-offered",
                params={"video_field": "video", "transcript_field": "transcript"},
            ),
            StaticRewriteAgent(
                directive="modality_substitution",
                match="modality_substitution:output",
                params={"video_field": "video", "unexpected": True},
            ),
        )

        for agent in cases:
            with self.subTest(agent=agent):
                with self.assertRaises(MMDSRewriteError):
                    rewrite_once(
                        _program(),
                        agent=agent,
                        directives=[ModalitySubstitution()],
                    )

    def test_duplicate_directive_names_raise(self) -> None:
        agent = StaticRewriteAgent(
            directive="modality_substitution",
            match="modality_substitution:output",
            params={"video_field": "video", "transcript_field": "transcript"},
        )

        with self.assertRaises(MMDSRewriteError):
            rewrite_once(
                _program(),
                agent=agent,
                directives=[ModalitySubstitution(), ModalitySubstitution()],
            )

    def test_matching_and_application_share_one_plan_index(self) -> None:
        class EmptyParams(BaseModel):
            model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

        class IdentityDirective:
            metadata = DirectiveMetadata(
                "identity",
                "Return the original plan.",
                "Testing only.",
            )
            params_type = EmptyParams

            def __init__(self) -> None:
                self.matched_index = None
                self.same_index = False

            def find_matches(self, index: PlanIndex):
                self.matched_index = index
                return (make_match(self.metadata, index.entries[0].path, "root"),)

            def apply(self, index, match, params):
                self.same_index = index is self.matched_index
                return index.root

        directive = IdentityDirective()
        rewrite_once(
            _program(),
            agent=StaticRewriteAgent(
                directive="identity",
                match="identity:output",
                params={},
            ),
            directives=[directive],
        )

        self.assertTrue(directive.same_index)

    def test_projection_rejects_known_temporary_field_collision(self) -> None:
        program = parse_query(
            '''
from mmds import Input, Map, Record
rows = Input("data.jsonl")
output = Map(
    rows,
    [Record["video"], Record["_mmds_projection"]],
    schema={"answer": "string"},
)
'''
        )
        agent = StaticRewriteAgent(
            directive="projection_before_map",
            match="projection_before_map:output",
            params={
                "source_field": "video",
                "intermediate_field": "_mmds_projection",
                "projection_prompt": "Project",
            },
        )

        with self.assertRaisesRegex(MMDSRewriteError, "collides"):
            rewrite_once(
                program,
                agent=agent,
                directives=[ProjectionBeforeMap()],
            )

    def test_global_validation_rejects_changed_input_and_schema(self) -> None:
        class EmptyParams(BaseModel):
            model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

        class MutatingDirective:
            params_type = EmptyParams

            def __init__(self, name, mutation):
                self.metadata = DirectiveMetadata(name, "Invalid mutation", "Testing")
                self._mutation = mutation

            def find_matches(self, index: PlanIndex):
                return (make_match(self.metadata, index.entries[0].path, "root"),)

            def apply(self, index, match, params):
                return self._mutation(index.root)

        def changed_input(plan):
            return replace(plan, source=Input("other.jsonl"))

        def changed_schema(plan):
            return replace(
                plan,
                spec=replace(plan.spec, output_schema={"different": "string"}),
            )

        for name, mutation, message in (
            ("changed_input", changed_input, "Input"),
            ("changed_schema", changed_schema, "schema"),
        ):
            with self.subTest(name=name):
                directive = MutatingDirective(name, mutation)
                agent = StaticRewriteAgent(
                    directive=name,
                    match=f"{name}:output",
                    params={},
                )
                with self.assertRaisesRegex(MMDSRewriteError, message):
                    rewrite_once(
                        _program(),
                        agent=agent,
                        directives=[directive],
                    )

    def test_search_keeps_baseline_records_rejections_and_deduplicates(self) -> None:
        selections = (
            RewriteSelection(
                directive="modality_substitution",
                match_id="modality_substitution:output",
                params={"video_field": "video", "transcript_field": "transcript"},
            ),
            RewriteSelection(
                directive="modality_substitution",
                match_id="modality_substitution:output",
                params={"video_field": "video", "transcript_field": "transcript"},
            ),
            RewriteSelection(
                directive="unknown",
                match_id="unknown:output",
                params={},
            ),
        )
        result = search_rewrites(
            _program(),
            agent=StaticRewriteAgent(selections=selections),
            directives=[ModalitySubstitution()],
            max_candidates=3,
        )

        self.assertEqual(len(result.candidates), 2)
        self.assertIsNone(result.candidates[0].trace)
        self.assertEqual(len(result.rejections), 1)
        self.assertIn("unknown directive", result.rejections[0].reason)

    def test_search_rejects_unsupported_depth_and_bad_limit(self) -> None:
        agent = StaticRewriteAgent(
            directive="modality_substitution",
            match="modality_substitution:output",
            params={"video_field": "video", "transcript_field": "transcript"},
        )
        for kwargs in ({"max_depth": 2}, {"max_candidates": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(MMDSRewriteError):
                    search_rewrites(
                        _program(),
                        agent=agent,
                        directives=[ModalitySubstitution()],
                        **kwargs,
                    )

    def test_search_reports_candidate_truncation(self) -> None:
        selections = tuple(
            RewriteSelection(
                directive="modality_substitution",
                match_id="modality_substitution:output",
                params={
                    "video_field": "video",
                    "transcript_field": f"transcript_{index}",
                },
            )
            for index in range(3)
        )

        result = search_rewrites(
            _program(),
            agent=StaticRewriteAgent(selections=selections),
            directives=[ModalitySubstitution()],
            max_candidates=2,
        )

        self.assertTrue(result.truncated)
        self.assertEqual(len(result.candidates), 3)

    def test_search_does_not_swallow_programming_errors(self) -> None:
        class EmptyParams(BaseModel):
            model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

        class BrokenDirective:
            metadata = DirectiveMetadata("broken", "Broken test", "Testing only")
            params_type = EmptyParams

            def find_matches(self, index: PlanIndex):
                return (make_match(self.metadata, index.entries[0].path, "root"),)

            def apply(self, index, match: RewriteMatch, params):
                raise RuntimeError("programming bug")

        agent = StaticRewriteAgent(
            directive="broken",
            match="broken:output",
            params={},
        )

        with self.assertRaisesRegex(RuntimeError, "programming bug"):
            search_rewrites(
                _program(),
                agent=agent,
                directives=[BrokenDirective()],
            )


if __name__ == "__main__":
    unittest.main()
