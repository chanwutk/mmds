from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import (  # noqa: E402
    Input,
    PromptSpec,
    QueryProgram,
    RecordPath,
    UdfSpec,
    parse_query,
)
from mmds.model import DatasetExpr  # noqa: E402
from mmds.optimizers.rewriter import (  # noqa: E402
    DirectiveMetadata,
    MMDSRewriteError,
    NodePath,
    PlanIndex,
    RewriteMatch,
    apply_rewrite,
    validate_rewrite_structure,
)


class RenameFieldParams(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    old_field: str = Field(min_length=1)
    new_field: str = Field(min_length=1)


class RenamePromptField:
    metadata = DirectiveMetadata(
        name="rename_prompt_field",
        description="Replace one top-level prompt field reference.",
        when_to_use="Use in tests of deterministic directive application.",
    )
    params_type = RenameFieldParams

    def find_matches(self, index: PlanIndex) -> tuple[RewriteMatch, ...]:
        return tuple(
            RewriteMatch(
                path=entry.path,
                summary="Prompt-backed Map",
            )
            for entry in index.entries
            if entry.node.kind == "map"
            and isinstance(entry.node.spec, PromptSpec)
        )

    def apply(
        self,
        index: PlanIndex,
        match: RewriteMatch,
        params: BaseModel,
    ) -> DatasetExpr:
        if not isinstance(params, RenameFieldParams):
            raise MMDSRewriteError("RenamePromptField received invalid parameters.")
        node = index.node_at(match.path)
        if not isinstance(node.spec, PromptSpec):
            raise MMDSRewriteError("RenamePromptField requires a prompt-backed node.")
        old = RecordPath((params.old_field,))
        new = RecordPath((params.new_field,))
        if old not in node.spec.parts:
            raise MMDSRewriteError(
                f"Matched prompt does not reference {params.old_field!r}."
            )
        replacement = replace(
            node,
            spec=replace(
                node.spec,
                parts=tuple(new if part == old else part for part in node.spec.parts),
            ),
        )
        return index.replace(match.path, replacement)


def _program() -> QueryProgram:
    return parse_query(
        '''
from mmds import Input, Map, Record
rows = Input("rows.jsonl")
output = Map(
    rows,
    ["Inspect ", Record["video"]],
    schema={"answer": "string"},
    name="answer_video",
)
'''
    )


class NodePathTests(unittest.TestCase):
    def test_paths_are_rendered_from_the_output(self) -> None:
        root = NodePath()

        self.assertEqual(str(root), "output")
        self.assertEqual(str(root.source()), "output.source")
        self.assertEqual(
            str(root.source().source()),
            "output.source.source",
        )

    def test_only_unary_source_steps_are_supported(self) -> None:
        invalid_steps = (["source"], ("left",), ("source", "right"))
        for steps in invalid_steps:
            with self.subTest(steps=steps):
                with self.assertRaisesRegex(MMDSRewriteError, "unary 'source'"):
                    NodePath(steps)  # type: ignore[arg-type]


class PlanIndexTests(unittest.TestCase):
    def test_build_indexes_existing_nodes_without_copying_them(self) -> None:
        root = _program().output_expr

        index = PlanIndex.build(root)

        self.assertEqual(
            [str(entry.path) for entry in index.entries],
            ["output", "output.source"],
        )
        self.assertIs(index.entries[0].node, root)
        self.assertIs(index.entries[1].node, root.source)
        self.assertIs(index.node_at(NodePath().source()), root.source)

    def test_replace_rebuilds_only_ancestors_and_preserves_original(self) -> None:
        root = _program().output_expr
        original_input = root.source
        replacement = Input("replacement.jsonl")

        rewritten = PlanIndex.build(root).replace(
            NodePath().source(),
            replacement,
        )

        self.assertIsNot(rewritten, root)
        self.assertIs(rewritten.source, replacement)
        self.assertIs(root.source, original_input)
        self.assertEqual(root.source.input_path, "rows.jsonl")

    def test_replace_can_replace_the_root(self) -> None:
        replacement = Input("replacement.jsonl")

        rewritten = PlanIndex.build(_program().output_expr).replace(
            NodePath(),
            replacement,
        )

        self.assertIs(rewritten, replacement)

    def test_missing_paths_and_invalid_replacements_fail_explicitly(self) -> None:
        index = PlanIndex.build(_program().output_expr)
        missing = NodePath().source().source()

        with self.assertRaisesRegex(MMDSRewriteError, "does not exist"):
            index.node_at(missing)
        with self.assertRaisesRegex(MMDSRewriteError, "does not exist"):
            index.replace(missing, Input("replacement.jsonl"))
        with self.assertRaisesRegex(TypeError, "DatasetExpr"):
            index.replace(NodePath(), object())  # type: ignore[arg-type]


class DirectiveContractTests(unittest.TestCase):
    def test_metadata_and_match_text_must_be_non_empty(self) -> None:
        with self.assertRaisesRegex(MMDSRewriteError, "metadata"):
            DirectiveMetadata(name="", description="description", when_to_use="use")
        with self.assertRaisesRegex(MMDSRewriteError, "summaries"):
            RewriteMatch(path=NodePath(), summary="   ")

    def test_apply_rewrite_validates_params_and_preserves_original(self) -> None:
        original = replace(_program(), path="queries/example.py")
        directive = RenamePromptField()
        match = directive.find_matches(PlanIndex.build(original.output_expr))[0]

        rewritten = apply_rewrite(
            original,
            directive=directive,
            match=match,
            params={"old_field": "video", "new_field": "transcript"},
        )

        self.assertEqual(rewritten.path, original.path)
        self.assertEqual(
            rewritten.output_expr.spec.parts,
            ("Inspect ", RecordPath(("transcript",))),
        )
        self.assertEqual(
            original.output_expr.spec.parts,
            ("Inspect ", RecordPath(("video",))),
        )

    def test_apply_rewrite_rejects_matches_not_offered_by_the_directive(self) -> None:
        with self.assertRaisesRegex(MMDSRewriteError, "did not offer"):
            apply_rewrite(
                _program(),
                directive=RenamePromptField(),
                match=RewriteMatch(
                    path=NodePath().source(),
                    summary="Input node",
                ),
                params={"old_field": "video", "new_field": "transcript"},
            )

    def test_apply_rewrite_rejects_invalid_parameters(self) -> None:
        program = _program()
        directive = RenamePromptField()
        match = directive.find_matches(PlanIndex.build(program.output_expr))[0]
        invalid_params = (
            {},
            {"old_field": "video", "new_field": 7},
            {
                "old_field": "video",
                "new_field": "transcript",
                "unexpected": True,
            },
        )

        for params in invalid_params:
            with self.subTest(params=params):
                with self.assertRaisesRegex(MMDSRewriteError, "Invalid parameters"):
                    apply_rewrite(
                        program,
                        directive=directive,
                        match=match,
                        params=params,
                    )


class RewriteValidationTests(unittest.TestCase):
    def test_input_paths_must_be_preserved(self) -> None:
        original = _program()
        rewritten = replace(
            original.output_expr,
            source=Input("different.jsonl"),
        )

        with self.assertRaisesRegex(MMDSRewriteError, "Input"):
            validate_rewrite_structure(original, rewritten)

    def test_declared_output_schema_must_be_preserved(self) -> None:
        original = _program()
        rewritten = replace(
            original.output_expr,
            spec=replace(
                original.output_expr.spec,
                output_schema={"different": "string"},
            ),
        )

        with self.assertRaisesRegex(MMDSRewriteError, "output schema"):
            validate_rewrite_structure(original, rewritten)

    def test_udf_output_schema_is_explicitly_unknown(self) -> None:
        original = _program()
        rewritten = replace(
            original.output_expr,
            spec=UdfSpec(module="udfs.test_ops", name="annotate"),
        )

        validate_rewrite_structure(original, rewritten)

    def test_rewritten_plan_must_round_trip_through_the_dsl(self) -> None:
        original = parse_query(
            '''
from mmds import Input
output = Input("rows.jsonl")
'''
        )
        rewritten = DatasetExpr(
            kind="coalesce",
            source=original.output_expr,
            group_by=("id",),
            field="clip",
        )

        with self.assertRaisesRegex(MMDSRewriteError, "round-trip"):
            validate_rewrite_structure(original, rewritten)


if __name__ == "__main__":
    unittest.main()
