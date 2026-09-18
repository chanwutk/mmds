from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds import Filter, Input, Map, Record  # noqa: E402
from mmds.optimizers.rewriter.directive import (  # noqa: E402
    MMDSRewriteError,
    NodePath,
    PlanIndex,
)


class NodePathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.input = Input("data.jsonl")
        self.mapped = Map(
            self.input,
            ["Read ", Record["document"]],
            schema={"summary": "string"},
        )
        self.plan = Filter(
            self.mapped,
            ["Keep ", Record["summary"]],
        )

    def test_paths_resolve_from_output(self) -> None:
        index = PlanIndex.build(self.plan)

        self.assertIs(index.node_at(NodePath()), self.plan)
        self.assertIs(index.node_at(NodePath(("source",))), self.mapped)
        self.assertIs(
            index.node_at(NodePath(("source", "source"))),
            self.input,
        )

    def test_paths_have_readable_names(self) -> None:
        self.assertEqual(str(NodePath()), "output")
        self.assertEqual(str(NodePath(("source",))), "output.source")

    def test_rejects_unknown_path_steps(self) -> None:
        with self.assertRaises(MMDSRewriteError):
            NodePath(("left",))

    def test_rejects_stale_paths(self) -> None:
        with self.assertRaises(MMDSRewriteError):
            PlanIndex.build(self.input).node_at(NodePath(("source",)))

    def test_replace_rebuilds_ancestors_without_mutating_original(self) -> None:
        replacement = Map(
            self.input,
            ["Extract ", Record["document"]],
            schema={"extract": "string"},
        )

        rewritten = PlanIndex.build(self.plan).replace(
            NodePath(("source",)),
            replacement,
        )

        self.assertEqual(rewritten.source, replacement)
        self.assertIs(self.plan.source, self.mapped)
        self.assertIsNot(rewritten, self.plan)


class PlanIndexTests(unittest.TestCase):
    def test_indexes_each_node_once_with_field_effects(self) -> None:
        plan = Filter(
            Map(
                Input("data.jsonl"),
                ["Read ", Record["document"]],
                schema={"summary": "string"},
            ),
            ["Keep ", Record["summary"]],
        )

        index = PlanIndex.build(plan)

        self.assertEqual(
            [str(entry.path) for entry in index.entries],
            ["output", "output.source", "output.source.source"],
        )
        self.assertIs(index.root, plan)
        self.assertEqual(index.entries[0].field_effects.reads, {"summary"})
        self.assertEqual(index.entries[1].field_effects.reads, {"document"})
        self.assertEqual(index.entries[1].field_effects.writes, {"summary"})
        self.assertEqual(index.known_fields(), {"document", "summary"})


if __name__ == "__main__":
    unittest.main()
