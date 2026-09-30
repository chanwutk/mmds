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

from examples.semantic_join_cross_camera_vehicle import (  # noqa: E402
    stitch_prompt_preamble,
)
from udfs.vehicle_labels import (  # noqa: E402
    VEHICLE_CAR_SUBTYPES,
    VEHICLE_CLASSES,
    VEHICLE_COLOR_VOCAB,
    VEHICLE_SUBTYPES,
    VEHICLE_TRUCK_SUBTYPES,
    format_attribute_vocab_for_prompt,
)


class VehicleLabelsTests(unittest.TestCase):
    def test_formatter_lists_every_canonical_label(self) -> None:
        text = format_attribute_vocab_for_prompt()
        for label in sorted(VEHICLE_CLASSES | VEHICLE_COLOR_VOCAB | VEHICLE_SUBTYPES):
            self.assertIn(label, text)
        self.assertNotIn("tractor trailer", text)
        self.assertIn("source video", text)

    def test_subtype_sets_are_disjoint_partitions(self) -> None:
        self.assertEqual(
            VEHICLE_SUBTYPES,
            VEHICLE_CAR_SUBTYPES | VEHICLE_TRUCK_SUBTYPES,
        )
        self.assertTrue(VEHICLE_CAR_SUBTYPES.isdisjoint(VEHICLE_TRUCK_SUBTYPES))

    def test_semantic_prompt_uses_shared_formatter(self) -> None:
        preamble = stitch_prompt_preamble()
        shared = format_attribute_vocab_for_prompt()
        self.assertIn(shared, preamble)
        self.assertIn("tractor_trailer", preamble)
        self.assertIn("flatbed_truck", preamble)
        for color in sorted(VEHICLE_COLOR_VOCAB):
            self.assertIn(color, preamble)


if __name__ == "__main__":
    unittest.main()
