from __future__ import annotations

import sys
import unittest
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmds.model import MMDSValidationError  # noqa: E402
from mmds.utilities.timestamps import (  # noqa: E402
    parse_iso_timestamp,
    require_iso_timestamp,
)


class TimestampParseTests(unittest.TestCase):
    def test_parse_accepts_z_and_naive(self) -> None:
        zulu = parse_iso_timestamp("2024-01-01T00:00:00Z")
        naive = parse_iso_timestamp("2024-01-01T00:00:00")
        assert zulu is not None and naive is not None
        self.assertEqual(zulu, naive)
        self.assertEqual(zulu.tzinfo, timezone.utc)

    def test_parse_returns_none_for_invalid(self) -> None:
        self.assertIsNone(parse_iso_timestamp(None))
        self.assertIsNone(parse_iso_timestamp(""))
        self.assertIsNone(parse_iso_timestamp("not-a-timestamp"))
        self.assertIsNone(parse_iso_timestamp(123))

    def test_require_raises_for_invalid(self) -> None:
        with self.assertRaises(MMDSValidationError):
            require_iso_timestamp("not-a-timestamp", label="recorded_at")
        with self.assertRaises(MMDSValidationError):
            require_iso_timestamp(None, label="recorded_at")

    def test_require_returns_parsed(self) -> None:
        parsed = require_iso_timestamp("2024-06-01T12:00:00Z", label="recorded_at")
        self.assertEqual(parsed.year, 2024)
        self.assertEqual(parsed.tzinfo, timezone.utc)


if __name__ == "__main__":
    unittest.main()
