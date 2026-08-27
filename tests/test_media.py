from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mmds.model import MMDSValidationError
from mmds.utilities.media import resolve_local_media_path


class ResolveLocalMediaPathTests(unittest.TestCase):
    def test_allows_relative_and_absolute_paths_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "clips" / "demo.mp4"
            media.parent.mkdir()
            media.write_bytes(b"video")

            self.assertEqual(
                resolve_local_media_path("clips/demo.mp4", media_root=root),
                media.resolve(),
            )
            self.assertEqual(
                resolve_local_media_path(media, media_root=root),
                media.resolve(),
            )

    def test_rejects_traversal_and_absolute_paths_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "root"
            root.mkdir()
            outside = base / "outside.mp4"
            outside.write_bytes(b"video")

            for candidate in ("../outside.mp4", outside):
                with self.subTest(candidate=candidate):
                    with self.assertRaisesRegex(MMDSValidationError, "outside media_root"):
                        resolve_local_media_path(candidate, media_root=root)

    def test_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "root"
            root.mkdir()
            outside = base / "outside.mp4"
            outside.write_bytes(b"video")
            link = root / "linked.mp4"
            link.symlink_to(outside)

            with self.assertRaisesRegex(MMDSValidationError, "outside media_root"):
                resolve_local_media_path(link, media_root=root)

    def test_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(MMDSValidationError, "does not exist"):
                resolve_local_media_path("missing.mp4", media_root=tmp)


if __name__ == "__main__":
    unittest.main()
