"""Guards on the import-time dependency surface of the package.

`import mmds` must work without the heavy computer-vision stack (OpenCV/NumPy,
and transitively torch/ultralytics) installed. Those are only needed by the
`Detect` operator and `VideoView`, which are imported lazily.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a fresh interpreter with the project on PYTHONPATH.

    A subprocess is used so the assertions see a clean ``sys.modules`` that is
    not polluted by other test modules importing cv2/numpy in this process.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), str(ROOT)])
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )


class EagerImportSurfaceTests(unittest.TestCase):
    def test_importing_mmds_does_not_pull_in_cv2_or_video(self) -> None:
        result = _run_isolated(
            "import sys\n"
            "import mmds\n"
            "assert 'cv2' not in sys.modules, 'cv2 imported eagerly by `import mmds`'\n"
            "assert 'mmds.utilities.video' not in sys.modules, "
            "'video module imported eagerly by `import mmds`'\n"
            "for name in ('Input', 'Map', 'Filter', 'Reduce', 'Unnest', 'execute', "
            "'GeminiPromptExecutor'):\n"
            "    assert hasattr(mmds, name), name\n"
            "print('OK')\n"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OK", result.stdout)


class LazyVideoViewExportTests(unittest.TestCase):
    def test_unknown_attribute_still_raises_attribute_error(self) -> None:
        import mmds

        with self.assertRaises(AttributeError):
            mmds.ThisDoesNotExist  # noqa: B018

    def test_videoview_export_matches_direct_import_when_cv2_present(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 not installed; VideoView export is loaded lazily")

        from mmds import VideoView
        from mmds.utilities.video import VideoView as DirectVideoView

        self.assertIs(VideoView, DirectVideoView)


if __name__ == "__main__":
    unittest.main()
