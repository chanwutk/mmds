"""Guards on the import-time dependency surface of the package.

`import mmds` must work without the heavy computer-vision stack (OpenCV/NumPy,
and transitively torch/ultralytics) installed. Those are only needed by the
`Detect` operator and `VideoView`, which are imported lazily.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
import warnings
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


def _run_package_only(code: str) -> subprocess.CompletedProcess:
    """Run with only the wheel-included ``src`` package importable."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    with tempfile.TemporaryDirectory() as cwd:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
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

    def test_importing_media_utilities_does_not_pull_in_cv_stack(self) -> None:
        result = _run_isolated(
            "import sys\n"
            "from mmds.utilities.media import resolve_video_source\n"
            "assert resolve_video_source('clip.mp4') == 'clip.mp4'\n"
            "for name in ('cv2', 'numpy', 'torch', 'ultralytics'):\n"
            "    assert name not in sys.modules, f'{name} imported eagerly'\n"
            "print('OK')\n"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OK", result.stdout)

    def test_deprecated_join_exports_resolve_lazily(self) -> None:
        import mmds.join
        from mmds.case_studies import same_vehicle

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = mmds.join.same_vehicle

        self.assertIs(resolved, same_vehicle)
        self.assertTrue(
            any(issubclass(item.category, DeprecationWarning) for item in caught)
        )

    def test_case_study_namespace_is_lazy(self) -> None:
        result = _run_package_only(
            "import sys\n"
            "import mmds.case_studies\n"
            "assert 'mmds.case_studies.predicates' not in sys.modules\n"
            "from mmds.case_studies import same_vehicle\n"
            "assert same_vehicle.__module__ == 'mmds.case_studies.predicates'\n"
            "print('OK')\n"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OK", result.stdout)

    def test_deprecated_join_modules_work_with_package_only(self) -> None:
        result = _run_package_only(
            "import warnings\n"
            "warnings.simplefilter('always', DeprecationWarning)\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    from mmds.join.predicates import same_vehicle\n"
            "    from mmds.join.trajectory import timeline_segment_from_track\n"
            "    from mmds.join.text_retrieval import tag_gallery_video\n"
            "    from mmds.join.eval_trajectories import evaluate_trajectories\n"
            "assert same_vehicle.__module__ == 'mmds.case_studies.predicates'\n"
            "assert timeline_segment_from_track.__module__ == 'mmds.case_studies.trajectory'\n"
            "assert tag_gallery_video.__module__ == 'mmds.case_studies.text_retrieval'\n"
            "assert evaluate_trajectories.__module__ == "
            "'mmds.case_studies.trajectory_evaluation'\n"
            "assert len(caught) == 4\n"
            "assert all(issubclass(item.category, DeprecationWarning) for item in caught)\n"
            "print('OK')\n"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OK", result.stdout)

    def test_udf_entrypoints_keep_udfs_module_paths(self) -> None:
        from udfs.join_ops import same_vehicle, vehicle_match_score
        from udfs.retrieval_ops import format_gt_caption_clip, group_gt_clips_by_video
        from udfs.trajectory_ops import join_match_to_trajectory

        for udf in (
            same_vehicle,
            vehicle_match_score,
            format_gt_caption_clip,
            group_gt_clips_by_video,
            join_match_to_trajectory,
        ):
            with self.subTest(udf=udf.__name__):
                self.assertTrue(udf.__module__.startswith("udfs."))


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
