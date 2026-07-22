from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from udfs import vehicle_reid_model as reid  # noqa: E402
from udfs.reid_ops import appearance_match_score, attach_track_embedding  # noqa: E402


class CosineSimilarityTests(unittest.TestCase):
    def test_identical_vectors(self):
        self.assertAlmostEqual(reid.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(reid.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_vectors_clamped_to_zero(self):
        self.assertEqual(reid.cosine_similarity([1.0, 0.0], [-1.0, 0.0]), 0.0)

    def test_missing_or_mismatched_returns_zero(self):
        self.assertEqual(reid.cosine_similarity([1.0], [1.0, 2.0]), 0.0)
        self.assertEqual(reid.cosine_similarity([], [1.0]), 0.0)
        self.assertEqual(reid.cosine_similarity(None, [1.0]), 0.0)


class HistogramEmbedTests(unittest.TestCase):
    def test_histogram_dim_and_normalized(self):
        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        crop[:, :] = (10, 20, 200)  # BGR
        vec = reid._histogram_embed(crop)
        self.assertEqual(len(vec), reid._HIST_BINS**3)
        self.assertAlmostEqual(sum(v * v for v in vec) ** 0.5, 1.0, places=5)

    def test_similar_crops_more_similar_than_different(self):
        red = np.zeros((20, 20, 3), dtype=np.uint8); red[:, :] = (0, 0, 200)      # BGR red
        red2 = np.zeros((20, 20, 3), dtype=np.uint8); red2[:, :] = (0, 0, 210)
        blue = np.zeros((20, 20, 3), dtype=np.uint8); blue[:, :] = (200, 0, 0)    # BGR blue
        e_red, e_red2, e_blue = (reid._histogram_embed(c) for c in (red, red2, blue))
        self.assertGreater(
            reid.cosine_similarity(e_red, e_red2),
            reid.cosine_similarity(e_red, e_blue),
        )

    def test_empty_crop_returns_none(self):
        self.assertIsNone(reid._histogram_embed(np.zeros((0, 0, 3), dtype=np.uint8)))
        self.assertIsNone(reid._histogram_embed(None))


class EmbedCropFallbackTests(unittest.TestCase):
    def setUp(self):
        reid._reset_cache_for_tests()

    def tearDown(self):
        reid._reset_cache_for_tests()

    def test_falls_back_to_histogram_when_backbone_unavailable(self):
        crop = np.full((20, 20, 3), 50, dtype=np.uint8)
        with patch.object(reid, "_load_backbone", return_value=None):
            vec = reid.embed_crop(crop)
        self.assertEqual(len(vec), reid._HIST_BINS**3)  # histogram path used

    def test_unusable_crop_returns_none(self):
        with patch.object(reid, "_load_backbone", return_value=None):
            self.assertIsNone(reid.embed_crop(None))


class AppearanceMatchScoreTests(unittest.TestCase):
    def test_uses_cosine_when_embeddings_present(self):
        left = {"embedding": [1.0, 0.0, 0.0], "confidence": 0.1}
        right = {"embedding": [1.0, 0.0, 0.0], "confidence": 0.1}
        self.assertAlmostEqual(appearance_match_score(left, right), 1.0)

    def test_falls_back_to_confidence_when_embedding_missing(self):
        left = {"embedding": [], "confidence": 0.8}
        right = {"confidence": 0.6}
        # mean confidence = 0.7
        self.assertAlmostEqual(appearance_match_score(left, right), 0.7)


class AttachTrackEmbeddingTests(unittest.TestCase):
    def test_attaches_embedding_list(self):
        from udfs import reid_ops

        with patch.object(reid_ops, "crop_from_track_row", return_value=object()):
            with patch.object(reid_ops, "embed_crop", return_value=[0.1, 0.2, 0.3]):
                result = attach_track_embedding({"track_id": "t1"})
        self.assertEqual(result["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(result["track_id"], "t1")

    def test_attaches_empty_list_when_no_crop(self):
        from udfs import reid_ops

        with patch.object(reid_ops, "crop_from_track_row", return_value=None):
            result = attach_track_embedding({"track_id": "t1"})
        self.assertEqual(result["embedding"], [])


if __name__ == "__main__":
    unittest.main()
