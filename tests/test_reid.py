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
from udfs.reid_ops import (  # noqa: E402
    appearance_match_score,
    attach_track_embedding,
    attach_track_summary_embeddings,
)


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

    def test_batch_histogram_fallback_matches_single_results(self):
        crops = [
            np.full((20, 20, 3), value, dtype=np.uint8)
            for value in (20, 80, 160)
        ]
        with patch.object(reid, "_load_backbone", return_value=None):
            batched = reid.embed_crops(crops, batch_size=2)
            expected = [reid.embed_crop(crop) for crop in crops]
        self.assertEqual(batched, expected)

    def test_cnn_batches_are_bounded(self):
        import torch

        class FakeBackbone:
            def __init__(self):
                self.batch_sizes = []

            def __call__(self, tensor):
                self.batch_sizes.append(len(tensor))
                means = tensor.mean(dim=(1, 2, 3))
                return torch.stack((means, means + 1.0), dim=1)

        crops = [
            np.full((12, 12, 3), value, dtype=np.uint8)
            for value in (10, 20, 30, 40, 50)
        ]
        backbone = FakeBackbone()
        with patch.object(reid, "_load_backbone", return_value=backbone):
            vectors = reid.embed_crops(crops, batch_size=2)
        self.assertEqual(len(vectors), len(crops))
        self.assertEqual(backbone.batch_sizes, [2, 2, 1])
        for vector in vectors:
            self.assertAlmostEqual(sum(value * value for value in vector) ** 0.5, 1.0)


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

    def test_mismatched_embedding_dimensions_return_zero(self):
        left = {"embedding": [1.0, 0.0], "confidence": 0.9}
        right = {"embedding": [1.0, 0.0, 0.0], "confidence": 0.9}

        self.assertEqual(appearance_match_score(left, right), 0.0)


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

    def test_embeds_track_summaries_before_unnest_with_one_frame_read(self):
        from udfs import reid_ops

        frame = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
        row = {
            "camera_id": "cam-a",
            "video": {"path": "/tmp/video.mp4"},
            "track_summaries": [
                {"track_id": "t1", "rep_frame_id": 7, "rep_bbox": [0, 0, 6, 6]},
                {"track_id": "t2", "rep_frame_id": 7, "rep_bbox": [6, 6, 12, 12]},
            ],
        }
        with patch.object(
            reid_ops, "_video_path_from_row", return_value="/tmp/video.mp4"
        ):
            with patch(
                "mmds.utilities.video.read_frames_at_indices",
                return_value={7: frame},
            ) as read_frames:
                with patch.object(
                    reid_ops,
                    "embed_crops",
                    return_value=[[1.0, 0.0], [0.0, 1.0]],
                ) as embed_batch:
                    result = attach_track_summary_embeddings(row, batch_size=2)

        read_frames.assert_called_once_with("/tmp/video.mp4", [7, 7])
        self.assertEqual(embed_batch.call_args.kwargs["batch_size"], 2)
        self.assertEqual(
            [summary["embedding"] for summary in result["track_summaries"]],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertNotIn("embedding", row["track_summaries"][0])


if __name__ == "__main__":
    unittest.main()
