from __future__ import annotations

import os
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

from udfs.vehicle_color_model import (  # noqa: E402
    CHEN_COLOR_CLASSES,
    DEFAULT_WEIGHTS_PATH,
    _preprocess_crop,
    _reset_cache_for_tests,
    build_color_model,
    predict_color_from_crop,
    predict_colors_from_crops,
    vocab_color_for_chen_label,
    weights_path,
)


class VocabMappingTests(unittest.TestCase):
    def test_identity_colors(self) -> None:
        for color in ("black", "blue", "gray", "green", "red", "white", "yellow"):
            self.assertEqual(vocab_color_for_chen_label(color), color)

    def test_cyan_maps_to_blue(self) -> None:
        self.assertEqual(vocab_color_for_chen_label("cyan"), "blue")

    def test_unknown_maps_to_gray(self) -> None:
        self.assertEqual(vocab_color_for_chen_label("mauve"), "gray")

    def test_every_chen_class_has_a_mapping(self) -> None:
        for color in CHEN_COLOR_CLASSES:
            # None of them should silently collapse to the unknown default
            # unless intended (cyan is the only remap, and it maps to blue).
            self.assertIn(vocab_color_for_chen_label(color), {
                "black", "blue", "gray", "green", "red", "white", "yellow",
            })


class WeightsPathTests(unittest.TestCase):
    def test_default_when_env_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMDS_VEHICLE_COLOR_WEIGHTS", None)
            self.assertEqual(weights_path(), DEFAULT_WEIGHTS_PATH)

    def test_env_override(self) -> None:
        with patch.dict(os.environ, {"MMDS_VEHICLE_COLOR_WEIGHTS": "/tmp/w.pt"}):
            self.assertEqual(weights_path(), "/tmp/w.pt")


class PreprocessTests(unittest.TestCase):
    def test_shape_is_normalized_batch_tensor(self) -> None:
        crop = np.full((40, 30, 3), 128, dtype=np.uint8)
        tensor = _preprocess_crop(crop)
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))

    def test_none_crop_returns_none(self) -> None:
        self.assertIsNone(_preprocess_crop(None))

    def test_empty_crop_returns_none(self) -> None:
        self.assertIsNone(_preprocess_crop(np.zeros((0, 0, 3), dtype=np.uint8)))

    def test_non_3channel_returns_none(self) -> None:
        self.assertIsNone(_preprocess_crop(np.zeros((10, 10), dtype=np.uint8)))


class BuildModelTests(unittest.TestCase):
    def test_output_matches_num_classes(self) -> None:
        import torch

        model = build_color_model()
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (1, len(CHEN_COLOR_CLASSES)))


class _FakeModel:
    """Stand-in model whose argmax is a fixed class index."""

    def __init__(self, index: int) -> None:
        self._index = index

    def eval(self) -> "_FakeModel":
        return self

    def __call__(self, tensor):  # noqa: ANN001
        import torch

        logits = torch.full((1, len(CHEN_COLOR_CLASSES)), -10.0)
        logits[0, self._index] = 10.0
        return logits


class _FakeBatchModel:
    def __init__(self, index: int) -> None:
        self._index = index
        self.batch_sizes: list[int] = []

    def __call__(self, tensor):  # noqa: ANN001
        import torch

        self.batch_sizes.append(len(tensor))
        logits = torch.full((len(tensor), len(CHEN_COLOR_CLASSES)), -10.0)
        logits[:, self._index] = 10.0
        return logits


class PredictColorTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_cache_for_tests()

    def tearDown(self) -> None:
        _reset_cache_for_tests()

    def test_missing_weights_returns_none(self) -> None:
        crop = np.full((20, 20, 3), 200, dtype=np.uint8)
        self.assertIsNone(
            predict_color_from_crop(crop, path="/definitely/not/here.pt")
        )

    def test_predicts_mapped_vocab_color(self) -> None:
        crop = np.full((20, 20, 3), 10, dtype=np.uint8)
        # index 5 == "red"
        with patch("udfs.vehicle_color_model.os.path.exists", return_value=True):
            with patch(
                "udfs.vehicle_color_model._load_classifier",
                return_value=_FakeModel(5),
            ):
                self.assertEqual(predict_color_from_crop(crop), "red")

    def test_cyan_prediction_remaps_to_blue(self) -> None:
        crop = np.full((20, 20, 3), 10, dtype=np.uint8)
        # index 2 == "cyan" -> mapped to "blue"
        with patch("udfs.vehicle_color_model.os.path.exists", return_value=True):
            with patch(
                "udfs.vehicle_color_model._load_classifier",
                return_value=_FakeModel(2),
            ):
                self.assertEqual(predict_color_from_crop(crop), "blue")

    def test_load_failure_returns_none(self) -> None:
        crop = np.full((20, 20, 3), 10, dtype=np.uint8)
        with patch("udfs.vehicle_color_model.os.path.exists", return_value=True):
            with patch(
                "udfs.vehicle_color_model._load_classifier", return_value=None
            ):
                self.assertIsNone(predict_color_from_crop(crop))

    def test_unusable_crop_returns_none_even_with_model(self) -> None:
        with patch("udfs.vehicle_color_model.os.path.exists", return_value=True):
            with patch(
                "udfs.vehicle_color_model._load_classifier",
                return_value=_FakeModel(0),
            ):
                self.assertIsNone(predict_color_from_crop(None))

    def test_batch_matches_single_predictions_and_is_bounded(self) -> None:
        crops = [
            np.full((20, 20, 3), value, dtype=np.uint8)
            for value in (10, 20, 30, 40, 50)
        ]
        batch_model = _FakeBatchModel(5)
        with patch("udfs.vehicle_color_model.os.path.exists", return_value=True):
            with patch(
                "udfs.vehicle_color_model._load_classifier",
                return_value=batch_model,
            ):
                batched = predict_colors_from_crops(crops, batch_size=2)
        self.assertEqual(batched, ["red"] * len(crops))
        self.assertEqual(batch_model.batch_sizes, [2, 2, 1])

        with patch("udfs.vehicle_color_model.os.path.exists", return_value=True):
            with patch(
                "udfs.vehicle_color_model._load_classifier",
                return_value=_FakeModel(5),
            ):
                singles = [predict_color_from_crop(crop) for crop in crops]
        self.assertEqual(batched, singles)

    def test_batch_missing_weights_returns_aligned_fallback_markers(self) -> None:
        crops = [np.full((10, 10, 3), 100, dtype=np.uint8), None]
        self.assertEqual(
            predict_colors_from_crops(crops, path="/definitely/not/here.pt"),
            [None, None],
        )


class PredictVehicleAttributesIntegrationTests(unittest.TestCase):
    def test_uses_classifier_when_available(self) -> None:
        from udfs.detection_ops import predict_vehicle_attributes

        crop = np.full((20, 20, 3), 10, dtype=np.uint8)
        with patch(
            "udfs.detection_ops.predict_color_from_crop", return_value="red"
        ):
            attrs = predict_vehicle_attributes("sedan", [0, 0, 40, 20], crop)
        self.assertEqual(attrs["vehicle_color"], "red")

    def test_falls_back_to_heuristic_when_classifier_unavailable(self) -> None:
        from udfs.detection_ops import predict_vehicle_attributes

        crop = np.full((20, 20, 3), 10, dtype=np.uint8)
        with patch(
            "udfs.detection_ops.predict_color_from_crop", return_value=None
        ):
            attrs = predict_vehicle_attributes("sedan", [0, 0, 40, 20], crop)
        # Heuristic still yields a valid vocabulary color (not None).
        self.assertIn(
            attrs["vehicle_color"],
            {"silver", "white", "gray", "black", "beige", "yellow",
             "red", "blue", "green", "brown"},
        )

    def test_batch_falls_back_to_hsv_per_crop(self) -> None:
        from udfs.detection_ops import predict_vehicle_attributes_batch

        red = np.zeros((20, 20, 3), dtype=np.uint8)
        red[:, :] = (0, 0, 220)
        blue = np.zeros((20, 20, 3), dtype=np.uint8)
        blue[:, :] = (220, 0, 0)
        with patch(
            "udfs.detection_ops.predict_colors_from_crops",
            return_value=[None, None],
        ):
            attrs = predict_vehicle_attributes_batch(
                ["sedan", "suv"],
                [[0, 0, 40, 20], [0, 0, 40, 20]],
                [red, blue],
            )
        self.assertEqual([item["vehicle_color"] for item in attrs], ["red", "blue"])


if __name__ == "__main__":
    unittest.main()
