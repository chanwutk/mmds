from __future__ import annotations

import unittest

import numpy as np

from udfs.detection_ops import _crop_from_frame


class CropFromFrameTests(unittest.TestCase):
    def test_clamps_bbox_to_frame_bounds(self) -> None:
        frame = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)

        crop = _crop_from_frame(frame, [-5.0, -2.0, 7.0, 6.0])

        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(crop.shape, (6, 7, 3))
        np.testing.assert_array_equal(crop, frame[0:6, 0:7])

    def test_rejects_empty_or_invalid_bbox(self) -> None:
        frame = np.zeros((10, 12, 3), dtype=np.uint8)

        self.assertIsNone(_crop_from_frame(frame, [5.0, 5.0, 5.0, 8.0]))
        self.assertIsNone(_crop_from_frame(frame, [1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
