"""视频稳定和语义压缩辅助函数的回归检查。"""
from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from segment_video import stabilize_prediction  # noqa: E402
from semantic_compress_video import parse_class_indices, protect_foreground  # noqa: E402


class VideoToolTests(unittest.TestCase):
    def test_three_frame_vote_uses_majority_and_recent_on_tie(self):
        first = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        middle = np.array([[1, 9, 8], [7, 5, 0]], dtype=np.uint8)
        recent = np.array([[2, 2, 7], [4, 8, 6]], dtype=np.uint8)
        history = deque((first, middle, recent), maxlen=3)
        result = stabilize_prediction(history, recent, class_count=10)
        np.testing.assert_array_equal(result, [[1, 2, 7], [4, 5, 6]])

    def test_foreground_pixels_are_preserved(self):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(16, dtype=np.uint8)
        frame[:, :, 1] = np.arange(12, dtype=np.uint8)[:, None]
        prediction = np.zeros((12, 16), dtype=np.uint8)
        prediction[4:8, 6:10] = 1
        result = protect_foreground(frame, prediction, [1], background_scale=0.25, dilate=0)
        np.testing.assert_array_equal(result[4:8, 6:10], frame[4:8, 6:10])
        self.assertGreater(np.mean(result != frame), 0.0)

    def test_class_list_validation(self):
        self.assertEqual(parse_class_indices("6, 1, 1", 8), [1, 6])
        with self.assertRaises(ValueError):
            parse_class_indices("1,8", 8)
        with self.assertRaises(ValueError):
            parse_class_indices("", 8)


if __name__ == "__main__":
    unittest.main()
