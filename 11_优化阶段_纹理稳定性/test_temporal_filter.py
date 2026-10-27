import unittest
from unittest.mock import patch
import cv2
import numpy as np
from temporal_filter import TemporalFilter, correspondence, lowpass, warp


class TemporalTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.image = np.full((64, 96, 3), 120, np.uint8)

    def test_static_image_preserved(self):
        f = TemporalFilter()
        for _ in range(3):
            got, _ = f.apply(self.image)
            np.testing.assert_array_equal(got, self.image)

    def test_cut_and_reset_do_not_mix(self):
        f = TemporalFilter()
        f.apply(self.image)
        white = np.full_like(self.image, 240)
        got, stats = f.apply(white)
        self.assertTrue(stats['reset'])
        np.testing.assert_array_equal(got, white)
        f.reset()
        got, stats = f.apply(self.image)
        self.assertTrue(stats['reset'])
        np.testing.assert_array_equal(got, self.image)

    def test_translation_direction_and_exposed_border(self):
        backward = np.zeros((64, 96, 2), np.float32)
        backward[..., 0] = -4
        forward = -backward
        mx, my, valid, error = correspondence(backward, forward)
        self.assertFalse(valid[:, :6].any())
        self.assertEqual(float(error.max()), 0)
        ramp = np.tile(np.arange(96, dtype=np.float32), (64, 1))
        np.testing.assert_array_equal(warp(ramp, mx, my)[3:-3, 6:-3], ramp[3:-3, 2:-7])

    def test_inconsistent_flow_rejected(self):
        other = self.image.copy()
        other[::2] += 10
        wrong = np.full((64, 96, 2), 8, np.float32)
        f = TemporalFilter()
        f.apply(self.image)
        with patch('temporal_filter.motion', return_value=(wrong, wrong)):
            got, stats = f.apply(other)
        self.assertEqual(stats['active_fraction'], 0)
        np.testing.assert_array_equal(got, other)

    def test_alternating_texture_suppressed_without_brightness_shift(self):
        checker = ((np.indices((64, 96)).sum(0) % 2) * 2 - 1)[..., None]
        a = (self.image + checker * 8).astype(np.uint8)
        b = (self.image - checker * 8).astype(np.uint8)
        flow = np.zeros((64, 96, 2), np.float32)
        f = TemporalFilter(); f.apply(a)
        with patch('temporal_filter.motion', return_value=(flow, flow)):
            got, _ = f.apply(b)
        area = np.s_[8:-8, 8:-8]
        self.assertLess(np.std((got - lowpass(got))[area]), np.std((b - lowpass(b))[area]))
        self.assertAlmostEqual(float(got[area].mean()), float(b[area].mean()), places=4)

    def test_prefix_is_causal_and_input_not_mutated(self):
        rng = np.random.default_rng(8)
        frames = [rng.integers(80, 160, self.image.shape, dtype=np.uint8) for _ in range(3)]
        copies = [x.copy() for x in frames]
        a, b = TemporalFilter(), TemporalFilter()
        short = [a.apply(x)[0] for x in frames[:2]]
        full = [b.apply(x)[0] for x in frames]
        for x, y in zip(short, full): np.testing.assert_array_equal(x, y)
        for x, y in zip(frames, copies): np.testing.assert_array_equal(x, y)


if __name__ == '__main__':
    unittest.main()
