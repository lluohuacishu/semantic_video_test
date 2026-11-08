"""Check the three-frame method's alignment and conservative fallback."""

import unittest

import cv2
import numpy as np

from window_filter import blend_center


class CenteredFilterTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        yy, xx = np.indices((64, 96))
        self.texture = np.repeat((120 + ((xx // 3 + yy // 3) % 2 * 2 - 1) * 16)
                                 [..., None], 3, axis=2).astype(np.uint8)
        self.flicker = 240 - self.texture
        self.zero = np.zeros((*self.texture.shape[:2], 2), np.float32)

    def test_identical_frames_pass_through(self):
        flow = (self.zero, self.zero)
        got, _ = blend_center(self.texture, self.texture, self.texture, flow, flow)
        np.testing.assert_array_equal(got, self.texture)

    def test_bidirectional_translation_matches_aligned_control(self):
        motion = self.zero.copy()
        motion[..., 0] = -7
        pair = (motion, -motion)
        center = self.flicker.copy()
        shifted_past = np.roll(self.texture, -7, axis=1)
        shifted_future = np.roll(self.texture, 7, axis=1)
        actual, _ = blend_center(shifted_past, center, shifted_future, pair, pair)
        expected, _ = blend_center(self.texture, center, self.texture,
                                   (self.zero, self.zero), (self.zero, self.zero))
        inner = np.s_[10:-10, 15:-15]
        self.assertTrue(np.any(expected[inner] != center[inner]))
        np.testing.assert_array_equal(actual[inner], expected[inner])
        np.testing.assert_array_equal(actual[:, :7], center[:, :7])
        np.testing.assert_array_equal(center, self.flicker)

    def test_one_unreliable_neighbor_preserves_current(self):
        wrong = self.zero + 8
        for fusion in ('consensus', 'mean'):
            got, active = blend_center(self.texture, self.flicker, self.texture,
                                       (wrong, wrong), (self.zero, self.zero), fusion=fusion)
            np.testing.assert_array_equal(got, self.flicker)
            self.assertEqual(active, 0.0)

    def test_mean_can_use_a_correction_from_one_of_two_reliable_matches(self):
        flow = (self.zero, self.zero)
        consensus, _ = blend_center(self.texture, self.flicker, self.flicker, flow, flow)
        mean, _ = blend_center(self.texture, self.flicker, self.flicker, flow, flow, fusion='mean')
        np.testing.assert_array_equal(consensus, self.flicker)
        self.assertTrue(np.any(mean != self.flicker))

    def test_cut_or_missing_neighbor_preserves_current(self):
        flow = (self.zero, self.zero)
        for neighbor in (None, np.full_like(self.texture, 240)):
            got, active = blend_center(self.texture, self.flicker, neighbor, flow, flow)
            np.testing.assert_array_equal(got, self.flicker)
            self.assertEqual(active, 0.0)

    def test_unreliable_distant_pair_falls_back_to_three_frames(self):
        near = (self.zero, self.zero)
        wrong = self.zero + 8
        baseline, _ = blend_center(self.texture, self.flicker, self.texture, near, near, fusion='mean')
        got, _ = blend_center(self.texture, self.flicker, self.texture, near, near, fusion='mean',
                              distant=(self.flicker, self.flicker, (wrong, wrong), near))
        self.assertTrue(np.any(baseline != self.flicker))
        np.testing.assert_array_equal(got, baseline)

    def test_distant_pair_uses_both_flow_directions(self):
        zero = (self.zero, self.zero)
        backward = self.zero.copy()
        backward[..., 0] = -14
        pair = (backward, -backward)
        got, _ = blend_center(self.flicker, self.flicker, self.flicker, zero, zero, fusion='mean',
                              distant=(np.roll(self.texture, -14, axis=1), np.roll(self.texture, 14, axis=1),
                                       pair, pair))
        expected, _ = blend_center(self.flicker, self.flicker, self.flicker, zero, zero, fusion='mean',
                                   distant=(self.texture, self.texture, zero, zero))
        interior = np.s_[10:-10, 22:-22]
        self.assertTrue(np.any(expected[interior] != self.flicker[interior]))
        np.testing.assert_array_equal(got[interior], expected[interior])
        np.testing.assert_array_equal(got[:, :14], self.flicker[:, :14])


if __name__ == '__main__':
    unittest.main()
