"""Checks for the background-only jump repair."""
import unittest

import cv2
import numpy as np

from reef_repair import feather, repair_frame


class ReefRepairTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        y, x = np.indices((64, 80))
        texture = 115 + ((x // 4 + y // 4) % 2 * 2 - 1) * 22
        self.reference = np.repeat(texture[..., None], 3, axis=2).astype(np.uint8)
        self.jump = 230 - self.reference
        self.region = feather((10, 8, 70, 56), self.reference.shape)
        flow = np.zeros((64, 80, 2), np.float32)
        self.flow = (flow, flow)

    def test_corrects_isolated_jump_only_in_the_selected_background(self):
        current = self.jump.copy()
        got = repair_frame(self.reference, current, self.reference, self.flow, self.flow, self.region)
        crop = np.s_[24:40, 24:56]
        before = np.abs(self.reference[crop].astype(np.int16) - current[crop]).mean()
        after = np.abs(self.reference[crop].astype(np.int16) - got[crop]).mean()
        self.assertLess(after, before)
        np.testing.assert_array_equal(got[self.region == 0], self.jump[self.region == 0])
        np.testing.assert_array_equal(current, self.jump)

    def test_neighbor_disagreement_keeps_current(self):
        got = repair_frame(self.reference, self.jump, self.jump, self.flow, self.flow, self.region)
        np.testing.assert_array_equal(got, self.jump)

    def test_inconsistent_motion_keeps_current(self):
        wrong = self.flow[0] + 8
        got = repair_frame(self.reference, self.jump, self.reference, (wrong, wrong), self.flow, self.region)
        np.testing.assert_array_equal(got, self.jump)


if __name__ == '__main__':
    unittest.main()
