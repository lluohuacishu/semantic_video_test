"""CPU checks for exact RGB conversion and complete, unpadded frame grouping."""
import unittest
import numpy as np
import torch
from benchmark_batch import rgb_batch
from batch_common import groups
from role_codec import ref


class BatchChecks(unittest.TestCase):
    def test_batched_pixel_conversion_matches_original_per_frame(self):
        rng = torch.Generator().manual_seed(31)
        value = torch.randn(4, 3, 8, 10, generator=rng, dtype=torch.float32)
        actual = rgb_batch(value, 9, 7)
        expected = np.stack([ref.rgb_array(value[i:i+1], 9, 7) for i in range(4)])
        self.assertTrue(np.array_equal(actual, expected))
        self.assertEqual(actual.shape, (4, 7, 9, 3))

    def test_nonfinite_and_half_pixel_conversion_are_rejected(self):
        value = torch.zeros(2, 3, 2, 2)
        with self.assertRaises(ValueError):rgb_batch(value.half(), 2, 2)
        value[1, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):rgb_batch(value, 2, 2)

    def test_grouping_keeps_order_and_never_invents_tail_frames(self):
        for count in (1, 19, 20, 21):
            for batch in (1, 2, 4):
                spans = groups(count, batch)
                self.assertEqual([i for begin,end in spans for i in range(begin,end)], list(range(count)))
                self.assertTrue(all(1 <= end-begin <= batch for begin,end in spans))


if __name__ == '__main__':unittest.main()
