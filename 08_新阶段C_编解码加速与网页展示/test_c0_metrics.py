import unittest
from c0_metrics import Timings, distribution


class TimingsTest(unittest.TestCase):
    def test_nested_times_are_not_double_counted(self):
        clock = iter([0, 1, 3, 5]).__next__
        timer = Timings(clock)
        with timer.scope("parent"):
            with timer.scope("child"):
                pass
        result = timer.snapshot()
        self.assertEqual(result["parent"]["inclusive_seconds"], 5)
        self.assertEqual(result["parent"]["exclusive_seconds"], 3)
        self.assertEqual(result["child"]["exclusive_seconds"], 2)
        self.assertEqual(sum(row["exclusive_seconds"] for row in result.values()), 5)

    def test_exception_closes_scope(self):
        timer = Timings(iter([10, 12]).__next__)
        with self.assertRaises(ValueError):
            with timer.scope("failed"):
                raise ValueError("expected")
        self.assertEqual(timer.stack, [])
        self.assertEqual(timer.snapshot()["failed"]["inclusive_seconds"], 2)

    def test_small_sample_reports_range_and_no_fake_tail(self):
        self.assertEqual(distribution([3, 1, 2]), {"n": 3, "median": 2, "min": 1, "max": 3, "values": [3, 1, 2]})
        self.assertIsNone(distribution([]))


if __name__ == "__main__":
    unittest.main()
