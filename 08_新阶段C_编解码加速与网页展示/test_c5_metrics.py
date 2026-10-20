import unittest

from c5_metrics import compression_metrics


class CompressionTests(unittest.TestCase):
    header = {"width": 640, "height": 360, "frames": 125, "fps": 12.5}

    def encoded(self, start=0, end=10, duration="10"):
        return {"input": {"input_bytes": 1000000, "sampling_plan": {
            "start_seconds": start, "end_seconds": end, "probe": {"duration": duration}}},
            "mean_psnr_db": 30.5}

    def test_stream_size_baseline_and_rate(self):
        result = compression_metrics(self.header, 25000)
        self.assertEqual(10, result["duration_seconds"])
        self.assertEqual(20, result["video_kbps"])
        self.assertEqual(86400000, result["raw_rgb24_bytes"])
        self.assertEqual(3456, result["raw_rgb24_ratio"])
        self.assertIsNone(result["source_file_ratio"])

    def test_full_video_file_ratio(self):
        result = compression_metrics(self.header, 25000, self.encoded())
        self.assertTrue(result["whole_source_processed"])
        self.assertEqual(40, result["source_file_ratio"])
        self.assertEqual(97.5, result["source_file_reduction_percent"])

    def test_partial_ranges_do_not_compare_against_entire_file(self):
        for encoded in (self.encoded(duration="30"), self.encoded(start=5, end=15, duration="15")):
            with self.subTest(encoded=encoded):
                result = compression_metrics(self.header, 25000, encoded)
                self.assertFalse(result["whole_source_processed"])
                self.assertIsNone(result["source_file_ratio"])
                self.assertIsNone(result["source_file_reduction_percent"])

    def test_short_video_tail_quantization_still_covers_full_source(self):
        header = {**self.header, "frames": 13}
        result = compression_metrics(header, 25000, self.encoded(end=1.01, duration="101/100"))
        self.assertTrue(result["whole_source_processed"])
        self.assertEqual(1.04, result["duration_seconds"])

    def test_legacy_unknown_range_and_nonfinite_psnr(self):
        result = compression_metrics(self.header, 25000,
            {"input": {"input_bytes": 1000000}, "mean_psnr_db": float("inf")})
        self.assertIsNone(result["whole_source_processed"])
        self.assertIsNone(result["source_file_ratio"])
        self.assertIsNone(result["mean_psnr_db"])


if __name__ == "__main__":
    unittest.main()
