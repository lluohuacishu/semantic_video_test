import unittest

from c5_metrics import compression_metrics, timing_metrics


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


class TimingTests(unittest.TestCase):
    def test_model_load_is_not_added_twice_and_total_is_separate(self):
        result = {"role": "sender", "encoded_this_task": True, "service_seconds": 45.1,
                  "preprocessing": {"seconds": 22.97}, "transfer": {"elapsed_seconds": .02},
                  "remote": {"durations": {"decode_seconds": 4.87}}}
        encode = {"codec_acquire_seconds": 3.57, "task_seconds": {"bitstream_closed_seconds": 9.03}}
        metrics = timing_metrics(result, encode)
        self.assertAlmostEqual(metrics["codec_total_seconds"], 13.9)
        self.assertEqual(metrics["task_total_seconds"], 45.1)
        self.assertEqual(metrics["preprocess_seconds"], 22.97)
        self.assertEqual(metrics["transfer_seconds"], .02)

    def test_missing_or_invalid_measurements_stay_unknown(self):
        result = {"role": "sender", "encoded_this_task": True, "service_seconds": float("inf"),
                  "transfer": {"elapsed_seconds": -1}}
        metrics = timing_metrics(result, {"task_seconds": {"bitstream_closed_seconds": True}})
        for key in ("encode_seconds", "decode_seconds", "codec_total_seconds", "task_total_seconds", "transfer_seconds"):
            self.assertIsNone(metrics[key])

    def test_reuse_and_receiver_do_not_claim_a_new_encode(self):
        decode = {"task_seconds": {"decode_verified_seconds": 4}}
        reuse = timing_metrics({"role": "sender", "encoded_this_task": False}, decode=decode)
        self.assertEqual(reuse["encode_seconds"], 0)
        self.assertEqual(reuse["codec_total_seconds"], 4)
        receiver = timing_metrics({"role": "receiver", "timing": decode,
                                   "transfer": {"payload_receive_seconds": .1}})
        self.assertIsNone(receiver["encode_seconds"])
        self.assertIsNone(receiver["codec_total_seconds"])
        self.assertEqual(receiver["decode_seconds"], 4)
        self.assertEqual(receiver["transfer_seconds"], .1)


if __name__ == "__main__":
    unittest.main()
