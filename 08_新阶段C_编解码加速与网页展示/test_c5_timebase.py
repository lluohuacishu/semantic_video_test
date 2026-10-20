"""PTS and RGB tests use small lossless generated videos, not neural inference."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from c4_config import load_settings
from c5_timebase import clip_options, plan_samples, prepare, TimedFrames, hidden


class TimebaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.ffmpeg = load_settings()["runtime"]["ffmpeg"]
        cls.sources = {}
        for name, fps in (("25", "25"), ("2997", "30000/1001"), ("30", "30"), ("60", "60")):
            count = int(Fraction(fps) * 10.2) + 1
            raw = b"".join(np.full((32, 64, 3), i % 251, np.uint8).tobytes() for i in range(count))
            path = cls.root / (name + ".mkv")
            subprocess.run([str(cls.ffmpeg), "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", "64x32", "-framerate", fps, "-i", "pipe:0", "-c:v", "ffv1",
                "-pix_fmt", "bgr0", "-n", str(path)], input=raw, check=True, creationflags=hidden())
            cls.sources[name] = path

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_input_rates_cover_ten_seconds_and_exact_pixels(self):
        for name, source in self.sources.items():
            for preset, count in (("compact", 125), ("motion", 250)):
                with self.subTest(source=name, preset=preset):
                    plan = prepare(source, self.ffmpeg, preset)
                    self.assertEqual(count, plan["frames"])
                    self.assertEqual(10, plan["duration_seconds"])
                    folder = self.root / (name + preset)
                    folder.mkdir()
                    args = argparse.Namespace(input=source, ffmpeg=self.ffmpeg, workdir=folder, time_plan=plan)
                    reader = TimedFrames(args)
                    try:
                        for index, rgb in zip(plan["sampled_source_frame_indices"], reader, strict=True):
                            self.assertTrue(np.all(rgb == index % 251), (name, index, rgb[0, 0]))
                    finally:
                        reader.close()
                    self.assertEqual(count, reader.count)

    def test_fractional_start_and_eof_have_bounded_quantization(self):
        source = self.sources["2997"]
        plan = prepare(source, self.ffmpeg, "compact", {"mode": "segment", "start": 1.125, "end": 30})
        self.assertTrue(plan["clipped_to_eof"])
        self.assertLess(plan["duration_seconds"] - plan["selected_duration_seconds"], 1 / 12.5)
        self.assertGreater(plan["sampled_source_frame_indices"][0], 0)
        for actual, requested in zip(plan["sampled_source_pts"], plan["target_source_pts"]):
            self.assertLessEqual(Fraction(actual), Fraction(requested))

    def test_full_is_not_silently_limited_to_125_frames(self):
        plan = prepare(self.sources["60"], self.ffmpeg, "compact", {"mode": "full"})
        self.assertGreater(plan["frames"], 125)

    def test_known_vfr_intervals_and_duplication(self):
        probe = {"width": 64, "height": 32, "pts": ["0", "1/10", "1/2", "3/5"],
                 "duration": "4/5", "source_frames": 4}
        plan = plan_samples(probe, "compact", {"mode": "full"})
        self.assertEqual([0, 0, 1, 1, 1, 1, 1, 2, 3, 3], plan["sampled_source_frame_indices"])
        self.assertEqual(6, plan["duplicated_output_frames"])

    def test_real_vfr_pts_match_decoded_pixels(self):
        path = self.root / "vfr.mkv"
        raw = b"".join(np.full((32, 64, 3), i * 20, np.uint8).tobytes() for i in range(4))
        subprocess.run([str(self.ffmpeg), "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", "64x32", "-framerate", "10", "-i", "pipe:0",
            "-vf", "setpts='if(lt(N,2),N, N+3)'", "-fps_mode", "passthrough",
            "-c:v", "ffv1", "-pix_fmt", "bgr0", "-n", str(path)],
            input=raw, check=True, creationflags=hidden())
        plan = prepare(path, self.ffmpeg, "compact", {"mode": "full"})
        self.assertEqual(["0", "1/10", "1/2", "3/5"], plan["probe"]["pts"])
        folder = self.root / "vfr-read"
        folder.mkdir()
        reader = TimedFrames(argparse.Namespace(input=path, ffmpeg=self.ffmpeg, workdir=folder, time_plan=plan))
        try:
            for index, rgb in zip(plan["sampled_source_frame_indices"], reader, strict=True):
                self.assertTrue(np.all(rgb == index * 20))
        finally:
            reader.close()

    def test_invalid_bounds_and_frame_budget(self):
        for options in ({"mode": "bad"}, {"mode": "first10", "end": 4},
                        {"mode": "segment", "start": 2, "end": 1},
                        {"mode": "segment", "start": True, "end": 2},
                        {"mode": "segment", "start": 0, "end": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                clip_options(options)
        with self.assertRaises(ValueError):
            prepare(self.sources["25"], self.ffmpeg, "compact", {"mode": "segment", "start": 100, "end": 110})
        probe = {"width": 64, "height": 32, "pts": ["0"], "duration": "801"}
        with self.assertRaisesRegex(ValueError, "10000"):
            plan_samples(probe, "compact", {"mode": "full"})


if __name__ == "__main__":
    unittest.main()
