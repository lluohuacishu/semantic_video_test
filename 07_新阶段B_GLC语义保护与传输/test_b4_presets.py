"""Check preset selection at the real encoder CLI without running a GPU job."""
from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ADAPTER = Path(__file__).resolve().parents[1] / "04_外部依赖快照" / "semantic-video-training" / "scripts"
sys.path.insert(0, str(ADAPTER))
from glc_presets import apply_encode_preset


class PresetTests(unittest.TestCase):
    def test_all_four_cli_choices_keep_same_source_time_span(self):
        import glc_video_codec

        expected = {
            "detail_motion": (960, 1, 250), "detail": (960, 2, 125),
            "motion": (640, 1, 250), "compact": (640, 2, 125),
        }
        for name, settings in expected.items():
            with self.subTest(preset=name), patch.object(sys, "argv", [
                "glc_video_codec.py", "encode", "--preset", name,
                "--input", __file__, "--ffmpeg", __file__,
                "--output", "unused.glcv", "--workdir", "unused_work",
            ]), patch.object(glc_video_codec, "encode") as encode:
                glc_video_codec.main()
                args = encode.call_args.args[0]
                self.assertEqual((args.width, args.frame_stride, args.frames), settings)
                self.assertEqual(args.frames * args.frame_stride, 250)
                self.assertEqual((args.q, args.intra_period, args.reset_interval, args.container_version),
                                 (0, 64, 32, 2))

    def test_old_cli_defaults_and_explicit_flags_still_work(self):
        args = apply_encode_preset(SimpleNamespace())
        self.assertEqual((args.frames, args.width, args.frame_stride, args.container_version,
                          args.q, args.intra_period, args.reset_interval), (50, 1280, 1, 1, 63, 32, 32))
        args = apply_encode_preset(SimpleNamespace(width=640, frames=9, q=0, container_version=2))
        self.assertEqual((args.width, args.frames, args.q, args.container_version), (640, 9, 0, 2))

    def test_frame_count_can_change_clip_length_without_changing_preset(self):
        args = apply_encode_preset(SimpleNamespace(preset="compact", frames=250))
        self.assertEqual((args.frames, args.frame_stride, args.width), (250, 2, 640))

    def test_conflicting_preset_option_exits_before_encoding(self):
        import glc_video_codec

        with patch.object(sys, "argv", [
            "glc_video_codec.py", "encode", "--preset", "compact", "--q", "63",
            "--input", __file__, "--ffmpeg", __file__, "--output", "unused.glcv", "--workdir", "unused",
        ]), patch.object(glc_video_codec, "encode") as encode, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                glc_video_codec.main()
            self.assertEqual(raised.exception.code, 2)
            encode.assert_not_called()

    def test_stride_quality_uses_sampled_input_instead_of_early_source_frames(self):
        import tempfile
        from PIL import Image
        from run_b4_pilot import resolution_and_time_quality

        with tempfile.TemporaryDirectory() as temporary:
            source, sampled, decoded = (Path(temporary) / name for name in ("source", "sampled", "decoded"))
            for folder in (source, sampled, decoded):
                folder.mkdir()
            for index, level in enumerate((0, 90, 180, 255), 1):
                Image.new("RGB", (960, 540), (level,) * 3).save(source / f"im{index:05d}.png")
            for index, level in enumerate((0, 180), 1):
                for folder in (sampled, decoded):
                    Image.new("RGB", (960, 540), (level,) * 3).save(folder / f"im{index:05d}.png")
            result = resolution_and_time_quality(source, sampled, decoded, stride=2,
                                                 reference_frames=4, width=960, height=540)
            self.assertEqual(result["metric_version"], 2)
            self.assertIsNone(result["mean_psnr_db_vs_sampled_encoder_input_at_native_resolution"])
            self.assertIsNotNone(result["mean_psnr_db_vs_960x540_25fps_timeline_after_lanczos_upscale"])


if __name__ == "__main__":
    unittest.main()
