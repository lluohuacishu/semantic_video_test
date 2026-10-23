"""Pixel preservation, original-reference alignment and finite web input contract."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from c5_enhancement import PreparedFrames, transform
from c5_server import options
from c5_samples import sample_path


class EnhancementTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.random.default_rng(8).integers(0, 256, (32, 48, 3), dtype=np.uint8)

    def test_semantic_core_background_and_fallback(self):
        alpha = np.zeros((32, 48), dtype=np.uint8)
        alpha[8:24, 16:32] = 255
        ordinary = transform(self.rgb, "ordinary")
        strong = transform(self.rgb, "lowpass")
        light = transform(self.rgb, "light_lowpass")
        for mode, target in (("semantic_original", ordinary), ("semantic_light", light)):
            coding = transform(self.rgb, mode, alpha)
            np.testing.assert_array_equal(coding[alpha == 255], target[alpha == 255])
            np.testing.assert_array_equal(coding[alpha == 0], strong[alpha == 0])
            for reason in ("empty_current_prediction", "protection_exceeds_half_frame"):
                np.testing.assert_array_equal(transform(self.rgb, mode, alpha, reason), self.rgb)
        self.assertLess(float(strong.var()), float(light.var()))
        self.assertLess(float(light.var()), float(ordinary.var()))

    def test_reject_unaligned_or_modified_prepared_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frames").mkdir()
            coding = transform(self.rgb, "lowpass")
            Image.fromarray(coding).save(root / "frames/00000.png")
            manifest = root / "prepared.json"
            manifest.write_text(json.dumps({"summary": {"frames": 1, "size": [48, 32], "state": "completed"},
                "records": [{"index": 0, "source_rgb_sha256": hashlib.sha256(self.rgb.tobytes()).hexdigest(),
                             "coding_rgb_sha256": hashlib.sha256(coding.tobytes()).hexdigest()}]}), encoding="utf-8")
            prepared = PreparedFrames(manifest, 1, 48, 32)
            np.testing.assert_array_equal(prepared.frame(0, self.rgb), coding)
            with self.assertRaisesRegex(ValueError, "alignment"):
                prepared.frame(0, np.zeros_like(self.rgb))
            Image.fromarray(self.rgb).save(root / "frames/00000.png")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepared.frame(0, self.rgb)
            with self.assertRaisesRegex(ValueError, "count"):
                PreparedFrames(manifest, 2, 48, 32)

    def test_http_parameters_cannot_select_gt_or_prepared_files(self):
        self.assertEqual(options({"preset": "compact"}, ["compact"])["enhancement"], "ordinary")
        for value in ("gt", "../prepared.json", {}, None):
            with self.assertRaises(ValueError):
                options({"preset": "compact", "enhancement": value}, ["compact"])
        with self.assertRaises(ValueError):
            options({"preset": "compact", "prepared_transform": "arbitrary.json"}, ["compact"])
        with self.assertRaises(ValueError):
            options({"preset": "compact", "enhancement": "lowpass", "sampling_mode": "legacy_frame_stride", "frames": 2}, ["compact"])

    def test_only_registered_sample_paths(self):
        for key in ("007", "../011", "011/../../connection-token.txt", ""):
            with self.assertRaises(ValueError):
                sample_path(key)


if __name__ == "__main__":
    unittest.main()
