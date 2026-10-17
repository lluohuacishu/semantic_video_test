"""实际文件码流的截断、尺寸和内容校验；不下载模型。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from image_codec import ImageCodec, read_package, validate_header, write_package
from prepare_image_codec import ARCHITECTURE, SOURCE_REVISION


class PackageTests(unittest.TestCase):
    def header(self):
        return {"architecture": ARCHITECTURE, "quality": 3, "entropy_coder": "ans",
                "source_revision": SOURCE_REVISION, "inference_backend": "cpu", "width": 117, "height": 53,
                "shape": [1, 2], "weights_sha256": "a" * 64}

    def test_package_preserves_non_square_geometry_and_exact_streams(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "图片.svic"
            header = write_package(path, self.header(), [b"actual-first-stream", b"actual-second-stream"])
            recovered, streams = read_package(path)
            self.assertEqual(recovered, header)
            self.assertEqual(streams, [b"actual-first-stream", b"actual-second-stream"])
            with self.assertRaises(FileExistsError):
                write_package(path, self.header(), streams)

    def test_truncation_extra_data_and_corrupt_payload_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.svic"
            write_package(path, self.header(), [b"first", b"second"])
            original = path.read_bytes()
            for index, data in enumerate((original[:-1], original + b"extra", original[:-1] + b"X", b"wrong-file")):
                changed = Path(folder) / f"bad-{index}.svic"
                changed.write_bytes(data)
                with self.assertRaises(ValueError):
                    read_package(changed)

    def test_invalid_dimensions_shapes_and_quality_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.svic"
            header = write_package(path, self.header(), [b"one", b"two"])
            for changes in ({"width": True}, {"height": 900000}, {"shape": [2, 2]},
                            {"quality": [3]}, {"stream_lengths": [-1, 2]}, {"weights_sha256": "unknown"},
                            {"inference_backend": "cuda"}):
                with self.assertRaises(ValueError):
                    validate_header(dict(header, **changes))

    def test_cross_device_entropy_decoding_is_not_silently_enabled(self):
        with self.assertRaises(ValueError):
            ImageCodec("cuda")


if __name__ == "__main__":
    unittest.main()
