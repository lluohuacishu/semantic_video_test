"""Packet corruption, temporal alignment, legacy dispatch and read-guard checks."""
from __future__ import annotations
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_主工程/semantic_mvp"))
import glc_layered as packet
import system
from receiver_b2 import LayeredReadPolicy


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.streams = self.root / "streams"
        self.streams.mkdir()
        semantics = {"meaning": "sender predictions; components are not verified object counts",
                     "frames": [{"frame": i, "targets": {"fish": {"pixels": 0, "regions": []}}} for i in range(2)]}
        (self.streams / "semantics.json.gz").write_bytes(gzip.compress(packet.compact(semantics), mtime=0))
        (self.streams / "base.glcv").write_bytes(b"test packet payload; not a decodable neural stream")
        (self.streams / "roi_0000.mp4").write_bytes(b"test roi payload")
        self.manifest = {"profile": packet.PROFILE, "version": 1, "mode": "video", "width": 64, "height": 32, "frames": 2,
                         "fps": "15/1", "roi_layout": "halves", "compositor": "hard_replace_disjoint_rectangles_v1",
                         "base": {"file": "base.glcv", "codec": "glcv-lab-f32-v1"}, "semantics_file": "semantics.json.gz",
                         "targets": {"fish": {"target": "fish", "class_id": 6, "model_sha256": "a" * 64, "threshold": 0.9}},
                         "configuration": {"roi": {"segment_frames": 32}},
                         "patches": [{"file": "roi_0000.mp4", "box": [0, 0, 16, 16], "start": 0, "frames": 2, "partition": "left"}]}
        self.path = self.root / "message.scv"
        self.ledger = packet.write_packet(self.streams, self.manifest, self.path)

    def rewrite(self, mutate_manifest=None, mutate_payload=None):
        with zipfile.ZipFile(self.path) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        if mutate_manifest:
            manifest = json.loads(members["manifest.json"])
            mutate_manifest(manifest)
            members["manifest.json"] = packet.compact(manifest)
        if mutate_payload:
            mutate_payload(members)
        destination = self.root / "changed.scv"
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED) as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        return destination

    def test_valid_package_ledger_and_existing_dispatch(self):
        self.assertEqual(sum(self.ledger.values()), self.path.stat().st_size)
        output = self.root / "received"
        output.mkdir()
        manifest, _, streams = system.unpack_packet(self.path, output)
        self.assertEqual(manifest["profile"], packet.PROFILE)
        self.assertEqual((streams / "base.glcv").read_bytes(), (self.streams / "base.glcv").read_bytes())

    def test_glc_dispatch_requires_declared_shared_deployment(self):
        with self.assertRaisesRegex(ValueError, "declared shared deployment"):
            system.decode_packet(self.path, self.root)

    def test_corrupt_payload_rejected(self):
        changed = self.rewrite(mutate_payload=lambda values: values.update({"base.glcv": b"corrupted"}))
        with self.assertRaises(ValueError):
            packet.inspect_packet(changed)

    def test_path_traversal_rejected_before_extraction(self):
        changed = self.rewrite(mutate_payload=lambda values: values.update({"../escape": b"bad"}))
        with self.assertRaisesRegex(ValueError, "member/path"):
            packet.inspect_packet(changed)
        self.assertFalse((self.root.parent / "escape").exists())

    def test_duplicate_members_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.path, "a") as archive:
                archive.writestr("base.glcv", b"duplicate")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            packet.inspect_packet(self.path)

    def test_wrong_version_rejected(self):
        with self.assertRaisesRegex(ValueError, "version"):
            packet.inspect_packet(self.rewrite(lambda m: m.update(version=2)))

    def test_unreferenced_stream_rejected(self):
        with self.assertRaises(ValueError):
            packet.inspect_packet(self.rewrite(mutate_payload=lambda values: values.update({"roi_0001.mp4": b"extra"})))

    def test_window_and_even_alignment_rejected(self):
        for key, value in (("start", 1), ("frames", 1), ("box", [1, 0, 16, 16])):
            manifest = copy.deepcopy(self.manifest)
            manifest["patches"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                packet.validate_manifest(manifest)

    def test_cross_partition_and_overlapping_patches_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["patches"][0]["box"] = [24, 0, 16, 16]
        with self.assertRaisesRegex(ValueError, "partition"):
            packet.validate_manifest(manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest["patches"].append({**manifest["patches"][0], "file": "roi_0001.mp4"})
        with self.assertRaisesRegex(ValueError, "overlap"):
            packet.validate_manifest(manifest)

    def test_semantic_frame_order_and_pixel_counts_rejected(self):
        semantics = packet.inspect_packet(self.path)[1]
        semantics["frames"][0]["frame"] = 1
        with self.assertRaisesRegex(ValueError, "order"):
            packet.validate_semantics(semantics, self.manifest)
        semantics["frames"][0]["frame"] = 0
        semantics["frames"][0]["targets"]["fish"]["pixels"] = 1
        with self.assertRaisesRegex(ValueError, "pixel mismatch"):
            packet.validate_semantics(semantics, self.manifest)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            packet.parse(b'{"version":1,"version":2}')

    def test_legacy_profiles_still_unpack(self):
        for profile in ("semantic-video-layered-v1", "semantic-video-layered-v2"):
            semantic = gzip.compress(json.dumps({"target": "fish", "frames": [{"frame": 0, "regions": []}]}).encode())
            members = {"context.mp4": b"legacy-video", "semantics.json.gz": semantic}
            manifest = {"profile": profile, "width": 64, "height": 32, "frames": 1, "fps": "15/1", "mode": "video", "roi_layout": "halves",
                        "context": {"file": "context.mp4"}, "semantics_file": "semantics.json.gz", "teacher": {"target": "fish"}, "patches": [],
                        "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in members.items()}}
            path = self.root / f"{profile}.scv"
            with zipfile.ZipFile(path, "x") as archive:
                archive.writestr("manifest.json", json.dumps(manifest))
                for name, data in members.items():
                    archive.writestr(name, data)
            output = self.root / profile
            output.mkdir()
            self.assertEqual(system.unpack_packet(path, output)[0]["profile"], profile)


class LayeredGuardTests(unittest.TestCase):
    def test_missing_optional_colour_tags_are_recorded(self):
        from glc_layered_decoder import validate_roi_colour
        result = validate_roi_colour({"color_range": "tv", "color_space": "bt709"})
        self.assertIsNone(result["transfer"])
        self.assertIsNone(result["primaries"])
        self.assertEqual(result["missing_optional_tags"], ["color_transfer", "color_primaries"])

    def test_wrong_matrix_range_and_conflicting_tags_rejected(self):
        from glc_layered_decoder import validate_roi_colour
        for change in ({"color_range": "pc"}, {"color_space": "bt2020nc"}, {"color_transfer": "smpte2084"}, {"color_primaries": "bt2020"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_roi_colour({"color_range": "tv", "color_space": "bt709", **change})

    def test_external_ffmpeg_inputs_and_unregistered_commands_denied(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "receiver"
            config = {"shared_models": {}, "code_hashes": {}, "glc_repo": str(root / "glc"), "ffmpeg": str(root / "ffmpeg.exe")}
            policy = LayeredReadPolicy(output, config)
            with self.assertRaisesRegex(ValueError, "External FFmpeg input"):
                policy.authorize([config["ffmpeg"], "-i", str(root / "source.mp4"), str(output / "preview.mp4")])
            with self.assertRaises(PermissionError):
                policy.hook("subprocess.Popen", (None, ["unexpected.exe"], None, {}))


if __name__ == "__main__":
    unittest.main()
