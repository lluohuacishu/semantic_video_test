"""Asset verification rejects incomplete, changed, or escaped deployment files."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from package_c5_receiver import sha, verify_bundle


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="c5 bundle ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.asset = self.root / "asset.bin"
        self.asset.write_bytes(b"known-model-asset")
        self.manifest = {"files": {"asset.bin": {"bytes": self.asset.stat().st_size, "sha256": sha(self.asset)}},
                         "glc_revision": "expected-revision"}
        self.save()

    def save(self):
        (self.root / "bundle_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_exact_assets_and_revision_pass(self):
        with patch("package_c5_receiver.git", side_effect=["expected-revision", ""]):
            self.assertEqual("passed", verify_bundle(self.root)["status"])

    def test_same_size_corruption_rejected(self):
        self.asset.write_bytes(b"broken-model-xxxx")
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            verify_bundle(self.root)

    def test_missing_asset_rejected(self):
        self.asset.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            verify_bundle(self.root)

    def test_parent_escape_rejected(self):
        self.manifest["files"] = {"../outside": {"bytes": 0, "sha256": ""}}
        self.save()
        with self.assertRaisesRegex(ValueError, "unsafe"):
            verify_bundle(self.root)

    def test_different_or_modified_glc_checkout_rejected(self):
        for responses in (["another-revision"], ["expected-revision", " M network.py"]):
            with self.subTest(responses=responses), patch("package_c5_receiver.git", side_effect=responses):
                with self.assertRaisesRegex(ValueError, "checkout differs"):
                    verify_bundle(self.root)


if __name__ == "__main__":
    unittest.main()
