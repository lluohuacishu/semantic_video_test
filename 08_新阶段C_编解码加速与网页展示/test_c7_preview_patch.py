"""Verify scoped patch application and rollback; no models, network or GPU."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from c7_apply_preview_patch import STAGE, install, sha
import c7_apply_preview_patch as installer


class PatchTests(unittest.TestCase):
    def test_venv_launcher_is_excluded_but_running_server_is_detected(self):
        project = Path("C:/fixture/semantic_video_test")
        script = str(Path(installer.__file__).resolve())
        command = 'python.exe "' + script + '" --project "' + str(project) + '"'
        rows = [{"ProcessId": 100, "ParentProcessId": 90, "CommandLine": command},
                {"ProcessId": 90, "ParentProcessId": 80, "CommandLine": command},
                {"ProcessId": 80, "ParentProcessId": 70, "CommandLine": str(project / STAGE / "c5_server.py")},
                {"ProcessId": 200, "ParentProcessId": 80, "CommandLine": str(project / STAGE / "c1_worker.py")},
                {"ProcessId": 300, "ParentProcessId": 80, "CommandLine": None}]
        # setUp replaces active_processes for installation tests. Restore it
        # here to exercise the Windows process parser, with only CIM mocked.
        self.active.stop()
        with patch.object(os, "getpid", return_value=100), patch.object(os, "getppid", return_value=90), \
                patch("c7_apply_preview_patch.subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(rows))):
            self.assertEqual(installer.active_processes(project), [80, 200])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "bundle/semantic_video_test"
        self.package = self.root / "patch"
        (self.project / STAGE).mkdir(parents=True)
        (self.package / "payload" / STAGE).mkdir(parents=True)
        self.old = self.project / STAGE / "c5_server.py"
        self.old.write_bytes(b"old server\n")
        self.config = self.project / STAGE / "runtime.local.json"
        self.config.write_bytes(b"user configuration")
        entries = {}
        for filename, body in (("c5_server.py", b"new server\n"), ("c7_diagnostics.py", b"new helper\n")):
            source = self.package / "payload" / STAGE / filename
            source.write_bytes(body)
            entries[STAGE + "/" + filename] = {"bytes": len(body), "sha256": sha(source),
                "previous_sha256": [sha(self.old)] if filename == "c5_server.py" else [None]}
        self.manifest = {"schema": "c7-preview-patch-v1", "producer_commit": "synthetic", "files": entries}
        self.save_manifest()
        self.bundle = self.project.parent / "bundle_manifest.json"
        self.bundle.write_text(json.dumps({"glc_revision": "fixed", "files": {}}), "utf-8")
        self.active = patch("c7_apply_preview_patch.active_processes", return_value=[])
        self.active.start()
        self.addCleanup(self.active.stop)

    def save_manifest(self):
        (self.package / "patch_manifest.json").write_text(json.dumps(self.manifest), "utf-8")

    def test_check_does_not_mutate_then_install_backs_up_and_is_idempotent(self):
        self.assertEqual(install(self.project, self.package, check=True)["state"], "checked")
        self.assertEqual(self.old.read_bytes(), b"old server\n")
        self.assertFalse((self.project / "C7_patch_backups").exists())
        result = install(self.project, self.package)
        self.assertEqual(result["state"], "installed")
        backup = Path(result["backup"])
        self.assertEqual((backup / "files" / STAGE / "c5_server.py").read_bytes(), b"old server\n")
        self.assertEqual(self.old.read_bytes(), b"new server\n")
        self.assertEqual(self.config.read_bytes(), b"user configuration")
        bundle = json.loads(self.bundle.read_text())
        self.assertEqual(bundle["glc_revision"], "fixed")
        self.assertEqual(len(bundle["files"]), 2)
        self.assertEqual(install(self.project, self.package)["state"], "already_installed")

    def test_unknown_local_edit_is_not_overwritten(self):
        self.old.write_bytes(b"user modified server")
        with self.assertRaisesRegex(ValueError, "未覆盖"):
            install(self.project, self.package)
        self.assertEqual(self.old.read_bytes(), b"user modified server")
        self.assertFalse((self.project / "C7_patch_backups").exists())

    def test_corrupt_patch_payload_is_rejected_before_writes(self):
        (self.package / "payload" / STAGE / "c7_diagnostics.py").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            install(self.project, self.package)
        self.assertEqual(self.old.read_bytes(), b"old server\n")
        self.assertFalse((self.project / "C7_patch_backups").exists())

    def test_configuration_and_path_escape_targets_are_rejected(self):
        for name in (STAGE + "/runtime.local.json", STAGE + "/../escape.py"):
            original = self.manifest["files"]
            self.manifest["files"] = {name: {"bytes": 0, "sha256": "0" * 64, "previous_sha256": [None]}}
            self.save_manifest()
            with self.assertRaisesRegex(ValueError, "unsupported target"):
                install(self.project, self.package)
            self.manifest["files"] = original

    def test_running_project_is_rejected_without_stopping_processes(self):
        with patch("c7_apply_preview_patch.active_processes", return_value=[1234]), self.assertRaisesRegex(ValueError, "1234"):
            install(self.project, self.package)
        self.assertEqual(self.old.read_bytes(), b"old server\n")

    def test_mid_install_failure_restores_files_and_manifest(self):
        original_replace = Path.replace
        original_manifest = self.bundle.read_bytes()
        def replace(path, target):
            if path.name == "c7_diagnostics.py.c7-new":
                raise OSError("simulated disk error")
            return original_replace(path, target)
        with patch.object(Path, "replace", replace), self.assertRaisesRegex(OSError, "disk error"):
            install(self.project, self.package)
        self.assertEqual(self.old.read_bytes(), b"old server\n")
        self.assertFalse((self.project / STAGE / "c7_diagnostics.py").exists())
        self.assertEqual(self.bundle.read_bytes(), original_manifest)
        self.assertEqual(list(self.project.rglob("*.c7-new")), [])
        receipt = next((self.project / "C7_patch_backups").glob("*/receipt.json"))
        self.assertEqual(json.loads(receipt.read_text())["state"], "rolled_back")


if __name__ == "__main__":
    unittest.main()
