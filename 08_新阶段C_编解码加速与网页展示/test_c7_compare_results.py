"""Synthetic CPU evidence only; no neural decoder or CUDA use."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "04_外部依赖快照/semantic-video-training/scripts"))
import glc_container as container
from c7_compare_results import compare_results
from c7_diagnostics import sha256, sequence_report, write_json


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.header = {"width": 2, "height": 2, "frames": 3, "fps": 15.0,
                       "intra_period": 64, "arithmetic": {"device": "synthetic"}}
        self.digest = hashlib.sha256(b"synthetic stream").hexdigest()

    def raw(self, frame):
        values = {"rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                  "ref_latent": np.zeros((1, 1, 1, 1), dtype=np.float32)}
        if frame != 1:
            values["ref_y"] = np.zeros((1, 2, 1, 1), dtype=np.float32)
        return values

    def hashes(self, frame, values):
        latent = hashlib.sha256()
        for key in ("ref_latent", "ref_y"):
            latent.update(key.encode())
            if key in values:
                latent.update(values[key].tobytes())
        return {"frame": frame, "rgb_sha256": hashlib.sha256(values["rgb"].tobytes()).hexdigest(),
                "latent_sha256": latent.hexdigest()}

    def run_folder(self, name, *, mutate=None, count=3, snapshots=(1, 2, 3), empty_trace=False):
        folder = self.root / name
        folder.mkdir()
        expected = [self.hashes(i, self.raw(i)) for i in range(1, 4)]
        expected_sequence, actual_sequence = container.SequenceChecksums(), container.SequenceChecksums()
        for i, row in enumerate(expected):
            expected_sequence.add(i, row)
        records, files = [], []
        for frame in range(1, count + 1):
            values = self.raw(frame)
            if mutate:
                mutate(frame, values)
            record = {**self.hashes(frame, values), "expected": expected[frame - 1],
                      "entropy_trace": [] if empty_trace else [
                          {"part": 0, **{key: self.digest for key in
                           ("compressed_sha256", "cdf_indexes_sha256", "symbols_sha256")}}]}
            record["mismatched_channels"] = [k for k in ("latent_sha256", "rgb_sha256")
                                             if record[k] != record["expected"][k]]
            actual_sequence.add(frame - 1, record)
            records.append(record)
            if frame in snapshots:
                path = folder / f"frame_{frame:05d}.npz"
                np.savez_compressed(path, **values)
                files.append({"frame": frame, "file": path.name, "sha256": sha256(path)})
        (folder / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), "utf-8")
        report = {"schema": "c7-probe-v1", "state": "failed" if mutate or count < 3 else "sample_passed",
                  "backend": "python", "header": self.header, "input_sha256": self.digest,
                  "code_sha256": {"fixture.py": self.digest}, "probe_code_sha256": {"probe.py": self.digest},
                  "reconstruction": sequence_report(self.header, {"device": name}, expected_sequence.values(),
                                                     actual_sequence.values(), records, self.digest),
                  "snapshots": files}
        write_json(folder / "report.json", report)
        return folder

    def edit_report(self, folder, change):
        path = folder / "report.json"
        report = json.loads(path.read_text("utf-8"))
        change(report)
        write_json(path, report)

    def test_exact_pixels_have_null_psnr_with_explicit_infinity_not_acceptance(self):
        report = compare_results(self.run_folder("left"), self.run_folder("right"))
        self.assertTrue(report["full_sample_pixels_compared"])
        self.assertFalse(report["cross_device_accepted"])
        self.assertEqual(report["provenance_confounders"], [])
        for row in report["measurements"]:
            self.assertTrue(row["rgb"]["byte_equal"])
            self.assertIsNone(row["rgb"]["psnr_db"])
            self.assertTrue(row["rgb"]["psnr_infinite"])
        json.dumps(report, allow_nan=False)

    def test_later_rgb_difference_and_partial_sequence_are_quantified(self):
        def mutate(frame, values):
            if frame == 2:
                values["rgb"][0, 0, 0] = 1
                values["rgb"][1, 0, 2] = 3
        report = compare_results(self.run_folder("left"), self.run_folder("right", mutate=mutate, count=2))
        self.assertEqual(report["snapshot_frames_compared"], [1, 2])
        self.assertFalse(report["full_sample_pixels_compared"])
        self.assertEqual(report["first_hash_difference_between_runs"]["rgb_sha256"], 2)
        self.assertTrue(report["first_difference_has_paired_snapshot"]["rgb_sha256"])
        metrics = report["measurements"][1]["rgb"]
        self.assertEqual(metrics["different_pixels"], 2)
        self.assertEqual(metrics["different_channels_rgb"], [1, 0, 1])
        self.assertAlmostEqual(metrics["mae"], 4 / 12)
        self.assertAlmostEqual(metrics["mse"], 10 / 12)
        self.assertEqual(metrics["max_abs_error"], 3)
        self.assertAlmostEqual(metrics["psnr_db"], 10 * math.log10(255 ** 2 / (10 / 12)))
        self.assertTrue(report["measurements"][1]["reference_state"]["ref_latent"]["byte_equal"])

    def test_reference_state_differences_and_signed_zero_remain_visible(self):
        def mutate(frame, values):
            values["ref_latent"][:] = -0.0 if frame == 1 else 0.25
        report = compare_results(self.run_folder("left"), self.run_folder("right", mutate=mutate))
        signed = report["measurements"][0]["reference_state"]["ref_latent"]
        self.assertFalse(signed["byte_equal"])
        self.assertTrue(signed["numeric_equal"])
        self.assertEqual(report["measurements"][1]["reference_state"]["ref_latent"]["max_abs_error"], 0.25)

    def test_backends_and_streams_cannot_be_mixed(self):
        left, right = self.run_folder("left"), self.run_folder("right")
        self.edit_report(right, lambda r: r.update(backend="native"))
        with self.assertRaisesRegex(ValueError, "backends differ"):
            compare_results(left, right)
        def change(r):
            r["backend"] = "python"
            r["input_sha256"] = r["reconstruction"]["input_sha256"] = "0" * 64
        self.edit_report(right, change)
        with self.assertRaisesRegex(ValueError, "same stream"):
            compare_results(left, right)

    def test_missing_raw_frames_are_not_assumed_identical(self):
        left = self.run_folder("left", snapshots=())
        right = self.run_folder("right")
        report = compare_results(left, right)
        self.assertEqual(report["state"], "insufficient_raw_snapshots")
        self.assertEqual(report["measurements"], [])
        self.assertEqual(report["common_frames_without_paired_snapshots"], [1, 2, 3])

    def test_sparse_old_output_flags_missing_divergence_snapshot(self):
        def mutate(frame, values):
            if frame == 2:
                values["rgb"][:] = 1
        report = compare_results(self.run_folder("left", snapshots=(1,)),
                                 self.run_folder("right", mutate=mutate, count=2))
        self.assertEqual(report["snapshot_frames_compared"], [1])
        self.assertFalse(report["first_difference_has_paired_snapshot"]["rgb_sha256"])

    def test_snapshot_file_tampering_is_rejected(self):
        left, right = self.run_folder("left"), self.run_folder("right")
        with (right / "frame_00001.npz").open("ab") as handle:
            handle.write(b"altered")
        with self.assertRaisesRegex(ValueError, "file checksum"):
            compare_results(left, right)

    def test_rehashed_snapshot_still_has_to_match_frame_pixels(self):
        left, right = self.run_folder("left"), self.run_folder("right")
        values = self.raw(1)
        values["rgb"][:] = 10
        path = right / "frame_00001.npz"
        np.savez_compressed(path, **values)
        self.edit_report(right, lambda r: r["snapshots"][0].update(sha256=sha256(path)))
        with self.assertRaisesRegex(ValueError, "RGB does not match"):
            compare_results(left, right)

    def test_nonfinite_reference_and_incorrect_rgb_dtype_are_rejected(self):
        left = self.run_folder("left")
        def nonfinite(frame, values):
            values["ref_latent"][:] = np.nan
        with self.assertRaisesRegex(ValueError, "finite float32"):
            compare_results(left, self.run_folder("nan", mutate=nonfinite))
        def wrong_type(frame, values):
            values["rgb"] = values["rgb"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "uint8"):
            compare_results(left, self.run_folder("dtype", mutate=wrong_type))

    def test_frame_logs_and_aggregates_are_checked(self):
        left, right = self.run_folder("left"), self.run_folder("right")
        (right / "frames.jsonl").write_text("", "utf-8")
        with self.assertRaisesRegex(ValueError, "frames.jsonl differs"):
            compare_results(left, right)
        self.edit_report(right, lambda r: r["reconstruction"]["actual_sequence_checksums"].update(rgb_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "sequence aggregate"):
            compare_results(left, right)

    def test_snapshot_paths_cannot_escape_the_result_folder(self):
        left, right = self.run_folder("left"), self.run_folder("right")
        self.edit_report(right, lambda r: r["snapshots"][0].update(file="../outside.npz"))
        with self.assertRaisesRegex(ValueError, "local .npz"):
            compare_results(left, right)

    def test_missing_trace_and_changed_code_are_explicit(self):
        left, right = self.run_folder("left", empty_trace=True), self.run_folder("right")
        self.edit_report(right, lambda r: r["code_sha256"].update(fixture="0" * 64))
        report = compare_results(left, right)
        self.assertEqual(report["provenance_confounders"], ["code_sha256 missing or different"])
        self.assertIsNone(report["entropy_comparison"][0]["matches"])
        self.assertFalse(report["entropy_comparison"][0]["available"])

    def test_cli_uses_no_torch_and_preserves_existing_output(self):
        left, right = self.run_folder("left"), self.run_folder("right")
        output = self.root / "comparison.json"
        command = [sys.executable, "-B", str(Path(__file__).with_name("c7_compare_results.py")),
                   "--left", str(left), "--right", str(right), "--output", str(output)]
        first = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertFalse(json.loads(first.stdout)["torch_imported"])
        original = output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(second.returncode, 2)
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
