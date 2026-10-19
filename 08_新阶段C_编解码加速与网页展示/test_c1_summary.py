"""Regression tests for evidence selection and repaired timing interpretation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from summarize_c1 import build, delta, load_records


class SummaryTests(unittest.TestCase):
    def test_old_cold_endpoints_are_excluded_but_outer_boundary_is_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "job"
            job.mkdir()
            result = job / "result.json"
            result.write_text(json.dumps({"frames": [{}], "frame_processing_seconds": 1,
                "peak_allocated_mib": None, "peak_reserved_mib": None}), encoding="utf-8")
            timing = {"run_id": "job", "result_path": str(result), "startup_before_first_task_seconds": 2,
                "outer_call_wall_seconds": 10, "process_launch_to_evidence_ready_seconds": 9,
                "process_launch_to_view_ready_seconds": 11}
            path = job / "timing.json"
            path.write_text(json.dumps(timing), encoding="utf-8")
            self.assertEqual(load_records(root), [])  # unfinished/unchecked jobs are excluded
            (job / "validation.json").write_text("{}", encoding="utf-8")
            record = load_records(root)[0]
            self.assertEqual(record["process_launch_to_task_return_seconds"], 12)
            self.assertIsNone(record["process_launch_to_evidence_ready_seconds"])
            self.assertIsNone(record["process_launch_to_view_ready_seconds"])
            self.assertIsNone(record["startup_before_first_task_seconds"])
            timing.update(task_timer_origin_perf_counter=100, process_launch_to_task_return_seconds=12.5)
            path.write_text(json.dumps(timing), encoding="utf-8")
            record = load_records(root)[0]
            self.assertEqual(record["process_launch_to_view_ready_seconds"], 11)
            self.assertEqual(record["process_launch_to_task_return_seconds"], 12.5)

    def test_overlapping_recovery_records_are_rejected(self):
        with patch("summarize_c1.load_records", return_value=[{"run_id": "duplicate"}]):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                build(Path("original"), Path("gates"), [Path("recovery")])

    def test_outer_wall_clock_is_not_replaced_with_inner_timer(self):
        before = {"run_id": "before", "outer_call_wall_seconds": 20,
                  "task_seconds": {"offline_view_ready_seconds": 10}}
        after = {"run_id": "after", "outer_call_wall_seconds": 15,
                 "task_seconds": {"offline_view_ready_seconds": 5}}
        self.assertEqual(delta(before, after, "outer_call_wall_seconds")["reduction_percent"], 25)
        self.assertEqual(delta(before, after, "offline_view_ready_seconds")["reduction_percent"], 50)


if __name__ == "__main__":
    unittest.main()
