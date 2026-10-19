from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from c2_watch import write
from run_c2_paired import summarize
from run_c2 import worker
from recover_c2 import recover
from run_c0 import HERE, REPO, sha


class ResultsTests(unittest.TestCase):
    def test_zero_exit_does_not_accept_missing_task_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('run_c2.supervise', return_value={'returncode': 0, 'state': 'completed'}):
                with self.assertRaisesRegex(RuntimeError, 'missing terminal'):
                    worker(root, 'encode_process', 'encode', [{'id': 'missing', 'output': str(root / 'missing')}])

    def fixture(self, root, pair, role, backend, seconds):
        name = f'{role}_{pair}_{backend}'
        folder = root / name
        folder.mkdir()
        write(folder / 'timing.json', {'run_id': name, 'role': role, 'entropy_backend': backend,
              'group': 'paired', 'pair': pair, 'model_was_resident': True,
              'codec_acquire_seconds': 0, 'task_seconds': {'offline_view_ready_seconds': seconds,
                                                          'evidence_ready_seconds': seconds - 0.1}})
        write(folder / 'validation.json', {'all_rgb_and_latent_hashes_match_b4': True})

    def test_aggregate_uses_matched_pair_sums_and_median_of_reductions(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            write(root / 'paired_config.json', {'pairs': 2, 'preset': 'compact'})
            for pair, values in ((1, (30, 20, 20, 10)), (2, (30, 25, 30, 20))):
                for (role, backend), seconds in zip(
                    [('encode', 'python'), ('encode', 'native'), ('decode', 'python'), ('decode', 'native')], values):
                    self.fixture(root, pair, role, backend, seconds)
            summarize(root)
            result = json.loads((root / 'paired_summary.json').read_text(encoding='utf-8'))
            total = next(row for row in result['groups'] if row['role'] == 'sum')
            self.assertEqual(result['complete_pair_count'], 2)
            self.assertEqual(total['before_seconds']['median'], 55)
            self.assertEqual(total['after_seconds']['median'], 37.5)
            self.assertAlmostEqual(total['reduction_percent']['median'], 32.5)

    def test_unvalidated_or_partial_pairs_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            write(root / 'paired_config.json', {'pairs': 1, 'preset': 'compact'})
            self.fixture(root, 1, 'encode', 'python', 30)
            self.fixture(root, 1, 'encode', 'native', 20)
            (root / 'encode_1_native/validation.json').unlink()
            summarize(root)
            result = json.loads((root / 'paired_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(result['validated_tasks'], 1)
            self.assertEqual(result['complete_pair_count'], 0)
            self.assertEqual(result['groups'], [])

    def test_recovery_refuses_live_or_unresolved_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'encode_process').mkdir()
            for state in ('running', 'orphan_running', 'liveness_unknown'):
                with patch('recover_c2.inspect', return_value={'observed_state': state}):
                    with self.assertRaises(RuntimeError):
                        recover(root, root / 'report.json')

    def test_recovery_excludes_partial_task_without_running_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validator = HERE / 'run_c1.py'
            write(root / 'manifest.json', {'code_sha256': {str(validator.relative_to(REPO)): sha(validator)}})
            write(root / 'encode_process_config.json', {'role': 'encode', 'tasks': [{'id': 'partial', 'output': str(root / 'partial')}]})
            with patch('recover_c2.validate') as validate:
                result = recover(root, root / 'report.json')
                validate.assert_not_called()
            self.assertEqual(len(result['excluded']), 1)
            self.assertEqual(result['validated'], [])


if __name__ == '__main__':
    unittest.main()
