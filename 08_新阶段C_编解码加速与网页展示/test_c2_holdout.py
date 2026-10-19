from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from c2_holdout import compare_frames, require_distinct_source
from c2_watch import write
from recover_c2 import recover
from run_c2_paired import run
from run_c0 import INPUT_SHA


class SecondClipTests(unittest.TestCase):
    def test_combined_control_uses_legacy_only_for_python_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('run_c2_paired.worker') as worker:
                run(Path(tmp), 'compact', 1, reference_implementation='legacy', frames=8)
            for call in worker.call_args_list:
                for task in call.args[3]:
                    self.assertEqual(task['implementation'], 'legacy' if task['backend'] == 'python' else 'c1')

    def test_renamed_copy_of_main_clip_is_not_a_second_source(self):
        with self.assertRaisesRegex(ValueError, 'content'):
            require_distinct_source('renamed.mkv', INPUT_SHA)

    def setUp(self):
        self.reference = [{'frame': i, 'rgb_sha256': f'rgb{i}', 'latent_sha256': f'latent{i}'} for i in (1, 2, 3)]

    def test_equal_sequence_is_accepted(self):
        compare_frames([dict(row) for row in self.reference], self.reference, 3)

    def test_short_sequence_is_rejected_before_zip(self):
        with self.assertRaisesRegex(ValueError, 'frame count'):
            compare_frames(self.reference[:2], self.reference, 3)

    def test_latent_change_is_rejected(self):
        candidate = [dict(row) for row in self.reference]
        candidate[1]['latent_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'latent_sha256'):
            compare_frames(candidate, self.reference, 3)

    def test_reordered_frames_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Out-of-order'):
            compare_frames(list(reversed(self.reference)), self.reference, 3)

    def test_source_override_and_decoder_stream_stay_in_same_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, validator = root / 'second.mkv', object()
            with patch('run_c2_paired.worker') as worker:
                run(root, 'compact', 1, source=source, validator=validator, frames=125)
            calls = worker.call_args_list
            self.assertEqual(len(calls), 2)
            encode, decode = calls[0].args[3], calls[1].args[3]
            self.assertTrue(all(task['input'] == str(source) for task in encode))
            self.assertTrue(all(task['input'] == str(root / 'encode_1_python/stream.glcv')
                                for task in decode if task['group'] == 'paired'))
            self.assertTrue(all(call.kwargs['validator'] is validator for call in calls))

    def test_unknown_recovery_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'manifest.json', {'validation_target': 'unknown'})
            with self.assertRaisesRegex(ValueError, 'Unknown validation'):
                recover(root, root / 'report.json')


if __name__ == '__main__':
    unittest.main()
