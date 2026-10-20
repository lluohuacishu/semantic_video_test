from pathlib import Path
import unittest

from run_c2_cold import cold_tasks, endpoints


class ColdTests(unittest.TestCase):
    def test_cold_endpoint_includes_startup_exactly_once(self):
        timing = {'task_timer_origin_perf_counter': 105.0, 'model_was_resident': False,
                  'codec_acquire_seconds': 3.0,
                  'task_seconds': {'evidence_ready_seconds': 30.0, 'offline_view_ready_seconds': 31.0}}
        result = endpoints(timing, 100.0)
        self.assertEqual(result['dispatch_to_evidence_ready_seconds'], 35.0)
        self.assertEqual(result['dispatch_to_view_ready_seconds'], 36.0)
        # Model acquisition is already part of the task, never add another 3 s.
        self.assertEqual(result['model_acquire_seconds_included_in_task'], 3.0)
        timing['model_was_resident'] = True
        with self.assertRaises(ValueError):
            endpoints(timing, 100.0)
        timing['model_was_resident'] = False
        with self.assertRaises(ValueError):
            endpoints(timing, 106.0)

    def test_full_length_cold_roles_are_independent_and_decode_own_stream(self):
        root = Path('experiment')
        tasks = list(cold_tasks(root))
        self.assertEqual([(r, b) for r, b, _ in tasks],
                         [('encode', 'python'), ('encode', 'native'), ('decode', 'native'), ('decode', 'python')])
        self.assertEqual(len({t['output'] for _, _, t in tasks}), 4)
        for role, backend, task in tasks:
            self.assertEqual(task['frames'], 125)
            self.assertEqual(task['group'], 'cold_single_pair')
            self.assertFalse(task['profile'])
            self.assertEqual(task['implementation'], 'legacy' if backend == 'python' else 'c1')
            if role == 'decode':
                self.assertEqual(task['input'], str(root / 'encode_cold_python/stream.glcv'))


if __name__ == '__main__':
    unittest.main()
