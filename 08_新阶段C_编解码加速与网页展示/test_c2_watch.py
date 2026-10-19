import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from c2_watch import Heartbeat, identity, inspect, supervise, write


class WatchTests(unittest.TestCase):
    def test_current_process_identity(self):
        value = identity(os.getpid())
        self.assertTrue(value['alive'])
        self.assertIsNotNone(value['creation_token'])
        self.assertEqual(value, identity(os.getpid()))

    def test_normal_and_abrupt_exit_are_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for code in (0, 7):
                folder = root / str(code)
                result = supervise([sys.executable, '-c', f'import os; print("alive", flush=True); os._exit({code})'],
                                   folder, root, root / 'STOP', poll_seconds=0.05)
                saved = json.loads((folder / 'process.json').read_text(encoding='utf-8'))
                self.assertEqual(saved['returncode'], code)
                self.assertEqual(saved['state'], 'completed' if code == 0 else 'failed')
                self.assertIn('alive', (folder / 'worker.log').read_text(encoding='utf-8'))
                self.assertEqual(inspect(folder)['observed_state'], result['state'])
                self.assertIsNone(saved['exit_cause'])

    def test_launch_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'failed'
            with self.assertRaises(OSError):
                supervise([str(Path(tmp) / 'missing.exe')], folder, tmp, Path(tmp) / 'STOP')
            value = json.loads((folder / 'process.json').read_text(encoding='utf-8'))
            self.assertEqual(value['state'], 'supervisor_exception')
            self.assertIn('exception', value)

    def test_pid_reuse_and_orphan_are_not_reported_as_normal_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write(folder / 'process.json', {'pid': 20, 'state': 'running', 'returncode': None,
                  'child_identity': {'creation_token': 1}, 'supervisor': {'pid': 10, 'creation_token': 2}})
            with patch('c2_watch.identity', side_effect=[{'alive': True, 'creation_token': 9}, {'alive': False}]):
                self.assertEqual(inspect(folder)['observed_state'], 'interrupted_exit_unknown')
            with patch('c2_watch.identity', side_effect=[{'alive': True, 'creation_token': 1}, {'alive': False}]):
                self.assertEqual(inspect(folder)['observed_state'], 'orphan_running')

    def test_heartbeat_retains_last_progress_and_terminal_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Heartbeat(tmp, interval=0.02) as beat:
                beat.update(state='completed', last_progress={'frame_completed': 35})
            value = json.loads((Path(tmp) / 'heartbeat.json').read_text(encoding='utf-8'))
            self.assertEqual(value['state'], 'completed')
            self.assertEqual(value['last_progress']['frame_completed'], 35)
            self.assertIn('heartbeat_utc', value)


if __name__ == '__main__':
    unittest.main()
