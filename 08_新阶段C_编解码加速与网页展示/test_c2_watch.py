import json
import ctypes
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from c2_watch import Heartbeat, identity, inspect, supervise, write


class WatchTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows file-sharing semantics')
    def test_real_windows_reader_lock_is_retried(self):
        from ctypes import wintypes as w
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
        kernel.CreateFileW.restype = w.HANDLE
        kernel.CloseHandle.argtypes = [w.HANDLE]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'heartbeat.json'
            write(path, {'state': 'old'})
            handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            release = threading.Timer(0.08, lambda: kernel.CloseHandle(handle))
            release.start()
            try:
                write(path, {'state': 'completed'})
            finally:
                release.join()
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['state'], 'completed')

    def test_transient_windows_replace_failure_retries_and_preserves_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'status.json'
            original = os.replace
            error = PermissionError('temporary sharing conflict')
            error.winerror = 5
            calls = 0
            def replace(a, b):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise error
                return original(a, b)
            with patch('c2_watch.os.replace', side_effect=replace), patch('c2_watch.time.sleep'):
                write(target, {'state': 'completed'})
            self.assertEqual(calls, 3)
            self.assertEqual(json.loads(target.read_text(encoding='utf-8'))['state'], 'completed')

    def test_permanent_replace_failure_remains_bounded_and_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            error = PermissionError('persistent access denied')
            error.winerror = 5
            with patch('c2_watch.os.replace', side_effect=error) as replace, patch('c2_watch.time.sleep'):
                with self.assertRaises(PermissionError):
                    write(Path(tmp) / 'state.json', {'state': 'completed'})
            self.assertEqual(replace.call_count, 10)

    def test_actual_worker_surviving_redirector_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'process.json', {'pid': 20, 'state': 'failed', 'returncode': 1,
                  'child_identity': {'creation_token': 1}, 'supervisor': {'pid': 10, 'creation_token': 2}})
            write(root / 'heartbeat.json', {'pid': 30, 'identity': {'creation_token': 3}})
            with patch('c2_watch.identity', side_effect=[{'alive': False}, {'alive': False},
                                                        {'alive': True, 'creation_token': 3}]):
                self.assertEqual(inspect(root)['observed_state'], 'orphan_running')

    def test_current_process_identity(self):
        value = identity(os.getpid())
        self.assertTrue(value['alive'])
        self.assertIsNotNone(value['creation_token'])
        self.assertEqual(value, identity(os.getpid()))

    def test_recorded_exit_does_not_override_uncertain_worker_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'process.json', {'pid': 20, 'state': 'completed', 'returncode': 0,
                  'child_identity': {'creation_token': 1}, 'supervisor': {'pid': 10, 'creation_token': 2}})
            write(root / 'heartbeat.json', {'pid': 30, 'identity': {'creation_token': 3}})
            for worker in ({'alive': None, 'query_error': 5}, {'alive': True}):
                with self.subTest(worker=worker), patch('c2_watch.identity', side_effect=[
                        {'alive': False}, {'alive': False}, worker]):
                    result = inspect(root)
                    self.assertEqual(result['observed_state'], 'liveness_unknown')
                    self.assertEqual(result['recorded']['returncode'], 0)
            with patch('c2_watch.identity', side_effect=[{'alive': False}, {'alive': False},
                                                        {'alive': True, 'creation_token': 99}]):
                self.assertEqual(inspect(root)['observed_state'], 'completed')

    def test_invalid_or_unreadable_heartbeat_blocks_terminal_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'process.json', {'pid': 20, 'state': 'failed', 'returncode': 1,
                  'child_identity': {'creation_token': 1}, 'supervisor': {'pid': 10, 'creation_token': 2}})
            for invalid in ('{', '{}', 'null', '{"pid": 0, "identity": {}}'):
                (root / 'heartbeat.json').write_text(invalid, encoding='utf-8')
                with self.subTest(invalid=invalid), patch('c2_watch.identity', return_value={'alive': False}):
                    result = inspect(root)
                    self.assertEqual(result['observed_state'], 'liveness_unknown')
                    self.assertIsNotNone(result['heartbeat_read_error'])
            original = Path.read_text
            def read(path, *args, **kwargs):
                if path.name == 'heartbeat.json':
                    raise PermissionError('locked reader')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'read_text', read), patch('c2_watch.identity', return_value={'alive': False}):
                result = inspect(root)
                self.assertEqual(result['observed_state'], 'liveness_unknown')
                self.assertEqual(result['heartbeat_read_error']['type'], 'PermissionError')

    def test_live_launcher_and_uncertain_supervisor_are_not_reported_as_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = {'pid': 20, 'state': 'completed', 'returncode': 0,
                     'child_identity': {'creation_token': 1}, 'supervisor': {'pid': 10, 'creation_token': 2}}
            write(root / 'process.json', value)
            with patch('c2_watch.identity', side_effect=[{'alive': True, 'creation_token': 1}, {'alive': False}]):
                self.assertEqual(inspect(root)['observed_state'], 'orphan_running')
            value.update(returncode=None, state='running')
            write(root / 'process.json', value)
            with patch('c2_watch.identity', side_effect=[{'alive': True, 'creation_token': 1}, {'alive': None}]):
                self.assertEqual(inspect(root)['observed_state'], 'liveness_unknown')

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
