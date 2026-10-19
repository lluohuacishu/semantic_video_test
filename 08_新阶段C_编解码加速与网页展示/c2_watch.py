"""Durable process supervision without importing CUDA; stale state is not success."""
from datetime import datetime, timezone
import ctypes
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def identity(pid):
    """Creation token avoids treating a reused PID as the original process."""
    if os.name == 'nt':
        from ctypes import wintypes as w
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        kernel.OpenProcess.restype = w.HANDLE
        kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
        kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            error = ctypes.get_last_error()
            return {'pid': pid, 'alive': False if error == 87 else None, 'query_error': error}
        try:
            created, exited, system, user = [w.FILETIME() for _ in range(4)]
            code = w.DWORD()
            if not kernel.GetProcessTimes(handle, *map(ctypes.byref, (created, exited, system, user))):
                return {'pid': pid, 'alive': None, 'query_error': ctypes.get_last_error()}
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return {'pid': pid, 'alive': None, 'query_error': ctypes.get_last_error()}
            return {'pid': pid, 'alive': code.value == 259,
                    'creation_token': (created.dwHighDateTime << 32) | created.dwLowDateTime}
        finally:
            kernel.CloseHandle(handle)
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'alive': fields[0] != 'Z', 'creation_token': fields[19]}
    except FileNotFoundError:
        return {'pid': pid, 'alive': False}
    except OSError:
        return {'pid': pid, 'alive': None}


def inspect(folder):
    value = json.loads((Path(folder) / 'process.json').read_text(encoding='utf-8'))
    current = identity(value['pid']) if value.get('pid') else {'alive': False}
    owner = identity(value['supervisor']['pid'])
    same = (current.get('creation_token') is not None and
            current.get('creation_token') == value.get('child_identity', {}).get('creation_token'))
    supervisor_same = (owner.get('creation_token') is not None and
                       owner.get('creation_token') == value['supervisor'].get('creation_token'))
    if value.get('returncode') is not None:
        observed = value['state']
    elif current['alive'] and same:
        observed = 'running' if owner['alive'] and supervisor_same else 'orphan_running'
    elif current['alive'] is None:
        observed = 'liveness_unknown'
    else:
        observed = 'interrupted_exit_unknown'
    return {'observed_state': observed, 'recorded': value, 'current_process': current,
            'current_supervisor': owner, 'cause': None}


class Heartbeat:
    def __init__(self, folder, interval=2):
        self.folder, self.interval = Path(folder), interval
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.data = {'pid': os.getpid(), 'identity': identity(os.getpid()),
                     'started_utc': utc(), 'state': 'importing', 'last_progress': None}
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def update(self, **values):
        with self.lock:
            self.data.update(values)

    def _flush(self):
        with self.lock:
            value = dict(self.data, heartbeat_utc=utc())
        write(self.folder / 'heartbeat.json', value)

    def _loop(self):
        while not self.stop.wait(self.interval):
            try:
                self._flush()
            except BaseException:
                # Visible stderr plus failure surfaced by close; never silently succeed.
                with self.lock:
                    self.data['heartbeat_error'] = traceback.format_exc()
                traceback.print_exc()
                return

    def __enter__(self):
        self._flush()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self._flush()
        if self.data.get('heartbeat_error'):
            raise RuntimeError('Heartbeat persistence failed; inspect worker.log')


def supervise(argv, folder, cwd, stop_file, poll_seconds=1):
    folder, stop_file = Path(folder), Path(stop_file)
    folder.mkdir(parents=True, exist_ok=False)
    value = {'argv': list(map(str, argv)), 'started_utc': utc(), 'state': 'starting',
             'supervisor': identity(os.getpid()), 'pid': None, 'returncode': None,
             'exit_cause': None, 'stop_file': str(stop_file)}
    write(folder / 'process.json', value)
    process, stop_at = None, None
    started = time.perf_counter()
    try:
        with (folder / 'worker.log').open('wb', buffering=0) as log:
            process = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1'),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            value.update(pid=process.pid, child_identity=identity(process.pid), state='running')
            write(folder / 'process.json', value)
            while process.poll() is None:
                if stop_file.exists():
                    if stop_at is None:
                        stop_at = time.monotonic()
                        value.update(stop_requested_utc=utc(), state='stop_requested')
                    if time.monotonic() - stop_at > 60:
                        # Only our still-live child tree; never infer targets from old logs.
                        value['forced_termination'] = True
                        if os.name == 'nt':
                            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                           stdout=log, stderr=log, check=False,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                        else:
                            process.terminate()
                value.update(supervisor_heartbeat_utc=utc(), elapsed_seconds=time.perf_counter() - started)
                write(folder / 'process.json', value)
                time.sleep(poll_seconds)
            value.update(returncode=process.wait(), ended_utc=utc())
            value['state'] = 'completed' if value['returncode'] == 0 else 'failed'
            if value['returncode'] == 2 and stop_file.exists():
                value['state'] = 'cancelled'
            value['returncode_hex'] = f"0x{value['returncode'] & 0xffffffff:08X}"
    except BaseException:
        value.update(state='supervisor_exception', exception=traceback.format_exc(), ended_utc=utc())
        if process is not None and process.poll() is None:
            stop_file.touch()
            try:
                value['returncode'] = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                value['child_may_still_be_running'] = True
        raise
    finally:
        value['elapsed_seconds'] = time.perf_counter() - started
        write(folder / 'process.json', value)
    return value
