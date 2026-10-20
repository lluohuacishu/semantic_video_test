"""Own and supervise one resident role. Task failure discards the whole worker."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import secrets
import time

from c2_watch import supervise, inspect, write
from c4_config import HERE, REPO
from c3_catalog import read_json


class ResidentService:
    def __init__(self, role, runtime, directory):
        self.role, self.runtime, self.directory = role, runtime, Path(directory)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="c5-worker-supervisor")
        self.future = None
        self.folder = None
        self.closed = False

    def launch(self):
        if self.closed:
            raise RuntimeError("Resident service is closed")
        if self.future is not None:
            if not self.future.done():
                return
            self.reap()
        self.folder = self.directory / ("resident_" + secrets.token_hex(12))
        self.folder.mkdir()
        mailbox = self.folder / "mailbox"
        mailbox.mkdir()
        config = {k: str(v) for k, v in self.runtime.items() if k != "python"}
        config.update(role=self.role, device="cuda", process_folder=str(self.folder / "process"),
                      mailbox=str(mailbox), stop_file=str(self.folder / "STOP_REQUESTED"))
        path = self.folder / "config.json"
        write(path, config)
        self.future = self.pool.submit(supervise,
            [str(self.runtime["python"]), "-X", "utf8", "-B", str(HERE / "c5_worker.py"), "--config", str(path)],
            self.folder / "process", REPO, self.folder / "STOP_REQUESTED", .2)

    def reap(self):
        if self.future is None:
            return
        self.future.result(timeout=90)
        state = inspect(self.folder / "process")["observed_state"]
        if state in ("running", "orphan_running", "liveness_unknown"):
            raise RuntimeError("Resident worker liveness is uncertain; new work is blocked")
        self.future = None

    def execute(self, task, job_folder):
        self.launch()
        stop = Path(job_folder) / "STOP_REQUESTED"
        request = self.folder / "mailbox" / (secrets.token_hex(16) + ".request.json")
        response = request.with_name(request.name.replace(".request.json", ".response.json"))
        write(Path(job_folder) / "resident.json", {"role": self.role, "process_folder": str(self.folder / "process"),
                                                 "task_id": task["id"], "request": str(request)})
        started = time.perf_counter()
        write(request, {"task": task, "stop_file": str(stop)})
        while True:
            if stop.exists():
                (self.folder / "STOP_REQUESTED").touch()
                self.reap()  # supervisor owns cooperative timeout and process-tree cleanup
                raise InterruptedError("Resident task cancelled; worker released")
            if response.exists():
                result = read_json(response)
                if result["state"] != "done":
                    self.reap()
                    raise RuntimeError(result.get("message", "Resident task failed"))
                result["dispatch_seconds"] = time.perf_counter() - started
                write(Path(job_folder) / "resident-result.json", result)
                return result
            if self.future.done():
                self.reap()
                raise RuntimeError("Resident worker exited before publishing task result")
            time.sleep(.1)

    def close(self):
        if self.closed:
            return
        if self.future is not None:
            (self.folder / "STOP_REQUESTED").touch()
            self.reap()
        self.pool.shutdown(wait=True)
        self.closed = True
