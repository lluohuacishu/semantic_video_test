"""Single-role resident process. Starts only on explicit task dispatch."""
import argparse
import faulthandler
import json
from pathlib import Path
import time
import traceback

from c2_watch import Heartbeat, utc, write


class Stop:
    def __init__(self, global_path):
        self.global_path = Path(global_path)
        self.task_path = None

    def is_set(self):
        return self.global_path.exists() or bool(self.task_path and self.task_path.exists())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text("utf-8"))
    folder = Path(config["process_folder"])
    root = Path(config["mailbox"])
    stop = Stop(config["stop_file"])
    with (folder / "fatal.log").open("w", encoding="utf-8") as fatal, Heartbeat(folder) as beat:
        faulthandler.enable(file=fatal, all_threads=True)
        response = None
        try:
            from c1_runtime import ResidentRunner, TaskCancelled, ref
            from run_c0 import gpu_snapshot
            backend = [None]
            current_id = [None]

            def factory(*a, **kw):
                codec = ref.NeuralCodec(*a, **kw)
                codec.c5_backends = {"python": codec.entropy}
                original = codec.frame
                def frame(header, index, *a, **kw):
                    if stop.is_set():
                        raise TaskCancelled("Cancelled at a frame boundary")
                    result = original(header, index, *a, **kw)
                    beat.update(last_progress={"task": current_id[0], "frame_completed": index + 1,
                                                "frames": header["frames"], "utc": utc()})
                    return result
                codec.frame = frame
                return codec

            class Runner(ResidentRunner):
                def acquire(self):
                    codec = super().acquire()
                    selected = backend[0]
                    if selected not in ("native", "python"):
                        raise ValueError("Invalid entropy backend")
                    if selected not in codec.c5_backends:
                        from c2_native import NativeGaussianRans
                        codec.c5_backends[selected] = NativeGaussianRans(config["native_library"], codec.c5_backends["python"])
                    codec.entropy = codec.c5_backends[selected]
                    return codec

            runner = Runner(config["role"], config["repo"], config["weights"], config["ffmpeg"],
                            config.get("device", "cuda"), codec_factory=factory)
            consumed = set()
            while not stop.global_path.exists():
                requests = sorted(p for p in root.glob("*.request.json") if p.name not in consumed)
                if not requests:
                    beat.update(state="idle", task=None)
                    time.sleep(.1)
                    continue
                request = requests[0]
                consumed.add(request.name)
                message = json.loads(request.read_text("utf-8"))
                response = request.with_name(request.name.replace(".request.json", ".response.json"))
                task = message["task"]
                current_id[0] = task["id"]
                stop.task_path = Path(message["stop_file"])
                backend[0] = task.get("backend", "native")
                beat.update(state="running", task=task["id"], last_progress=None)
                before = gpu_snapshot()
                timing = runner.run(task, cancel_event=stop, on_event=lambda event: beat.update(task_state=event))
                runner.synchronize()
                timing.update(worker_mode="resident", entropy_backend=backend[0], gpu_before=before,
                              gpu_after=gpu_snapshot(), torch_threads=ref.torch.get_num_threads(),
                              idle_allocated_mib=ref.torch.cuda.memory_allocated() / 2**20,
                              idle_reserved_mib=ref.torch.cuda.memory_reserved() / 2**20)
                write(Path(task["output"]) / "timing.json", timing)
                write(response, {"state": "done", "task_id": task["id"], "timing": timing})
                response = None
                stop.task_path = None
                beat.update(state="idle", task=None, task_state={"state": "idle"})
            beat.update(state="completed", ended_utc=utc())
            return 0
        except BaseException as exc:
            cancelled = type(exc).__name__ == "TaskCancelled" or stop.is_set()
            result = {"state": "cancelled" if cancelled else "failed", "type": type(exc).__name__,
                      "message": str(exc), "traceback": traceback.format_exc()}
            if response:
                write(response, result)
            write(folder / "exception.json", result)
            beat.update(**result)
            traceback.print_exc()
            return 2 if cancelled else 1
        finally:
            faulthandler.disable()


if __name__ == "__main__":
    raise SystemExit(main())
