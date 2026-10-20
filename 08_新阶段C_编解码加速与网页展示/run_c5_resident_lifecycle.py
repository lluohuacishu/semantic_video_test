"""Ten short tasks per role, real cancellation/failure, then clean recovery."""
import argparse
import json
from pathlib import Path
import threading
import time

from c4_config import load_settings
from c2_watch import inspect, write
from c3_catalog import read_json, sha256
from c5_resident import ResidentService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--stream", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    runtime = load_settings()["runtime"]
    rows, recovery = [], []
    reference_stream = sha256(args.stream)
    for role in ("encode", "decode"):
        role_folder = root / role
        role_folder.mkdir()
        service = ResidentService(role, runtime, role_folder)
        try:
            def execute(name, override=None):
                folder = role_folder / name
                folder.mkdir()
                task = {"id": name, "input": str(args.source if role == "encode" else args.stream),
                        "output": str(folder / role), "preset": "compact", "backend": "native",
                        "output_mode": "display", "sampling_mode": "legacy_frame_stride", "frames": 8}
                task.update(override or {})
                result = service.execute(task, folder)
                if role == "encode" and sha256(folder / "encode/stream.glcv") != reference_stream:
                    raise ValueError("Repeated resident task changed the legacy stream")
                if role == "decode":
                    result_json = read_json(folder / "decode/receiver/decode.json")
                    if not result_json["all_frame_checksums_match"] or result_json["source_media_reads"]:
                        raise ValueError("Decode isolation failed")
                return folder, result

            for i in range(10):
                folder, result = execute(f"task{i}")
                timing = result["timing"]
                rows.append({"role": role, "task": i, "pid": timing["pid"], "resident": timing["model_was_resident"],
                             "idle_allocated_mib": timing["idle_allocated_mib"],
                             "idle_reserved_mib": timing["idle_reserved_mib"], "verified": True})
            # Encoder cancellation during a real long task; decoder corruption after warm reuse.
            previous_pid = rows[-1]["pid"]
            if role == "encode":
                cancelled = role_folder / "cancel"
                errors = []
                def request_cancel():
                    try:
                        deadline = time.monotonic() + 60
                        while time.monotonic() < deadline:
                            beat_path = service.folder / "process/heartbeat.json"
                            beat = read_json(beat_path) if beat_path.exists() else {}
                            progress = beat.get("last_progress") or {}
                            if progress.get("task") == "cancel" and progress.get("frame_completed", 0) >= 2:
                                (cancelled / "STOP_REQUESTED").touch()
                                return
                            time.sleep(.05)
                        errors.append("No cancellation boundary observed")
                    except BaseException as exc:
                        errors.append(str(exc))
                thread = threading.Thread(target=request_cancel)
                thread.start()
                try:
                    execute("cancel", {"frames": 125})
                    raise ValueError("Expected cancelled resident encode")
                except InterruptedError:
                    pass
                finally:
                    thread.join(65)
                if errors or thread.is_alive():
                    raise ValueError(str(errors))
            else:
                corrupt = role_folder / "corrupt.glcv"
                data = bytearray(args.stream.read_bytes())
                data[-1] ^= 1
                corrupt.write_bytes(data)
                try:
                    execute("corrupt", {"input": str(corrupt)})
                    raise ValueError("Expected corrupt stream rejection")
                except RuntimeError:
                    pass
            folder, result = execute("recovery")
            if result["timing"]["pid"] == previous_pid or result["timing"]["model_was_resident"]:
                raise ValueError("Recovery did not create a fresh worker")
            recovery.append({"role": role, "previous_pid": previous_pid, "new_pid": result["timing"]["pid"], "verified": True})
            print("C5_RESIDENT_LIFECYCLE " + json.dumps({"role": role, "successes": 10, "recovery": recovery[-1]}), flush=True)
        finally:
            service.close()
    states = {str(p.relative_to(root)): inspect(p.parent)["observed_state"] for p in root.glob("*/resident_*/process/process.json")}
    if any(s in ("running", "orphan_running", "liveness_unknown") for s in states.values()):
        raise ValueError("Worker remained alive")
    write(root / "summary.json", {"status": "passed", "rows": rows, "recovery": recovery, "process_states": states,
                                 "scope": "ten sequential 8-frame tasks per role; not a long-duration soak"})


if __name__ == "__main__":
    main()
