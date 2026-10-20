"""Three alternating cold/warm pairs per role, one GPU process at a time."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

from c2_watch import supervise, write
from c3_catalog import read_json, sha256
from c4_config import HERE, REPO, load_settings
from c5_resident import ResidentService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--short-stream", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    runtime = load_settings()["runtime"]
    rows = []
    reference_enc = None
    reference_dec = None
    write(root / "manifest.json", {"code_sha256": {p.name: sha256(p) for p in HERE.glob("c5_*.py")},
          "source_sha256": sha256(args.source), "stream_sha256": sha256(args.stream),
          "neural_concurrency": 1, "pairs": 3, "rate_limited_experiments_performed": False})
    for pair in range(3):
        for role in ("encode", "decode"):
            modes = ("cold", "resident") if pair % 2 == 0 else ("resident", "cold")
            for mode in modes:
                folder = root / f"pair{pair}_{role}_{mode}"
                folder.mkdir()
                task = {"id": folder.name, "input": str(args.source if role == "encode" else args.stream),
                        "output": str(folder / role), "preset": "compact", "backend": "native",
                        "output_mode": "display", "sampling_mode": "time", "clip": {"mode": "first10"}}
                service = None
                try:
                    if mode == "resident":
                        service = ResidentService(role, runtime, folder)
                        warmup_folder = folder / "warmup_job"
                        warmup_folder.mkdir()
                        warmup = {**task, "id": folder.name + "_warmup", "output": str(folder / "warmup"),
                                  "clip": {"mode": "segment", "start": 0, "end": .64}}
                        if role == "decode":
                            warmup["input"] = str(args.short_stream)
                        warmup_started = time.perf_counter()
                        service.execute(warmup, warmup_folder)
                        warmup_seconds = time.perf_counter() - warmup_started
                        started = time.perf_counter()
                        service.execute(task, folder)
                    else:
                        config = {k: str(v) for k, v in runtime.items() if k != "python"}
                        config.update(role=role, device="cuda", process_folder=str(folder / "process"),
                                      stop_file=str(folder / "STOP_REQUESTED"), tasks=[task])
                        write(folder / "config.json", config)
                        warmup_seconds = None
                        started = time.perf_counter()
                        result = supervise([str(runtime["python"]), "-X", "utf8", "-B", str(HERE / "c2_worker.py"),
                                            "--config", str(folder / "config.json")],
                                           folder / "process", REPO, folder / "STOP_REQUESTED", .2)
                        if result["returncode"] != 0:
                            raise RuntimeError("Cold worker failed")
                    elapsed = time.perf_counter() - started
                    timing = read_json(folder / role / "timing.json")
                    if role == "encode":
                        enc = read_json(folder / "encode/encoder/encode.json")
                        key = (enc["file_sha256"], [(f["rgb_sha256"], f["latent_sha256"]) for f in enc["frames"]])
                        if reference_enc is None:
                            reference_enc = key
                        if key != reference_enc or enc["file_sha256"] != sha256(args.stream):
                            raise ValueError("Resident encode changed bytes/checksums")
                        evidence = enc
                    else:
                        dec = read_json(folder / "decode/receiver/decode.json")
                        key = [(f["rgb_sha256"], f["latent_sha256"]) for f in dec["frames"]]
                        if reference_dec is None:
                            reference_dec = key
                        if key != reference_dec or dec["source_media_reads"] or not dec["all_frame_checksums_match"]:
                            raise ValueError("Resident decode changed reconstruction/independence")
                        evidence = dec
                    rows.append({"pair": pair, "role": role, "mode": mode, "dispatch_seconds": elapsed,
                                 "task_seconds": timing["task_seconds"]["offline_view_ready_seconds"],
                                 "acquire_seconds": timing["codec_acquire_seconds"], "warmup_seconds": warmup_seconds,
                                 "model_was_resident": timing["model_was_resident"],
                                 "peak_allocated_mib": evidence["peak_allocated_mib"],
                                 "peak_reserved_mib": evidence["peak_reserved_mib"],
                                 "idle_allocated_mib": timing.get("idle_allocated_mib"),
                                 "idle_reserved_mib": timing.get("idle_reserved_mib"), "checks_passed": True})
                    write(root / "summary.json", {"status": "running", "rows": rows})
                    print("C5_RESIDENT_DONE " + json.dumps(rows[-1]), flush=True)
                finally:
                    if service:
                        service.close()
    comparisons = []
    for pair in range(3):
        cold = sum(r["dispatch_seconds"] for r in rows if r["pair"] == pair and r["mode"] == "cold")
        hot = sum(r["dispatch_seconds"] for r in rows if r["pair"] == pair and r["mode"] == "resident")
        comparisons.append({"pair": pair, "cold_seconds": cold, "warm_seconds": hot, "reduction_percent": (1-hot/cold)*100})
    write(root / "summary.json", {"status": "passed", "rows": rows, "pairs": comparisons,
          "median_reduction_percent": statistics.median(r["reduction_percent"] for r in comparisons),
          "timing_scope": "sum of independent role dispatches; excludes browser/network/queue; one GPU worker at a time",
          "browser_target_52_seconds_verified": False, "two_physical_computers": False,
          "triton_import_available": importlib.util.find_spec("triton") is not None})
    print("C5_RESIDENT_PASSED " + json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()
