"""Transmit both selected B3 controls at three rates and verify real receipts."""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time
from b2_common import HERE, ROOT, Journal, git, read, require, sha, write
from run_b0_b1 import child_env, validate_schedule, verify_received
from run_b2 import verify as verify_layered


def deployment(baseline):
    deps = baseline["runtime_dependencies"]
    adapter = (ROOT / baseline["adapter"]["repository_path"]).resolve()
    files = [adapter.parent / name for name in ("glc_video_codec.py", "glc_container.py", "glc_entropy.py", "glc_video_probe.py")]
    files += [ROOT / "01_主工程/semantic_mvp" / name for name in ("transport.py", "runtime.py", "roi_layout.py", "system.py", "glc_layered.py", "glc_layered_decoder.py")]
    files += [HERE / name for name in ("receiver_b2.py", "receiver_baseline.py", "b2_common.py")]
    return {"adapter": str(adapter), "fps": baseline["source"]["fps"], "glc_repo": str(Path(deps["glc_repo"]).resolve()),
            "weights": str(Path(deps["weights"]).resolve()), "ffmpeg": str(Path(deps["ffmpeg"]).resolve()),
            "transport": str(ROOT / "01_主工程/semantic_mvp/transport.py"),
            "shared_models": {str((Path(deps["weights"])/name).resolve()): digest for name, digest in baseline["codec"]["expected_weight_hashes_from_a"].items()},
            "code_hashes": {str(p): sha(p) for p in files}, "wait_for_decode_gate": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--wait-for-searches", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    require(not git("status", "--porcelain"), "Commit runner before executing")
    network = run / "network"
    network.mkdir()
    journal = Journal(network)
    invocation = journal.begin([sys.executable, *sys.argv], code_commit=git("rev-parse", "HEAD"), pid=os.getpid())
    tick = time.perf_counter()
    try:
        searches = [run / f"{method}_search" for method in ("glc", "geometry")]
        if args.wait_for_searches:
            deadline = time.monotonic()+14400
            while not all((p / "selection.json").is_file() for p in searches):
                require(not any((p / "failure.json").exists() for p in searches), "Encoding failed; retain evidence before recovery")
                require(time.monotonic() < deadline, "Search wait expired")
                time.sleep(1)
        saved = read(searches[0]/"config.json")
        baseline, config = saved["baseline"], saved["b3"]
        deps = baseline["runtime_dependencies"]
        os.environ["FFMPEG_PATH"] = deps["ffmpeg"]
        sys.path.insert(0, str(ROOT / "01_主工程/semantic_mvp"))
        sys.path.insert(0, str((ROOT / baseline["adapter"]["repository_path"]).parent))
        from transport import send_file
        from glc_container import read_stream
        import runtime
        runtime.run = journal.run
        deploy = deployment(baseline)
        write(network / "environment.json", {"code_commit": invocation["code_commit"], "runner_sha256": sha(__file__), "deployment": deploy,
              "monotonic_clock": vars(time.get_clock_info("monotonic")), "python": sys.version, "executable": sys.executable,
              "ffmpeg_sha256": sha(deps["ffmpeg"]), "GPU_schedule": "serial; start only after both byte searches finish"})
        methods = {}
        for method, search in zip(("glc", "geometry"), searches):
            selection = read(search/"selection.json")
            chosen = selection["selected"]
            packet = Path(chosen["packet"])
            require(sha(packet) == chosen["sha256"], "Selected packet changed")
            if method == "glc":
                actual = copy.deepcopy(baseline)
                actual["baseline_stream"] = {"path": str(packet), "bytes": packet.stat().st_size, "sha256": sha(packet)}
                header, packets = read_stream(packet)
                require([header[k] for k in ("width","height","frames")] == [1280,720,250], "Control geometry mismatch")
                mapping = [{"decoded_filename": f"im{i+1:05d}.png", "source_frame": i} for i in range(header["frames"])]
            results = []
            for rate in config["rates_kbps"]:
                directory = network / method / f"rate_{rate}kbps"
                receiver = directory / "receiver"
                receiver.mkdir(parents=True)
                write(receiver/"deployment.json", deploy)
                script = "receiver_baseline.py" if method == "glc" else "receiver_b2.py"
                argv = [sys.executable, "-B", str(HERE/script), "--config", str(receiver/"deployment.json"), "--output", str(receiver)]
                started = time.perf_counter()
                log_path = directory/"receiver_process.log"
                entry = journal.begin(argv, cwd=str(receiver), log=str(log_path))
                with log_path.open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(argv, cwd=receiver, stdout=log, stderr=subprocess.STDOUT, env=child_env())
                    entry["pid"] = process.pid
                    write(journal.path, journal.entries)
                    try:
                        deadline = time.monotonic()+120
                        while not (receiver/"ready.json").exists():
                            require(process.poll() is None and time.monotonic()<deadline, "Receiver did not become ready")
                            time.sleep(0.2)
                        ready = read(receiver/"ready.json")
                        require(ready["pid"] != os.getpid(), "Receiver process not independent")
                        write(network/"progress.json", {"method": method, "rate_kbps": rate, "phase": "sending", "receiver_pid": process.pid})
                        print(f"B3 {method} {rate} kbps: sending {packet.stat().st_size} bytes", flush=True)
                        transfer = send_file(packet, "127.0.0.1", ready["port"], rate, trace=directory/"transfer.csv", schedule_trace=directory/"send_schedule.csv")
                        write(directory/"transfer.json", transfer)
                        write(network/"progress.json", {"method": method, "rate_kbps": rate, "phase": "independent_decode", "receiver_pid": process.pid})
                        print(f"B3 {method} {rate} kbps: sent in {transfer['elapsed_seconds']:.3f}s; decoding", flush=True)
                        code = process.wait(timeout=3600)
                        journal.finish(entry, code, time.perf_counter()-started)
                        require(code == 0, f"Receiver failed with {code}: {directory}")
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=30)
                            journal.finish(entry, process.returncode, time.perf_counter()-started)
                if method == "geometry":
                    # Same immutable q0 base, so B2's explicit B1-equality check is valid here.
                    result = verify_layered(rate, directory, packet.parent, journal)
                else:
                    result = verify_received(actual, directory, packets, mapping)
                    require(result["receiver_pid"] != os.getpid(), "Receiver independence failed")
                    pacing = validate_schedule(directory/"send_schedule.csv", rate, transfer["application_bytes"], time.get_clock_info("monotonic").resolution)
                    write(directory/"pacing_verification.json", pacing)
                    probe = runtime.probe(result["preview"])
                    require([probe["width"],probe["height"],int(probe["nb_read_frames"])] == [1280,720,250] and runtime.fps_string(probe["avg_frame_rate"]) == "15/1", "Playback differs")
                    journal.run([deps["ffmpeg"], "-v", "error", "-i", result["preview"], "-f", "null", "-"])
                    write(directory/"playback_probe.json", probe)
                    result.update(status="passed", rate_kbps=rate, transfer=transfer, pacing=pacing, playback_verified=True)
                    write(directory/"result.json", result)
                result["directory"] = str(directory)
                results.append(result)
                print(f"B3 {method} {rate} kbps: all 250 frames verified", flush=True)
                write(network/f"{method}_progress.json", results)
            key = "rgb_sequence_sha256" if method == "glc" else "enhanced_rgb_sequence_sha256"
            require(len({r[key] for r in results}) == 1, "RGB varies by transmission rate")
            methods[method] = {"selection": selection, "results": results, "identical_RGB_across_rates": True}
        write(network/"summary.json", {"status": "B3_development_controls_engineering_passed", "methods": methods, "code_commit": invocation["code_commit"],
              "limitations": ["same-host TCP loopback", "application-byte pacing only", "Python audit, not OS isolation", "video91 is development data", "human review pending"]})
        journal.finish(invocation, 0, time.perf_counter()-tick)
    except BaseException as exc:
        write(network/"failure.json", {"type": type(exc).__name__, "message": str(exc)})
        journal.finish(invocation, 1, time.perf_counter()-tick)
        raise


if __name__ == "__main__":
    main()
