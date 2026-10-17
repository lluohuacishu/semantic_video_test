"""Sequential GPU experiments; each encode/decode/verification is a new process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=r"C:\Users\lenovo\Desktop\临时\语义\test5_5fps_reconstructed.avi")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    log_dir = ROOT / "audit/phase_a_commands"
    log_dir.mkdir(exist_ok=True)
    commands = json.loads((log_dir / "commands.json").read_text("utf-8")) if args.resume else []
    def run(name, arguments):
        if args.resume and any(c["name"] == name and c["exit_code"] == 0 for c in commands):
            print(f"{name}: previous success retained", flush=True)
            return
        cmd = [sys.executable, *arguments]
        started = time.perf_counter()
        with (log_dir / f"{name}.log").open("a" if args.resume else "x", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        commands.append({"name": name, "command": cmd, "exit_code": result.returncode, "seconds": time.perf_counter()-started})
        (log_dir / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{name}: exit {result.returncode}, {commands[-1]['seconds']:.2f}s", flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed, see {log_dir / (name+'.log')}")
    for q in [63, 21, 42]:
        directory = f"runs/baseline_720p_dmci_q{q}"
        if q != 63:
            run(f"encode_q{q}", ["adapters/mlvc_file.py", "encode", "--source", args.source, "--run", directory,
                                "--q-index", str(q), "--i-weight", "checkpoints/DMCI-6.0-mirror.pth.tar"])
        run(f"decode_q{q}", ["adapters/mlvc_file.py", "decode", "--file", f"{directory}/video.mlvcv", "--output", f"{directory}/receiver",
                            "--i-weight", "checkpoints/DMCI-6.0-mirror.pth.tar", "--preview"])
        run(f"verify_q{q}", ["adapters/mlvc_file.py", "verify", "--run", directory, "--decoded", f"{directory}/receiver"])
    run("train_probe_v2", ["adapters/train_probe.py", "--frames", "runs/baseline_720p_dmci_q0/reference", "--output", "runs/train_probe_fp32_v2",
                         "--i-weight", "checkpoints/DMCI-6.0-mirror.pth.tar", "--steps", "20"])
    run("probe_encode_v2", ["adapters/mlvc_file.py", "encode", "--source", args.source, "--run", "runs/probe_weight_bitstream_v2",
                           "--frames", "4", "--width", "256", "--height", "256", "--q-index", "32",
                           "--weight", "runs/train_probe_fp32_v2/probe_finetuned.ckpt", "--i-weight", "checkpoints/DMCI-6.0-mirror.pth.tar"])
    run("probe_decode_v2", ["adapters/mlvc_file.py", "decode", "--file", "runs/probe_weight_bitstream_v2/video.mlvcv", "--output", "runs/probe_weight_bitstream_v2/receiver",
                           "--weight", "runs/train_probe_fp32_v2/probe_finetuned.ckpt", "--i-weight", "checkpoints/DMCI-6.0-mirror.pth.tar"])
    run("probe_verify_v2", ["adapters/mlvc_file.py", "verify", "--run", "runs/probe_weight_bitstream_v2", "--decoded", "runs/probe_weight_bitstream_v2/receiver"])
    run("comparison", ["adapters/compare_baselines.py", "--reference", "runs/baseline_720p_dmci_q0/reference", "--output", "runs/comparison_native",
                         "--mlvc-runs", *[f"runs/baseline_720p_dmci_q{q}" for q in [0, 21, 42, 63]]])


if __name__ == "__main__":
    main()
