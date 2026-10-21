"""Manually run a bounded C7 decode diagnostic using an existing C5 installation."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import time
import traceback

from c7_diagnostics import HASH_KEYS, sha256, sequence_report, validate_reference, write_json

STAGE = "08_新阶段C_编解码加速与网页展示"
SCRIPTS = "04_外部依赖快照/semantic-video-training/scripts"


def inspect_inputs(project, stream, reference, backend):
    """No torch import, CUDA initialization, model loading, or inference."""
    project = Path(project).resolve()
    stage = project / STAGE
    if not (stage / "c1_runtime.py").is_file():
        raise ValueError("请选择已有 C5 包内的 semantic_video_test 文件夹")
    sys.path.insert(0, str(stage))
    sys.path.insert(0, str(project / SCRIPTS))
    import glc_container as container
    from c5_config import load_settings
    runtime = load_settings("receiver")["runtime"]
    needed = ["repo", "weights"] + (["native_library"] if backend == "native" else [])
    for key in needed:
        if not runtime[key].exists():
            raise ValueError("Missing runtime asset: " + key + " = " + str(runtime[key]))
    header, packets, integrity = container.read_stream(stream, with_integrity=True)
    if integrity["container_version"] != 2 or not 1 <= len(packets) <= 64:
        raise ValueError("C7 diagnostic accepts only v2 samples of 1–64 frames; use the supplied prefix sample")
    reference_value = json.loads(Path(reference).read_text("utf-8-sig"))
    expected_frames = validate_reference(reference_value, sha256(stream), header, integrity, container)
    return project, runtime, container, header, packets, integrity, expected_frames


def run_probe(inputs, stream, output, backend):
    project, runtime, container, header, packets, integrity, expected_frames = inputs
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records, receiver_profile, first_mismatch, snapshots = [], None, None, []
    checksums = container.SequenceChecksums()
    report = {"schema": "c7-probe-v1", "state": "running", "backend": backend,
              "input_sha256": sha256(stream), "header": header,
              "source_media_reads": [], "cross_device_accepted": False,
              "python": sys.version, "platform": platform.platform(),
              "started_at": datetime.now().astimezone().isoformat(),
              "note": "A passing short sample is diagnostic evidence, not full cross-device acceptance."}
    started = time.perf_counter()
    write_json(output / "report.json", report)
    try:
        # Import and load neural assets only after --check has returned and all
        # byte-level evidence has been validated.
        from c1_runtime import ref, receiver_audit
        from c5_compat import check_arithmetic
        import numpy as np

        files = [project / SCRIPTS / name for name in
                 ("glc_video_codec.py", "glc_container.py", "glc_entropy.py", "glc_video_probe.py")]
        files += [project / STAGE / name for name in ("c1_runtime.py", "c2_native.py", "c5_compat.py")]
        report["code_sha256"] = {str(p.relative_to(project)): sha256(p) for p in files}
        report["probe_code_sha256"] = {p.name: sha256(p) for p in
                                       (Path(__file__), Path(__file__).with_name("c7_diagnostics.py"))}
        try:
            info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10)
            report["gpu_driver"] = info.stdout.strip() if info.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            report["gpu_driver"] = None
        if header["revision"] != ref.REVISION or header["entropy"] != ref.ENTROPY_ID:
            raise ValueError("Unsupported model or entropy coder version")
        if header["weights"] != {key: value["sha256"] for key, value in ref.ASSETS.items()}:
            raise ValueError("Stream requires different model weights")
        codec = ref.NeuralCodec(runtime["repo"], runtime["weights"], "cuda", decoder_only=True)
        receiver_profile = codec.profile
        report["arithmetic_admission"] = check_arithmetic(header["arithmetic"], codec.profile,
                                                         "device_pair_test", 2)
        if backend == "native":
            from c2_native import NativeGaussianRans
            codec.entropy = NativeGaussianRans(runtime["native_library"], codec.entropy)
            report["native_library_sha256"] = sha256(runtime["native_library"])
        if header["table_sha256"] != codec.entropy.table_sha256:
            raise ValueError("Entropy tables differ from sender")

        entropy_trace = []
        original_decode = codec.entropy.decode

        def traced_decode(data, indexes):
            result = original_decode(data, indexes)
            entropy_trace.append({"part": len(entropy_trace),
                                  "compressed_sha256": hashlib.sha256(data).hexdigest(),
                                  "cdf_indexes_sha256": hashlib.sha256(np.asarray(indexes, dtype="<u2").tobytes()).hexdigest(),
                                  "symbols_sha256": hashlib.sha256(np.asarray(result, dtype="<i4").tobytes()).hexdigest()})
            return result

        codec.entropy.decode = traced_decode
        with receiver_audit() as reads:
            report["source_media_reads"] = reads
            for index, packet in enumerate(packets):
                entropy_trace.clear()
                rgb, checked = codec.frame(header, index, packet=packet)
                checksums.add(index, checked)
                mismatches = [k for k in HASH_KEYS if checked[k] != expected_frames[index][k]]
                record = {"frame": index + 1, **{k: checked[k] for k in HASH_KEYS},
                          "expected": expected_frames[index], "mismatched_channels": mismatches,
                          "entropy_trace": list(entropy_trace)}
                records.append(record)
                # Preserve the first frame and first divergence for later numeric
                # comparison between machines. These are outputs, never decoder inputs.
                if index == 0 or mismatches:
                    snapshot = output / f"frame_{index + 1:05d}.npz"
                    values = {"rgb": rgb}
                    values.update({k: v.detach().cpu().contiguous().numpy()
                                   for k, v in codec.dpb.items() if v is not None})
                    np.savez_compressed(snapshot, **values)
                    snapshots.append({"file": snapshot.name, "frame": index + 1})
                with (output / "frames.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log.flush()
                print(f"C7 frame {index + 1}/{len(packets)}: " + (", ".join(mismatches) or "hashes match"), flush=True)
                if mismatches:
                    first_mismatch = index + 1
                    raise ValueError(f"Frame {index + 1}: {', '.join(mismatches)} differs from sender")
        checksums.verify(integrity["sequence_reconstruction_checksums"])
        report["state"] = "sample_passed"
        return_code = 0
    except BaseException as exc:
        report.update(state="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        return_code = 2 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        # Hash saved outputs after leaving the independent-receiver read guard.
        for snapshot in snapshots:
            snapshot["sha256"] = sha256(output / snapshot["file"])
        report["reconstruction"] = sequence_report(header, receiver_profile,
            integrity["sequence_reconstruction_checksums"], checksums.values(), records, sha256(stream))
        report["reconstruction"].update(first_mismatch_frame=first_mismatch,
                                        first_mismatch_known=first_mismatch is not None)
        report.update(snapshots=snapshots, elapsed_seconds=time.perf_counter() - started,
                      finished_at=datetime.now().astimezone().isoformat())
        write_json(output / "report.json", report)
    print("C7 report: " + str((output / "report.json").resolve()), flush=True)
    return return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--stream", type=Path, default=Path(__file__).parent / "samples/prefix_008.glcv")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("native", "python"), default="native")
    parser.add_argument("--check", action="store_true", help="Validate paths/bytes only; no torch import or model run")
    args = parser.parse_args()
    reference = args.reference or args.stream.with_suffix(".reference.json")
    inputs = inspect_inputs(args.project, args.stream, reference, args.backend)
    if args.check:
        print(json.dumps({"status": "inputs_verified", "frames": len(inputs[4]),
                          "gpu_used": False, "neural_inference_performed": False,
                          "torch_imported": "torch" in sys.modules}, ensure_ascii=False))
        return 0
    output = args.output or (Path(__file__).parent / "results" /
                            (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)))
    return run_probe(inputs, args.stream, output, args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
