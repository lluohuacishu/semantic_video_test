"""Frozen-E external holdout check; CPU-only independent H265 receiver included."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import zipfile
from external_preflight import WORKSPACE, PROJECT, RUN, OUT, DATA, read, write, digest
sys.path.insert(0, str(PROJECT/"semantic_mvp"))
from runtime import encode_h265, decode_png, preview, probe, run, sha256, FFMPEG, FFPROBE
from system import deliver, install_read_guard
from transport import send_file, receive_connection
from roi_layout import plan_halves

TARGET = 100000
SYSTEM = PROJECT/"semantic_mvp/system.py"
INITIAL = RUN/"initial_crf32"
SOURCE = INITIAL/"private_frames"


def protocol():
    data = read(DATA/"manifest.json")
    fixed = read(PROJECT/"runs/phase_c_20261013/fixed_teacher.json")
    old = read(WORKSPACE/"outputs/semantic_roi_phase_f/step2_box_coverage/步骤2冻结协议.json")
    for path, expected in old["input_sha256"].items():
        if Path(path).name in {"teacher.py", "system.py", "roi_layout.py", "fixed_teacher.json", "best.pt"}:
            assert sha256(path) == expected
    files = [Path(__file__), DATA/"manifest.json", PROJECT/"semantic_mvp/system.py",
             PROJECT/"semantic_mvp/roi_layout.py", PROJECT/"semantic_mvp/teacher.py",
             PROJECT/"semantic_mvp/runtime.py", PROJECT/"semantic_mvp/transport.py",
             Path(fixed["checkpoint"]), PROJECT/"runs/phase_c_20261013/fixed_teacher.json"]
    value = {
        "stage": "F", "step": "3a", "date": "2026-10-15", "source": data,
        "test_role": "new local holdout, official-train video6; capture independence not established",
        "scope": "one preselected 50-frame clip, one complete-file budget; keep negative results",
        "method": "unchanged E with fresh predictions; GT never passed to production encoder",
        "teacher": fixed, "configuration": {
            "threshold": .9, "min_pixels": 64, "segment_frames": 32, "padding": 16,
            "roi_layout": "halves", "background_scale": .5, "background_crf": 42,
            "ROI": "libx265 medium P-only keyint32 BT709 limited YUV420",
            "H265": "libx265 medium offline B4 keyint32 BT709 limited YUV420",
        },
        "byte_selection": {
            "E_target": TARGET, "E_tolerance": .02,
            "candidate_CRFS": "start32, up to11 bisection steps within0..51; stop within1%; all complete packet bytes",
            "H265_target": "selected actual complete E packet bytes; same bisection; full MP4",
            "fallback": "if frozen E cannot reach100000B withinCRF0..51, retain closest feasible packet; H265 matches its actual bytes; no padding and no background/model changes",
            "scores_not_used_for_selection": True,
        },
        "network": "20 decimal kbps, real TCP loopback, separate receive/decode processes for both methods",
        "evaluation": "all50 received RGB global PSNR; 4 sparse publisher GT frames only for fish/background/task; no-GT frames are not labeled background",
        "empty_gt": "no positive GT here; report FP and absent-target behavior, not positive-target protection",
        "human_readability_evaluation": "not performed; examples are prompted visual diagnostics only",
        "not_real_acoustic_or_live_streaming": True,
        "input_sha256": {str(path): sha256(path) for path in files},
        "ffmpeg_version": run([FFMPEG, "-version"]).splitlines()[0],
    }
    path = RUN/"experiment_protocol.json"
    if path.exists():
        assert read(path) == value
    else:
        write(path, value)
    write(OUT/"步骤3a冻结实验协议.json", value)
    return value


def initial_encode():
    if (INITIAL/"encode.json").exists():
        return
    data = read(DATA/"manifest.json")
    command = [
        sys.executable, SYSTEM, "encode", "--input", data["sequence_dir"], "--output", INITIAL,
        "--target", "fish", "--mode", "video", "--first", "0", "--frames", "50",
        "--fps", "25", "--roi-layout", "halves", "--background-scale", "0.5",
        "--background-crf", "42", "--roi-crf", "32", "--segment-frames", "32",
        "--padding", "16", "--min-pixels", "64",
    ]
    write(RUN/"initial_encode_command.json", [str(x) for x in command])
    started = time.perf_counter()
    output = run(command)
    (RUN/"initial_encode.log").write_text(output, "utf-8")
    write(RUN/"initial_encode_elapsed.json", {
        "seconds": time.perf_counter()-started,
        "includes": "fresh model loading, source preparation, 50-frame perception and CRF32 encoding",
        "excludes": "byte search, transfer, receive/decode, evaluation, dataset download"})
    print(output, flush=True)
    source = read(INITIAL/"source_audit.json")
    assert source["source_frame_indices"] == list(range(50))
    assert source["rgb_sha256"] == data["source_rgb_sha256"]


def inputs():
    with zipfile.ZipFile(INITIAL/"message.scv") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        semantic_bytes = archive.read("semantics.json.gz")
        context_bytes = archive.read("context.mp4")
    semantics = json.loads(gzip.decompress(semantic_bytes))
    return manifest, semantics, semantic_bytes, context_bytes


def candidate(method, crf, target):
    folder = RUN/method/"candidates"/f"crf_{crf:.3f}"
    if (folder/"candidate.json").exists():
        return read(folder/"candidate.json")
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    commands = []
    if method == "H265":
        path = folder/"video.mp4"
        commands.append(encode_h265(SOURCE, path, "25/1", crf, count=50, offline=True))
        parts = {"full_frame_mp4": path.stat().st_size}
    else:
        original, semantics, semantic_bytes, context_bytes = inputs()
        if crf == 32:
            path = folder/"message.scv"
            path.write_bytes((INITIAL/"message.scv").read_bytes())
            manifest = original
            write(folder/"manifest.json", manifest)
        else:
            streams = folder/"streams"; streams.mkdir()
            (streams/"context.mp4").write_bytes(context_bytes)
            (streams/"semantics.json.gz").write_bytes(semantic_bytes)
            patches = []
            for patch in original["patches"]:
                command = encode_h265(
                    SOURCE, streams/patch["file"], "25/1", crf, bbox=patch["box"],
                    start=patch["start"], count=patch["frames"])
                commands.append(command)
                patches.append(patch)
            file_info = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in sorted(streams.iterdir())}
            manifest = {**original, "configuration": {**original["configuration"], "roi_crf": crf},
                        "files": file_info, "patches": patches}
            path = folder/"message.scv"
            with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
                for name in file_info:
                    archive.write(streams/name, name)
            write(folder/"manifest.json", manifest)
        parts = {
            "context": manifest["files"]["context.mp4"]["bytes"],
            "roi": sum(manifest["files"][p["file"]]["bytes"] for p in manifest["patches"]),
            "semantic": manifest["files"]["semantics.json.gz"]["bytes"],
            "container": path.stat().st_size-sum(v["bytes"] for v in manifest["files"].values()),
        }
    value = {
        "method": method, "crf": crf, "file": str(path), "file_bytes": path.stat().st_size,
        "sha256": sha256(path), "parts": parts, "target_bytes": target,
        "relative_byte_difference": (path.stat().st_size-target)/target,
        "candidate_encode_seconds": time.perf_counter()-started,
        "cached_source_predictions_and_context": method == "E",
        "commands": commands,
    }
    write(folder/"candidate.json", value)
    print("CANDIDATE", method, crf, value["file_bytes"], flush=True)
    return value


def select(method, target):
    destination = RUN/method/"selection.json"
    if destination.exists():
        return read(destination)
    values = [candidate(method, 32., target)]
    low, high = (32., 51.) if values[0]["file_bytes"] > target else (0., 32.)
    for _ in range(11):
        best = min(values, key=lambda x: abs(x["file_bytes"]-target))
        if abs(best["file_bytes"]-target)/target <= .01:
            break
        # Empty ROI cannot spend more bytes by changing ROI CRF.
        if method == "E" and not inputs()[0]["patches"]:
            break
        crf = round((low+high)/2, 3)
        if crf in {x["crf"] for x in values}:
            break
        row = candidate(method, crf, target); values.append(row)
        if row["file_bytes"] > target: low = crf
        else: high = crf
        write(RUN/method/"search.json", values)
    best = min(values, key=lambda x: abs(x["file_bytes"]-target))
    selected = {**best, "matched_within_2_percent": abs(best["file_bytes"]-target)/target <= .02,
                "selection": "absolute complete-file byte difference only; no decoded/task/GT scores"}
    write(destination, selected); write(RUN/method/"search.json", values)
    print("SELECTED", method, selected["file_bytes"], selected["crf"], flush=True)
    return selected


def receive_h265(output, ready):
    # This branch loads no teacher/model or source manifest.
    output.mkdir(parents=True, exist_ok=False)
    audit = install_read_guard(output)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(120)
        host, port = listener.getsockname()
        write(ready, {"host": host, "port": port, "pid": os.getpid()})
        connection, _ = listener.accept()
        with connection:
            received = receive_connection(connection, output/"video.mp4")
    write(output/"receive.json", received)
    started = time.perf_counter()
    metadata = probe(output/"video.mp4")
    commands = [decode_png(output/"video.mp4", output/"rgb")]
    preview(output/"rgb", output/"reconstructed.mp4", metadata["avg_frame_rate"])
    from PIL import Image
    import numpy as np
    paths = sorted((output/"rgb").glob("*.png"))
    hashes = [hashlib.sha256(np.asarray(Image.open(p).convert("RGB")).tobytes()).hexdigest() for p in paths]
    write(output/"decode.json", {
        "success": True, "independent_process_pid": os.getpid(), "frames": len(paths),
        "width": metadata["width"], "height": metadata["height"],
        "fps": metadata["avg_frame_rate"], "file_bytes": (output/"video.mp4").stat().st_size,
        "file_sha256": sha256(output/"video.mp4"), "decoded_rgb_sha256": hashes,
        "decode_and_preview_seconds": time.perf_counter()-started,
        "inputs": "received native MP4 only; FFmpeg, no source/GT/model", "commands": commands,
        "preview": "H264 viewing copy generated from received RGB, not transmitted or used for metrics"})
    write(output/"receiver_read_audit.json", {
        "guard_installed": True, "python_reads": audit,
        "guard_scope": "Python reads and explicit FFmpeg paths, not an OS sandbox"})


def transfer_h265(selected):
    folder = RUN/"H265/network"; folder.mkdir(parents=True, exist_ok=True)
    if (folder/"end_to_end.json").exists():
        return
    ready = folder/"ready.json"
    command = [sys.executable, Path(__file__), "receive-h265", "--output", folder/"receiver", "--ready", ready]
    with (folder/"receiver_process.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(x) for x in command], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+30
            while not ready.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Native H265 receiver startup failed")
                time.sleep(.1)
            details = read(ready)
            sent = send_file(Path(selected["file"]), details["host"], details["port"], 20,
                             trace=folder/"transfer.csv")
            write(folder/"transfer.json", sent)
            if process.wait(timeout=300):
                raise RuntimeError("Native H265 receiver decode failed")
        finally:
            if process.poll() is None:
                process.terminate(); process.wait(timeout=10)
    decoded = read(folder/"receiver/decode.json")
    assert decoded["file_sha256"] == selected["sha256"] and decoded["frames"] == 50
    value = {
        "success": True, "file_bytes": selected["file_bytes"],
        "application_transmit_bytes": sent["application_bytes"],
        "transfer_seconds": sent["elapsed_seconds"], "packet_sha256_equal": True,
        "separate_sender_receiver_processes": os.getpid() != decoded["independent_process_pid"],
        "sender_pid": os.getpid(), "receiver_pid": decoded["independent_process_pid"], "frames": 50,
        "link": "TCP loopback, 20 decimal kbps; no acoustic channel",
    }
    write(folder/"end_to_end.json", value)
    print("TRANSFERRED_H265", value, flush=True)


def audit():
    rows = []
    for name in ("E", "H265"):
        selected = read(RUN/name/"selection.json")
        if name == "E":
            with zipfile.ZipFile(selected["file"]) as archive:
                m = json.loads(archive.read("manifest.json"))
                destination = RUN/"E/audit_streams"; destination.mkdir(exist_ok=True)
                paths = []
                for filename in m["files"]:
                    if filename.endswith(".mp4"):
                        path = destination/filename; path.write_bytes(archive.read(filename)); paths.append(path)
        else:
            paths = [Path(selected["file"])]
        for path in paths:
            types = json.loads(run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                                    "-show_entries", "frame=pict_type", "-of", "json", path]))
            counts = dict(Counter(x["pict_type"] for x in types["frames"]))
            rows.append({"method": name, "file": path.name, "bytes": path.stat().st_size, "frame_types": counts})
    write(OUT/"实际码流结构核验.json", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "receive-h265"])
    parser.add_argument("--output", type=Path); parser.add_argument("--ready", type=Path)
    args = parser.parse_args()
    if args.command == "receive-h265":
        receive_h265(args.output, args.ready)
        return
    frozen = protocol()
    initial_encode()
    selected_e = select("E", TARGET)
    selected_h = select("H265", selected_e["file_bytes"])
    # Complete selection before any reconstructed quality or task score is computed.
    write(OUT/"新视频完整文件选择.json", {"E": selected_e, "H265": selected_h})
    network = RUN/"E/network"; network.mkdir(parents=True, exist_ok=True)
    if not (network/"end_to_end.json").exists():
        deliver(Path(selected_e["file"]), network, 20)
    transfer_h265(selected_h)
    audit()
    assert all(sha256(p) == value for p, value in frozen["input_sha256"].items())
    write(OUT/"真实收发核验.json", {
        name: read(RUN/name/"network/end_to_end.json") for name in ("E", "H265")})
    print("EXTERNAL_TRANSPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
