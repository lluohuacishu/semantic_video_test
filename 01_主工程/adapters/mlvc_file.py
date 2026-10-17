"""Complete MLVC-S byte file and independent decoder. This container is local."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

from common import (ROOT, COMMIT, P_WEIGHT_SHA, torch, configure, load_models, probe,
                    extract, rgb_tensor, prepare, gray_dpb, rgb8, save_rgb, dpb_hash,
                    metrics, preview, sha256, save_json, sync, environment)

MAGIC = b"MLVCV001"


def write_file(path, manifest, packets):
    header = json.dumps(manifest, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<I", len(header)))
        stream.write(header)
        for packet in packets:
            stream.write(struct.pack("<I", len(packet)))
            stream.write(packet)


def read_file(path):
    with Path(path).open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError("Invalid container magic")
        data = stream.read(4)
        if len(data) != 4:
            raise ValueError("Truncated header length")
        length = struct.unpack("<I", data)[0]
        if length > 1024 * 1024:
            raise ValueError("Header exceeds initial-profile limit")
        manifest = json.loads(stream.read(length))
        if manifest["profile"] != "mlvc-s-fp32-yuv420-v1":
            raise ValueError("Unsupported coding profile")
        packets = []
        for frame in manifest["frames"]:
            data = stream.read(4)
            if len(data) != 4:
                raise ValueError("Truncated packet length")
            length = struct.unpack("<I", data)[0]
            if length != frame["bytes"]:
                raise ValueError("Packet length mismatch")
            packet = stream.read(length)
            if hashlib.sha256(packet).hexdigest() != frame["sha256"]:
                raise ValueError("Packet SHA256 mismatch")
            packets.append(packet)
        if len(packets) != manifest["frame_count"] or stream.read(1):
            raise ValueError("Unexpected container length")
    return manifest, packets


def encode(args):
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = Path(args.source).resolve()
    info = probe(source)
    if args.reference_dir:
        if not args.fps or args.frames or args.width or args.height:
            raise ValueError("Prepared RGB input requires --fps and no frame/resize overrides")
        prepared = sorted(Path(args.reference_dir).resolve().glob("*.png"))
        if not prepared:
            raise ValueError("Prepared reference directory has no PNG frames")
        (run / "reference").mkdir()
        for idx, path in enumerate(prepared):
            os.link(path, run / "reference" / f"{idx:05d}.png")
        refs = sorted((run / "reference").glob("*.png"))
    else:
        refs = extract(source, run / "reference", args.width, args.height, args.frames)
    p_model, i_model, model_config = load_models(args.weight, args.i_weight, args.device)
    model_load_seconds = time.perf_counter() - start
    first = rgb_tensor(refs[0], args.device)
    height, width = first.shape[-2:]
    padded = prepare(first, p_model.padding_size)
    padded_height, padded_width = padded.shape[-2:]
    manifest = {"profile": "mlvc-s-fp32-yuv420-v1", "container": "local MLVCV001; official MLVC entropy streams", "upstream_commit": COMMIT,
                "p_weight_sha256": sha256(args.weight), "i_weight_sha256": sha256(args.i_weight) if args.i_weight else None,
                "width": width, "height": height, "padded_width": padded_width, "padded_height": padded_height,
                "fps": args.fps or info["avg_frame_rate"], "frame_count": len(refs), "intra_period": args.intra_period,
                "initialization": "dmci-6.0" if i_model else "shared_gray_yuv_0.5", "i_frame_pass_count": 1,
                "color": "BT.709 full-range, chroma average downsample and nearest upsample, replicate padding bottom-right",
                "p_model_config": model_config["p_frame"], "frames": []}
    (run / "encoder_rgb").mkdir()
    packets, frame_records = [], []
    dpb = None
    torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
    with torch.inference_mode():
        for idx, path in enumerate(refs):
            x = prepare(rgb_tensor(path, args.device), p_model.padding_size)
            intra_idx = idx % args.intra_period
            reset = intra_idx == 0
            frame_type = "I" if reset and i_model else "P"
            fa_idx = p_model.frame_index_map[(intra_idx + (0 if i_model else 1)) % len(p_model.frame_index_map)]
            sync(args.device)
            t0 = time.perf_counter()
            if frame_type == "I":
                result = i_model.compress(x, q_index=args.q_index)
                dpb = {"ref_frame": result["x_hat"], "ref_feature": None}
            else:
                if reset:
                    dpb = gray_dpb(x.shape, args.device)
                result = p_model.compress(x, dpb, q_index=args.q_index, fa_idx=fa_idx)
                dpb = result["dpb"]
            sync(args.device)
            elapsed = time.perf_counter() - t0
            packet = bytes(result["bit_stream"])
            packets.append(packet)
            manifest["frames"].append({"index": idx, "type": frame_type, "reset": reset, "q_index": args.q_index,
                                       "fa_idx": fa_idx, "bytes": len(packet), "sha256": hashlib.sha256(packet).hexdigest()})
            pixels = rgb8(result["x_hat"], height, width)
            save_rgb(run / "encoder_rgb" / f"{idx:05d}.png", pixels)
            frame_records.append({"index": idx, "codec_seconds": elapsed, "state": dpb_hash(dpb),
                                  "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
            print(f"encoded {idx+1}/{len(refs)} {frame_type} {len(packet)} bytes {elapsed:.3f}s", flush=True)
    write_file(run / "video.mlvcv", manifest, packets)
    total = (run / "video.mlvcv").stat().st_size
    fps_a, fps_b = map(int, manifest["fps"].split("/"))
    duration = len(refs) * fps_b / fps_a
    payload = sum(map(len, packets))
    result = {"source": str(source), "source_sha256": sha256(source), "prepared_reference_dir": str(Path(args.reference_dir).resolve()) if args.reference_dir else None,
              "native_source_probe": info, "file_sha256": sha256(run / "video.mlvcv"),
              "file_bytes": total, "payload_bytes": payload, "container_bytes": total-payload,
              "i_bytes": sum(f["bytes"] for f in manifest["frames"] if f["type"] == "I"),
              "p_bytes": sum(f["bytes"] for f in manifest["frames"] if f["type"] == "P"),
              "duration_seconds": duration, "average_video_kbps": total*8/duration/1000,
              "theoretical_transfer_seconds_at_20kbps": total*8/20000,
              "codec_encode_seconds": sum(f["codec_seconds"] for f in frame_records),
              "setup_and_extract_seconds": model_load_seconds, "wall_seconds": time.perf_counter()-start,
              "parameters_p": sum(p.numel() for p in p_model.parameters()),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
              "environment": environment(), "frames": frame_records}
    save_json(run / "manifest.json", manifest)
    save_json(run / "encode.json", result)


def decode(args):
    # No source video argument, reference PNGs, cached state, or encoder outputs.
    start = time.perf_counter()
    manifest, packets = read_file(args.file)
    if manifest["upstream_commit"] != COMMIT:
        raise ValueError("Upstream version mismatch")
    allowed_weights = {Path(args.weight).resolve()}
    if args.i_weight:
        allowed_weights.add(Path(args.i_weight).resolve())
    media_reads = []
    def receiver_audit(event, audit_args):
        if event != "open" or not isinstance(audit_args[0], (str, bytes)):
            return
        path = Path(audit_args[0]).resolve()
        mode, flags = audit_args[1:3]
        is_read = (isinstance(mode, str) and "r" in mode) or (mode is None and flags & 3 == 0)
        if not is_read:
            return
        if path.suffix.lower() in {".avi", ".mp4", ".mkv", ".mov", ".webm", ".png", ".jpg", ".jpeg", ".npy", ".npz", ".ckpt", ".tar", ".pt", ".pth"}:
            media_reads.append(str(path))
            if path not in allowed_weights:
                raise RuntimeError(f"Cold decoder attempted non-shared media/state read: {path}")
    sys.addaudithook(receiver_audit)
    if sha256(args.weight) != manifest["p_weight_sha256"]:
        raise ValueError("P-frame weight hash mismatch")
    if manifest["i_weight_sha256"]:
        if not args.i_weight or sha256(args.i_weight) != manifest["i_weight_sha256"]:
            raise ValueError("I-frame weight hash mismatch")
    elif args.i_weight:
        raise ValueError("Gray profile must not use an I-frame model")
    p_model, i_model, config = load_models(args.weight, args.i_weight, args.device)
    if manifest["p_model_config"] != config["p_frame"]:
        raise ValueError("Model architecture mismatch")
    setup_seconds = time.perf_counter() - start
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "rgb").mkdir()
    dpb, records = None, []
    torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
    with torch.inference_mode():
        for frame, packet in zip(manifest["frames"], packets, strict=True):
            sync(args.device)
            t0 = time.perf_counter()
            if frame["type"] == "I":
                if not i_model:
                    raise ValueError("I-frame model missing")
                result = i_model.decompress(packet, q_index=frame["q_index"], height=manifest["padded_height"], width=manifest["padded_width"])
                dpb = {"ref_frame": result["x_hat"], "ref_feature": None}
            else:
                if frame["reset"]:
                    dpb = gray_dpb((1, 3, manifest["padded_height"], manifest["padded_width"]), args.device)
                result = p_model.decompress(packet, dpb, q_index=frame["q_index"], fa_idx=frame["fa_idx"],
                                            height=manifest["padded_height"], width=manifest["padded_width"])
                dpb = result["dpb"]
            sync(args.device)
            elapsed = time.perf_counter() - t0
            pixels = rgb8(result["x_hat"], manifest["height"], manifest["width"])
            save_rgb(output / "rgb" / f"{frame['index']:05d}.png", pixels)
            records.append({"index": frame["index"], "codec_seconds": elapsed, "state": dpb_hash(dpb),
                            "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
            print(f"decoded {frame['index']+1}/{len(packets)} {elapsed:.3f}s", flush=True)
    save_json(output / "decode.json", {"file_sha256": sha256(args.file), "file_bytes": Path(args.file).stat().st_size,
               "codec_decode_seconds": sum(f["codec_seconds"] for f in records), "setup_seconds": setup_seconds,
               "wall_seconds": time.perf_counter()-start, "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
               "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
               "environment": environment(), "receiver_inputs": [str(Path(args.file).resolve()), str(Path(args.weight).resolve())] + ([str(Path(args.i_weight).resolve())] if args.i_weight else []), "frames": records})
    save_json(output / "receiver_read_audit.json", {"non_shared_media_and_state_reads_blocked": True, "observed_media_and_weight_reads": sorted(set(media_reads))})
    if args.preview:
        preview(output / "rgb", output / "reconstructed.mp4", manifest["fps"])


def verify(args):
    run, decoded = Path(args.run), Path(args.decoded)
    enc = json.loads((run / "encode.json").read_text("utf-8"))
    dec = json.loads((decoded / "decode.json").read_text("utf-8"))
    if enc["file_sha256"] != dec["file_sha256"] or len(enc["frames"]) != len(dec["frames"]):
        raise ValueError("Encoded and decoded sequence mismatch")
    for a, b in zip(enc["frames"], dec["frames"], strict=True):
        if a["rgb_sha256"] != b["rgb_sha256"] or a["state"] != b["state"]:
            raise ValueError(f"Independent reconstruction/state mismatch at frame {a['index']}")
    result = metrics(run / "reference", decoded / "rgb", args.device)
    result.update({"independent_decode_exact_rgb_and_state": True, "verified_frames": len(enc["frames"]),
                   "file_bytes": enc["file_bytes"], "average_video_kbps": enc["average_video_kbps"],
                   "theoretical_transfer_seconds_at_20kbps": enc["theoretical_transfer_seconds_at_20kbps"],
                   "codec_encode_seconds": enc["codec_encode_seconds"], "codec_decode_seconds": dec["codec_decode_seconds"]})
    save_json(run / "verification.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "frames"}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    enc = sub.add_parser("encode")
    enc.add_argument("--source", required=True)
    enc.add_argument("--run", required=True)
    enc.add_argument("--weight", default=str(ROOT / "checkpoints/mlvc-s-psnr-v1.ckpt"))
    enc.add_argument("--i-weight")
    enc.add_argument("--q-index", type=int, default=0, choices=range(64))
    enc.add_argument("--intra-period", type=int, default=32)
    enc.add_argument("--frames", type=int)
    enc.add_argument("--width", type=int)
    enc.add_argument("--height", type=int)
    enc.add_argument("--reference-dir", help="Read-only, already prepared RGB PNG sequence; hardlink into run")
    enc.add_argument("--fps", help="Explicit playback fps for a prepared sequence, e.g. 5/1")
    enc.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    dec = sub.add_parser("decode")
    dec.add_argument("--file", required=True)
    dec.add_argument("--output", required=True)
    dec.add_argument("--weight", default=str(ROOT / "checkpoints/mlvc-s-psnr-v1.ckpt"))
    dec.add_argument("--i-weight")
    dec.add_argument("--preview", action="store_true")
    dec.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ver = sub.add_parser("verify")
    ver.add_argument("--run", required=True)
    ver.add_argument("--decoded", required=True)
    ver.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    configure(args.device)
    if args.command == "encode" and args.intra_period <= 0:
        raise ValueError("Initial profile requires positive intra period")
    {"encode": encode, "decode": decode, "verify": verify}[args.command](args)


if __name__ == "__main__":
    main()
