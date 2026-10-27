"""Residual bit-error trials on local GLC streams; production checks stay intact."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "04_外部依赖快照/semantic-video-training/scripts"
sys.path.insert(0, str(SCRIPTS))
import numpy as np
from glc_container import read_stream, SequenceChecksums

RATES = (0., 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bit_flips(rng, nbits, rate):
    count = int(rng.binomial(nbits, rate))
    return np.sort(rng.choice(nbits, size=count, replace=False)).astype(np.int64)


def flip(data, positions):
    changed = bytearray(data)
    for bit in positions:
        changed[int(bit) // 8] ^= 1 << (7 - int(bit) % 8)
    return bytes(changed)


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    scale = 1 + z*z / total
    center = (p + z*z/(2*total)) / scale
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / scale
    return [max(0., center-radius), min(1., center+radius)]


def strict_trials(args, raw):
    rng = np.random.default_rng(args.seed)
    candidate = args.output / "channel_trial.glcv"
    summaries = []
    with (args.output / "strict_trials.jsonl").open("w", encoding="utf-8") as log:
        for rate in RATES:
            accepted = corrupted = corrupt_accepted = errors = 0
            reasons = Counter()
            for trial in range(args.file_trials):
                positions = bit_flips(rng, len(raw)*8, rate)
                changed = flip(raw, positions)
                candidate.write_bytes(changed)
                reason = None
                try:
                    read_stream(candidate)
                    ok = True
                except ValueError as exc:
                    ok, reason = False, str(exc)
                    reasons[reason] += 1
                k = len(positions)
                assert sum((a ^ b).bit_count() for a, b in zip(raw, changed)) == k
                accepted += ok
                corrupted += k > 0
                corrupt_accepted += bool(k and ok)
                errors += k
                log.write(json.dumps({"rate": rate, "trial": trial, "flips": k,
                    "accepted": ok, "error": reason, "bit_positions": positions.tolist()}) + "\n")
            row = {"ber": rate, "trials": args.file_trials, "accepted": accepted,
                "rejected": args.file_trials-accepted, "corrupted_trials": corrupted,
                "corrupt_accepted": corrupt_accepted, "actual_flips": errors,
                "measured_ber": errors/(args.file_trials*len(raw)*8),
                "acceptance": accepted/args.file_trials,
                "acceptance_wilson95": wilson(accepted, args.file_trials),
                "predicted_no_error_probability": math.exp(len(raw)*8*math.log1p(-rate)),
                "errors": dict(reasons)}
            summaries.append(row)
            write(args.output / "strict_summary.json", summaries)
            print(f"STRICT ber={rate:g} accepted={accepted}/{args.file_trials}", flush=True)
    candidate.unlink()
    return summaries


def payload_layout(raw, packets):
    cursor = 12 + struct.unpack_from(">I", raw, 8)[0]
    components, chunks = [], []
    payload_offset = 0
    for index, packet in enumerate(packets):
        for kind, part, data in [("z", None, packet["z"])] + [
                ("y", i, value) for i, value in enumerate(packet["y"])]:
            if kind == "y":
                assert struct.unpack_from(">I", raw, cursor)[0] == len(data)
                cursor += 4
            assert raw[cursor:cursor+len(data)] == data
            components.append({"frame": index, "kind": kind, "part": part, "file_offset": cursor,
                               "start": payload_offset, "end": payload_offset+len(data)})
            chunks.append(data)
            payload_offset += len(data)
            cursor += len(data)
    assert cursor + 64 + 32 == len(raw)
    return b"".join(chunks), components


def decoder_trials(args, settings, header, packets, integrity, payload, components):
    from c1_runtime import receiver_audit
    import glc_video_codec as ref
    from c2_native import NativeGaussianRans

    torch = ref.torch
    runtime = settings["runtime"]
    codec = ref.NeuralCodec(runtime["repo"], runtime["weights"], "cuda", decoder_only=True)
    codec.entropy = NativeGaussianRans(runtime["native_library"], codec.entropy)
    assert codec.profile == header["arithmetic"]
    assert codec.entropy.table_sha256 == header["table_sha256"]
    expected = read(args.stream.parent / "encoder/encode.json")["frames"]
    assert len(expected) == len(packets) == header["frames"]
    clean_rgb, clean_meta, states = [], [], []
    aggregate = SequenceChecksums()

    def snapshot():
        return {k: None if v is None else v.detach().cpu().clone() for k, v in codec.dpb.items()}

    with receiver_audit() as reads:
        for index, packet in enumerate(packets):
            states.append(snapshot())
            rgb, checked = codec.frame(header, index, packet=packet)
            assert (checked["rgb_sha256"], checked["latent_sha256"]) == (
                expected[index]["rgb_sha256"], expected[index]["latent_sha256"])
            aggregate.add(index, checked)
            clean_rgb.append(rgb.copy())
            clean_meta.append(checked)
            if index % 16 == 0:
                print(f"BASELINE {index+1}/{len(packets)}", flush=True)
        aggregate.verify(integrity["sequence_reconstruction_checksums"])
        write(args.output / "baseline_validation.json", {"frames": len(packets), "all_hashes_match": True,
            "source_media_reads": reads, "entropy_backend": type(codec.entropy).__name__,
            "arithmetic": codec.profile, "reconstruction_checksums": aggregate.values()})

        trial_rows = []
        rng = np.random.default_rng(args.seed+1)
        representatives = []

        def run_case(label, positions, rate=None, recover=False):
            changed = flip(payload, positions)
            touched, altered = [], {}
            for comp in components:
                a, b = comp["start"], comp["end"]
                if changed[a:b] == payload[a:b]:
                    continue
                index = comp["frame"]
                touched.append(index)
                if index not in altered:
                    altered[index] = {**packets[index], "y": list(packets[index]["y"])}
                if comp["kind"] == "z":
                    altered[index]["z"] = changed[a:b]
                else:
                    altered[index]["y"][comp["part"]] = changed[a:b]
            touched = sorted(set(touched))
            first = touched[0] if touched else None
            row = {"label": label, "ber": rate, "flips": len(positions),
                "bit_positions": [int(v) for v in positions], "touched_frames_1based": [i+1 for i in touched],
                "first_error_frame_1based": first+1 if first is not None else None,
                "clean_prefix_frames": first if first is not None else len(packets),
                "decoded_frames_before_stop": first if first is not None else len(packets),
                "changed_rgb_frames_1based": [], "changed_latent_frames_1based": [],
                "quality_vs_clean_reconstruction": [], "failure_frame_1based": None,
                "failure": None, "decoded_complete": True, "exact_complete": True,
                "cached_clean_prefix_only": True}
            if first is not None:
                codec.dpb = {k: None if v is None else v.to("cuda") for k, v in states[first].items()}
                for index in range(first, len(packets)):
                    try:
                        rgb, result = codec.frame(header, index, packet=altered.get(index, packets[index]))
                    except (ValueError, IndexError) as exc:
                        row.update(failure_frame_1based=index+1, failure=str(exc),
                                   decoded_complete=False, exact_complete=False)
                        break
                    row["decoded_frames_before_stop"] += 1
                    if result["latent_sha256"] != clean_meta[index]["latent_sha256"]:
                        row["changed_latent_frames_1based"].append(index+1)
                        row["exact_complete"] = False
                    if result["rgb_sha256"] != clean_meta[index]["rgb_sha256"]:
                        row["changed_rgb_frames_1based"].append(index+1)
                        row["exact_complete"] = False
                        delta = rgb.astype(np.float64)-clean_rgb[index]
                        mse = float(np.mean(delta**2))
                        row["quality_vs_clean_reconstruction"].append({"frame_1based": index+1,
                            "psnr_db": 10*math.log10(255**2/mse), "mae": float(np.mean(np.abs(delta)))})
                        if len(representatives) < 3:
                            representatives.append((label, index, rgb.copy()))
                    # Once an unchanged state is reached after the last damaged packet,
                    # the remaining clean suffix is provably identical on this setup.
                    if index >= touched[-1] and result["latent_sha256"] == clean_meta[index]["latent_sha256"]:
                        row["cached_clean_suffix_frames"] = len(packets)-index-1
                        row["decoded_frames_before_stop"] = len(packets)
                        break
                if recover:
                    end_damage = touched[-1]
                    next_i = (end_damage//header["intra_period"]+1)*header["intra_period"]
                    recovered = []
                    if next_i < len(packets):
                        codec.dpb = {"ref_latent": None, "ref_y": None}
                        for index in range(next_i, len(packets)):
                            _, result = codec.frame(header, index, packet=packets[index])
                            assert (result["rgb_sha256"], result["latent_sha256"]) == (
                                clean_meta[index]["rgb_sha256"], clean_meta[index]["latent_sha256"])
                            recovered.append(index+1)
                    row["diagnostic_seek_next_clean_I"] = {
                        "supported_by_production_receiver": False,
                        "requires_protected_header_and_frame_boundaries": True,
                        "exact_recovered_frames_1based": recovered}
            trial_rows.append(row)
            with (args.output / "decoder_trials.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            return row

        for rate in RATES:
            for trial in range(args.decoder_trials):
                positions = bit_flips(rng, len(payload)*8, rate)
                run_case(f"iid_{rate:g}_{trial:03}", positions, rate)
            group = [r for r in trial_rows if r["ber"] == rate]
            print(f"PAYLOAD ber={rate:g} complete={sum(r['decoded_complete'] for r in group)}/{len(group)}", flush=True)

        for index in (0, 1, 32):
            for kind in ("z", "y"):
                comp = next(c for c in components if c["frame"] == index and c["kind"] == kind)
                start = comp["start"] + (12 if kind == "y" else 0)
                end = comp["end"]
                if kind == "y":
                    end = start + struct.unpack_from(">I", payload, comp["start"]+4)[0]
                for trial in range(8):
                    bit = int(rng.integers(start*8, end*8))
                    run_case(f"single_f{index+1}_{kind}_{trial}", np.array([bit]), recover=trial==0)
            print(f"SINGLE frame={index+1} complete", flush=True)

        for burst in (8, 32, 128):
            for trial in range(8):
                start = int(rng.integers(0, len(payload)*8-burst+1))
                run_case(f"burst_{burst}_{trial}", np.arange(start, start+burst))
            print(f"BURST bits={burst} complete", flush=True)
        assert reads == []

    if representatives:
        from PIL import Image
        for number, (label, index, rgb) in enumerate(representatives):
            Image.fromarray(np.concatenate([clean_rgb[index], rgb], axis=1)).save(args.output / f"changed_example_{number+1}.png")
        write(args.output / "changed_examples.json", [{"file": f"changed_example_{n+1}.png",
            "case": label, "frame_1based": index+1, "left": "clean reconstruction", "right": "corrupt payload decode"}
            for n, (label, index, rgb) in enumerate(representatives)])
    grouped = []
    for rate in RATES:
        group = [r for r in trial_rows if r["ber"] == rate]
        damaged = [r for r in group if r["flips"]]
        grouped.append({"ber": rate, "trials": len(group), "corrupted_trials": len(damaged),
            "error_free_trials": len(group)-len(damaged), "decoded_complete": sum(r["decoded_complete"] for r in group),
            "exact_complete": sum(r["exact_complete"] for r in group),
            "corrupt_decoded_complete": sum(r["decoded_complete"] for r in damaged),
            "corrupt_exact_complete": sum(r["exact_complete"] for r in damaged),
            "changed_rgb_any": sum(bool(r["changed_rgb_frames_1based"]) for r in group),
            "failure_reasons": dict(Counter(r["failure"] for r in group if r["failure"]))})
    forced = []
    for prefix in ("single_f1_z", "single_f1_y", "single_f2_z", "single_f2_y", "single_f33_z", "single_f33_y",
                   "burst_8_", "burst_32_", "burst_128_"):
        group = [r for r in trial_rows if r["label"].startswith(prefix)]
        forced.append({"case": prefix, "trials": len(group),
            "decoded_complete": sum(r["decoded_complete"] for r in group),
            "exact_complete": sum(r["exact_complete"] for r in group),
            "changed_rgb_any": sum(bool(r["changed_rgb_frames_1based"]) for r in group),
            "failure_reasons": dict(Counter(r["failure"] for r in group if r["failure"]))})
    write(args.output / "decoder_summary.json", {"iid": grouped, "forced": forced,
        "source_media_reads": [], "recovery": [r for r in trial_rows if "diagnostic_seek_next_clean_I" in r]})
    return grouped, forced


def main():
    from c4_config import load_settings
    settings = load_settings()
    base = settings["workspace"] / "02_实验数据/C_编解码加速与网页展示"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, default=base / "GLC_precision_20261026_native_05/fp32/encode/run_1/stream.glcv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file-trials", type=int, default=1000)
    parser.add_argument("--decoder-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.stream = args.stream.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.stream.read_bytes()
    header, packets, integrity = read_stream(args.stream, with_integrity=True)
    assert integrity["container_version"] == 2
    payload, components = payload_layout(raw, packets)
    write(args.output / "manifest.json", {"utc": datetime.now(timezone.utc).isoformat(),
        "source_stream": str(args.stream), "stream_sha256": sha(raw), "stream_bytes": len(raw),
        "payload_bytes": len(payload), "header": header, "seed": args.seed,
        "file_trials_per_rate": args.file_trials, "decoder_trials_per_rate": args.decoder_trials,
        "rates": list(RATES), "script_sha256": sha(Path(__file__).read_bytes()),
        "channel": "Independent residual bit flips (BSC), not physical-channel or TCP simulation",
        "decoder_scope": "Post-parser z/y payload corruption; header, boundaries and outer metadata protected; outer and reconstruction hash checks bypassed only in this harness; native entropy validity checks retained",
        "cache": "Only unchanged prefix and proven identical latent-state suffix reuse; never replace a damaged reference with clean state during propagation",
        "production_changes": False})
    (args.output / "bench_glc_errors.snapshot.py").write_bytes(Path(__file__).read_bytes())
    write(args.output / "payload_layout.json", components)
    strict = strict_trials(args, raw)
    iid, forced = decoder_trials(args, settings, header, packets, integrity, payload, components)
    write(args.output / "summary.json", {"strict": strict, "payload_iid": iid, "forced": forced,
        "stream_bytes": len(raw), "payload_bytes": len(payload), "complete": True})
    print("BER CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
