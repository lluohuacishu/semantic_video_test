"""Official GLC networks with a local byte-stream encoder and independent decoder.

The public upstream test() reports estimated bits only. This adapter adds rANS,
codebook packing and a research container; it is not an official GLC bitstream.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from glc_container import StreamWriter, read_stream, validate_header
from glc_entropy import ENTROPY_ID, GaussianRans, pack_indices, unpack_indices
from glc_video_probe import (ROOT, REVISION, ASSETS, check_source, check_weights,
                             load_weights, extract_frames, make_preview, write_json)

SCHEDULE = [0, 1, 0, 2, 0, 2, 0, 2]


def configure(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the GLC environment.")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "implementation": "glcv-lab-f32-v1", "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "capability": list(torch.cuda.get_device_capability()) if device == "cuda" else None,
        "tf32": False, "deterministic": True,
    }


def reference_hash(dpb):
    digest = hashlib.sha256()
    for key in ("ref_latent", "ref_y"):
        digest.update(key.encode())
        if dpb[key] is not None:
            digest.update(dpb[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rgb_array(tensor, width, height):
    if not torch.isfinite(tensor).all():
        raise ValueError("Neural decoder produced non-finite pixels.")
    rgb = tensor[0, :, :height, :width].clamp(-1, 1).add(1).mul(127.5)
    return rgb.round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


class NeuralCodec:
    def __init__(self, repo, weights, device, decoder_only=False):
        check_source(repo)
        check_weights(weights)
        self.profile = configure(device)
        sys.path.insert(0, str(repo.resolve()))
        from src.models.image_model import GLC_Image
        from src.models.video_model import GLC_Video

        self.entropy, self.device = GaussianRans(), device
        self.image, self.video = GLC_Image(), GLC_Video(hyper_K=16)
        self.loading = {
            "image": load_weights(self.image, weights / "GLC_image.pth.tar", torch),
            "video": load_weights(self.video, weights / "GLC_Video.pth.tar", torch),
        }
        self.image.interpolate_q()
        for model in (self.image, self.video):
            model.eval().to(device)
        self.dpb = {"ref_latent": None, "ref_y": None}
        if decoder_only:
            def reject_source_encoding(*_args, **_kwargs):
                raise RuntimeError("Source-frame encoder called from independent decoder.")
            for module in (self.image.vqgan.encoder, self.image.enc, self.image.hyper_enc,
                           self.video.vqgan.encoder, self.video.encoder, self.video.hyper_tokenizer):
                module.forward = reject_source_encoding

    def codebook(self, model, z_shape, z=None, stream=None):
        if z is not None:
            indexes = model.z_vq.get_indices(z)
            stream = pack_indices(indexes.cpu().numpy())
        else:
            values = unpack_indices(stream, z_shape[1] * z_shape[2])
            indexes = torch.from_numpy(values).to(self.device).reshape(-1, 1)
        # Both ends use a discrete lookup. Upstream P-frame test() uses an STE
        # expression whose floating-point cancellation cannot be reconstructed
        # from the codebook index alone.
        quantized = model.z_vq.get_quan_feat(indexes, z_shape)
        return quantized, stream

    def prior(self, model, params, is_i, y=None, streams=None):
        q_enc, q_dec, scales, means = model.separate_prior(params)
        batch, channels, height, width = means.shape
        masks = (model.get_mask_four_parts if is_i else model.get_mask_dual)(
            batch, channels, height, width, means.dtype, means.device)
        common = model.y_spatial_prior_reduction(params) if is_i else params
        scaled_y = y * q_enc if y is not None else None
        reconstructed, encoded = None, []
        for stage, mask in enumerate(masks):
            if stage:
                context = torch.cat((reconstructed, common), dim=1)
                if is_i:
                    context = getattr(model, f"y_spatial_prior_adaptor_{stage}")(context)
                scales, means = model.y_spatial_prior(context).chunk(2, 1)
            active = mask.bool()
            table_indexes = self.entropy.indexes(scales[active].cpu().numpy())
            if scaled_y is not None:
                symbols = torch.round((scaled_y - means * mask) * mask)[active].cpu().numpy()
                stream = self.entropy.encode(symbols, table_indexes)
            else:
                stream = streams[stage]
                symbols = self.entropy.decode(stream, table_indexes)
            encoded.append(stream)
            # Rebuild from the same integers in both processes, in mask order.
            residual = torch.zeros_like(means)
            residual[active] = torch.from_numpy(symbols).to(device=self.device, dtype=means.dtype)
            part = residual + means * mask
            reconstructed = part if reconstructed is None else reconstructed + part
        return reconstructed * q_dec, encoded

    @torch.inference_mode()
    def frame(self, header, index, x=None, packet=None):
        is_i = index % header["intra_period"] == 0
        q = header["q"]
        hp = (header["height"] + 63) // 64 * 64
        wp = (header["width"] + 63) // 64 * 64
        y, z = None, None
        if is_i:
            model = self.image
            if x is not None:
                y = model.enc(model.vqgan.encoder(x), model.q_enc[q:q + 1])
                z = model.hyper_enc(y)
            z_hat, z_stream = self.codebook(model, (1, hp // 64, wp // 64, 256),
                                           z=z, stream=None if packet is None else packet["z"])
            params = model.y_prior_fusion(model.hyper_dec(z_hat))
            y_hat, y_streams = self.prior(model, params, True, y, None if packet is None else packet["y"])
            latent = model.dec(y_hat, model.q_dec[q:q + 1])
            self.dpb = {"ref_latent": latent, "ref_y": None}
        else:
            model = self.video
            if self.dpb["ref_latent"] is None:
                raise ValueError("P-frame has no decoded reference.")
            if header["reset_interval"] and index % header["reset_interval"] == 1:
                self.dpb["ref_y"] = None
            context = model.context_generation(self.dpb, SCHEDULE[index % 8])
            temporal = model.temporal_prior_encoder(context)
            weight_map = model.weight_generator(temporal)
            if x is not None:
                y = model.encoder(model.vqgan.encoder(x), context, model.y_q_enc[q:q + 1])
                hyper_input = model.y_temporal_combine(y, model.temporal_cond(context))
                hyper_input, _ = model.pad_for_y(hyper_input)
                z = model.hyper_tokenizer(hyper_input, weight_map)
            z_hat, z_stream = self.codebook(model, (1, 16, 1, 256),
                                           z=z, stream=None if packet is None else packet["z"])
            hierarchical = model.hyper_inv_tokenizer(z_hat, weight_map)
            # Input is padded to 64 pixels, so latent dimensions are multiples
            # of four and pad_for_y / slice_to_y have no spatial padding here.
            params = model.res_prior_param_decoder(hierarchical, self.dpb, temporal)
            y_hat, y_streams = self.prior(model, params, False, y, None if packet is None else packet["y"])
            latent = model.decoder(y_hat, context, model.y_q_dec[q:q + 1])
            self.dpb = {"ref_latent": latent, "ref_y": y_hat}
        rgb = rgb_array(model.vqgan.generator(latent), header["width"], header["height"])
        result = {"kind": 0 if is_i else 1, "z": z_stream, "y": y_streams,
                  "latent_sha256": reference_hash(self.dpb),
                  "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
        if packet is not None:
            for key in ("latent_sha256", "rgb_sha256"):
                if result[key] != packet[key]:
                    raise ValueError(f"Frame {index + 1}: {key} differs; decoder arithmetic diverged.")
        return rgb, result


def empty_directory(path):
    path = path.resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Use an empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def encode(args):
    import cv2

    started = time.perf_counter()
    output = empty_directory(args.workdir)
    if args.output.exists() or args.output.with_name(args.output.name + ".part").exists():
        raise FileExistsError(args.output)
    source = extract_frames(args, output, cv2, Image)
    frames_dir = output / "frames"
    frames_dir.mkdir()
    codec = NeuralCodec(args.repo, args.weights, args.device)
    header = {key: source[key] for key in ("width", "height", "fps")}
    header.update({"frames": source["processed_frames"], "q": args.q,
                   "intra_period": args.intra_period, "reset_interval": args.reset_interval,
                   "revision": REVISION, "entropy": ENTROPY_ID,
                   "table_sha256": codec.entropy.table_sha256, "arithmetic": codec.profile,
                   "weights": {key: value["sha256"] for key, value in ASSETS.items()}})
    validate_header(header)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = StreamWriter(args.output, header)
    records = []
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        for index in range(header["frames"]):
            name = f"im{index + 1:05d}.png"
            with Image.open(output / "source" / name) as image:
                source_rgb = np.array(image.convert("RGB"))
            tensor = torch.from_numpy(source_rgb.copy()).to(args.device).permute(2, 0, 1).unsqueeze(0).float()
            # Match upstream's replicated border before normalization.
            tensor = tensor.div(255)
            tensor = F.pad(tensor, (0, (-header["width"]) % 64, 0, (-header["height"]) % 64), mode="replicate")
            tensor = tensor.mul(2).sub(1)
            tick = time.perf_counter()
            rgb, packet = codec.frame(header, index, x=tensor)
            seconds = time.perf_counter() - tick
            frame_bytes = writer.add(packet)
            Image.fromarray(rgb).save(frames_dir / name)
            mse = float(np.mean((source_rgb.astype(np.float64) - rgb) ** 2))
            record = {"frame": index + 1, "type": "I" if packet["kind"] == 0 else "P",
                      "record_bytes": frame_bytes, "z_bytes": len(packet["z"]),
                      "y_bytes": sum(map(len, packet["y"])), "seconds": seconds,
                      "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
                      "rgb_sha256": packet["rgb_sha256"], "latent_sha256": packet["latent_sha256"]}
            records.append(record)
            print(f"encode {index + 1}/{header['frames']} {record['type']} {frame_bytes} bytes {seconds:.2f}s", flush=True)
        writer.finish()
    finally:
        writer.close()
    pipeline_seconds = time.perf_counter() - started
    size = args.output.stat().st_size
    result = {
        "codec": "GLC official pretrained networks + local entropy coding/container v1",
        "official_bitstream_format": False, "actual_bitstream": True, "training_performed": False,
        "input": source, "header": header, "weight_loading": codec.loading,
        "bitstream_path": str(args.output.resolve()), "file_bytes": size,
        "file_sha256": sha256_file(args.output), "actual_bpp": size * 8 / (header["width"] * header["height"] * header["frames"]),
        "actual_video_kbps": size * 8 / source["duration_seconds"] / 1000,
        "mean_psnr_db": float(np.mean([r["psnr_db"] for r in records if r["psnr_db"] is not None])),
        "frame_processing_seconds": sum(r["seconds"] for r in records),
        "pipeline_seconds_before_preview": pipeline_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if args.device == "cuda" else None,
        "container_overhead_bytes": size - sum(r["z_bytes"] + r["y_bytes"] for r in records),
        "audio": "not encoded", "frames": records,
        "input_transform": "RGB / 255; replicate right/bottom to multiples of 64; x * 2 - 1",
    }
    write_json(output / "encode.json", result)
    make_preview(args.ffmpeg, output / "source", output / "source_preview.mp4", header["fps"])
    make_preview(args.ffmpeg, frames_dir, output / "reconstruction_preview.mp4", header["fps"])
    print(f"Encoded real stream: {size} bytes; {result['actual_video_kbps']:.3f} kbps; {args.output}", flush=True)


def decode(args):
    started = time.perf_counter()
    header, packets = read_stream(args.input)
    if header["revision"] != REVISION or header["entropy"] != ENTROPY_ID:
        raise ValueError("Unsupported model or entropy coder version.")
    if header["weights"] != {key: value["sha256"] for key, value in ASSETS.items()}:
        raise ValueError("Stream requires different model weights.")
    profile = configure(args.device)
    if header["arithmetic"] != profile:
        raise ValueError("This experimental stream requires the sender's arithmetic profile (PyTorch/GPU).")
    if header["table_sha256"] != GaussianRans().table_sha256:
        raise ValueError("Entropy tables differ from the sender.")
    output = empty_directory(args.output)
    frames_dir = output / "frames"
    frames_dir.mkdir()
    # Record any source-media reads made by this Python decoder. Preview FFmpeg
    # runs afterwards and reads only the newly decoded output PNGs.
    media_reads = []
    def audit(event, values):
        if event != "open" or not isinstance(values[0], (str, bytes)):
            return
        path, mode, flags = values
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        # Pillow creates output images with w+b; this is not an input read.
        reading = (isinstance(mode, str) and "r" in mode) or (mode is None and not flags & (os.O_WRONLY | os.O_RDWR))
        if reading and Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".avi", ".mp4", ".mkv", ".npz"}:
            media_reads.append(str(Path(path).resolve()))
    sys.addaudithook(audit)
    codec = NeuralCodec(args.repo, args.weights, args.device, decoder_only=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    records = []
    for index, packet in enumerate(packets):
        tick = time.perf_counter()
        rgb, checked = codec.frame(header, index, packet=packet)
        seconds = time.perf_counter() - tick
        Image.fromarray(rgb).save(frames_dir / f"im{index + 1:05d}.png")
        records.append({"frame": index + 1, "seconds": seconds, "rgb_sha256": checked["rgb_sha256"],
                        "latent_sha256": checked["latent_sha256"], "sender_checksums_match": True})
        print(f"decode {index + 1}/{header['frames']} verified {seconds:.2f}s", flush=True)
    if media_reads:
        raise RuntimeError(f"Independent decoder unexpectedly read source media: {media_reads}")
    result = {"input_bitstream": str(args.input.resolve()), "input_sha256": sha256_file(args.input),
              "header": header, "independent_decoder": True, "encoder_modules_disabled": True,
              "source_media_reads": media_reads, "all_frame_checksums_match": True,
              "frame_processing_seconds": sum(r["seconds"] for r in records),
              "pipeline_seconds_before_preview": time.perf_counter() - started,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if args.device == "cuda" else None,
              "frames": records}
    write_json(output / "decode.json", result)
    make_preview(args.ffmpeg, frames_dir, output / "reconstruction_preview.mp4", header["fps"])
    print(f"Independent decoding complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    encoder = sub.add_parser("encode")
    decoder = sub.add_parser("decode")
    for child in (encoder, decoder):
        child.add_argument("--input", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--repo", type=Path, default=ROOT / "tools/GLC")
        child.add_argument("--weights", type=Path, default=ROOT / "checkpoints/glc")
        child.add_argument("--ffmpeg", type=Path, default=ROOT / "tools/ffmpeg/bin/ffmpeg.exe")
        child.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    encoder.add_argument("--workdir", type=Path, required=True)
    encoder.add_argument("--frames", type=int, default=50)
    encoder.add_argument("--start-frame", type=int, default=0)
    encoder.add_argument("--width", type=int, default=1280)
    encoder.add_argument("--q", type=int, default=63)
    encoder.add_argument("--intra-period", type=int, default=32)
    encoder.add_argument("--reset-interval", type=int, default=32)
    args = parser.parse_args()
    if not args.input.is_file() or not args.ffmpeg.is_file():
        parser.error("Input and FFmpeg must exist.")
    if args.command == "encode":
        if not 1 <= args.frames <= 10000 or args.start_frame < 0 or not 0 <= args.width <= 4096:
            parser.error("Invalid frame count, start frame or width.")
        if not 0 <= args.q <= 63 or not 1 <= args.intra_period <= 10000 or not 0 <= args.reset_interval <= 10000:
            parser.error("Invalid q, intra-period or reset-interval.")
        encode(args)
    else:
        decode(args)


if __name__ == "__main__":
    main()
