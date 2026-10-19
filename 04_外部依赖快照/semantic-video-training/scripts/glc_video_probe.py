"""Run the official GLC networks and record reconstruction / estimated rate.

Upstream 126db25 does not implement a byte-stream encoder or standalone decoder.
The MP4 files made here are viewing copies, not GLC compressed bitstreams.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REVISION = "126db25f9c093508cd0c99cee32b53fd60074f9a"
ASSETS = {
    "GLC_image.pth.tar": {
        "url": "https://github.com/jzyustc/GLC/releases/download/GLC-Image/GLC_image.pth.tar",
        "sha256": "139c1d49a56b5c14b0fbd7e0c43b53a4f0b093d35aa11c79ed9d0b9bfede4c94",
        "bytes": 667371692,
    },
    "GLC_Video.pth.tar": {
        "url": "https://github.com/jzyustc/GLC/releases/download/GLC-Video/GLC_Video.pth.tar",
        "sha256": "c55a46f33b92945dcdc24bbbc023909368de78fce7830d3137e4d2ef7249ee62",
        "bytes": 502245134,
    },
}


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def check_source(repo):
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected official revision {REVISION}, found {revision}")
    if subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--"], text=True):
        raise ValueError("The official GLC checkout has tracked modifications.")
    return revision


def check_weights(directory):
    records = {}
    for name, spec in ASSETS.items():
        path = directory / name
        if path.stat().st_size != spec["bytes"]:
            raise ValueError(f"Incomplete checkpoint: {path}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != spec["sha256"]:
            raise ValueError(f"Official release checksum mismatch: {path}")
        records[name] = dict(spec)
    return records


def load_weights(model, path, torch):
    weights = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "net", "net_g"):
        if key in weights:
            weights = weights[key]
    weights = {k.removeprefix("module."): v for k, v in weights.items()}
    # Fail if any inference parameter would otherwise remain randomly initialized.
    mismatch = model.load_state_dict(weights, strict=False)
    # The released P-frame checkpoint retains an unused older z entropy estimator.
    # Upstream also loads this model with strict=False. All live parameters must match.
    allowed_unused = {
        f"bit_estimator_z.f{stage}.{field}"
        for stage in range(1, 5)
        for field in (("h", "b", "a") if stage < 4 else ("h", "b"))
    } if type(model).__name__ == "GLC_Video" else set()
    if mismatch.missing_keys or set(mismatch.unexpected_keys) - allowed_unused:
        raise ValueError(f"Checkpoint does not exactly match {type(model).__name__}: {mismatch}")
    return {"missing_keys": [], "ignored_unused_checkpoint_keys": mismatch.unexpected_keys}


def extract_frames(args, output, cv2, Image):
    source = output / "source"
    source.mkdir()
    frame_stride = getattr(args, "frame_stride", 1)
    if type(frame_stride) is not int or frame_stride < 1:
        raise ValueError("Frame stride must be a positive integer.")
    reader = cv2.VideoCapture(str(args.input))
    if not reader.isOpened():
        raise ValueError(f"Cannot open video: {args.input}")
    try:
        fps = reader.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Input has no usable nominal frame rate.")
        reported = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(args.start_frame):
            if not reader.grab():
                raise ValueError("start-frame lies beyond the video.")
        count = 0
        source_indices = []
        original_size = None
        while count < args.frames:
            ok, frame = reader.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if original_size is None:
                original_size = (w, h)
                out_w = min(w, args.width) if args.width else w
                out_h = max(1, round(h * out_w / w))
            if (w, h) != original_size:
                raise ValueError("Changing frame dimensions are not supported.")
            if (w, h) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
                source / f"im{count + 1:05d}.png"
            )
            source_indices.append(args.start_frame + count * frame_stride)
            count += 1
            if count < args.frames:
                for _ in range(frame_stride - 1):
                    if not reader.grab():
                        break
    finally:
        reader.release()
    if count == 0:
        raise ValueError("No frames were read.")
    return {
        "input_path": str(args.input.resolve()), "input_bytes": args.input.stat().st_size,
        "source_width": original_size[0], "source_height": original_size[1],
        "width": out_w, "height": out_h, "fps": fps / frame_stride,
        "reported_source_frames": reported, "processed_frames": count,
        "start_frame": args.start_frame, "frame_stride": frame_stride,
        "sampled_source_frame_indices": source_indices,
        "duration_seconds": count / (fps / frame_stride),
        "audio": "not processed", "timebase": "nominal constant frame rate",
    }


def make_preview(ffmpeg, folder, output, fps):
    subprocess.run(
        [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
         "-framerate", str(fps), "-start_number", "1",
         "-i", str(folder / "im%05d.png"), "-an",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         "-c:v", "libx264", "-preset", "fast", "-crf", "16",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(output)],
        check=True,
    )


def make_comparison(output, summary, ffmpeg):
    """Make a labeled viewing copy; never count it as the GLC payload."""
    from PIL import Image, ImageDraw, ImageFont

    info = summary["source"]
    panels = [(output / "source", "Input")]
    panels += [
        (output / f"q{result['q_index_i']:02d}" / "frames", f"GLC q={result['q_index_i']}")
        for result in summary["results"][:2]
    ]
    tile_width = min(640, info["width"])
    tile_height = round(info["height"] * tile_width / info["width"])
    width, height = tile_width * len(panels), tile_height + 40
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22) if Path("C:/Windows/Fonts/arial.ttf").is_file() else ImageFont.load_default()
    process = subprocess.Popen(
        [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
         "-framerate", str(info["fps"]), "-i", "pipe:0", "-an",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         "-c:v", "libx264", "-preset", "fast", "-crf", "16",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(output / "comparison_preview.mp4")],
        stdin=subprocess.PIPE,
    )
    try:
        for index in range(info["processed_frames"]):
            canvas = Image.new("RGB", (width, height), "#18242e")
            draw = ImageDraw.Draw(canvas)
            for column, (folder, label) in enumerate(panels):
                with Image.open(folder / f"im{index + 1:05d}.png") as frame:
                    canvas.paste(frame.convert("RGB").resize((tile_width, tile_height), Image.Resampling.LANCZOS), (column * tile_width, 40))
                draw.text((column * tile_width + 12, 6), label, font=font, fill="white")
            if index == min(34, info["processed_frames"] - 1):
                canvas.save(output / "comparison.png")
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"ffmpeg comparison export failed ({returncode})")


def run_quality(q, args, output, source_info, models, libraries):
    torch, np, Image, F = libraries
    image_model, video_model = models
    device = next(image_model.parameters()).device
    gpu = device.type == "cuda"
    width, height = source_info["width"], source_info["height"]
    frames = source_info["processed_frames"]
    folder = output / f"q{q:02d}"
    folder.mkdir()
    reconstruction = folder / "frames"
    reconstruction.mkdir()
    padding = image_model.get_padding_size(height, width, 64)
    schedule = [0, 1, 0, 2, 0, 2, 0, 2]
    records = []
    dpb = None
    if gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
    with torch.inference_mode():
        for index in range(frames):
            rgb = np.asarray(Image.open(output / "source" / f"im{index + 1:05d}.png").convert("RGB"))
            x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255
            if gpu:
                torch.cuda.synchronize()
            start = time.perf_counter()
            padded = F.pad(x, padding, mode="replicate") * 2 - 1
            is_intra = index % args.intra_period == 0
            if is_intra:
                encoded = image_model.test(padded, q)
                dpb = {"ref_frame": encoded["x_hat"], "ref_latent": encoded["ref_latent"], "ref_y": None}
                restored = encoded["x_hat"]
            else:
                if args.reset_interval > 0 and index % args.reset_interval == 1:
                    dpb["ref_y"] = None
                encoded = video_model.test(padded, dpb, q, fa_idx=schedule[index % 8])
                dpb = encoded["dpb"]
                restored = dpb["ref_frame"]
            restored = (restored.clamp(-1, 1) + 1) / 2
            restored = restored[:, :, :height, :width]
            if gpu:
                torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - start
            estimated_bits = float(encoded["bit"])
            if not math.isfinite(estimated_bits) or estimated_bits < 0:
                raise ValueError(f"Invalid rate estimate at frame {index}: {estimated_bits}")
            if not torch.isfinite(restored).all():
                raise ValueError(f"Non-finite reconstruction at frame {index}")
            mse = float((restored - x).square().mean())
            psnr = -10 * math.log10(mse) if mse > 0 else None
            psnr_display = f"{psnr:.2f}" if psnr is not None else "inf"
            restored_u8 = restored[0].permute(1, 2, 0).mul(255).round().clamp(0, 255).byte().cpu().numpy()
            Image.fromarray(restored_u8).save(reconstruction / f"im{index + 1:05d}.png")
            row = {
                "frame_index": index, "frame_type": "I" if is_intra else "P",
                "estimated_bits": estimated_bits,
                "estimated_y_bits": float(encoded["bit_y"]),
                "estimated_z_bits": float(encoded["bit_z"]),
                "psnr_db_float_reconstruction": psnr, "forward_seconds": forward_seconds,
            }
            records.append(row)
            print(
                f"q={q:02d} frame={index+1}/{frames} {row['frame_type']} "
                f"estimated_bits={estimated_bits:.1f} PSNR={psnr_display} "
                f"forward={forward_seconds:.3f}s", flush=True,
            )
            del x, padded, encoded, restored, rgb, restored_u8
    pipeline_seconds = time.perf_counter() - total_start
    estimated_bits = sum(row["estimated_bits"] for row in records)
    forward_seconds = sum(row["forward_seconds"] for row in records)
    psnrs = [row["psnr_db_float_reconstruction"] for row in records]
    result = {
        "q_index_i": q, "q_index_p": q,
        "rate_measurement": "upstream probability-based estimate; no actual entropy-coded file",
        "actual_glc_file_bytes": None, "standalone_bitstream_decode_verified": False,
        "container_and_protocol_overhead_included": False,
        "estimated_total_bits": estimated_bits,
        "estimated_bpp": estimated_bits / (frames * width * height),
        "estimated_video_kbps": estimated_bits / source_info["duration_seconds"] / 1000,
        "mean_psnr_db_float_reconstruction": sum(psnrs) / frames if all(p is not None for p in psnrs) else None,
        "exact_reconstruction_frames": sum(p is None for p in psnrs),
        "forward_seconds": forward_seconds, "forward_fps": frames / forward_seconds,
        "pipeline_seconds_without_mp4": pipeline_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if gpu else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if gpu else None,
        "precision": "float32", "padding_lrtb": list(padding),
        "intra_period": args.intra_period, "reset_interval": args.reset_interval,
        "rate_gop_size": 8, "frames": records,
    }
    write_json(folder / "estimate.json", result)
    make_preview(args.ffmpeg, reconstruction, folder / "reconstruction_preview.mp4", source_info["fps"])
    return result


def make_report(output, summary):
    info = summary["source"]
    rows, cards = [], []
    for result in summary["results"]:
        q = result["q_index_i"]
        memory = result["peak_allocated_mib"]
        memory_display = f"{memory:.0f}" if memory is not None else "CPU"
        psnr = result["mean_psnr_db_float_reconstruction"]
        psnr_display = f"{psnr:.2f}" if psnr is not None else "∞"
        rows.append(
            f"<tr><td>q={q}</td><td>{result['estimated_bpp']:.5f}</td>"
            f"<td>{result['estimated_video_kbps']:.2f}</td>"
            f"<td>{psnr_display}</td>"
            f"<td>{result['forward_fps']:.2f}</td>"
            f"<td>{memory_display}</td></tr>"
        )
        cards.append(
            f'<article><h2>GLC-Video · q={q}</h2><video controls muted preload="metadata" '
            f'src="q{q:02d}/reconstruction_preview.mp4"></video>'
            f'<p><a href="q{q:02d}/estimate.json">逐帧测量记录</a></p></article>'
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLC-Video 本机试跑</title>
<style>body{font:16px/1.7 system-ui,sans-serif;margin:32px auto;max-width:1240px;padding:0 24px;color:#24313b;background:#f4f6f8}
h1{font-size:28px;margin-bottom:4px}h2{font-size:18px}article{background:white;padding:18px;border-radius:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}video{width:100%;background:#111}
.note{background:#fff3d9;padding:16px;border-left:4px solid #b97912}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}
td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}button{padding:9px 16px;margin:14px 6px 14px 0;cursor:pointer}a{color:#175891}</style>
<h1>GLC-Video 本机试跑</h1>"""
    page += (
        f"<p>{html.escape(Path(info['input_path']).name)} · {info['width']} × {info['height']} · "
        f"{info['processed_frames']} 帧 · {info['fps']:g} FPS · {info['duration_seconds']:.2f} 秒</p>"
        '<p class="note"><b>本页验证官方模型的前向重建能力。</b>'
        "当前官方源码仅估计码量，没有实际熵编码文件和独立码流解码器。"
        "下方 MP4 是重建帧另存的 H.264 观看副本，大小不能作为 GLC 压缩率。"
        "这里也没有测量语义分割准确率或实际传输耗时。</p>"
        '<button onclick="playAll()">从头同步播放</button><button onclick="pauseAll()">全部暂停</button>'
        '<section class="grid"><article><h2>输入片段</h2>'
        '<video controls muted preload="metadata" src="source_preview.mp4"></video></article>'
        + "".join(cards) + "</section>"
        '<p><a href="comparison_preview.mp4">打开并排对比视频</a> · '
        '<a href="comparison.png">查看一帧并排对比图</a></p>'
        "<table><tr><th>参数</th><th>估计 bpp</th><th>估计视频码率 kbps</th>"
        "<th>PSNR dB</th><th>模型前向 FPS</th><th>PyTorch 峰值显存 MiB</th></tr>"
        + "".join(rows) + "</table>"
        "<p>模型前向时间包括编码网络和重建网络，不含加载权重、逐帧读写和 MP4 封装；"
        "这是离线处理速度。显存为 PyTorch 分配峰值，不是整块显卡总占用。"
        "PSNR 在裁去补边后的浮点重建画面上计算。</p>"
        '<p><a href="summary.json">实验配置与完整记录</a> · '
        '<a href="https://github.com/jzyustc/GLC">官方源码</a></p>'
        "<script>const clips=[...document.querySelectorAll('video')];"
        "function pauseAll(){clips.forEach(v=>v.pause())}"
        "function playAll(){pauseAll();clips.forEach(v=>v.currentTime=0);"
        "Promise.allSettled(clips.map(v=>v.play()))}</script></html>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT / "tools/GLC")
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/glc")
    parser.add_argument("--ffmpeg", type=Path, default=ROOT / "tools/ffmpeg/bin/ffmpeg.exe")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280, help="Maximum width; 0 keeps input size.")
    parser.add_argument("--q", type=int, nargs="+", default=[0, 63])
    parser.add_argument("--intra-period", type=int, default=32)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.frames <= 0 or args.start_frame < 0 or args.width < 0 or args.intra_period <= 0:
        parser.error("frames and intra-period must be positive; start-frame and width cannot be negative.")
    if len(set(args.q)) != len(args.q) or any(q < 0 or q > 63 for q in args.q):
        parser.error("q indexes must be unique integers between 0 and 63.")
    if not args.input.is_file() or not args.ffmpeg.is_file():
        parser.error("input video and ffmpeg must exist.")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory; existing experiment files are preserved.")
    revision = check_source(args.repo)
    checkpoints = check_weights(args.weights)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo.resolve()))
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from src.models.image_model import GLC_Image
    from src.models.video_model import GLC_Video
    from src.utils.test_utils import set_torch_env

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")
    set_torch_env()
    source_info = extract_frames(args, output, cv2, Image)
    make_preview(args.ffmpeg, output / "source", output / "source_preview.mp4", source_info["fps"])
    print(f"Loading official models; device={args.device}", flush=True)
    image_model, video_model = GLC_Image(), GLC_Video(hyper_K=16)
    loading = {
        "image": load_weights(image_model, args.weights / "GLC_image.pth.tar", torch),
        "video": load_weights(video_model, args.weights / "GLC_Video.pth.tar", torch),
    }
    image_model.interpolate_q()
    image_model.eval().to(args.device)
    video_model.eval().to(args.device)
    gc.collect()
    summary = {
        "upstream_repository": "https://github.com/jzyustc/GLC",
        "upstream_revision": revision, "checkpoints": checkpoints, "weight_loading": loading,
        "runner": "local video I/O wrapper calling unmodified official model.test methods",
        "actual_bitstream_supported": False, "training_performed": False,
        "perceptual_metrics": "LPIPS and DISTS not run; no auxiliary metric networks loaded",
        "source": source_info,
        "environment": {
            "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        }, "results": [],
    }
    write_json(output / "summary.json", summary)
    for q in args.q:
        result = run_quality(q, args, output, source_info, (image_model, video_model), (torch, np, Image, F))
        summary["results"].append(result)
        write_json(output / "summary.json", summary)
    make_comparison(output, summary, args.ffmpeg)
    make_report(output, summary)
    print(f"Finished: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
