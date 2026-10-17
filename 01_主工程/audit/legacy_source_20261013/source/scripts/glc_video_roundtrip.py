"""Encode, pace a real TCP file transfer, then decode in a fresh process."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageFont

from file_transfer import loopback_transfer
from glc_video_probe import ROOT, write_json


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Progress:
    """Optional atomic status file for the local web console."""
    def __init__(self, path=None):
        self.path = path
        self.last_transfer = 0.0

    def update(self, stage, phase, progress, **fields):
        if self.path is None:
            return
        temporary = self.path.with_suffix(".tmp")
        write_json(temporary, {"stage": stage, "phase": phase, "progress": progress, **fields})
        temporary.replace(self.path)

    def line(self, line):
        match = re.match(r"^(encode|decode) (\d+)/(\d+) ", line)
        if not match:
            return
        name, done, total = match.group(1), int(match.group(2)), int(match.group(3))
        if name == "encode":
            phase = f"编码 {done}/{total} 帧" if done < total else "编码完成，正在保存预览"
            self.update("encoding", phase, 2 + 50 * done / total, processed_frames=done, total_frames=total)
        else:
            phase = f"解码 {done}/{total} 帧" if done < total else "解码完成，正在保存预览"
            self.update("decoding", phase, 72 + 24 * done / total, processed_frames=done, total_frames=total)

    def transfer(self, sent, total, elapsed):
        tick = time.monotonic()
        if sent != total and tick - self.last_transfer < 0.25:
            return
        self.last_transfer = tick
        self.update("transfer", "正在限速传输压缩文件", 54 + 16 * sent / total,
                    transferred_bytes=sent, transfer_total_bytes=total, transfer_elapsed_seconds=elapsed)


def run_process(command, cwd, log_path, on_line=None):
    with log_path.open("x", encoding="utf-8") as log:
        environment = dict(os.environ, PYTHONIOENCODING="utf-8")
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line.rstrip(), flush=True)
                log.write(line)
                log.flush()
                if on_line:
                    on_line(line)
            if process.wait():
                raise RuntimeError(f"GLC subprocess failed; see {log_path}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def comparison(folder, ffmpeg, info):
    index = min(34, info["processed_frames"] - 1)
    panels = [folder / "sender/source", folder / "receiver/frames"]
    width, height = info["width"], info["height"]
    canvas = Image.new("RGB", (width * 2, height + 40), "#18242e")
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for column, (directory, label) in enumerate(zip(panels, ["Input", "Decoded from .glcv"])):
        with Image.open(directory / f"im{index + 1:05d}.png") as frame:
            canvas.paste(frame.convert("RGB"), (column * width, 40))
        draw.text((column * width + 12, 6), label, font=font, fill="white")
    canvas.save(folder / "comparison.png")
    subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-i", str(folder / "sender/source_preview.mp4"),
                    "-i", str(folder / "receiver/reconstruction_preview.mp4"),
                    "-filter_complex", "[0:v]scale=640:-2[a];[1:v]scale=640:-2[b];[a][b]hstack=inputs=2[v]",
                    "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(folder / "comparison_preview.mp4")], check=True)


def report(output, results):
    rows, sections = [], []
    for result in results:
        enc, dec, transfer = result["encode"], result["decode"], result["transfer"]
        q, folder = enc["header"]["q"], result["folder"]
        rows.append(f"<tr><td>{q}</td><td>{enc['file_bytes']:,} B</td><td>{enc['actual_video_kbps']:.2f}</td>"
                    f"<td>{enc['mean_psnr_db']:.2f}</td><td>{transfer['elapsed_seconds']:.2f} s</td>"
                    f"<td>{enc['frame_processing_seconds']:.1f} / {dec['frame_processing_seconds']:.1f} s</td></tr>")
        sections.append(f"<section><h2>q={q}</h2><p>左：输入；右：接收端独立解码。MP4 仅用于播放，传输文件是 .glcv。</p>"
                        f'<video controls preload="metadata" src="{folder}/comparison_preview.mp4"></video>'
                        f'<p><a href="{folder}/sender.glcv">压缩文件</a> · '
                        f'<a href="{folder}/receiver/reconstruction_preview.mp4">接收端视频</a> · '
                        f'<a href="{folder}/verification.json">校验记录</a> · '
                        f'<a href="{folder}/sender/encode.json">编码数据</a> · '
                        f'<a href="{folder}/receiver/decode.json">解码数据</a></p></section>')
    source = results[0]["encode"]["input"]
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GLC 视频文件传输实验</title>'
            '<style>body{font:16px/1.7 system-ui;margin:35px auto;padding:0 20px;max-width:1200px;color:#18242e;background:#f5f7f9}'
            'table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #cad2da}'
            'video{width:100%;background:#18242e}section{margin:30px 0}a{color:#175d93}</style>'
            '<h1>GLC 视频编码 → 文件传输 → 独立解码</h1>'
            '<p>使用官方预训练网络和权重。本项目补充了熵编码与 .glcv 文件格式；这不是官方码流格式，也不是完整语义通信系统。</p>'
            f"<p>素材：{html.escape(Path(source['input_path']).name)}；{source['width']} × {source['height']}，"
            f"{source['processed_frames']} 帧，{source['fps']:g} FPS，{source['duration_seconds']:g} 秒；无音轨。</p>"
            '<table><tr><th>q</th><th>实际文件大小</th><th>视频码率 kbps</th><th>PSNR dB</th><th>传输耗时</th><th>编码 / 解码处理耗时</th></tr>'
            + ''.join(rows) + '</table>'
            f"<p>TCP 本机实传限速 {results[0]['transfer']['limit_kbps']:g} kbps，包含应用层传输头，不含 TCP/IP 包头和反向确认。"
            '视频码率按完整 .glcv 文件大小计算，含 I/P 帧、码本索引、帧头和校验。处理耗时包括神经网络、熵编码和逐帧校验，不含模型加载、PNG 保存和预览视频制作。</p>'
            '<p>每帧接收结果与发送端重建结果逐像素一致；接收进程没有读取原始图片或视频。两端预先共享模型权重，权重不计入每次文件大小。'
            '目前仅验证同机、相同 PyTorch / GPU 环境，不保证跨硬件码流兼容。</p>'
            + ''.join(sections) + '<p><a href="summary.json">完整实验数据</a> · <a href="https://github.com/jzyustc/GLC">官方仓库</a></p></html>')
    (output / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--q", type=int, nargs="+", default=[63])
    parser.add_argument("--kbps", type=float, default=20)
    parser.add_argument("--intra-period", type=int, default=32)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--progress-file", type=Path, help="Optional status JSON for the local web console.")
    args = parser.parse_args()
    if (not args.input.is_file() or len(set(args.q)) != len(args.q) or any(not 0 <= q <= 63 for q in args.q)
            or not 1 <= args.kbps <= 100000 or not 1 <= args.frames <= 10000
            or args.start_frame < 0 or not 0 <= args.width <= 4096
            or not 1 <= args.intra_period <= 10000 or not 0 <= args.reset_interval <= 10000):
        parser.error("Invalid input, rate, frame range, q indexes or coding periods.")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Use an empty output directory; previous runs are preserved.")
    output.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.progress_file)
    codec = ROOT / "scripts/glc_video_codec.py"
    ffmpeg = ROOT / "tools/ffmpeg/bin/ffmpeg.exe"
    results = []
    for q in args.q:
        folder = output / f"q{q:02d}"
        folder.mkdir()
        command = [sys.executable, "-u", str(codec), "encode", "--input", str(args.input.resolve()),
                   "--output", str(folder / "sender.glcv"), "--workdir", str(folder / "sender"),
                   "--frames", str(args.frames), "--width", str(args.width), "--q", str(q),
                   "--start-frame", str(args.start_frame), "--intra-period", str(args.intra_period),
                   "--reset-interval", str(args.reset_interval), "--device", args.device]
        progress.update("encoding", "抽取视频片段并加载 GLC 模型", 1, processed_frames=0, total_frames=args.frames)
        run_process(command, ROOT, folder / "encode.log", progress.line)
        print(f"Sending q={q} over loopback TCP at {args.kbps:g} kbps...", flush=True)
        progress.update("transfer", "开始传输压缩文件", 54, transferred_bytes=0,
                        transfer_total_bytes=(folder / "sender.glcv").stat().st_size)
        transfer = loopback_transfer(folder / "sender.glcv", folder / "received.glcv", args.kbps,
                                     trace=folder / "transfer.csv", progress=progress.transfer)
        write_json(folder / "transfer.json", transfer)
        # A separate working directory contains no source frames or video.
        decoder_cwd = folder / "decoder_workdir"
        decoder_cwd.mkdir()
        command = [sys.executable, "-u", str(codec), "decode", "--input", str(folder / "received.glcv"),
                   "--output", str(folder / "receiver"), "--device", args.device]
        progress.update("decoding", "文件接收完成，加载独立解码器", 71, processed_frames=0)
        run_process(command, decoder_cwd, folder / "decode.log", progress.line)
        enc = json.loads((folder / "sender/encode.json").read_text(encoding="utf-8"))
        dec = json.loads((folder / "receiver/decode.json").read_text(encoding="utf-8"))
        expected = [f"im{i + 1:05d}.png" for i in range(enc["header"]["frames"])]
        checks = []
        progress.update("verifying", "逐帧核对接收结果", 97)
        for name in expected:
            with Image.open(folder / "sender/frames" / name) as sent, Image.open(folder / "receiver/frames" / name) as received:
                match = sent.size == received.size and sent.convert("RGB").tobytes() == received.convert("RGB").tobytes()
            if not match:
                raise ValueError(f"Independent frame reconstruction differs: {name}")
            checks.append({"file": name, "pixels_identical": True})
        same_stream = digest(folder / "sender.glcv") == digest(folder / "received.glcv")
        if not same_stream or dec["source_media_reads"] or not dec["all_frame_checksums_match"]:
            raise ValueError("Round-trip verification failed.")
        verification = {"separate_encoder_decoder_processes": True, "decoder_workdir_initially_empty": True,
                        "sent_received_files_identical": same_stream, "source_media_reads_in_decoder": [],
                        "all_rgb_frames_identical": True, "all_reference_latents_identical": True,
                        "frames": checks}
        write_json(folder / "verification.json", verification)
        progress.update("verifying", "核对通过，正在生成对比视频与报告", 99)
        comparison(folder, ffmpeg, enc["input"])
        results.append({"folder": folder.name, "encode": enc, "decode": dec,
                        "transfer": transfer, "verification": verification})
        write_json(output / "summary.json", {"actual_bitstream": True, "official_format": False, "results": results})
        report(output, results)
        progress.update("done", "处理完成", 100, processed_frames=len(checks), total_frames=len(checks))
        print(f"q={q}: all {len(checks)} frames verified; TCP {transfer['elapsed_seconds']:.2f}s.", flush=True)
    print(f"Report: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
