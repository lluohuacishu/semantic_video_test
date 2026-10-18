"""Build browser-friendly, synchronized viewing copies from verified decoded frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_frame(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size and image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return image.copy()


def label(image: Image.Image, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    tile = Image.new("RGB", (image.width, image.height + 36), "#0c1825")
    tile.paste(image, (0, 36))
    draw = ImageDraw.Draw(tile, "RGBA")
    draw.text((12, 5), text, font=font, fill=(255, 255, 255, 255))
    return tile


def compose(root: Path, index: int, panel_names: list[str], layout: str,
            crop: tuple[int, int, int, int] | None, font: ImageFont.FreeTypeFont) -> Image.Image:
    path = lambda name: root / "source" / f"im{index:05d}.png" if name == "source" else root / name / "receiver" / "frames" / f"im{index:05d}.png"
    if layout == "three":
        panels = []
        for name, title in zip(panel_names, ["源图", "GLC 540p", "GLC 原生 720p"]):
            panel = read_frame(path(name), (1280, 720))
            if crop:
                panel = panel.crop(crop).resize((640, 360), Image.Resampling.NEAREST)
            else:
                panel = panel.resize((640, 360), Image.Resampling.LANCZOS)
            panels.append(label(panel, title, font))
        output = Image.new("RGB", (1920, 396), "#111a24")
        for x, panel in zip((0, 640, 1280), panels):
            output.paste(panel, (x, 0))
        return output

    output = Image.new("RGB", (1280, 792), "#111a24")
    tile_positions = [(0, 0), (640, 0), (0, 396), (640, 396)]
    titles = {"source": "源图", "glc_540_g64": "GLC 540p · 108.5 KB",
              "h265_540_for_540_g64": "H.265 540p · 109.5 KB",
              "av1_540_for_540_g64": "AV1 540p · 109.6 KB"}
    for name, (x, y) in zip(panel_names, tile_positions):
        # Crop in the same 1280x720 coordinates for EVERY reconstruction.
        image = read_frame(path(name), (1280, 720))
        if crop:
            image = image.crop(crop).resize((640, 360), Image.Resampling.NEAREST)
        else:
            image = image.resize((640, 360), Image.Resampling.LANCZOS)
        output.paste(label(image, titles.get(name, name), font), (x, y))
    return output


def make_movie(ffmpeg: Path, root: Path, output: Path, frames: int, fps: int,
               names: list[str], layout: str, crop: tuple[int, int, int, int] | None,
               quality: int, start_index: int = 0, speed: float = 1.0) -> dict:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 19)
    sample = compose(root, start_index + 1, names, layout, crop, font)
    width, height = sample.size
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps * speed), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
               "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p",
               "-crf", str(quality), "-pix_fmt", "yuv420p", "-profile:v", "high",
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
               "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for index in range(start_index + 1, start_index + frames + 1):
            image = compose(root, index, names, layout, crop, font)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        process.stdin = None
        code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if code:
        raise RuntimeError(f"FFmpeg exited with status {code}: {output}")
    with output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    result = {"path": str(output.resolve()), "frames": frames, "fps": fps,
              "duration_seconds": frames / (fps * speed), "size": [width, height],
              "source_index_start": start_index, "playback_speed": speed,
              "view_copy_codec": "H.264 / yuv420p", "view_copy_crf": quality,
              "source_frame_sequences": names, "fixed_crop_xyxy": crop,
              "purpose": "synchronized viewing copy; experiment metrics are from PNG reconstructions, not this preview encode",
              "bytes": output.stat().st_size, "sha256": digest, "command": command}
    output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="long91 experiment directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--crf", type=int, default=11)
    args = parser.parse_args()

    results = []
    results.append(make_movie(args.ffmpeg, args.run, args.output / "video91_同预算四路全画面.mp4",
                              args.frames, args.fps,
                              ["source", "glc_540_g64", "h265_540_for_540_g64", "av1_540_for_540_g64"],
                              "grid", None, args.crf))
    results.append(make_movie(args.ffmpeg, args.run, args.output / "video91_末段鱼纹_四路半速.mp4",
                              40, args.fps,
                              ["source", "glc_540_g64", "h265_540_for_540_g64", "av1_540_for_540_g64"],
                              "grid", (256, 576, 512, 720), args.crf, start_index=210, speed=0.5))
    results.append(make_movie(args.ffmpeg, args.run, args.output / "video91_GLC分辨率三路对照.mp4",
                              args.frames, args.fps, ["source", "glc_540_g64", "glc_720_g64"],
                              "three", None, args.crf))
    results.append(make_movie(args.ffmpeg, args.run, args.output / "video91_末段鱼纹_GLC分辨率_半速.mp4",
                              40, args.fps, ["source", "glc_540_g64", "glc_720_g64"],
                              "three", (256, 576, 512, 720), args.crf, start_index=210, speed=0.5))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "viewing_videos.json").write_text(
        json.dumps({"note": "观看副本经过统一 H.264 预览编码；真实压缩效果与指标均来自它们所显示的独立解码 PNG。",
                    "videos": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"videos": [{"path": r["path"], "bytes": r["bytes"],
                                   "duration_seconds": r["duration_seconds"], "size": r["size"]}
                                  for r in results]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
