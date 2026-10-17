"""按固定帧间隔抽取 PNG，保留源目录和扩展名以区分同名视频。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
from tqdm import tqdm

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".webm", ".m4v", ".mpeg", ".mpg"}


def extract(video: Path, target: Path, stride: int, max_frames: int | None) -> int:
    if target.exists():
        raise FileExistsError(f"输出目录已存在，请换一个目录以免混入旧帧：{target}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开视频：{video}")
    index, saved = 0, 0
    reported_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    total = int(reported_total) if math.isfinite(reported_total) and reported_total > 0 else None
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        target.mkdir(parents=True, exist_ok=False)
        with tqdm(total=total, desc=video.name, unit="frame") as progress:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                progress.update(1)
                if index % stride == 0:
                    output = target / f"{index:08d}.png"
                    ok, encoded = cv2.imencode(".png", frame)
                    if not ok:
                        raise RuntimeError(f"PNG 编码失败：{output}")
                    # numpy.tofile 支持 Windows 中文路径。
                    encoded.tofile(output)
                    saved += 1
                    if max_frames is not None and saved >= max_frames:
                        break
                index += 1
    finally:
        capture.release()
    if not saved:
        raise RuntimeError(f"视频没有读到可解码帧：{video}")
    report = {"source": str(video.resolve()), "stride": stride, "saved_frames": saved,
              "max_frames_per_video": max_frames, "source_reported_frames": total,
              "source_fps": fps if math.isfinite(fps) and fps > 0 else None}
    (target / "extraction.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path)
    source.add_argument("--input", type=Path, help="只抽取一个视频")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, help="每个视频最多保存的帧数")
    args = parser.parse_args()
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("stride 和 max-frames 必须大于 0")
    if args.input:
        videos = [args.input.resolve()] if args.input.is_file() and args.input.suffix.lower() in VIDEO_EXTENSIONS else []
        root = args.input.resolve().parent
    else:
        root = args.input_dir.resolve()
        videos = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        parser.error("没有找到支持的视频，请检查输入路径")
    targets = [args.output_dir / video.relative_to(root) for video in videos]
    for target in targets:
        if target.exists():
            parser.error(f"输出目录已存在，请使用新目录：{target}")
    total = sum(extract(video, target, args.stride, args.max_frames) for video, target in zip(videos, targets))
    print(f"完成：{len(videos)} 个视频，保存 {total} 帧到 {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
