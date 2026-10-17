"""Re-score existing complete files and encode HEVC against identical RGB frames."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from common import ROOT, FFMPEG, FFPROBE, configure, metrics, save_json, sha256

LEGACY = Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-training\runs\glc_stream_720p_final_20261002")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crfs", nargs="+", type=int, default=[20, 28, 36, 44])
    parser.add_argument("--fps", default="5/1")
    parser.add_argument("--mlvc-runs", nargs="*", default=[])
    parser.add_argument("--append", action="store_true", help="Add missing rate points, retain prior results")
    args = parser.parse_args()
    configure()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.append)
    previous = json.loads((output / "comparison.json").read_text("utf-8")) if args.append else None
    rows = previous["rows"] if previous else []
    count = len(list(Path(args.reference).glob("*.png")))
    fps_a, fps_b = map(int, args.fps.split("/"))
    duration = count*fps_b/fps_a
    if previous and (previous["frames"] != count or previous["fps"] != args.fps or previous["source_reference"] != str(Path(args.reference).resolve())):
        raise ValueError("Append run changed source profile")
    for folder in ["q00", "q63"]:
        if any(r["codec"] == "GLC" and r["setting"] == folder for r in rows):
            continue
        data = json.loads((LEGACY / folder / "sender/encode.json").read_text("utf-8"))
        if data["input"]["processed_frames"] != count:
            raise ValueError("Legacy frame count differs")
        file = LEGACY / folder / "sender.glcv"
        if sha256(file) != data["file_sha256"]:
            raise ValueError("Legacy file hash changed")
        result = metrics(args.reference, LEGACY / folder / "receiver/frames")
        save_json(output / f"glc_{folder}_metrics.json", result)
        rows.append({"codec": "GLC", "setting": folder, "file_bytes": file.stat().st_size,
                     "rgb_psnr_global_db": result["rgb_psnr_global_db"], "rgb_psnr_mean_per_frame_db": result["rgb_psnr_mean_per_frame_db"],
                     "rgb_ms_ssim_mean": result["rgb_ms_ssim_mean"], "complete_file": str(file),
                     "timing_source": "historical run, different Torch/cuDNN; speed not strictly comparable"})
    for crf in args.crfs:
        if any(r["codec"] == "HEVC/libx265" and r["setting"] == f"crf{crf}" for r in rows):
            continue
        run = output / f"hevc_crf{crf}"
        run.mkdir()
        file = run / "video.mp4"
        encode_command = [str(FFMPEG), "-v", "error", "-framerate", args.fps, "-start_number", "0", "-i", str(Path(args.reference) / "%05d.png"),
                          "-an", "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709", "-c:v", "libx265", "-preset", "medium",
                          "-tune", "zerolatency", "-crf", str(crf), "-x265-params", "keyint=32:min-keyint=32:scenecut=0:bframes=0:log-level=error",
                          "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709", str(file)]
        start = time.perf_counter()
        subprocess.run(encode_command, check=True)
        encode_seconds = time.perf_counter()-start
        (run / "rgb").mkdir()
        decode_command = [str(FFMPEG), "-v", "error", "-i", str(file), "-an", "-fps_mode", "passthrough", "-start_number", "0", str(run / "rgb/%05d.png")]
        start = time.perf_counter()
        subprocess.run(decode_command, check=True)
        decode_seconds = time.perf_counter()-start
        frames = json.loads(subprocess.check_output([str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pict_type", "-of", "json", str(file)]))["frames"]
        if len(frames) != count or sum(f["pict_type"] == "I" for f in frames) != 2:
            raise ValueError("Unexpected HEVC frame count/GOP")
        result = metrics(args.reference, run / "rgb")
        result.update({"file_bytes": file.stat().st_size, "file_sha256": sha256(file), "encode_process_seconds": encode_seconds,
                       "decode_process_and_png_seconds": decode_seconds, "i_frames": 2, "p_frames": count-2,
                       "commands": {"encode": encode_command, "decode": decode_command},
                       "profile": "libx265 medium zerolatency; GOP32; no B frames; BT.709 limited-range 8-bit YUV420; MP4 complete-file size"})
        save_json(run / "result.json", result)
        rows.append({"codec": "HEVC/libx265", "setting": f"crf{crf}", "file_bytes": file.stat().st_size,
                     "rgb_psnr_global_db": result["rgb_psnr_global_db"], "rgb_psnr_mean_per_frame_db": result["rgb_psnr_mean_per_frame_db"],
                     "rgb_ms_ssim_mean": result["rgb_ms_ssim_mean"], "complete_file": str(file),
                     "encode_process_seconds": encode_seconds, "decode_process_and_png_seconds": decode_seconds})
        print(f"HEVC CRF{crf}: {file.stat().st_size} bytes, PSNR {result['rgb_psnr_global_db']:.3f} dB", flush=True)
    for run in args.mlvc_runs:
        run = Path(run)
        data = json.loads((run / "verification.json").read_text("utf-8"))
        manifest = json.loads((run / "manifest.json").read_text("utf-8"))
        if manifest["frame_count"] != count or manifest["fps"] != args.fps:
            raise ValueError("MLVC sequence profile differs")
        rows.append({"codec": "MLVC-S + DMCI (mirror weights)", "setting": f"q{manifest['frames'][0]['q_index']}",
                     "file_bytes": data["file_bytes"], "rgb_psnr_global_db": data["rgb_psnr_global_db"],
                     "rgb_psnr_mean_per_frame_db": data["rgb_psnr_mean_per_frame_db"], "rgb_ms_ssim_mean": data["rgb_ms_ssim_mean"],
                     "complete_file": str(run / "video.mlvcv"), "codec_encode_seconds": data["codec_encode_seconds"], "codec_decode_seconds": data["codec_decode_seconds"]})
    for row in rows:
        row["video_kbps"] = row["file_bytes"]*8/duration/1000
        row["transfer_seconds_at_20kbps"] = row["file_bytes"]*8/20000
    save_json(output / "comparison.json", {"source_reference": str(Path(args.reference).resolve()), "frames": count, "fps": args.fps,
               "duration_seconds": duration, "rows": rows,
               "limitations": ["One reconstructed source video only; no semantic evaluation yet", "Codec native color domains differ (GLC normalized RGB, MLVC float YUV420, HEVC 8-bit YUV420)", "Historical GLC timings and HEVC process times are not directly comparable to MLVC kernel/API timings", "Rate points are not equal file sizes; compare RD curves/interpolated matched budgets only"]})


if __name__ == "__main__":
    main()
