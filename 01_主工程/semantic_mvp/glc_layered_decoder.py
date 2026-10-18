"""Decode a validated GLC/ROI packet with explicitly supplied shared deployment."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import runtime
from glc_layered import PROFILE, require, unpack_packet


def decode(packet, output, deployment, run_command, authorize):
    tick = time.perf_counter()
    packet, output = Path(packet), Path(output)
    manifest, semantics, streams = unpack_packet(packet, output)
    w, h, n, fps = (manifest[k] for k in ("width", "height", "frames", "fps"))
    runtime.run = run_command
    runtime.FFMPEG = Path(deployment["ffmpeg"])
    runtime.FFPROBE = runtime.FFMPEG.with_name("ffprobe.exe")
    sys.path.insert(0, str(Path(deployment["adapter"]).parent))
    import glc_video_codec
    from glc_container import read_stream
    header, packets = read_stream(streams / "base.glcv")
    require([header["width"], header["height"], header["frames"]] == [w, h, n]
            and runtime.fps_string(header["fps"]) == fps, "GLC header/packet geometry or timing mismatch")
    base = output / "base_decoded"
    for args in (("rev-parse", "HEAD"), ("diff", "HEAD", "--")):
        authorize(["git", "-C", deployment["glc_repo"], *args])
    authorize([deployment["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin", "-framerate", str(float(header["fps"])),
               "-start_number", "1", "-i", str(base / "frames/im%05d.png"), "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(base / "reconstruction_preview.mp4")])
    glc_video_codec.decode(SimpleNamespace(input=streams / "base.glcv", output=base, device="cuda",
                           repo=Path(deployment["glc_repo"]), weights=Path(deployment["weights"]), ffmpeg=Path(deployment["ffmpeg"])))
    decoded_base = json.loads((base / "decode.json").read_text("utf-8"))
    require(decoded_base["independent_decoder"] and not decoded_base["source_media_reads"] and len(decoded_base["frames"]) == n, "GLC independence/count check failed")
    patch_frames = {}
    patch_checks = []
    for index, patch in enumerate(manifest["patches"]):
        stream = streams / patch["file"]
        probe_command = [runtime.FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                         "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,pix_fmt,color_range,color_space,color_transfer,color_primaries", "-of", "json", stream]
        probe = json.loads(run_command(probe_command))["streams"]
        require(len(probe) == 1, "Expected one ROI video stream")
        probe = probe[0]
        require(probe["codec_name"] == "hevc" and probe["pix_fmt"] == "yuv420p", "ROI codec/pixel format mismatch")
        require([probe["width"], probe["height"], int(probe["nb_read_frames"])] == [patch["box"][2], patch["box"][3], patch["frames"]]
                and runtime.fps_string(probe["avg_frame_rate"]) == fps, "ROI geometry/frame count/timing mismatch")
        require(probe["color_range"] == "tv" and all(probe[k] == "bt709" for k in ("color_space", "color_transfer", "color_primaries")), "ROI colour declaration mismatch")
        directory = output / f"roi_rgb_{index:04d}"
        runtime.decode_png(stream, directory)
        files = sorted(directory.glob("*.png"))
        require([p.name for p in files] == [f"{i:05d}.png" for i in range(patch["frames"])], "ROI decoded sequence mismatch")
        hashes = []
        for local, path in enumerate(files):
            with Image.open(path) as image:
                require(image.size == tuple(patch["box"][2:]), "Decoded ROI dimensions mismatch")
                hashes.append(hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest())
            patch_frames.setdefault(patch["start"] + local, []).append((path, patch["box"]))
        patch_checks.append({"file": patch["file"], "probe": probe, "decoded_rgb_sha256": hashes})
    frames = output / "rgb"
    frames.mkdir()
    base_hashes, enhanced_hashes = [], []
    for index in range(n):
        with Image.open(base / "frames" / f"im{index + 1:05d}.png") as image:
            image = image.convert("RGB")
        digest = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        require(digest == packets[index]["rgb_sha256"] == decoded_base["frames"][index]["rgb_sha256"], "GLC RGB mismatch")
        require(decoded_base["frames"][index]["latent_sha256"] == packets[index]["latent_sha256"], "GLC latent mismatch")
        base_hashes.append(digest)
        for path, box in patch_frames.get(index, []):
            with Image.open(path) as roi:
                image.paste(roi.convert("RGB"), tuple(box[:2]))
        enhanced_hashes.append(hashlib.sha256(np.asarray(image).tobytes()).hexdigest())
        image.save(frames / f"{index:05d}.png", compress_level=1)
    runtime.preview(frames, output / "enhanced_preview.mp4", fps)
    runtime.write_json(output / "received_manifest.json", manifest)
    runtime.write_json(output / "target_information.json", semantics)
    result = {"success": True, "profile": PROFILE, "receiver_pid": os.getpid(), "packet_sha256": runtime.sha256(packet),
              "frames": n, "width": w, "height": h, "fps": fps, "base_rgb_sha256": base_hashes, "enhanced_rgb_sha256": enhanced_hashes,
              "base_rgb_sequence_sha256": hashlib.sha256("".join(base_hashes).encode()).hexdigest(),
              "enhanced_rgb_sequence_sha256": hashlib.sha256("".join(enhanced_hashes).encode()).hexdigest(),
              "patch_checks": patch_checks, "glc_source_media_reads": decoded_base["source_media_reads"],
              "glc_encoder_modules_disabled": decoded_base["encoder_modules_disabled"], "decode_and_preview_seconds": time.perf_counter() - tick,
              "semantics_are": "sender predictions; not receiver detection accuracy or ground truth",
              "preview_is": "post-reception H.264 CRF18 YUV420, not transmitted; metrics use PNG RGB"}
    runtime.write_json(output / "decode.json", result)
    return result
