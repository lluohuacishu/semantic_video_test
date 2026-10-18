"""Versioned GLC base + disjoint source-derived H.265 ROI packet.

Legacy profiles remain in system.py. This profile requires shared GLC deployment.
"""
from __future__ import annotations
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile
from runtime import fps_string, sha256
from roi_layout import spatial_overlap

PROFILE = "semantic-video-glc-roi-v1"
MAX_BYTES = 256 * 1024 * 1024
MAX_MANIFEST = 2 * 1024 * 1024
MAX_SEMANTICS = 16 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def parse(payload):
    return json.loads(payload, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid number: {value}")))


def packet_profile(packet):
    with zipfile.ZipFile(packet) as archive:
        require(sum(item.filename == "manifest.json" for item in archive.infolist()) == 1, "Missing/duplicate manifest")
        require(archive.getinfo("manifest.json").file_size <= MAX_MANIFEST, "Manifest too large")
        return parse(archive.read("manifest.json")).get("profile")


def validate_manifest(manifest):
    require(manifest.get("profile") == PROFILE and manifest.get("version") == 1, "Unsupported GLC packet profile/version")
    require(manifest.get("mode") == "video", "GLC packet must contain video")
    w, h, n = (manifest[key] for key in ("width", "height", "frames"))
    require(all(type(v) is int for v in (w, h, n)), "Noninteger geometry/count")
    require(32 <= min(w, h) and max(w, h) <= 4096 and w % 2 == h % 2 == 0 and 1 <= n <= 10000, "Invalid geometry/count")
    require(manifest["fps"] == fps_string(manifest["fps"]), "FPS must use canonical rational form")
    require(manifest.get("roi_layout") == "halves" and manifest.get("compositor") == "hard_replace_disjoint_rectangles_v1", "Unsupported compositor/layout")
    require(manifest["base"] == {"file": "base.glcv", "codec": "glcv-lab-f32-v1"}, "Invalid GLC base dispatch")
    require(manifest["semantics_file"] == "semantics.json.gz", "Invalid semantic stream")
    targets = manifest["targets"]
    require(isinstance(targets, dict) and bool(targets) and set(targets) <= {"fish", "diver"}, "Invalid targets")
    for target, profile in targets.items():
        require(profile["target"] == target and profile["class_id"] == {"fish": 6, "diver": 1}[target], "Target/class mismatch")
        require(re.fullmatch(r"[a-f0-9]{64}", profile["model_sha256"]) is not None, "Invalid teacher identity")
        require(type(profile["threshold"]) in (float, int) and 0 <= profile["threshold"] <= 1, "Invalid threshold")
    patches = manifest["patches"]
    require(isinstance(patches, list) and len(patches) <= 998, "Too many ROI patches")
    expected = {"base.glcv", "semantics.json.gz"}
    windows = {}
    segment = manifest["configuration"]["roi"]["segment_frames"]
    require(type(segment) is int and 1 <= segment <= 256, "Invalid ROI window")
    for index, patch in enumerate(patches):
        require(patch["file"] == f"roi_{index:04d}.mp4", "ROI names must be unique and ordered")
        expected.add(patch["file"])
        x, y, pw, ph = patch["box"]
        start, count = patch["start"], patch["frames"]
        require(all(type(v) is int for v in (x, y, pw, ph, start, count)), "Noninteger ROI geometry/time")
        require(min(x, y, start) >= 0 and min(pw, ph) >= 16 and all(v % 2 == 0 for v in (x, y, pw, ph)), "Invalid ROI alignment")
        require(x + pw <= w and y + ph <= h and start < n and start % segment == 0 and count == min(segment, n - start), "ROI bounds/window mismatch")
        midpoint = (w // 4) * 2
        require((patch["partition"] == "left" and x + pw <= midpoint) or (patch["partition"] == "right" and x >= midpoint), "ROI crosses partition")
        prior = windows.setdefault(start, [])
        require(len(prior) < 2 and not any(spatial_overlap(patch["box"], p["box"]) for p in prior), "ROI overlap")
        require(not any(p["partition"] == patch["partition"] for p in prior), "Duplicate partition/window")
        prior.append(patch)
    require(set(manifest["files"]) == expected, "Missing or unreferenced stream")
    for record in manifest["files"].values():
        require(type(record["bytes"]) is int and 0 < record["bytes"] <= MAX_BYTES, "Invalid file length")
        require(re.fullmatch(r"[a-f0-9]{64}", record["sha256"]) is not None, "Invalid stream checksum")


def validate_semantics(semantics, manifest):
    w, h, n = (manifest[k] for k in ("width", "height", "frames"))
    require(semantics["meaning"] == "sender predictions; components are not verified object counts", "Unknown semantic meaning")
    require(len(semantics["frames"]) == n, "Semantic frame count mismatch")
    for index, record in enumerate(semantics["frames"]):
        require(type(record["frame"]) is int and record["frame"] == index, "Semantic order mismatch")
        require(set(record["targets"]) == set(manifest["targets"]), "Semantic target mismatch")
        for prediction in record["targets"].values():
            require(type(prediction["pixels"]) is int and 0 <= prediction["pixels"] <= w * h, "Invalid pixel count")
            for region in prediction["regions"]:
                x, y, rw, rh = region["box"]
                require(all(type(v) is int for v in (x, y, rw, rh)) and min(x, y) >= 0 and min(rw, rh) > 0 and x + rw <= w and y + rh <= h, "Invalid semantic box")
                require(type(region["pixels"]) is int and 0 < region["pixels"] <= rw * rh, "Invalid component size")
                confidence = region["confidence"]
                require(type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1, "Invalid confidence")
            require(sum(r["pixels"] for r in prediction["regions"]) == prediction["pixels"], "Component/pixel mismatch")


def inspect_packet(packet):
    require(Path(packet).stat().st_size <= MAX_BYTES, "Packet exceeds transport implementation limit")
    with zipfile.ZipFile(packet) as archive:
        entries = archive.infolist()
        names = [item.filename for item in entries]
        require(len(entries) <= 1000 and len(set(names)) == len(names), "Duplicate/excessive packet members")
        require(all(re.fullmatch(r"manifest\.json|base\.glcv|semantics\.json\.gz|roi_[0-9]{4}\.mp4", name) for name in names), "Unsupported member/path")
        require(all(item.compress_type == zipfile.ZIP_STORED and not item.flag_bits & 1 for item in entries), "Packet must be unencrypted ZIP_STORED")
        require(sum(item.file_size for item in entries) <= MAX_BYTES, "Packet content too large")
        require(archive.getinfo("manifest.json").file_size <= MAX_MANIFEST, "Manifest too large")
        manifest = parse(archive.read("manifest.json"))
        validate_manifest(manifest)
        require(set(names) == {"manifest.json", *manifest["files"]}, "Manifest/member mismatch")
        payloads = {}
        for name, record in manifest["files"].items():
            require(archive.getinfo(name).file_size == record["bytes"], "Declared file size mismatch")
            payload = archive.read(name)
            require(hashlib.sha256(payload).hexdigest() == record["sha256"], "Stream hash mismatch")
            payloads[name] = payload
    with gzip.GzipFile(fileobj=io.BytesIO(payloads["semantics.json.gz"])) as reader:
        raw = reader.read(MAX_SEMANTICS + 1)
    require(len(raw) <= MAX_SEMANTICS, "Semantic metadata too large")
    semantics = parse(raw)
    validate_semantics(semantics, manifest)
    return manifest, semantics, payloads


def unpack_packet(packet, output):
    manifest, semantics, payloads = inspect_packet(packet)
    directory = Path(output) / "received_streams"
    directory.mkdir(exist_ok=False)
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
    return manifest, semantics, directory


def write_packet(streams, manifest, output):
    streams, output = Path(streams), Path(output)
    manifest["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(streams.iterdir())}
    validate_manifest(manifest)
    encoded = compact(manifest)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", encoded)
        for name in manifest["files"]:
            archive.write(streams / name, name)
    inspect_packet(output)
    coordinates = len(compact(manifest["patches"]))
    ledger = {"glc_file_bytes": manifest["files"]["base.glcv"]["bytes"],
              "roi_h265_bytes": sum(manifest["files"][p["file"]]["bytes"] for p in manifest["patches"]),
              "semantic_message_gzip_bytes": manifest["files"]["semantics.json.gz"]["bytes"],
              "roi_position_time_value_bytes": coordinates, "manifest_other_bytes": len(encoded) - coordinates,
              "zip_container_bytes": output.stat().st_size - len(encoded) - sum(r["bytes"] for r in manifest["files"].values())}
    require(sum(ledger.values()) == output.stat().st_size, "Byte ledger mismatch")
    return ledger
