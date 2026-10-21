"""Build a small, hash-bound C7 update for the original C5 receiver bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STAGE = HERE.name
PAYLOAD = (
    "c1_runtime.py", "c5_compat.py", "c5_server.py", "c5_timebase.py", "c5_metrics.py",
    "c7_diagnostics.py", "c5_console/app.js", "c5_console/index.html", "c5_console/styles.css",
    "C7跨卡预览使用说明.md",
)
KNOWN_REVISIONS = ("3401050", "9d070e3", "7058477", "47ba6cf", "acf4f47", "8b960e3")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=check)


def newline_hashes(data):
    lf = data.replace(b"\r\n", b"\n")
    return {digest(data), digest(lf), digest(lf.replace(b"\n", b"\r\n"))}


def build(base, output, *, allow_dirty=False):
    base, output = Path(base).resolve(), Path(output).resolve()
    manifest_path = base.parent / "bundle_manifest.json"
    original = json.loads(manifest_path.read_text("utf-8-sig"))
    dirty = bool(git("status", "--porcelain").stdout.strip())
    if dirty and not allow_dirty:
        raise ValueError("Commit source changes before building a distributable patch")
    commit = git("rev-parse", "HEAD").stdout.decode().strip()
    archive_path = output.with_suffix(".zip")
    if output.exists() or archive_path.exists():
        raise FileExistsError("Refusing to replace an existing package: " + str(output))
    files = {}
    payload = {}
    for filename in PAYLOAD:
        relative = STAGE + "/" + filename
        current = (HERE / filename).read_bytes()
        before = base / relative
        previous = set()
        if before.is_file():
            data = before.read_bytes()
            entry = original["files"][base.name + "/" + relative]
            if len(data) != entry["bytes"] or digest(data) != entry["sha256"]:
                raise ValueError("Original C5 bundle code has changed: " + relative)
            previous.update(newline_hashes(data))
        else:
            previous.add(None)
        for revision in KNOWN_REVISIONS:
            historical = git("show", revision + ":" + relative, check=False)
            if historical.returncode == 0:
                previous.update(newline_hashes(historical.stdout))
        previous.update(newline_hashes(current))
        files[relative] = {"bytes": len(current), "sha256": digest(current),
                           "previous_sha256": sorted(previous, key=lambda v: v or "")}
        payload[relative] = current
    output.mkdir(parents=True)
    for relative, data in payload.items():
        target = output / "payload" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for filename in ("c7_apply_preview_patch.py", "安装C7跨卡预览补丁.cmd", "C7跨卡预览使用说明.md"):
        data = (HERE / filename).read_bytes()
        if filename.endswith(".cmd"):
            data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        (output / filename).write_bytes(data)
    manifest = {"schema": "c7-preview-patch-v1", "created": "2026-10-21",
                "producer_commit": commit, "producer_worktree_dirty": dirty,
                "base_bundle_commit": original["producer_commit"],
                "policy": "reference_preview", "gpu_validation_performed": False,
                "files": files}
    (output / "patch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, output.name + "/" + path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError("Archive CRC failed: " + bad)
        for relative, entry in files.items():
            data = archive.read(output.name + "/payload/" + relative)
            if len(data) != entry["bytes"] or digest(data) != entry["sha256"]:
                raise ValueError("Archived payload differs: " + relative)
    result = {"archive": str(archive_path), "bytes": archive_path.stat().st_size,
              "sha256": digest(archive_path.read_bytes()), "producer_commit": commit,
              "producer_worktree_dirty": dirty, "payload_files": len(files), "archive_crc_verified": True}
    archive_path.with_suffix(".zip.sha256").write_text(
        result["sha256"] + "  " + archive_path.name + "\n", encoding="utf-8")
    archive_path.with_suffix(".receipt.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Original C5 semantic_video_test folder")
    parser.add_argument("--output", type=Path, required=True, help="New package directory")
    parser.add_argument("--allow-dirty", action="store_true", help="Internal installation validation only")
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output, allow_dirty=args.allow_dirty), ensure_ascii=False, indent=2))
