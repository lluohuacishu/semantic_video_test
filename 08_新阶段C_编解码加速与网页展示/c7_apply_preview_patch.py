"""Install a bounded C7 code patch with backups. Standard library; no model startup."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

STAGE = "08_新阶段C_编解码加速与网页展示"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def local_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Unsafe patch path: " + name)
    return path


def active_processes(project):
    if os.name != "nt":
        return []
    command = ('[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); '
               'Get-CimInstance Win32_Process -Filter "Name LIKE \'python%\'" | '
               'Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress')
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, encoding="utf-8", timeout=30, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    values = json.loads(result.stdout.strip() or "[]")
    if isinstance(values, dict):
        values = [values]
    # A Windows venv Python launcher remains alive as our parent process with
    # the same command line. It is part of this installer, not a live service.
    own_pids = {os.getpid()}
    by_pid = {row["ProcessId"]: row for row in values}
    script = str(Path(__file__).resolve()).replace("/", "\\").casefold()
    ancestor = os.getppid()
    while ancestor in by_pid and ancestor not in own_pids:
        row = by_pid[ancestor]
        if script not in (row.get("CommandLine") or "").replace("/", "\\").casefold():
            break
        own_pids.add(ancestor)
        ancestor = row.get("ParentProcessId")
    target = str(project).replace("/", "\\").casefold()
    return [row["ProcessId"] for row in values if row["ProcessId"] not in own_pids
            and target in (row.get("CommandLine") or "").replace("/", "\\").casefold()]


def install(project, package, *, check=False):
    project, package = Path(project).resolve(), Path(package).resolve()
    if not (project / STAGE / "c5_server.py").is_file():
        raise ValueError("请选择原 C5 包内的 semantic_video_test 文件夹")
    manifest = json.loads((package / "patch_manifest.json").read_text("utf-8-sig"))
    if manifest.get("schema") != "c7-preview-patch-v1":
        raise ValueError("Unsupported patch manifest")
    pending = []
    for name, entry in manifest["files"].items():
        # Code only: never touch runtime environments, user configs or task data.
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or relative.parts[0] != STAGE or ".local." in relative.name
                or relative.suffix not in (".py", ".js", ".html", ".css", ".md")):
            raise ValueError("Patch contains an unsupported target: " + name)
        source, target = local_path(package / "payload", name), local_path(project, name)
        if source.stat().st_size != entry["bytes"] or sha(source) != entry["sha256"]:
            raise ValueError("Patch file checksum differs: " + name)
        current = sha(target) if target.is_file() else None
        if current == entry["sha256"]:
            continue
        if current not in entry["previous_sha256"]:
            raise ValueError("目标文件已被修改或版本未知，未覆盖：" + name)
        pending.append((name, source, target, current))
    active = active_processes(project)
    if active:
        raise ValueError("请先关闭本项目的 C5/C7 进程，再安装；PID: " + ", ".join(map(str, active)))
    if check or not pending:
        return {"state": "checked" if check else "already_installed", "changed_files": len(pending),
                "gpu_used": False, "producer_commit": manifest["producer_commit"]}
    backup = project / "C7_patch_backups" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3))
    backup.mkdir(parents=True)
    bundle_path = project.parent / "bundle_manifest.json"
    bundle_bytes = bundle_path.read_bytes() if bundle_path.is_file() else None
    if bundle_bytes is not None:
        (backup / "bundle_manifest.json").write_bytes(bundle_bytes)
    for name, source, target, current in pending:
        if current is not None:
            saved = local_path(backup / "files", name)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    receipt = {"schema": "c7-preview-patch-receipt-v1", "project": str(project),
               "producer_commit": manifest["producer_commit"], "patch_manifest_sha256": sha(package / "patch_manifest.json"),
               "files": {name: {"before": current, "after": manifest["files"][name]["sha256"]}
                         for name, source, target, current in pending}, "state": "installing"}
    receipt_path = backup / "receipt.json"
    def save():
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save()
    changed, temporary_files = [], []
    try:
        for name, source, target, current in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".c7-new")
            with temporary.open("xb") as f:
                temporary_files.append(temporary)
                f.write(source.read_bytes())
            temporary.replace(target)
            changed.append((name, target, current))
            if sha(target) != manifest["files"][name]["sha256"]:
                raise ValueError("Installed checksum differs: " + name)
        if bundle_bytes is not None:
            bundle = json.loads(bundle_bytes.decode("utf-8-sig"))
            for name, entry in manifest["files"].items():
                bundle["files"][project.name + "/" + name] = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
            bundle.setdefault("code_patches", []).append({"producer_commit": manifest["producer_commit"],
                                                         "manifest_sha256": receipt["patch_manifest_sha256"]})
            bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        receipt["state"] = "installed"
        save()
    except BaseException:
        for name, target, current in reversed(changed):
            if current is None:
                target.unlink(missing_ok=True)
            else:
                shutil.copy2(local_path(backup / "files", name), target)
        if bundle_bytes is not None:
            bundle_path.write_bytes(bundle_bytes)
        receipt["state"] = "rolled_back"
        save()
        raise
    finally:
        for temporary in temporary_files:
            temporary.unlink(missing_ok=True)
    return {"state": "installed", "changed_files": len(pending), "backup": str(backup),
            "gpu_used": False, "producer_commit": manifest["producer_commit"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        result = install(args.project, Path(__file__).resolve().parent, check=args.check)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print("C7 补丁未安装：" + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("重启两端服务；在接收页选择“跨显卡预览”，保存高级设置，再发起新任务。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
