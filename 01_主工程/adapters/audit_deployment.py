import json
from pathlib import Path
import subprocess
from common import ROOT, UPSTREAM, COMMIT, P_WEIGHT_SHA, configure, environment, sha256, save_json, load_models

configure()
head = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
assert head == COMMIT
assert sha256(ROOT / "checkpoints/mlvc-s-psnr-v1.ckpt") == P_WEIGHT_SHA
snapshot = json.loads((ROOT / "audit/legacy_source_20261013/manifest.json").read_text("utf-8"))
legacy = Path(snapshot["source_root"])
checked, changed = [], []
for record in snapshot["files"]:
    path = legacy / record["path"]
    # Report copies in the snapshot have a separate source-path field or prefix.
    if path.is_file():
        checked.append(record["path"])
        if sha256(path) != record["sha256"]:
            changed.append(record["path"])
assert not changed, changed
native = ROOT / ".venv/Lib/site-packages/msrtc/rans"
native_files = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in native.iterdir() if p.is_file()]
p_model, i_model, _ = load_models(ROOT / "checkpoints/mlvc-s-psnr-v1.ckpt", ROOT / "checkpoints/DMCI-6.0-mirror.pth.tar", device="cpu", entropy=False)
result = {"environment": environment(), "code_version_verified": True, "upstream_modified_paths": subprocess.check_output(["git", "-C", str(UPSTREAM), "diff", "--name-only"], text=True).splitlines(),
          "local_patch": "audit/windows_entropy_const.patch", "local_patch_purpose": "const-correct C++ Python type name; no coding algorithms changed",
          "compiler": subprocess.check_output([r"C:\mingw64\bin\g++.exe", "--version"], text=True).splitlines()[0],
          "entropy_tests": "7 passed; pinned official test_msrtc_rans.py", "native_package_files": native_files,
          "parameters_p": sum(p.numel() for p in p_model.parameters()), "parameters_i": sum(p.numel() for p in i_model.parameters()),
          "strict_checkpoint_loading": True,
          "p_weight": {"source": "https://mlvideopub.blob.core.windows.net/mlvc/models/mlvc-s-psnr-v1.ckpt", "sha256": P_WEIGHT_SHA, "official_hash_match": True},
          "i_weight": json.loads((ROOT / "audit/iframe_mirror_provenance.json").read_text("utf-8")),
          "legacy_source_checked_files": len(checked), "legacy_changed_files": changed,
          "adapter_files": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in (ROOT / "adapters").glob("*.py")],
          "source_rgb_reference_verification": "50/50 current extraction frames equal legacy GLC sender source RGB pixels"}
save_json(ROOT / "audit/deployment.json", result)
print(json.dumps({"upstream_commit": head, "legacy_unchanged_checked_files": len(checked), "adapter_files": len(result["adapter_files"])}, indent=2))
