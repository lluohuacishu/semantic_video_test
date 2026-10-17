"""Checks real decoder failure behavior against a successfully encoded file."""
import json
from pathlib import Path
import subprocess
import sys
from mlvc_file import read_file
from common import ROOT, save_json

output = ROOT / "runs/container_failure_checks"
output.mkdir(exist_ok=False)
original = ROOT / "runs/baseline_720p_dmci_q0/video.mlvcv"
data = original.read_bytes()
checks = []
for name, payload in [("truncated", data[:-1]), ("corrupted", data[:-1] + bytes([data[-1] ^ 1]))]:
    file = output / f"{name}.mlvcv"
    file.write_bytes(payload)
    try:
        read_file(file)
    except (ValueError, EOFError) as exc:
        checks.append({"case": name, "rejected": True, "error": str(exc)})
    else:
        raise AssertionError(f"Decoder accepted {name} file")
wrong = output / "wrong.ckpt"
wrong.write_bytes(b"incorrect shared model")
command = [sys.executable, str(ROOT / "adapters/mlvc_file.py"), "decode", "--file", str(original), "--weight", str(wrong),
           "--i-weight", str(ROOT / "checkpoints/DMCI-6.0-mirror.pth.tar"), "--output", str(output / "must_not_decode")]
result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
if result.returncode == 0 or "P-frame weight hash mismatch" not in result.stderr:
    raise AssertionError("Decoder failed to reject incorrect shared weights")
checks.append({"case": "wrong_weights", "rejected": True, "error": "P-frame weight hash mismatch", "exit_code": result.returncode})
save_json(output / "result.json", {"all_expected_failures_rejected": True, "checks": checks})
print(json.dumps(checks, indent=2))
