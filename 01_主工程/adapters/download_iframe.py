"""Pinned research-project mirror fallback; does not claim an official checksum."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7718ac66480ed572b64e616cec1a59b87653da1b"
URL = f"https://huggingface.co/GSVC-Project/gsvc-asset/resolve/{REVISION}/models/dcvc/cvpr2025_image.pth.tar?download=true"
EXPECTED_SHA = "555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9"
EXPECTED_SIZE = 182764252
destination = ROOT / "checkpoints/DMCI-6.0-mirror.pth.tar"
temporary = destination.with_suffix(".part")
digest = hashlib.sha256()
with urllib.request.urlopen(URL, timeout=90) as response, temporary.open("xb") as output:
    for block in iter(lambda: response.read(1024 * 1024), b""):
        output.write(block)
        digest.update(block)
if digest.hexdigest() != EXPECTED_SHA or temporary.stat().st_size != EXPECTED_SIZE:
    raise RuntimeError("Mirror weight integrity mismatch; incomplete file retained")
temporary.rename(destination)
provenance = {"model": "DCVC-RT cvpr2025_image.pth.tar, used as DMCI-6.0", "source": URL,
              "source_kind": "GSVC research project mirror on Hugging Face; NOT Microsoft-hosted",
              "revision": REVISION, "sha256": digest.hexdigest(), "bytes": EXPECTED_SIZE,
              "verification": "Matches Hugging Face LFS SHA256. Microsoft original checksum unavailable; official-source equivalence not yet verified.",
              "official_reference": "https://github.com/microsoft/mlvc#prepare-auxiliary-models",
              "official_download_attempts": ["OneDrive share API returned HTTP 401", "Public share page TLS access failed", "MLVC blob candidate URLs returned HTTP 404"]}
(ROOT / "audit/iframe_mirror_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(provenance, ensure_ascii=False, indent=2))
