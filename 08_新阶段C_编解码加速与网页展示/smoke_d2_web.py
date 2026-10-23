"""Explicit manual smoke: five methods, eight frames each, real local HTTP/TCP/GLC.

Requires an already-running local loopback service. This is functional QA,
not D2 reproduction, budget acceptance, or a new research comparison.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing smoke record")
    base = "http://127.0.0.1:8773"
    def api(path, value=None):
        data = None if value is None else json.dumps(value).encode("utf-8")
        with urlopen(Request(base + path, data=data, headers={"Content-Type": "application/json"}), timeout=180) as response:
            return json.load(response)
    config = api("/api/config")
    if config.get("version") != "D2-web-demo-v1" or not config.get("loopback_urls"):
        raise ValueError("Expected the D2 local loopback demo")
    if any(j["status"] in ("running", "queued", "cancelling") for j in api("/api/jobs")):
        raise ValueError("Another sender task is active")
    rows = []
    for mode in ("ordinary", "lowpass", "light_lowpass", "semantic_original", "semantic_light"):
        opts = {"preset": "compact", "backend": "native", "enhancement": mode, "sampling_mode": "time"}
        job = api("/api/demo-jobs", {"sample": "021", "options": opts})
        identifier = job["id"]
        opts = job["options"]
        opts["clip"]["end"] = str(Fraction(opts["clip"]["start"]) + Fraction(8 * 1001, 30000))
        plan = api(f"/api/jobs/{identifier}/probe", {"options": opts})
        assert plan["frames"] == 8, plan
        api(f"/api/jobs/{identifier}/start", {})
        deadline, phase = time.monotonic() + 600, None
        while time.monotonic() < deadline:
            job = api(f"/api/jobs/{identifier}")
            if job["phase"] != phase:
                phase = job["phase"]
                print(mode, identifier, job["status"], phase, flush=True)
            if job["status"] in ("done", "error", "cancelled", "interrupted"):
                break
            time.sleep(.8)
        assert job["status"] == "done", job
        result = job["result"]
        assert result["verified"] and result["frames"] == 8 and result["header"]["q"] == 0
        assert result["remote"]["source_media_reads"] == []
        assert result["transfer"]["network_transfer_performed"]
        enc = api(f"/jobs/{identifier}/encode.json")
        assert "before any lowpass" in enc["psnr_reference"]
        paths = [base + f"/jobs/{identifier}/source.mp4",
                 config["loopback_urls"]["receiver"].rstrip("/") + f'/jobs/{job["receiver_job_id"]}/decoded.mp4']
        if mode != "ordinary":
            prepared = api(f"/jobs/{identifier}/preprocessing.json")
            assert prepared["summary"]["mode"] == mode
            for original, coding in zip(enc["frames"], prepared["records"]):
                assert original["source_rgb_sha256"] == coding["source_rgb_sha256"]
                assert original["coding_rgb_sha256"] == coding["coding_rgb_sha256"]
            paths.append(base + f"/jobs/{identifier}/coding-input.mp4")
        for path in paths:
            with urlopen(Request(path, headers={"Range": "bytes=0-31"}), timeout=10) as response:
                assert response.status == 206 and len(response.read()) == 32, path
        rows.append({"method": mode, "sender_id": identifier, "receiver_id": job["receiver_job_id"],
                     "frames": result["frames"], "bytes": result["file_bytes"], "verified": result["verified"],
                     "network_transfer": True, "receiver_source_media_reads": [], "mp4_range_verified": True,
                     "service_seconds": result["service_seconds"], "preprocessing": result.get("preprocessing")})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"scope": "functional smoke, not research/acceptance", "state": "running",
                              "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.write_text(json.dumps({"scope": "functional smoke, not research/acceptance", "date": "2026-10-23",
                          "state": "completed", "source": "021 native source, 8 frames, INTER_AREA to 512x288",
                          "new_encodes": 5, "new_independent_decodes": 5, "new_semantic_frames": 16,
                          "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Five-method functional smoke passed", flush=True)


if __name__ == "__main__":
    main()
