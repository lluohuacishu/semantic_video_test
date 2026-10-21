"""Arithmetic admission and explicit, independently checked reconstruction outcomes."""
from __future__ import annotations

POLICIES = ("strict", "device_pair_test", "reference_preview")
FIELDS = {"implementation", "torch", "cuda", "cudnn", "device", "capability", "tf32", "deterministic"}
PREVIEW_WARNING = "参考状态校验通过；RGB 非逐位一致，仅供预览，画面误差未验收。"


def reconstruction_verdict(expected, actual, policy, *, completed_frames, total_frames):
    """Never infer success from a worker/peer's boolean; compare ordered digests."""
    if policy not in POLICIES:
        raise ValueError("Unknown arithmetic compatibility policy")
    if (type(total_frames) is not int or total_frames < 1
            or type(completed_frames) is not int or completed_frames != total_frames):
        raise ValueError("Incomplete reconstructed sequence")
    keys = {"latent_sha256", "rgb_sha256"}
    for hashes in (expected, actual):
        if (not isinstance(hashes, dict) or set(hashes) != keys
                or any(not isinstance(v, str) or len(v) != 64
                       or any(c not in "0123456789abcdef" for c in v) for v in hashes.values())):
            raise ValueError("Missing or invalid reconstruction checksums")
    latent, rgb = (actual[k] == expected[k] for k in ("latent_sha256", "rgb_sha256"))
    if not latent or (not rgb and policy != "reference_preview"):
        raise ValueError("Ordered GLC frame checksum aggregate differs from the sender.")
    return {"policy": policy, "status": "fully_verified" if rgb else "reference_verified_preview",
            "verified": rgb, "reference_verified": True, "rgb_verified": rgb,
            "complete_sequence": True, "completed_frames": completed_frames,
            "warning": None if rgb else PREVIEW_WARNING,
            "pixel_error_measured": False}


def check_arithmetic(sender, receiver, policy="strict", container_version=2):
    if policy not in POLICIES:
        raise ValueError("Unknown arithmetic compatibility policy")
    if policy == "reference_preview" and container_version != 2:
        raise ValueError("Reference preview requires a v2 stream with ordered sequence checksums")
    if not isinstance(sender, dict) or not isinstance(receiver, dict):
        raise ValueError("Missing arithmetic profile")
    if sender == receiver:
        return {"policy": policy, "admission": "exact_profile", "differences": {},
                "reconstruction_verified": False}
    differences = {key: {"sender": sender.get(key), "receiver": receiver.get(key)}
                   for key in sorted(set(sender) | set(receiver)) if sender.get(key) != receiver.get(key)}
    if policy == "strict":
        raise ValueError("Sender and receiver arithmetic profiles differ: " + ", ".join(differences))
    if (container_version != 2 or set(sender) != FIELDS or set(receiver) != FIELDS
            or set(differences) != {"device"} or sender["capability"] is None
            or not sender["cuda"] or sender["tf32"] is not False
            or sender["deterministic"] is not True):
        raise ValueError("Device-pair diagnostic only permits a GPU name difference with identical CUDA arithmetic")
    return {"policy": policy, "admission": "reference_preview_candidate" if policy == "reference_preview" else "device_pair_diagnostic", "differences": differences,
            "reconstruction_verified": False,
            "note": ("Preview requires complete latent verification; RGB differences remain unverified and visible."
                     if policy == "reference_preview" else
                     "Admission is not compatibility proof; this stream must pass all RGB/latent sequence checks.")}
