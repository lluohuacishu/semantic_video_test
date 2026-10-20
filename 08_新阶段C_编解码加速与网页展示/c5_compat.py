"""Arithmetic admission policy; reconstruction checks are always mandatory."""
from __future__ import annotations

POLICIES = ("strict", "device_pair_test")
FIELDS = {"implementation", "torch", "cuda", "cudnn", "device", "capability", "tf32", "deterministic"}


def check_arithmetic(sender, receiver, policy="strict", container_version=2):
    if policy not in POLICIES:
        raise ValueError("Unknown arithmetic compatibility policy")
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
    return {"policy": policy, "admission": "device_pair_diagnostic", "differences": differences,
            "reconstruction_verified": False,
            "note": "Admission is not compatibility proof; this stream must pass all RGB/latent sequence checks."}
