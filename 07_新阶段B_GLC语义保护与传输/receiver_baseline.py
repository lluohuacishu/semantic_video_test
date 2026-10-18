"""Independent B1 receiver: received GLC stream plus declared shared dependencies.

Python file-read policy and explicit subprocess arguments, not an OS sandbox.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class ReadPolicy:
    """Deny undeclared media/models even when placed under a code/runtime root."""
    protected = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".mp4", ".avi", ".mkv", ".mov",
                 ".yuv", ".npz", ".npy", ".pt", ".pth", ".ckpt", ".safetensors", ".tar", ".glcv", ".scv"}

    def __init__(self, output, config):
        self.output = Path(output).resolve()
        self.config = config
        self.models = {Path(p).resolve(): digest for p, digest in config["shared_models"].items()}
        self.exact_code = {Path(p).resolve() for p in config["code_hashes"]}
        self.code_roots = [Path(config["glc_repo"]).resolve(), Path(__file__).resolve().parent]
        self.runtime_roots = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
        self.reads = Counter()
        self.denied = []
        self.commands = []

    def classify(self, path):
        path = Path(path).resolve()
        if path.is_relative_to(self.output):
            return "receiver_directory"
        if path in self.models:
            return "declared_shared_model"
        if path.suffix.lower() in self.protected:
            return None
        if path in self.exact_code:
            return "pinned_adapter_or_transport_code"
        if any(path.is_relative_to(root) for root in self.runtime_roots):
            return "python_runtime_or_installed_package"
        if any(path.is_relative_to(root) for root in self.code_roots) and path.suffix.lower() in {".py", ".pyc", ".json"}:
            return "decoder_code_or_configuration"
        return None

    def hook(self, event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path, mode, flags = args
            reading = ((isinstance(mode, str) and ("r" in mode or "+" in mode)) or
                       (mode is None and isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR) != os.O_WRONLY))
            if not reading:
                return
            path = Path(os.fsdecode(path)).resolve()
            category = self.classify(path)
            self.reads[(str(path), category or "denied")] += 1
            if category is None:
                self.denied.append(str(path))
                raise PermissionError(f"B1 receiver refused undeclared external input: {path}")
        elif event == "subprocess.Popen":
            executable, argv, cwd, _env = args
            argv = [os.fsdecode(x) for x in argv]
            git_args = [["git", "-C", self.config["glc_repo"], "rev-parse", "HEAD"],
                        ["git", "-C", self.config["glc_repo"], "diff", "HEAD", "--"]]
            allowed = argv in git_args
            if Path(os.fsdecode(executable)).resolve() == Path(self.config["ffmpeg"]).resolve():
                inputs = [Path(argv[i + 1]).resolve() for i, value in enumerate(argv[:-1]) if value == "-i"]
                allowed = bool(inputs) and all(p.is_relative_to(self.output) for p in inputs)
                allowed = allowed and Path(argv[-1]).resolve().is_relative_to(self.output)
            self.commands.append({"argv": argv, "allowed": allowed})
            if not allowed:
                raise PermissionError(f"B1 receiver refused subprocess: {argv}")

    def report(self):
        return {
            "scope": "Python file opens and explicit subprocess arguments; not OS isolation; native DLL reads are not audited",
            "guard_installed": True, "shared_model_allowlist": {str(p): h for p, h in self.models.items()},
            "python_reads": [{"path": p, "category": c, "count": n} for (p, c), n in sorted(self.reads.items())],
            "denied_reads": self.denied, "subprocesses": self.commands,
            "external_source_or_sender_cache_reads": [p for p, c in self.reads if c == "denied"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    config = json.loads(args.config.read_text("utf-8"))
    if not args.config.resolve().is_relative_to(output):
        raise ValueError("Receiver deployment config must be in its own directory")
    policy = ReadPolicy(output, config)
    sys.addaudithook(policy.hook)
    try:
        for path, expected in {**config["code_hashes"], **config["shared_models"]}.items():
            if sha(path) != expected:
                raise ValueError(f"Shared dependency hash mismatch: {path}")
        sys.path.insert(0, str(Path(config["transport"]).parent))
        from transport import receive_connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(120)
            write_json(output / "ready.json", {"port": listener.getsockname()[1], "pid": os.getpid()})
            connection, _ = listener.accept()
            with connection:
                received = receive_connection(connection, output / "received.glcv")
        write_json(output / "receive.json", {**received, "pid": os.getpid(), "source": "actual TCP reception"})
        print("TCP reception verified; starting independent GLC decode", flush=True)
        sys.path.insert(0, str(Path(config["adapter"]).parent))
        import glc_video_codec
        glc_video_codec.decode(SimpleNamespace(
            input=output / "received.glcv", output=output / "decoded", device="cuda",
            repo=Path(config["glc_repo"]), weights=Path(config["weights"]), ffmpeg=Path(config["ffmpeg"])))
        write_json(output / "completion.json", {"success": True, "pid": os.getpid(), "finished_unix_seconds": time.time()})
    except BaseException as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_json(output / "receiver_read_audit.json", policy.report())


if __name__ == "__main__":
    main()
