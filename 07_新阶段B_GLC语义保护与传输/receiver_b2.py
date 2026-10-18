"""Receive the versioned B2 packet and independently restore GLC plus ROI."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from b2_common import Journal, read, require, sha, write, now
from receiver_baseline import ReadPolicy


class LayeredReadPolicy(ReadPolicy):
    def __init__(self, output, config):
        super().__init__(output, config)
        self.allowed_commands = []

    def authorize(self, argv):
        argv = list(map(str, argv))
        if argv[:3] == ["git", "-C", self.config["glc_repo"]]:
            require(argv[3:] in (["rev-parse", "HEAD"], ["diff", "HEAD", "--"]), "Unexpected git command")
        else:
            tool = Path(argv[0]).resolve()
            ffmpeg = Path(self.config["ffmpeg"]).resolve()
            require(tool in {ffmpeg, ffmpeg.with_name("ffprobe.exe")}, "Undeclared executable")
            if tool.name.lower() == "ffprobe.exe":
                require(Path(argv[-1]).resolve().is_relative_to(self.output), "External FFprobe input denied")
            else:
                require("-i" in argv, "Missing FFmpeg input")
                for index, arg in enumerate(argv):
                    if arg == "-i":
                        require(Path(argv[index + 1]).resolve().is_relative_to(self.output), "External FFmpeg input denied")
                require(Path(argv[-1]).resolve().is_relative_to(self.output), "External FFmpeg output denied")
                require(not any(x in argv for x in ("-filter_complex_script", "-filter_script", "-attach")), "External auxiliary input denied")
        self.allowed_commands.append(argv)

    def hook(self, event, args):
        if event != "subprocess.Popen":
            return super().hook(event, args)
        executable, argv, _cwd, _env = args
        actual = os.fsdecode(argv) if isinstance(argv, (str, bytes)) else list(map(os.fsdecode, argv))
        matched = next((c for c in self.allowed_commands if actual == (subprocess.list2cmdline(c) if isinstance(actual, str) else c)), None)
        allowed = matched is not None and (executable is None or Path(os.fsdecode(executable)).resolve() == Path(matched[0]).resolve())
        self.commands.append({"argv": matched or actual, "allowed": allowed})
        if not allowed:
            raise PermissionError(f"B2 receiver denied unregistered subprocess: {argv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(args.config.resolve().is_relative_to(output), "Deployment must be receiver-local")
    config = read(args.config)
    os.environ["FFMPEG_PATH"] = config["ffmpeg"]
    sys.pycache_prefix = str(output / "python_cache")
    policy = LayeredReadPolicy(output, config)
    sys.addaudithook(policy.hook)
    journal = Journal(output, authorize=policy.authorize)
    try:
        for path, digest in {**config["code_hashes"], **config["shared_models"]}.items():
            require(sha(path) == digest, f"Shared dependency changed: {path}")
        sys.path.insert(0, str(Path(config["transport"]).parent))
        from transport import receive_connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(120)
            write(output / "ready.json", {"port": listener.getsockname()[1], "pid": os.getpid()})
            connection, _ = listener.accept()
            with connection:
                received = receive_connection(connection, output / "message.scv")
        write(output / "receive.json", {**received, "pid": os.getpid(), "source": "actual TCP reception"})
        if config.get("wait_for_decode_gate"):
            deadline = time.monotonic() + 3600
            while not (output / "decode_allowed.json").exists():
                require(time.monotonic() < deadline, "Decode scheduling timeout")
                time.sleep(0.25)
            require(read(output / "decode_allowed.json")["allowed"] is True, "Decode gate refused")
        write(output / "decode_started.json", {"started_at": now(), "pid": os.getpid()})
        print("B2 TCP file verified; independently decoding received base and ROI", flush=True)
        from system import decode_packet
        decode_packet(output / "message.scv", output, glc_deployment=config, run_command=journal.run, authorize=policy.authorize)
        write(output / "completion.json", {"success": True, "pid": os.getpid(), "finished_at": now()})
    except BaseException as exc:
        write(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write(output / "receiver_read_audit.json", policy.report())


if __name__ == "__main__":
    main()
