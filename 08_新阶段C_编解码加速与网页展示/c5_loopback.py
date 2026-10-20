"""Run both C5 endpoints on localhost, paired and ready for uploaded videos."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import threading

from c4_config import load_settings as load_runtime
from c5_server import DataServer, Endpoint, Server


def create_endpoints(stack, runtime, directory, sender_port=8773, receiver_port=8774, data_port=8771):
    # Bind every port before touching task stores. Never replace another service.
    sender_server = stack.enter_context(Server(("127.0.0.1", sender_port)))
    receiver_server = stack.enter_context(Server(("127.0.0.1", receiver_port)))
    data_server = stack.enter_context(DataServer(("127.0.0.1", data_port)))
    urls = {"sender": f"http://127.0.0.1:{sender_server.server_port}/",
            "receiver": f"http://127.0.0.1:{receiver_server.server_port}/"}
    apps = {}
    for role, server in (("receiver", receiver_server), ("sender", sender_server)):
        settings = {"role": role, "runtime": runtime, "data_dir": Path(directory) / role,
                    "host": "127.0.0.1", "http_port": server.server_port,
                    "data_port": data_server.server_address[1], "token": "",
                    "peer_url": urls["receiver"].rstrip("/"), "rate_mode": "max",
                    "backend": "native", "compatibility_policy": "strict", "worker_mode": "cold",
                    "loopback_urls": urls}
        app = Endpoint(settings)
        stack.callback(app.close)
        server.app = apps[role] = app
    data_server.app = apps["receiver"]
    # Dedicated local stores may remember preferences; restore the loopback defaults.
    apps["receiver"].preferences({"rate_mode": "max", "compatibility_policy": "strict", "backend": "native"})
    apps["sender"].set_peer(urls["receiver"].rstrip("/"), apps["receiver"].token)
    return (sender_server, receiver_server, data_server), urls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Runtime configuration, as used by C4")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--sender-port", type=int, default=8773)
    parser.add_argument("--receiver-port", type=int, default=8774)
    parser.add_argument("--data-port", type=int, default=8771)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    settings = load_runtime(args.config)
    directory = (args.data_dir or settings["data_dir"].parent / "C5_loopback").resolve()
    missing = [key for key, path in settings["runtime"].items() if not path.exists()]
    if args.check:
        print(json.dumps({"mode": "local_loopback", "missing": missing, "data_dir": str(directory),
                          "gpu_started": False}, ensure_ascii=False))
        return int(bool(missing))
    if missing:
        raise ValueError("运行环境缺少：" + ", ".join(missing))
    with ExitStack() as stack:
        servers, urls = create_endpoints(stack, settings["runtime"], directory,
            args.sender_port, args.receiver_port, args.data_port)
        threads = []
        try:
            for server in servers:
                thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .2}, daemon=True)
                thread.start()
                threads.append((thread, server))
            print("C5_LOOPBACK_READY " + json.dumps(urls), flush=True)
            print("本机回环已配对。选择视频后点击准备文件、开始发送；Ctrl+C 关闭两端。", flush=True)
            if args.open:
                import webbrowser
                webbrowser.open(urls["sender"])
            # Shutting down either HTTP endpoint stops the whole local session.
            while all(thread.is_alive() for thread, _ in threads):
                threads[0][0].join(.25)
        except KeyboardInterrupt:
            pass
        finally:
            for thread, server in threads:
                if thread.is_alive():
                    server.shutdown()
                thread.join(5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
