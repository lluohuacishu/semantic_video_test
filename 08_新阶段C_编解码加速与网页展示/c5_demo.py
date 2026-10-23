"""Open the existing local demo, or start its usual loopback service."""
import json
import sys
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


def main():
    if "--check" not in sys.argv:
        try:
            with build_opener(ProxyHandler({})).open("http://127.0.0.1:8773/api/config", timeout=2) as response:
                config = json.load(response)
        except (URLError, TimeoutError, OSError, ValueError):
            config = None
        if config and config.get("version") == "D2-web-demo-v1" and config.get("loopback_urls"):
            import webbrowser
            webbrowser.open("http://127.0.0.1:8773/")
            print("Opened the already-running local console.")
            return 0
    from c5_loopback import main as loopback
    return loopback()


if __name__ == "__main__":
    raise SystemExit(main())
