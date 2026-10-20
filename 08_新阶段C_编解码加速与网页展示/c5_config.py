"""Separate, portable endpoint settings. No CUDA import or model startup."""
import json
from pathlib import Path
from urllib.parse import urlsplit

from c4_config import HERE, RUNTIME_KEYS, load_settings as load_runtime, resolve_path
from c5_compat import POLICIES
from c5_transport import rate_limit


def peer_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("接收端地址须为 http://主机:端口")
    port = parsed.port or 80
    if not 1 <= port <= 65535:
        raise ValueError("无效端口")
    return value.rstrip("/")


def load_settings(role, config=None):
    if role not in ("sender", "receiver"):
        raise ValueError("Unknown endpoint role")
    path = Path(config).resolve() if config else HERE / ("c5_" + role + ".local.json")
    if config and not path.is_file():
        raise ValueError("找不到指定端点配置")
    raw = json.loads(path.read_text("utf-8-sig")) if path.is_file() else {}
    allowed = {"base_config", "runtime", "data_dir", "host", "http_port", "data_port", "token",
               "peer_url", "rate_mode", "backend", "compatibility_policy", "worker_mode"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("端点配置包含未知字段")
    base_config = resolve_path(raw["base_config"], path.parent) if raw.get("base_config") else None
    defaults = load_runtime(base_config, environ={})
    configured = raw.get("runtime", {})
    if not isinstance(configured, dict) or set(configured) - RUNTIME_KEYS:
        raise ValueError("runtime 配置包含未知字段")
    runtime = {**defaults["runtime"], **{k: resolve_path(v, path.parent) for k, v in configured.items()}}
    result = {"role": role, "runtime": runtime,
              "data_dir": defaults["data_dir"].parent / ("C5_" + role + "_jobs"),
              "host": "127.0.0.1" if role == "sender" else "0.0.0.0",
              "http_port": 8773 if role == "sender" else 8774, "data_port": 8771,
              "token": "", "peer_url": "http://127.0.0.1:8774", "rate_mode": "max",
              "backend": "native", "compatibility_policy": "strict", "worker_mode": "cold"}
    result.update({k: v for k, v in raw.items() if k not in ("runtime", "data_dir", "base_config")})
    if "data_dir" in raw:
        result["data_dir"] = resolve_path(raw["data_dir"], path.parent)
    for key in ("http_port", "data_port"):
        if type(result[key]) is not int or not 1 <= result[key] <= 65535:
            raise ValueError("无效端口：" + key)
    if not isinstance(result["host"], str) or not result["host"]:
        raise ValueError("无效监听地址")
    if not isinstance(result["token"], str) or (result["token"] and len(result["token"]) < 24):
        raise ValueError("连接口令至少 24 字符")
    if result["backend"] not in ("native", "python") or result["compatibility_policy"] not in POLICIES:
        raise ValueError("无效后端或设备配对策略")
    if result["worker_mode"] not in ("cold", "resident"):
        raise ValueError("无效工作进程模式")
    rate_limit(result["rate_mode"])
    result["peer_url"] = peer_url(result["peer_url"])
    result["config_path"] = path
    return result
