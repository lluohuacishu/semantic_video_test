"""Portable local demo settings; no CUDA import, installation, or inference."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LOCAL_CONFIG = HERE / 'runtime.local.json'
RUNTIME_KEYS = {'python', 'repo', 'weights', 'ffmpeg', 'native_library'}


def discover_workspace(repository):
    return next((p for p in Path(repository).parents if (p / '01_项目/运行环境').is_dir()), Path(repository))


def resolve_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('配置路径必须是非空字符串')
    path = Path(os.path.expandvars(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_settings(config_path=None, environ=None):
    env = os.environ if environ is None else environ
    explicit = config_path or env.get('GLC_CONFIG')
    path = Path(explicit).expanduser().resolve() if explicit else LOCAL_CONFIG
    if explicit and not path.is_file():
        raise ValueError('找不到指定配置文件：' + str(path))
    raw = json.loads(path.read_text('utf-8-sig')) if path.is_file() else {}
    if not isinstance(raw, dict) or set(raw) - {'runtime', 'workspace', 'data_dir', 'history_enabled'}:
        raise ValueError('配置包含未知字段')
    configured = raw.get('runtime', {})
    if not isinstance(configured, dict) or set(configured) - RUNTIME_KEYS:
        raise ValueError('runtime 配置包含未知字段')
    base = path.parent
    default_workspace = discover_workspace(REPO)
    workspace = resolve_path(env.get('GLC_WORKSPACE', raw.get('workspace', str(default_workspace))), base)
    links = workspace / '01_项目/运行环境'
    bundle = REPO / 'runtime'
    win = os.name == 'nt'
    python_link = links / ('GLC_Python环境/Scripts/python.exe' if win else 'GLC_Python环境/bin/python')
    defaults = {
        'python': python_link if python_link.is_file() else Path(sys.executable),
        'repo': links / 'GLC上游源码' if links.is_dir() else bundle / 'GLC',
        'weights': links / 'GLC权重' if links.is_dir() else bundle / 'checkpoints/glc',
        'ffmpeg': (links / 'FFmpeg' if links.is_dir() else bundle / 'ffmpeg/bin') / ('ffmpeg.exe' if win else 'ffmpeg'),
        'native_library': workspace / '02_实验数据/C_编解码加速与网页展示/C2_native_build_20261019_r3/glc_rans.dll' if links.is_dir() else bundle / ('glc_rans.dll' if win else 'libglc_rans.so'),
    }
    runtime = {key: resolve_path(env.get('GLC_' + key.upper(), configured.get(key, str(value))), base)
               for key, value in defaults.items()}
    default_data = workspace / '02_实验数据/C_编解码加速与网页展示/C3_web_jobs' if links.is_dir() else REPO / 'local_data/jobs'
    data_dir = resolve_path(env.get('GLC_DATA_DIR', raw.get('data_dir', str(default_data))), base)
    history = raw.get('history_enabled', True)
    if type(history) is not bool:
        raise ValueError('history_enabled 必须是 true 或 false')
    return {'runtime': runtime, 'workspace': workspace, 'data_dir': data_dir,
            'history_enabled': history, 'config_path': path if path.is_file() else None}


class UploadCatalog:
    """Allow uploaded videos to run without the large historical asset archive."""
    def __init__(self, reason='未配置历史成果'):
        self.presets = json.loads((REPO / '04_外部依赖快照/semantic-video-training/scripts/glc_presets.json').read_text('utf-8'))
        self.records = []
        self.assets = {}
        self.payload = {'records': [], 'presets': self.presets, 'asset_check': {'total': 0, 'available': 0},
                        'history_available': False, 'history_reason': reason}

    def path(self, key):
        raise FileNotFoundError('本机未配置该示例素材')
