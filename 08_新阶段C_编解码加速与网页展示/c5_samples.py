"""Existing local source clips available to the sender."""
from c4_config import REPO, discover_workspace

D_ROOT = discover_workspace(REPO) / "02_实验数据/D_语义目标增强"


def sample_path(key):
    if key not in ("011", "021"):
        raise ValueError("未知示例视频")
    path = D_ROOT / "D2_lowrate_20261022_r1/sources" / ("video_" + key + ".mp4")
    if not path.is_file():
        raise FileNotFoundError("本机没有该示例原视频，请上传自己的视频")
    return path
