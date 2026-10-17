from pathlib import Path
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

PROJECT = Path('C:/Users/lenovo/Desktop/works/work/semantic-video-mlvc')
LEGACY = Path('C:/Users/lenovo/Desktop/works/work/semantic-video-training')
AUDIT = PROJECT / 'audit/legacy_source_20261013'
AUDIT.mkdir(parents=True, exist_ok=False)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

patterns = ['!**/.git/**', '!**/.venv*/**', '!tools/**', '!data/**',
            '!runs/**', '!checkpoints/**', '!**/__pycache__/**']
command = ['rg', '--files', '--hidden']
for pattern in patterns:
    command += ['-g', pattern]
files = subprocess.check_output(command, cwd=LEGACY, text=True, encoding='utf-8').splitlines()
entries = []
for relative in files:
    source = LEGACY / relative
    target = AUDIT / 'source' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    entries.append({'path': relative.replace('\\', '/'), 'bytes': source.stat().st_size,
                    'sha256': digest(source)})
for name, args in [('git_status.txt', ['status', '--porcelain=v1']),
                   ('git_diff.patch', ['diff', '--binary']),
                   ('git_head.txt', ['rev-parse', 'HEAD'])]:
    data = subprocess.check_output(['git', '-C', str(LEGACY), *args])
    (AUDIT / name).write_bytes(data)
reports = [
    'runs/suim_segformer_b0_20260915/summary.json',
    'runs/suim_segformer_b0_20260915/comparison_with_baseline/summary.json',
    'runs/joint_semantic_validation_20261002_verified/summary.json',
    'runs/joint_semantic_independent_20261002/verification.json',
    'runs/glc_stream_720p_final_20261002/summary.json',
]
for relative in reports:
    source = LEGACY / relative
    target = AUDIT / 'reports' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    entries.append({'path': relative, 'bytes': source.stat().st_size, 'sha256': digest(source)})
plan = Path('C:/Users/lenovo/Documents/Codex/2026-10-12/segformer-c-users-lenovo-desktop-works/outputs/水下视频语义通信_现状与完整实施计划_2026-10-12.md')
shutil.copy2(plan, PROJECT / 'docs/plan_20261012.md')
manifest = {'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'source_root': str(LEGACY), 'snapshot_root': str(AUDIT),
            'legacy_source_modified': False, 'files': entries,
            'plan_sha256': digest(plan)}
(AUDIT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'snapshot': str(AUDIT), 'files': len(entries),
                  'source_bytes': sum(e['bytes'] for e in entries)}, ensure_ascii=False))
