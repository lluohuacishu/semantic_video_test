"""Two fixed clips and one higher-rate setting; original artifacts are read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / '08_新阶段C_编解码加速与网页展示'))
from c4_config import load_settings
from temporal_filter import description


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    settings = load_settings()
    old = settings['workspace'] / '02_实验数据/D_语义目标增强/D21_20261022_r1'
    root.mkdir(parents=True, exist_ok=False)
    runtime = {k: str(v) for k, v in settings['runtime'].items()}
    encode, decode, clips = [], [], []
    for number, q in ((11, 12), (21, 17)):
        config = read(old / f'v{number:03d}_512x288_A_q{q:02d}_encode_config.json')
        original = config['tasks'][0]
        baseline = Path(original['output'])
        for path in original['frames']:
            if not Path(path).is_file(): raise FileNotFoundError(path)
        source = root / f'v{number:03d}'
        source.mkdir()
        shutil.copy2(baseline / 'stream.glcv', source / 'baseline.glcv')
        shutil.copy2(baseline / 'encode.json', source / 'baseline_encode.json')
        enc = {**original, 'id': f'v{number:03d}_high_q32', 'q': 32,
               'output': str(source / 'high_encoder')}
        encode.append(enc)
        for variant, stream in (('baseline', source / 'baseline.glcv'),
                                ('high', source / 'high_encoder/stream.glcv')):
            decode.append(dict(id=f'v{number:03d}_{variant}', input=str(stream),
                               output=str(source / f'{variant}_decoder')))
        clips.append(dict(video_id=number, baseline_q=q, high_q=32, frames=original['frames'],
            source_sha256=[sha(p) for p in original['frames']], width=512, height=288,
            fps=original['fps'], count=64, baseline_origin=str(baseline),
            baseline_sha256=sha(source / 'baseline.glcv'),
            crop=[0, 192, 288, 288] if number == 11 else [144, 0, 432, 96]))
    worker = REPO / '09_新阶段D_语义目标增强/d2_codec_worker.py'
    for role, tasks in (('encode', encode), ('decode', decode)):
        write(root / f'{role}_config.json', dict(phase=f'temporal_{role}', role=role,
            run_root=str(root), runtime=runtime, tasks=tasks, record_legacy_q32_anchor=False))
    write(root / 'plan.json', dict(date='2026-10-27', clips=clips, filter=description(),
        fixed_before_run=True, rate_policy='single fixed q32; report actual bytes, not equal-rate comparison',
        evaluation='lossless decoded RGB; source-motion-compensated temporal residual error plus spatial error and viewing',
        source_used_only_for_offline_evaluation=True, no_parameter_sweep=True,
        scope='two existing development clips; no independent test set, no Orin, no semantic treatment, no FEC',
        runtime=runtime, worker=str(worker), worker_sha256=sha(worker)))
    for path in HERE.glob('*.py'):
        shutil.copy2(path, root / (path.stem + '.snapshot.py'))
    return runtime, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    runtime, worker = prepare(root)
    for role in ('encode', 'decode'):
        print(f'START {role}', flush=True)
        with (root / f'{role}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run([runtime['python'], '-u', str(worker), '--config',
                str(root / f'{role}_config.json')], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONUTF8': '1'})
        if result.returncode:
            raise RuntimeError(f'{role} failed; inspect {root / (role + ".log")}')
        print(f'DONE {role}', flush=True)
    print('CODEC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
