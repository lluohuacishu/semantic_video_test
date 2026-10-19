"""Second existing clip: 8-frame check followed by finite compact paired runs."""
import argparse
from functools import partial
from pathlib import Path
import traceback

from run_c0 import HERE, REPO, SOURCE, read, sha
from run_c1 import preview_info
from run_c2 import manifest
from run_c2_paired import run, summarize
from c2_holdout import require_distinct_source, validate
from c2_watch import utc, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('gate8', 'paired', 'summary'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--native-library', type=Path)
    parser.add_argument('--prior-gate', type=Path)
    parser.add_argument('--pairs', type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase == 'summary':
        summarize(root)
        return
    if not args.source or not args.native_library:
        parser.error('--source and --native-library are required')
    source = args.source.resolve(strict=True)
    if source == SOURCE.resolve():
        parser.error('Select a clip different from the main tuning input')
    input_sha = sha(source)
    require_distinct_source(source, input_sha)
    info = preview_info(source)
    a, b = map(int, info['avg_frame_rate'].split('/'))
    if (info['width'], info['height'], int(info['nb_read_frames']), a / b) != (960, 540, 250, 25):
        parser.error('This finite second-clip recipe requires the existing 960x540/25fps/250-frame input')
    if not args.prior_gate:
        parser.error('--prior-gate is required: main-clip gate66 for gate8; second-clip gate8 for paired')
    gate = read(args.prior_gate / 'gate.json')
    old = read(args.prior_gate / 'manifest.json')
    if old['native']['sha256'] != sha(args.native_library):
        parser.error('Native library differs from the gate')
    if old['code_sha256'][str((HERE / 'c2_native.py').relative_to(REPO))] != sha(HERE / 'c2_native.py'):
        parser.error('Candidate changed since the gate')
    if args.phase == 'paired':
        if not gate['passed'] or gate['frames'] != 8 or old['input']['sha256'] != input_sha:
            parser.error('A passed gate on this exact source is required')
        for name in ('c2_native.py', 'c2_holdout.py'):
            if old['code_sha256'][str((HERE / name).relative_to(REPO))] != sha(HERE / name):
                parser.error('Candidate or validator changed since the gate')
    else:
        if not gate['passed'] or gate['frames'] != 66:
            parser.error('A passed 66-frame candidate gate is required before the second clip')
        args.pairs = 1
    args.frames = 8 if args.phase == 'gate8' else 125
    root.mkdir(parents=True, exist_ok=False)
    manifest(root, args, source=source)
    value = read(root / 'manifest.json')
    value.update(validation_target='second_clip_python_reference', source_info=info, prior_gate=str(args.prior_gate.resolve()),
                 scope='Second existing clip unused for C2 candidate selection; capture-source independence not established')
    write(root / 'manifest.json', value)
    write(root / 'paired_config.json', {'preset': 'compact', 'pairs': args.pairs, 'frames': args.frames,
          'source_sha256': input_sha, 'source_path': str(source), 'created_utc': utc(),
          'performance_eligible': args.phase == 'paired'})
    write(root / 'stage.json', {'state': 'running', 'started_utc': utc()})
    try:
        run(root, 'compact', args.pairs, source=source, validator=partial(validate, root), frames=args.frames)
        if sha(source) != input_sha:
            raise ValueError('Source file changed during the experiment')
        summarize(root)
        stopped = (root / 'STOP_REQUESTED').exists()
        if args.phase == 'gate8' and not stopped:
            write(root / 'gate.json', {'passed': True, 'frames': 8, 'source_sha256': input_sha,
                                     'reference_kind': 'Second-clip C1 Python reference, not B4'})
        write(root / 'stage.json', {'state': 'stopped' if stopped else 'complete', 'ended_utc': utc()})
    except BaseException:
        summarize(root)
        write(root / 'stage.json', {'state': 'interrupted_or_failed', 'ended_utc': utc(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
