"""Same-session legacy versus final C1+native combination; explicit short gate first."""
import argparse
from pathlib import Path
import traceback

from run_c0 import HERE, REPO, read, sha
from run_c2 import manifest
from run_c2_paired import run, summarize
from c2_watch import utc, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('gate8', 'paired', 'summary'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-library', type=Path)
    parser.add_argument('--prior-gate', type=Path)
    parser.add_argument('--pairs', type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase == 'summary':
        summarize(root)
        return
    if not args.native_library or not args.prior_gate:
        parser.error('--native-library and --prior-gate are required')
    gate = read(args.prior_gate / 'gate.json')
    previous = read(args.prior_gate / 'manifest.json')
    if not gate['passed'] or previous['native']['sha256'] != sha(args.native_library):
        parser.error('Passed matching-library gate required')
    if args.phase == 'gate8':
        if gate['frames'] != 66:
            parser.error('Candidate must already pass the 66-frame cross-decode gate')
        args.pairs = 1
    else:
        if gate['frames'] != 8 or gate.get('comparison') != 'legacy_vs_c1_native':
            parser.error('Passed combined gate8 required')
        worker = HERE / 'c2_worker.py'
        if previous['code_sha256'][str(worker.relative_to(REPO))] != sha(worker):
            parser.error('Worker changed since the combined gate')
    candidate = HERE / 'c2_native.py'
    if previous['code_sha256'][str(candidate.relative_to(REPO))] != sha(candidate):
        parser.error('Candidate wrapper changed since the gate')
    args.frames = 8 if args.phase == 'gate8' else 125
    root.mkdir(parents=True, exist_ok=False)
    manifest(root, args)
    current = read(root / 'manifest.json')
    current.update(comparison='legacy_vs_c1_native', prior_gate=str(args.prior_gate.resolve()))
    write(root / 'manifest.json', current)
    write(root / 'paired_config.json', {'preset': 'compact', 'pairs': args.pairs, 'frames': args.frames,
          'created_utc': utc(), 'reference_implementation': 'legacy', 'performance_eligible': args.phase == 'paired'})
    write(root / 'stage.json', {'state': 'running', 'started_utc': utc()})
    try:
        run(root, 'compact', args.pairs, frames=args.frames, reference_implementation='legacy')
        summarize(root)
        stopped = (root / 'STOP_REQUESTED').exists()
        if args.phase == 'gate8' and not stopped:
            write(root / 'gate.json', {'passed': True, 'frames': 8, 'comparison': 'legacy_vs_c1_native'})
        write(root / 'stage.json', {'state': 'stopped' if stopped else 'complete', 'ended_utc': utc()})
    except BaseException:
        summarize(root)
        write(root / 'stage.json', {'state': 'interrupted_or_failed', 'ended_utc': utc(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
