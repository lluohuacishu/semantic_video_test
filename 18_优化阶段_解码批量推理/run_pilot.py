"""Run one capture and six small, sequential generator jobs; no web changes or training."""
from __future__ import annotations
import subprocess
import time
from batch_common import *


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,clocks.current.graphics,temperature.gpu,power.draw',
                                    '--format=csv'], text=True).strip()


def run(script, folder, args):
    result = folder / 'result.json'
    if not result.exists():
        idle();folder.parent.mkdir(parents=True, exist_ok=True)
        command = [str(load_settings()['runtime']['python']), '-X', 'utf8', '-u', str(HERE/script), *args]
        launch = dict(argv=command, code_sha256=code_hashes(), gpu_before=gpu_snapshot(), started_unix=time.time())
        save_json(folder.with_suffix('.launch.json'), launch)
        with folder.with_suffix('.log').open('w', encoding='utf-8') as log:
            process = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
        launch.update(returncode=process.returncode, ended_unix=time.time(), gpu_after=gpu_snapshot())
        save_json(folder.with_suffix('.launch.json'), launch)
        if process.returncode:
            raise RuntimeError(f'Job failed ({process.returncode}): {folder.with_suffix(".log")}')
    value = load_json(result)
    if value['code_sha256'] != code_hashes():
        raise ValueError('Cached result has different producer code')
    return value


def main():
    capture = run('capture_latents.py', TRIAL/'capture', [])
    print(f'Capture: {capture["frames_decoded"]} decoded references verified; {len(capture["selected_indices"])} real P frames', flush=True)
    reports, failures = [], []
    for repeat, order in ((1, (1, 2, 4)), (2, (4, 2, 1))):
        for batch in order:
            folder = TRIAL/f'batch{batch}'/f'run{repeat}'
            try:
                value = run('benchmark_batch.py', folder, ['--batch', str(batch), '--repeat', str(repeat)])
                reports.append(dict(batch=batch, repeat=repeat, folder=str(folder), timings=value['timings']))
                print(f'PASS batch={batch} round={repeat}', flush=True)
            except Exception as error:
                failures.append(dict(batch=batch, repeat=repeat, folder=str(folder), error=str(error)))
                print(f'FAIL batch={batch} round={repeat}: {error}', flush=True)
            save_json(TRIAL/'progress.json', dict(reports=reports, failures=failures))
    save_json(TRIAL/'campaign.json', dict(complete=True, reports=reports, failures=failures,
        scope='P generator only, 640x360 q0, 20 real latents, two process rounds in opposite order; three timed passes/mode/round',
        source_stream_reencoded=False, decoder_interface_changed=False, web_changed=False, training_steps=0))
    if failures:
        raise RuntimeError('Pilot contains failed configurations; inspect preserved logs')


if __name__ == '__main__':main()
