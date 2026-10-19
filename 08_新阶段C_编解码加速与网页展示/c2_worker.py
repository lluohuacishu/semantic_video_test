"""C1 display worker with optional CPU hotspot scopes and independent heartbeat."""
import argparse
import faulthandler
import json
from pathlib import Path
import sys
import time
import traceback

from c2_watch import Heartbeat, utc, write


class StopFile:
    def __init__(self, path):
        self.path = Path(path)

    def is_set(self):
        return self.path.exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    folder = Path(config['process_folder'])
    with (folder / 'fatal.log').open('w', encoding='utf-8') as fatal, Heartbeat(folder) as beat:
        faulthandler.enable(file=fatal, all_threads=True)
        try:
            # Heartbeat/fault handler are active before importing torch or loading CUDA.
            from c1_runtime import ResidentRunner, TaskCancelled, ref
            from c1_worker import reference_task
            from c0_metrics import Timings
            from run_c0 import gpu_snapshot
            active = {'profile': False, 'timings': Timings(), 'task': None, 'backend': 'python'}
            neural_class = ref.NeuralCodec  # legacy control temporarily patches ref.NeuralCodec

            def measure_entropy(entropy):
                for name in ('indexes', 'encode', 'decode'):
                    original = getattr(entropy, name)
                    def measured(*a, _original=original, _name=name, **kw):
                        if active['profile']:
                            with active['timings'].scope('entropy.' + _name):
                                return _original(*a, **kw)
                        return _original(*a, **kw)
                    setattr(entropy, name, measured)
                return entropy

            def factory(*a, **kw):
                codec = neural_class(*a, **kw)
                codec.c2_backends = {'python': measure_entropy(codec.entropy)}
                original_frame = codec.frame
                def frame(header, index, *a, **kw):
                    if stop.is_set():
                        raise TaskCancelled('Stopped at a neural frame boundary')
                    if active['profile']:
                        with active['timings'].scope('frame'):
                            result = original_frame(header, index, *a, **kw)
                    else:
                        result = original_frame(header, index, *a, **kw)
                    beat.update(last_progress={'task': active['task'], 'frame_completed': index + 1,
                                                'frames': header['frames'], 'utc': utc()})
                    return result
                codec.frame = frame
                return codec

            class BackendRunner(ResidentRunner):
                def acquire(self):
                    codec = super().acquire()
                    backend = active['backend']
                    if backend not in ('python', 'native'):
                        raise ValueError('Unknown entropy backend')
                    if backend not in codec.c2_backends:
                        from c2_native import NativeGaussianRans
                        codec.c2_backends[backend] = measure_entropy(NativeGaussianRans(
                            config['native_library'], codec.c2_backends['python']))
                    codec.entropy = codec.c2_backends[backend]
                    return codec

            runner = BackendRunner(config['role'], config['repo'], config['weights'], config['ffmpeg'],
                                   config.get('device', 'cuda'), codec_factory=factory)
            stop = StopFile(config['stop_file'])
            for task in config['tasks']:
                if stop.is_set():
                    raise TaskCancelled('Stopped before dispatching next task')
                active.update(profile=task.get('profile', False), timings=Timings(), task=task['id'],
                              backend=task.get('backend', 'python'))
                beat.update(state='running', task=task['id'], last_progress=None,
                            task_state={'state': 'running', 'role': config['role']})
                before = gpu_snapshot()
                implementation = task.get('implementation', 'c1')
                if implementation == 'legacy':
                    if active['backend'] != 'python':
                        raise ValueError('Legacy reference must use Python entropy')
                    timing = reference_task(runner, task)
                    status = {'state': 'ready', 'role': config['role'], 'implementation': 'legacy',
                              'task_seconds': timing['task_seconds']}
                    write(Path(task['output']) / 'status.json', status)
                    beat.update(task_state=status)
                elif implementation == 'c1':
                    timing = runner.run(task, cancel_event=stop,
                                        on_event=lambda event: beat.update(task_state=event))
                else:
                    raise ValueError('Unknown pipeline implementation')
                timing.update(run_id=task['id'], group=task.get('group'), pair=task.get('pair'),
                              task_config=task, detailed_instrumentation=active['profile'],
                              entropy_backend=active['backend'], gpu_before=before, gpu_after=gpu_snapshot(),
                              torch_threads=ref.torch.get_num_threads())
                write(Path(task['output']) / 'timing.json', timing)
                write(Path(task['output']) / 'hotspots.json', {
                    'enabled': active['profile'], 'scopes': active['timings'].snapshot(),
                    'units': 'CPU wall seconds; frame includes waiting and transfers',
                    'exclusions': 'entropy scopes exclude preceding .cpu().numpy(), GPU waits and transfers',
                    'not_gpu_busy_time': True, 'run_id': task['id']})
                beat.update(state='task_completed', task=task['id'])
                print('C2_TASK_DONE ' + task['id'] + ' ' + json.dumps(timing['task_seconds']), flush=True)
            beat.update(state='completed', ended_utc=utc())
            return 0
        except BaseException as exc:
            cancelled = type(exc).__name__ == 'TaskCancelled'
            record = {'state': 'cancelled' if cancelled else 'failed', 'ended_utc': utc(),
                      'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
            write(folder / 'exception.json', record)
            beat.update(**record)
            traceback.print_exc()
            return 2 if cancelled else 1
        finally:
            faulthandler.disable()


if __name__ == '__main__':
    sys.exit(main())
