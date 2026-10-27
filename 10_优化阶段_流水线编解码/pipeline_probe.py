"""Bounded local TCP proof with separate real GLC encoder/decoder processes."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / '08_新阶段C_编解码加速与网页展示'))
import stream_protocol as p
from glc_container import StreamWriter, read_stream
from c4_config import load_settings
WORK = load_settings()['workspace']


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Events:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()

    def __call__(self, name, **fields):
        now = (time.perf_counter_ns() - self.args.origin) / 1e9
        row = {'event': name, 'seconds': now, **fields}
        with self.lock, (self.args.output / 'events.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
        return now


class ChannelQueue:
    def __init__(self):
        self.queue = queue.Queue(maxsize=2)
        self.error = None
        self.stop = threading.Event()
        self.peak = 0

    def check(self):
        if self.error:
            raise RuntimeError('Network worker failed') from self.error
        if self.stop.is_set():
            raise InterruptedError('Pipeline stopped')

    def put(self, value):
        while True:
            self.check()
            try:
                self.queue.put(value, timeout=0.1)
                self.peak = max(self.peak, self.queue.qsize())
                return
            except queue.Full:
                pass

    def get(self):
        while True:
            self.check()
            try:
                return self.queue.get(timeout=0.1)
            except queue.Empty:
                pass

    def guard(self, fn):
        try:
            fn()
        except BaseException as exc:
            self.error = exc


def load_codec(role):
    import glc_video_codec as ref
    from c2_native import NativeGaussianRans
    runtime = load_settings()['runtime']
    codec = ref.NeuralCodec(runtime['repo'], runtime['weights'], 'cuda', decoder_only=role == 'receiver')
    codec.entropy = NativeGaussianRans(runtime['native_library'], codec.entropy)
    ref.torch.cuda.synchronize()
    ref.torch.cuda.reset_peak_memory_stats()
    return ref, codec


def input_reader(args, count):
    from c1_io import MemoryFrames
    return MemoryFrames(argparse.Namespace(input=args.input, frames=count,
        width=640, start_frame=0, frame_stride=2))


def make_header(reader, codec, count, ref):
    meta = reader.metadata()
    header = {key: meta[key] for key in ('width', 'height', 'fps')}
    header.update(frames=count, q=0, intra_period=64, reset_interval=32,
        revision=ref.REVISION, entropy=ref.ENTROPY_ID, arithmetic=codec.profile,
        table_sha256=codec.entropy.table_sha256, container_version=2,
        weights={key: value['sha256'] for key, value in ref.ASSETS.items()})
    return header


def input_tensor(source, header, ref):
    tensor = ref.torch.from_numpy(source.copy()).to('cuda').permute(2, 0, 1).unsqueeze(0).float().div(255)
    return ref.F.pad(tensor, (0, -header['width'] % 64, 0, -header['height'] % 64), mode='replicate').mul(2).sub(1)


def warmup(args, role, ref, codec, events):
    if not args.warmup_frames:
        return
    events('warmup_start', frames=args.warmup_frames)
    with ref.torch.inference_mode():
        if role == 'sender':
            reader = input_reader(args, args.warmup_frames)
            try:
                for i, source in enumerate(reader):
                    header = make_header(reader, codec, args.warmup_frames, ref)
                    codec.frame(header, i, x=input_tensor(source, header, ref))
                if reader.count != args.warmup_frames:
                    raise ValueError('Insufficient input for warmup')
            finally:
                reader.close()
        else:
            header, packets = read_stream(args.warmup_stream)
            if len(packets) < args.warmup_frames:
                raise ValueError('Insufficient warmup stream frames')
            for i, packet in enumerate(packets[:args.warmup_frames]):
                codec.frame(header, i, packet=packet)
    ref.torch.cuda.synchronize()
    codec.dpb = {'ref_latent': None, 'ref_y': None}
    ref.torch.cuda.reset_peak_memory_stats()
    events('warmup_end_reference_reset')


def sender(args):
    events = Events(args)
    events('model_load_start')
    ref, codec = load_codec('sender')
    events('model_ready')
    warmup(args, 'sender', ref, codec, events)
    reader = input_reader(args, args.frames)
    pending = ChannelQueue()
    stats = {'forward_bytes': 0, 'reverse_bytes': 0}
    records, digest = [], hashlib.sha256()
    writer = None
    buffered = []
    buffered_bytes = 0
    def emit(kind, index, body):
        nonlocal buffered_bytes
        if args.mode == 'serial':
            buffered_bytes += len(body)
            if buffered_bytes > 256 * 1024 * 1024:
                raise ValueError('Serial comparison buffer limit exceeded')
            buffered.append((kind, index, body))
        else:
            pending.put((kind, index, body))
    with socket.create_connection(('127.0.0.1', args.port), timeout=120) as conn:
        conn.settimeout(120)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
        kind, index, ready, count = p.receive_record(conn)
        if (kind, index) != (p.READY, 0) or json.loads(ready) != {'ready': True}:
            raise ValueError('Receiver not ready')
        stats['reverse_bytes'] += count

        def transmit():
            deadline = time.perf_counter()
            while True:
                kind, index, body = pending.get()
                raw = p.encode_record(kind, index, body)
                events('send_start', kind=kind, frame=index + 1 if kind == p.FRAME else None)
                for offset in range(0, len(raw), 256):
                    chunk = raw[offset:offset + 256]
                    if args.kbps:
                        deadline = max(deadline, time.perf_counter()) + len(chunk) * 8 / (args.kbps * 1000)
                        if pending.stop.wait(max(0, deadline - time.perf_counter())):
                            raise InterruptedError('Cancelled during pacing')
                    conn.sendall(chunk)
                stats['forward_bytes'] += len(raw)
                events('send_end', kind=kind, frame=index + 1 if kind == p.FRAME else None,
                       bytes=len(raw))
                if kind == p.END:
                    conn.shutdown(socket.SHUT_WR)
                    k, i, body, count = p.receive_record(conn)
                    if (k, i) != (p.READY, args.frames):
                        raise ValueError('Invalid completion acknowledgement')
                    stats['ack'] = json.loads(body)
                    stats['reverse_bytes'] += count
                    events('acknowledged')
                    break

        thread = threading.Thread(target=lambda: pending.guard(transmit), daemon=True)
        thread.start()
        try:
            first = next(reader)
            header = make_header(reader, codec, args.frames, ref)
            emit(p.HEADER, 0, p.json_bytes(header))
            writer = StreamWriter(args.output / 'encoded.glcv', header, version=2)
            with ref.torch.inference_mode():
                for index, source in enumerate(itertools.chain((first,), reader)):
                    pending.check()
                    start = events('encode_start', frame=index + 1)
                    rgb, packet = codec.frame(header, index, x=input_tensor(source, header, ref))
                    end = events('encode_end', frame=index + 1)
                    body = p.encode_frame(header, index, packet)
                    digest.update(body)
                    writer.add(packet)
                    put_start = time.perf_counter()
                    emit(p.FRAME, index, body)
                    records.append({'frame': index + 1, 'start': start, 'end': end,
                        'queue_put_seconds': time.perf_counter() - put_start,
                        'payload_bytes': len(packet['z']) + sum(map(len, packet['y'])),
                        'rgb_sha256': packet['rgb_sha256'], 'latent_sha256': packet['latent_sha256'],
                        'source_rgb_sha256': reader.hashes[index]})
            if len(records) != args.frames:
                raise ValueError('Input ended before the declared frame count')
            writer.finish()
            file_sha = sha(args.output / 'encoded.glcv')
            emit(p.END, args.frames, p.json_bytes({'frames': args.frames,
                'frame_records_sha256': digest.hexdigest(), 'archive_sha256': file_sha}))
            for item in buffered:
                pending.put(item)
            buffered.clear()
            thread.join(timeout=120)
            if thread.is_alive():
                raise TimeoutError('Sender network worker did not finish')
            pending.check()
            if stats['ack'] != {'verified': True, 'archive_sha256': file_sha}:
                raise ValueError('Receiver completion mismatch')
            dump(args.output / 'result.json', {'header': header, 'frames': records, **stats,
                'queue_peak': pending.peak, 'queue_capacity': 2, 'archive_sha256': file_sha,
                'archive_bytes': (args.output / 'encoded.glcv').stat().st_size,
                'backend': type(codec.entropy).__name__, 'source': reader.metadata(),
                'peak_allocated_mib': ref.torch.cuda.max_memory_allocated() / 2**20})
        finally:
            pending.stop.set()
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            thread.join(timeout=2)
            reader.close()
            if writer:
                writer.close()


def receiver(args):
    from c1_runtime import receiver_audit
    events = Events(args)
    events('model_load_start')
    ref, codec = load_codec('receiver')
    events('model_ready')
    warmup(args, 'receiver', ref, codec, events)
    pending = ChannelQueue()
    stats = {'forward_bytes': 0}
    records, digest = [], hashlib.sha256()
    writer = None
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(120)
        dump(args.output / 'ready.json', {'port': listener.getsockname()[1]})
        conn, _ = listener.accept()
        with conn:
            conn.settimeout(120)
            conn.sendall(p.encode_record(p.READY, 0, p.json_bytes({'ready': True})))
            kind, index, body, count = p.receive_record(conn)
            if (kind, index) != (p.HEADER, 0):
                raise ValueError('Missing initial header')
            header = p.decode_header(body)
            if (header['arithmetic'] != codec.profile or header['table_sha256'] != codec.entropy.table_sha256
                    or header['revision'] != ref.REVISION or header['entropy'] != ref.ENTROPY_ID
                    or header['weights'] != {key: value['sha256'] for key, value in ref.ASSETS.items()}):
                raise ValueError('Incompatible decoder configuration')
            stats['forward_bytes'] += count
            events('header_verified')

            def ingest():
                for expected in range(header['frames'] + 1):
                    want = p.FRAME if expected < header['frames'] else p.END
                    body, count = p.receive_expected(conn, want, expected)
                    kind, index = want, expected
                    stats['forward_bytes'] += count
                    at = events('frame_received_verified' if kind == p.FRAME else 'end_received',
                                frame=index + 1 if kind == p.FRAME else None)
                    if kind == p.END and conn.recv(1):
                        raise ValueError('Trailing bytes after final record')
                    pending.put((kind, index, body, at))

            thread = threading.Thread(target=lambda: pending.guard(ingest), daemon=True)
            thread.start()
            try:
                writer = StreamWriter(args.output / 'received.pending.glcv', header, version=2)
                with receiver_audit() as reads, ref.torch.inference_mode():
                    serial_records = []
                    if args.mode == 'serial':
                        size = 0
                        for _ in range(header['frames'] + 1):
                            item = pending.get()
                            size += len(item[2])
                            if size > 256 * 1024 * 1024:
                                raise ValueError('Serial receiver buffer limit exceeded')
                            serial_records.append(item)
                        trailer = json.loads(serial_records[-1][2])
                        transcript = hashlib.sha256()
                        for item in serial_records[:-1]:
                            transcript.update(item[2])
                        p.validate_trailer(trailer, header['frames'], transcript.hexdigest())
                    saved = iter(serial_records)
                    def take():
                        return next(saved) if args.mode == 'serial' else pending.get()
                    for index in range(header['frames']):
                        kind, sequence, body, received_at = take()
                        packet = p.decode_frame(header, index, body)
                        digest.update(body)
                        start = events('decode_start', frame=index + 1)
                        rgb, checked = codec.frame(header, index, packet=packet)
                        end = events('decode_end', frame=index + 1)
                        writer.add(checked)
                        records.append({'frame': index + 1, 'received_verified': received_at,
                            'start': start, 'end': end, 'rgb_sha256': checked['rgb_sha256'],
                            'latent_sha256': checked['latent_sha256']})
                    kind, sequence, body, _ = take()
                    trailer = json.loads(body)
                    p.validate_trailer(trailer, len(records), digest.hexdigest())
                    writer.finish()
                    file_sha = sha(args.output / 'received.pending.glcv')
                    if file_sha != trailer['archive_sha256']:
                        raise ValueError('Final archival file hash mismatch')
                    read_stream(args.output / 'received.pending.glcv')
                    (args.output / 'received.pending.glcv').rename(args.output / 'received.glcv')
                    assert reads == []
                thread.join(timeout=5)
                if thread.is_alive():
                    raise TimeoutError('Receiver network worker did not finish')
                pending.check()
                conn.sendall(p.encode_record(p.READY, len(records), p.json_bytes(
                    {'verified': True, 'archive_sha256': file_sha})))
                dump(args.output / 'result.json', {'header': header, 'frames': records, **stats,
                    'queue_peak': pending.peak, 'queue_capacity': 2, 'source_media_reads': reads,
                    'archive_sha256': file_sha, 'backend': type(codec.entropy).__name__,
                    'peak_allocated_mib': ref.torch.cuda.max_memory_allocated() / 2**20})
                events('complete_verified')
            finally:
                pending.stop.set()
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                thread.join(timeout=2)
                if writer:
                    writer.close()


def orchestrate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    for role in ('sender', 'receiver'):
        (args.output / role).mkdir()
    for name in ('pipeline_probe.py', 'stream_protocol.py', 'test_stream_protocol.py'):
        shutil.copy2(HERE / name, args.output / name)
    origin = time.perf_counter_ns()
    dump(args.output / 'manifest.json', {'date': '2026-10-27', 'input': str(args.input),
        'frames': args.frames, 'application_rate_kbps': args.kbps, 'ordinary_glc': True,
        'precision': 'FP32, deterministic, TF32 disabled', 'warmup_frames': args.warmup_frames,
        'warmup_stream': str(args.warmup_stream) if args.warmup_frames else None,
        'transport_version': 2, 'mode': args.mode, 'preview': False,
        'topology': 'two separate processes, one shared RTX 4060 Laptop GPU, localhost TCP',
        'scope': 'same-host scheduling measurement; not an acoustic-channel/FEC or Orin test',
        'clock': 'same-host perf_counter_ns; model initialization recorded separately',
        'queue_capacity_per_stage': 2,
        'code_sha256': {name: sha(args.output / name) for name in ('pipeline_probe.py', 'stream_protocol.py')}})
    processes, handles = [], []
    def launch(role, extra):
        handle = (args.output / role / 'worker.log').open('w', encoding='utf-8')
        handles.append(handle)
        cmd = [sys.executable, str(HERE / 'pipeline_probe.py'), role, '--output', str(args.output / role),
               '--origin', str(origin), '--mode', args.mode, '--warmup-frames', str(args.warmup_frames),
               '--warmup-stream', str(args.warmup_stream), *extra]
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                env={**os.environ, 'PYTHONUTF8': '1'})
        processes.append(proc)
        return proc
    try:
        recv = launch('receiver', [])
        ready = args.output / 'receiver/ready.json'
        deadline = time.monotonic() + 120
        while not ready.is_file():
            if recv.poll() is not None:
                raise RuntimeError('Receiver initialization failed; see worker.log')
            if time.monotonic() > deadline:
                raise TimeoutError('Receiver initialization timeout')
            time.sleep(0.1)
        port = json.loads(ready.read_text('utf-8'))['port']
        send = launch('sender', ['--input', str(args.input), '--frames', str(args.frames),
                                '--port', str(port), '--kbps', str(args.kbps)])
        deadline = time.monotonic() + 240
        while any(proc.poll() is None for proc in processes):
            if any(proc.poll() not in (None, 0) for proc in processes):
                raise RuntimeError('Pipeline child failed; see worker logs')
            if time.monotonic() > deadline:
                raise TimeoutError('Pipeline exceeded bounded trial time')
            time.sleep(0.1)
        a, b = (json.loads((args.output / role / 'result.json').read_text('utf-8')) for role in ('sender', 'receiver'))
        assert a['archive_sha256'] == b['archive_sha256'] and a['forward_bytes'] == b['forward_bytes']
        assert len(a['frames']) == len(b['frames']) == args.frames and b['source_media_reads'] == []
        assert all(x[key] == y[key] for x, y in zip(a['frames'], b['frames'])
                   for key in ('rgb_sha256', 'latent_sha256'))
        equal_baseline = None
        if args.baseline_root:
            _, baseline_packets = read_stream(args.baseline_root / 'stream.glcv')
            _, packets = read_stream(args.output / 'sender/encoded.glcv')
            baseline_frames = json.loads((args.baseline_root / 'encoder/encode.json').read_text('utf-8'))['frames']
            equal_baseline = all(packet[key] == baseline_packets[i][key]
                                 for i, packet in enumerate(packets) for key in ('z', 'y'))
            equal_baseline &= all(row[key] == baseline_frames[i][key]
                                  for i, row in enumerate(a['frames']) for key in ('rgb_sha256', 'latent_sha256'))
            assert equal_baseline
        start = a['frames'][0]['start']
        first = b['frames'][0]['end']
        last_encode = a['frames'][-1]['end']
        last_receive = b['frames'][-1]['received_verified']
        receiver_events = [json.loads(line) for line in (args.output / 'receiver/events.jsonl').read_text('utf-8').splitlines()]
        sender_events = [json.loads(line) for line in (args.output / 'sender/events.jsonl').read_text('utf-8').splitlines()]
        completed = next(r['seconds'] for r in receiver_events if r['event'] == 'complete_verified')
        acknowledged = next(r['seconds'] for r in sender_events if r['event'] == 'acknowledged')
        result = {'verified': True, 'frames': args.frames, 'mode': args.mode,
            'application_rate_kbps': args.kbps, 'warmup_frames': args.warmup_frames,
            'identical_to_baseline_prefix': equal_baseline,
            'independent_receiver_no_source_reads': True, 'archives_identical': True,
            'first_decoded_before_encode_finished': first < last_encode,
            'first_decoded_before_last_frame_received': first < last_receive,
            'seconds_from_first_encode_start': {'first_frame_decoded': first - start,
                'encoding_finished': last_encode - start, 'last_frame_received': last_receive - start,
                'last_frame_decoded': b['frames'][-1]['end'] - start,
                'receiver_verified': completed - start, 'sender_acknowledged': acknowledged - start},
            'archival_bytes': a['archive_bytes'], 'forward_application_bytes': a['forward_bytes'],
            'reverse_application_bytes': a['reverse_bytes'],
            'added_forward_bytes_over_glcv': a['forward_bytes'] - a['archive_bytes'],
            'queue_peaks': {'sender': a['queue_peak'], 'receiver': b['queue_peak']},
            'peak_allocated_mib': {'sender': a['peak_allocated_mib'], 'receiver': b['peak_allocated_mib']},
            'archive_sha256': a['archive_sha256'],
            'encode_active_seconds': sum(r['end'] - r['start'] for r in a['frames']),
            'decode_active_seconds': sum(r['end'] - r['start'] for r in b['frames']),
            'limitations': ['same GPU contention', 'paired comparison requires matching other runs',
                           'no FEC, packet loss or real acoustic channel', 'web not integrated']}
        dump(args.output / 'summary.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=('run', 'sender', 'receiver'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path, default=WORK / '02_实验数据/A_压缩路线选型/A2_GLC验证/long108/input_540.mkv')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--kbps', type=float, default=40)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--origin', type=int, default=0)
    parser.add_argument('--mode', choices=('serial', 'pipeline'), default='pipeline')
    parser.add_argument('--warmup-frames', type=int, default=0)
    parser.add_argument('--warmup-stream', type=Path, default=WORK / '02_实验数据/C_编解码加速与网页展示/GLC_precision_20261026_native_05/fp32/encode/run_1/stream.glcv')
    parser.add_argument('--baseline-root', type=Path)
    args = parser.parse_args()
    if not 1 <= args.frames <= 66 or not 0 <= args.kbps <= 10000:
        parser.error('Bounded probe requires 1..66 frames and 0..10000 kbps')
    if not 0 <= args.warmup_frames <= 8:
        parser.error('Warmup is limited to 0..8 frames')
    if args.warmup_frames and not args.warmup_stream.is_file():
        parser.error('Warmup requires a local known-good compressed stream')
    try:
        {'run': orchestrate, 'sender': sender, 'receiver': receiver}[args.role](args)
    except BaseException as exc:
        if args.output.is_dir():
            dump(args.output / 'failure.json', {'type': type(exc).__name__, 'error': str(exc),
                                               'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
