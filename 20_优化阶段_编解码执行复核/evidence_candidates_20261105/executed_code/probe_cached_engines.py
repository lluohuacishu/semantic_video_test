"""Inspect cached P engines on existing real inputs, without changing the codec.

Timing has no layer profiler attached. A separate diagnostic context reports TRT
layer timings afterwards; these are not exclusive CUPTI kernel measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parents[2]
DATA = WORKSPACE / '02_实验数据'
PACKAGES = DATA / '优化_输入编码器推理/engine_packages_1050'
sys.path.insert(0, str(PACKAGES))


def load(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def gpu_status():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,power.draw,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()


def check_frozen():
    records = {}
    for name in (
        '17_优化阶段_解码器推理加速/evidence_trt_20261031/execution_manifest.json',
        '18_优化阶段_解码批量推理/evidence_20261031/execution_manifest.json',
    ):
        for relative, expected in load(REPO / name)['producer_code_sha256'].items():
            actual = sha(REPO / relative)
            if actual != expected:
                raise ValueError('Frozen source differs: ' + relative)
            records[relative] = actual
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('decoder', 'encoder'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    frozen = check_frozen()
    import numpy as np
    import torch
    import torch.nn.functional as F
    import tensorrt as trt

    if trt.__version__ != '10.5.0' or str(torch.__version__) != '2.5.1+cu124':
        raise ValueError('Use the existing pinned runtime')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.utils.deterministic.fill_uninitialized_memory = False
    captured_dir = DATA / '优化_解码批量/batch_pilot_20261031_01/capture'
    captured = load(captured_dir / 'result.json')
    selected = captured['selected_indices']
    files = {}
    if args.role == 'decoder':
        source = captured_dir / 'latents.npy'
        if sha(source) != captured['latents_sha256'] or not captured['reference_verified']:
            raise ValueError('Real latent capture provenance differs')
        array = np.load(source)
        engine_path = DATA / ('优化_解码器推理/decoder_20261031_05/engines/'
            '340bde8808e40b72192150f2efd28d78bfaeda4159c754a818b845b757b7cbdd/generator.engine')
        receipt = load(engine_path.with_name('build.json'))
        input_name, output_name = 'latent', 'rgb'
    else:
        plan = load(DATA / '优化_输入编码器推理/inference_20261030_01/plan.json')
        clip = next(c for c in plan['clips'] if c['id'] == 'long108_q0_640')
        source = Path(clip['source'])
        if sha(source) != clip['source_sha256']:
            raise ValueError('Existing source RGB changed')
        rgb = np.array(np.load(source, mmap_mode='r')[selected])
        array = rgb.transpose(0, 3, 1, 2).astype(np.float32) / 255
        engine_path = DATA / ('优化_输入编码器推理/inference_20261030_01/'
            'engines/640x384/video/trt_fp16.engine')
        receipt = load(engine_path.with_name('trt_fp16_build.json'))
        input_name, output_name = 'x', 'latent'
    files[str(source)] = sha(source)
    engine_sha = sha(engine_path)
    if engine_sha != receipt['engine_sha256']:
        raise ValueError('Cached engine checksum differs')
    files[str(engine_path)] = engine_sha
    initial_gpu = gpu_status()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError('Cannot deserialize the existing engine')
    shape = tuple(engine.get_tensor_shape(input_name))
    output_shape = tuple(engine.get_tensor_shape(output_name))
    if shape[0] != 1 or engine.get_tensor_dtype(output_name) != trt.float32:
        raise ValueError('Unexpected engine tensor contract')
    x = torch.from_numpy(np.ascontiguousarray(array)).cuda()
    if args.role == 'encoder':
        x = F.pad(x, (0, (-x.shape[-1]) % 64, 0, (-x.shape[-2]) % 64), mode='replicate').mul(2).sub(1)
    if tuple(x.shape[1:]) != shape[1:] or len(x) != 20:
        raise ValueError('Expected twenty existing, distinct real P-frame inputs')
    original_x = x.clone()
    stream = torch.cuda.Stream()
    contexts = {mode: engine.create_execution_context() for mode in ('deployed', 'graph')}
    if any(v is None for v in contexts.values()):
        raise RuntimeError('Cannot create execution contexts')
    outputs = {mode: torch.empty(output_shape, device='cuda') for mode in contexts}
    static_x = {mode: x[:1].clone() for mode in contexts}
    for mode, ctx in contexts.items():
        ctx.set_tensor_address(input_name, static_x[mode].data_ptr())
        ctx.set_tensor_address(output_name, outputs[mode].data_ptr())
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            if not contexts['graph'].execute_async_v3(stream.cuda_stream):
                raise RuntimeError('Graph warmup failed')
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        if not contexts['graph'].execute_async_v3(stream.cuda_stream):
            raise RuntimeError('Graph capture failed')
    stream.synchronize()

    enqueue_samples = []

    def forward(value, mode, collect=False):
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            if mode == 'deployed':
                ctx = contexts['deployed']
                ctx.set_tensor_address(input_name, value.data_ptr())
            elif mode == 'fixed':
                static_x['deployed'].copy_(value)
                ctx = contexts['deployed']
                ctx.set_tensor_address(input_name, static_x['deployed'].data_ptr())
            else:
                static_x['graph'].copy_(value)
                ctx = contexts['graph']
            tick = time.perf_counter()
            if mode == 'graph':
                graph.replay()
            elif not ctx.execute_async_v3(stream.cuda_stream):
                raise RuntimeError('Inference failed')
            if collect:
                enqueue_samples.append((time.perf_counter() - tick) * 1000)
            result = outputs['graph' if mode == 'graph' else 'deployed'].clone()
            value.record_stream(stream)
        current.wait_stream(stream)
        result.record_stream(current)
        return result

    mode_checks = {}
    reference = []
    for i in range(len(x)):
        reference.append(forward(x[i:i + 1], 'deployed').cpu().numpy())
    expected = np.concatenate(reference)
    for mode in ('fixed', 'graph'):
        values = np.concatenate([forward(x[i:i + 1], mode).cpu().numpy() for i in range(len(x))])
        if not np.array_equal(expected, values):
            raise ValueError(mode + ' changes engine output')
        mode_checks[mode] = dict(bit_exact_fp32=True, compared_values=int(values.size))
    if not np.isfinite(expected).all():
        raise ValueError('Non-finite engine output')
    old_rgb_match = None
    if args.role == 'decoder':
        h, w = captured['header']['height'], captured['header']['width']
        pixels = np.rint((np.clip(expected[:, :, :h, :w], -1, 1) + 1) * np.float32(127.5))
        pixels = pixels.astype(np.uint8).transpose(0, 2, 3, 1)
        old_path = captured_dir.parent / 'batch1/run1/rgb.npy'
        old_result = load(old_path.with_name('result.json'))
        if sha(old_path) != old_result['rgb_sha256']:
            raise ValueError('Historical Batch1 output changed')
        old_rgb_match = bool(np.array_equal(pixels, np.load(old_path)))
        if not old_rgb_match:
            raise ValueError('Cached-engine output differs from the validated stage18 RGB')
    del reference, expected
    for mode in ('deployed', 'fixed', 'graph'):
        for i in range(4):
            forward(x[i:i + 1], mode)
    torch.cuda.synchronize()
    rows = []
    sequence = ['deployed', 'fixed', 'graph', 'graph', 'fixed', 'deployed'] * 2
    for number, mode in enumerate(sequence):
        before_gpu = gpu_status()
        enqueue_samples.clear()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        tick = time.perf_counter()
        for i in range(len(x)):
            forward(x[i:i + 1], mode, collect=True)
        end.record()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        row = dict(pass_number=number + 1, mode=mode, frames=len(x), wall_seconds=seconds,
            wall_ms_per_frame=seconds * 1000 / len(x), event_ms_per_frame=start.elapsed_time(end) / len(x),
            enqueue_cpu_ms_median=statistics.median(enqueue_samples), gpu_before=before_gpu)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    # Separate profiling pass. Layer event intervals can overlap on TRT auxiliary
    # streams, and enabling profiling perturbs execution; not a speed benchmark.
    inspector = engine.create_engine_inspector()
    layer_info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = layer_info.get('Layers', [])
    # Stage16 was built with LAYER_NAMES_ONLY, which returns string entries.
    # Stage17 DETAILED inspection returns full per-layer dictionaries.
    metadata = {layer['Name']: layer for layer in layers if isinstance(layer, dict)}
    times = defaultdict(list)

    class Profiler(trt.IProfiler):
        def __init__(self):
            trt.IProfiler.__init__(self)

        def report_layer_time(self, name, ms):
            times[name].append(float(ms))

    profiler = Profiler()
    ctx = engine.create_execution_context()
    ctx.profiler = profiler
    ctx.enqueue_emits_profile = True
    ctx.set_tensor_address(output_name, outputs['deployed'].data_ptr())
    for i in (0, 1, 2, 12, 18, 19):
        ctx.set_tensor_address(input_name, x[i:i + 1].data_ptr())
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('Profiling execution failed')
        stream.synchronize()
    profile_rows = []
    for name, values in times.items():
        info = metadata.get(name, {})
        kind = info.get('LayerType', 'unknown')
        profile_rows.append(dict(name=name, layer_type=kind, observations=len(values),
            mean_ms=statistics.mean(values), median_ms=statistics.median(values),
            inputs=info.get('Inputs', []), tactic=info.get('TacticName', '')))
    profile_rows.sort(key=lambda r: r['mean_ms'], reverse=True)
    if not profile_rows:
        raise RuntimeError('TensorRT did not emit layer profiles')
    totals = defaultdict(float)
    protected = 0.
    for row in profile_rows:
        totals[row['layer_type']] += row['mean_ms']
        if any(f'/blocks.{block}/conv/Conv' in row['name'] for block in (10, 14, 18, 22)) or row['name'] == '/blocks.28/Conv':
            protected += row['mean_ms']
    summaries = {}
    for mode in ('deployed', 'fixed', 'graph'):
        selected_rows = [r for r in rows if r['mode'] == mode]
        summaries[mode] = dict(median_ms_per_frame=statistics.median(r['wall_ms_per_frame'] for r in selected_rows),
            all_ms_per_frame=[r['wall_ms_per_frame'] for r in selected_rows],
            median_enqueue_cpu_ms=statistics.median(r['enqueue_cpu_ms_median'] for r in selected_rows))
    for value in summaries.values():
        value['speedup_vs_deployed'] = summaries['deployed']['median_ms_per_frame'] / value['median_ms_per_frame']
    if not torch.equal(original_x, x) or check_frozen() != frozen:
        raise ValueError('Input tensors or frozen sources changed')
    result = dict(role=args.role, shape=list(shape), output_shape=list(output_shape),
        inputs=20, source_indices=selected, engine_sha256=engine_sha,
        engine_context_bytes=int(engine.device_memory_size), engine_num_aux_streams=engine.num_aux_streams,
        versions=dict(torch=str(torch.__version__), tensorrt=trt.__version__, cuda=torch.version.cuda),
        summaries=summaries, timings=rows, exact_output_checks=mode_checks,
        historical_decoder_rgb_bit_exact=old_rgb_match, source_tensors_unchanged=True,
        profile=dict(frames=6, layers=profile_rows, layer_type_ms=dict(totals),
            summed_layer_mean_ms=sum(r['mean_ms'] for r in profile_rows), protected_fp32_conv_mean_ms=protected,
            warning='TRT layer event intervals; auxiliary-stream overlap and profiler perturbation. Not exclusive GPU kernel time.'),
        inspector_layer_type_counts=dict(Counter(r.get('LayerType', 'unknown') if isinstance(r, dict) else 'name_only' for r in layers)),
        inspector_profiling_verbosity=str(engine.profiling_verbosity),
        gpu_initial=initial_gpu, gpu_final=gpu_status(), provenance=files, frozen_code_sha256=frozen,
        producer_sha256=sha(__file__), full_codec_test=False, reference_path_modified=False,
        weights_changed=False, engine_rebuilt=False, new_precision=False,
        scope='P-module only, existing GPU-resident real inputs. All paths include private output cloning; fixed/graph include a static input copy. No entropy/reference/RGB-to-CPU/video I/O in timing.')
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (out / 'layers.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=('name', 'layer_type', 'observations', 'mean_ms', 'median_ms', 'inputs', 'tactic'))
        writer.writeheader()
        writer.writerows(profile_rows)
    print('PASS ' + json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    main()
