"""Publish compact decoder measurements; engines, weights and RGB remain in local data."""
from __future__ import annotations

import csv
from collections import Counter
from decoder_common import *


def main():
    target = HERE / 'evidence_trt_20261031'
    target.mkdir(exist_ok=True)
    measurements, qualities, receipts, engines = [], [], {}, {}
    for suite in ('short', 'multi', 'full'):
        summary = load_json(TRIAL / (suite + '_summary.json'))
        quality = load_json(TRIAL / (suite + '_quality.json'))
        if not summary['complete'] or not quality['complete']:
            raise ValueError('Cannot publish an incomplete decoder campaign')
        for row in summary['runs']:
            path = Path(row['folder']) / 'result.json'
            result = load_json(path)
            if result['code_sha256'] != code_hashes() or not result['reference_verified']:
                raise ValueError('Producer code or reference verification differs')
            receipts[str(path.relative_to(TRIAL))] = dict(sha256=file_sha(path), bytes=path.stat().st_size)
            measurements.append(dict(suite=suite, clip=row.get('clip','long108_q0_640' if suite=='short' else 'uploaded_half_1239_q0_640'),
                backend=row['backend'],repeat=row.get('repeat',1),frames=result['frames'],
                core_seconds=result['core_seconds'],fps=result['fps'],model_load_seconds=result['model_load_seconds'],
                engine_setup_seconds=result['engine_setup_seconds'],warmup_seconds=result['warmup_seconds'],
                file_operation_seconds=result['file_operation_seconds'],worker_seconds=result['worker_seconds'],
                peak_torch_allocated_mib=result['peak_torch_allocated_mib'],
                stream_sha256=result['stream_sha256'],entropy_symbols_and_cdf_sha256=result['entropy_symbols_and_cdf_sha256'],
                reference_sha256=result['reference_checksums']['latent_sha256']))
            for info in result['setup'].values():
                build=load_json(Path(info['engine_file']).with_name('build.json'))
                engines[info['engine_sha256']]=dict(identity=info['identity'],engine_sha256=info['engine_sha256'],
                    build_seconds=build['build_seconds'],onnx_sha256=build['onnx_sha256'],
                    parsed_onnx_sha256=build['parsed_onnx_sha256'],strongly_typed=build['strongly_typed'],
                    convolution_precision_counts=dict(Counter(p['compute'] for p in build['convolution_precision'])))
        for row in quality['reports']:
            entry=dict(row)
            frame_rows=entry['quality']['frames']
            name=row['clip']+'__'+row['backend']+'.csv'
            with (target/name).open('w',encoding='utf-8-sig',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(frame_rows[0]))
                writer.writeheader();writer.writerows(frame_rows)
            entry['quality']=dict(groups=entry['quality']['groups'],frame_csv=name)
            qualities.append(entry)
    with (target/'timing.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(measurements[0]));writer.writeheader();writer.writerows(measurements)
    failures=[]
    root=TRIAL.parent
    for number in range(1,5):
        previous=root/f'decoder_20261031_{number:02d}'
        for log in previous.glob('short/*.log'):
            if 'trt_fp16' not in log.name:continue
            failure=previous/'short'/log.stem/'numerical_failure.json'
            failures.append(dict(raw_directory=str(previous),log_file=str(log.relative_to(previous)),
                log_sha256=file_sha(log),failure=load_json(failure) if failure.exists() else None,
                error_tail=log.read_text(encoding='utf-8',errors='replace').splitlines()[-8:]))
    ranges=load_json(root/'decoder_20261031_04/numerical_probe/eager_ranges.json')
    numeric=dict(top_original_ranges=sorted(ranges,key=lambda r:r['max_abs'],reverse=True)[:16],
                 pytorch_amp_probe=load_json(root/'decoder_20261031_04/numerical_probe/amp_probe.json'),
                 naive_fp16_failure_evidence=failures,
                 excluded_timing_reason='Attempts 01–04 were development/debugging; user confirmed concurrent video playback. No speedup is derived from them.')
    hidden_cast=[]
    for receipt in (root/'decoder_20261031_04/engines').glob('*/build.json'):
        meta=load_json(receipt)
        if meta['identity']['role']!='image':continue
        layers=load_json(receipt.parent/'layers.json')['Layers']
        hidden_cast=[layer for layer in layers if '/blocks.22/conv/Conv' in layer['Name']]
    numeric['weak_precision_hidden_cast']=hidden_cast
    save_json(target/'numerical_diagnostics.json',numeric)
    save_json(target/'engines.json',list(engines.values()))
    web_files=[STAGE16/name for name in ('web_app.py','web_video_worker.py','live.js','viewer.html')]+[HERE/'generator_install.py']
    save_json(target/'execution_manifest.json',dict(raw_directory=str(TRIAL),producer_code_sha256=code_hashes(),raw_result_receipts=receipts,
        web_code_sha256={str(path.relative_to(REPO)):file_sha(path) for path in web_files}))
    save_json(target/'summary.json',dict(reports=qualities,runs=len(measurements),frames_processed=sum(r['frames'] for r in measurements),
        original_bitstreams_unchanged=True,per_frame_reference_and_entropy_identity_verified=True,
        RGB_generator_only=True,weights_changed=False,training_steps=0,Orin_tested=False,
        timing_policy='Warm independent processes; short 3 runs/backend, alternating order; full 1 run/backend. Core, load, build, warmup and file output separate.',
        precision_policy='Only eligible generator convolution compute is FP16. Original upsample/output convolutions and other graph arithmetic stay FP32 via explicit casts and strong typing.',
        quality_policy='Aggregate uint8 RGB MSE before H.264 preview compression; original source GT; I/P separated; annotation-union fish masks where available.',
        temporal_policy='Frame error-difference proxy; not a perceptual flicker or motion-compensated conclusion.',
        memory_policy='Torch peak excludes some TensorRT allocations; device free-memory readings include other applications. No claim of full device peak or Orin memory.'))
    print(f'Published {len(measurements)} independent decoder runs and {len(qualities)} quality comparisons',flush=True)


if __name__=='__main__':main()
