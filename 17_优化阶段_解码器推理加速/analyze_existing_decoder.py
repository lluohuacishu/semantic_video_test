"""Reaggregate archived decoder CUDA traces on CPU. No torch, model or video execution."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ranked(values):return sorted(values.items(),key=lambda item:(-item[1],item[0]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'evidence_20261031')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    previous=REPO/'15_优化阶段_算子精确排查/evidence_20261030'
    analysis=read(previous/'analysis.json');role=analysis['roles']['receiver']
    sources={str((previous/'analysis.json').relative_to(REPO)):digest(previous/'analysis.json')}
    frames=[];p_rows=None
    for frame in role['frames']:
        assert frame['graphs_exact'] and frame['label_agreement_between_two_native_runs']
        assert not any(frame['dropped_records'])
        path=previous/frame['complete_layer_operator_shape_table']
        sources[str(path.relative_to(REPO))]=digest(path)
        with path.open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
        total=sum(float(row['gpu_kernel_ms']) for row in rows)
        assert abs(total-frame['kernel_ms'])<1e-6,(path,total,frame['kernel_ms'])
        generator=sum(float(row['gpu_kernel_ms']) for row in rows if row['module'].startswith(('I.vqgan.generator','P.vqgan.generator')))
        frames.append(dict(index=frame['index'],kind=frame['kind'],core_baseline_ms=frame['baseline_core_median_ms'],
            gpu_kernel_ms=total,generator_ms=generator,generator_share_percent=generator/total*100,
            dma_ms=frame['memcpy_ms'],kernel_count=frame['kernel_count']))
        if frame['index']==2:p_rows=rows
    assert p_rows is not None
    ops=defaultdict(float);gen_ops=defaultdict(float);blocks=defaultdict(float);spatial=defaultdict(float)
    for row in p_rows:
        ms=float(row['gpu_kernel_ms']);ops[row['operator']]+=ms
        if not row['module'].startswith('P.vqgan.generator'):continue
        gen_ops[row['operator']]+=ms
        suffix=row['module'].removeprefix('P.vqgan.generator.')
        key='.'.join(suffix.split('.')[:2]) if suffix.startswith('blocks.') else suffix.split('.')[0]
        blocks[key]+=ms
        shapes=json.loads(row['input_shapes'])
        shape=next((s for s in shapes if isinstance(s,list) and len(s)==4 and all(type(x)is int for x in s)),None)
        spatial[f'{shape[-1]}x{shape[-2]}' if shape else 'other']+=ms
    p2=next(f for f in frames if f['index']==2);gen=p2['generator_ms'];total=p2['gpu_kernel_ms']
    assert abs(sum(gen_ops.values())-gen)<1e-6
    assert abs(sum(blocks.values())-gen)<1e-6
    assert abs(sum(spatial.values())-gen)<1e-6
    operators=[dict(operator=name,receiver_ms=ms,generator_ms=gen_ops[name],receiver_share_percent=ms/total*100) for name,ms in ranked(ops)]
    block_rows=[dict(block=name,ms=ms,generator_share_percent=ms/gen*100,receiver_share_percent=ms/total*100) for name,ms in ranked(blocks)]
    spatial_rows=[dict(first_4d_input_wh=name,ms=ms,generator_share_percent=ms/gen*100,receiver_share_percent=ms/total*100) for name,ms in ranked(spatial)]
    scenarios=[dict(assumed_generator_speedup=s,gpu_work_ms=total-gen+gen/s,gpu_work_speedup=total/(total-gen+gen/s)) for s in [1.5,2.,3.,4.]]
    full_path=REPO/'16_优化阶段_输入编码器推理加速/web_evidence_20261031/full_acceptance.json'
    full=read(full_path);sources[str(full_path.relative_to(REPO))]=digest(full_path)
    selected=next(row for row in full['modes'] if row['mode']=='trt_fp16_stream')
    for name in ['04_外部依赖快照/semantic-video-training/scripts/glc_video_codec.py',
                 '16_优化阶段_输入编码器推理加速/web_video_worker.py']:
        sources[name]=digest(REPO/name)
    result=dict(schema='glc-decoder-existing-evidence-review-v1',date='2026-10-31',
        scope='CPU reaggregation of 2026-10-30 fill-on CUDA traces; no new GPU measurements or codec runs',
        baseline_66_core_seconds=role['baseline_seconds'],baseline_66_median_seconds=role['baseline_median_seconds'],
        frames=frames,p002_operators=operators,p002_generator_blocks=block_rows,p002_generator_input_shapes=spatial_rows,
        last_three_resblocks_ms=sum(blocks[f'blocks.{i}'] for i in [23,24,25]),
        last_three_resblocks_share_percent=sum(blocks[f'blocks.{i}'] for i in [23,24,25])/total*100,
        fill_ablation=analysis['fill_ablation']['receiver'],gpu_work_hypotheses_not_measured=scenarios,
        archived_full_job=dict(frames=full['frames'],width=full['width'],height=full['height'],fps=full['fps'],q=full['q'],
            encoder_mode=selected['mode'],encode_core_seconds=selected['encode_core_seconds'],decode_core_seconds=selected['decode_core_seconds'],
            decode_fraction_of_encode_plus_decode_core=selected['decode_core_seconds']/(selected['encode_core_seconds']+selected['decode_core_seconds'])),
        inputs_sha256=sources,analysis_script_sha256=digest(Path(__file__)),new_gpu_jobs=0,new_training_steps=0)
    for name,data in [('decoder_frames.csv',frames),('p002_operators.csv',operators),('p002_generator_blocks.csv',block_rows),('p002_generator_input_shapes.csv',spatial_rows)]:
        with (args.output/name).open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (args.output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified_archived_frames=len(frames),p002_generator_ms=gen,p002_generator_share_percent=gen/total*100,
        full_resolution_input_ms=spatial['640x384'],last_three_resblocks_ms=result['last_three_resblocks_ms'],
        last_three_resblocks_share_percent=result['last_three_resblocks_share_percent'],
        archived_full_job_decode_core_fraction=result['archived_full_job']['decode_fraction_of_encode_plus_decode_core'],
        new_gpu_jobs=0),ensure_ascii=False))


if __name__=='__main__':main()
