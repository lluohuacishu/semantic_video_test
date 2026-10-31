"""CPU-only pilot summary. Distinguish batch error from existing mixed-precision error."""
from __future__ import annotations
import csv
import hashlib
import math
import statistics
from batch_common import *
import numpy as np


def psnr(mse):
    return 10*math.log10(255**2/max(mse,1e-12))


def compare(gt,teacher,single,actual,indices):
    frames=[]
    for index,g,t,b,a in zip(indices,gt,teacher,single,actual):
        av=a.astype(np.float64);tv=t.astype(np.float64);bv=b.astype(np.float64);gv=g.astype(np.float64)
        d=av-bv
        frames.append(dict(source_index=index,gt_mse=float(np.mean((av-gv)**2)),
            teacher_gt_mse=float(np.mean((tv-gv)**2)),batch1_gt_mse=float(np.mean((bv-gv)**2)),
            teacher_difference_mse=float(np.mean((av-tv)**2)),batch1_difference_mse=float(np.mean(d*d)),
            batch1_difference_max=int(np.abs(d).max()),batch1_changed_values=int(np.count_nonzero(d))))
    if len(frames)!=len(indices):raise ValueError('Incomplete quality comparison')
    mse=lambda key:statistics.mean(row[key] for row in frames)
    return dict(frames=frames,psnr_vs_gt_db=psnr(mse('gt_mse')),
        teacher_psnr_vs_gt_db=psnr(mse('teacher_gt_mse')),batch1_psnr_vs_gt_db=psnr(mse('batch1_gt_mse')),
        psnr_delta_vs_teacher_db=psnr(mse('gt_mse'))-psnr(mse('teacher_gt_mse')),
        additional_psnr_delta_vs_batch1_db=psnr(mse('gt_mse'))-psnr(mse('batch1_gt_mse')),
        mse_vs_teacher=mse('teacher_difference_mse'),mse_vs_batch1=mse('batch1_difference_mse'),
        maximum_rgb_error_vs_batch1=max(row['batch1_difference_max'] for row in frames),
        changed_rgb_values_vs_batch1=sum(row['batch1_changed_values'] for row in frames),
        bit_exact_vs_batch1=all(row['batch1_changed_values']==0 for row in frames))


def main():
    campaign=load_json(TRIAL/'campaign.json')
    capture=load_json(TRIAL/'capture/result.json')
    if not campaign['complete'] or campaign['failures']:raise ValueError('Pilot is incomplete or has failed configurations')
    plan=load_json(ARCHIVE/'plan.json')
    clip=next(c for c in plan['clips'] if c['id']=='long108_q0_640')
    if file_sha(clip['source'])!=clip['source_sha256']:raise ValueError('GT source changed')
    selected=capture['selected_indices'];gt=np.load(clip['source'],mmap_mode='r')[selected]
    teacher=np.load(TRIAL/'capture/teacher_rgb.npy',mmap_mode='r')
    single=np.load(TRIAL/'batch1/run1/rgb.npy',mmap_mode='r')
    previous=np.load(DECODER_TRIAL/'short/trt_fp16_safe_1/decoded_rgb.npy',mmap_mode='r')
    if not np.array_equal(single,previous[selected]):raise ValueError('Batch-1 RGB differs from the validated stage 17 baseline')
    latent_array=np.load(TRIAL/'capture/latents.npy',mmap_mode='r')
    unique_latents=len({hashlib.sha256(value.tobytes()).hexdigest() for value in latent_array})
    reports=[];runs=[];receipts={}
    for batch in (1,2,4):
        values=[]
        for repeat in (1,2):
            folder=TRIAL/f'batch{batch}'/f'run{repeat}';path=folder/'result.json';value=load_json(path)
            if value['code_sha256']!=code_hashes() or value['latent_source_sha256']!=capture['latents_sha256']:
                raise ValueError('Input or implementation differs')
            if file_sha(folder/'rgb.npy')!=value['rgb_sha256']:raise ValueError('Output RGB changed')
            values.append(value)
            receipts[str(path.relative_to(TRIAL))]=dict(sha256=file_sha(path),bytes=path.stat().st_size)
            for mode,rr in value['timings'].items():
                for i,row in enumerate(rr):runs.append(dict(batch=batch,process_round=repeat,mode=mode,pass_number=i+1,**row))
        if values[0]['rgb_sha256']!=values[1]['rgb_sha256']:
            raise ValueError('Same fixed engine and input did not repeat RGB exactly')
        actual=np.load(TRIAL/f'batch{batch}/run1/rgb.npy',mmap_mode='r')
        if actual.shape!=teacher.shape or actual.dtype!=np.uint8:raise ValueError('Output shape/dtype differs')
        modes={}
        for mode in ('generator','rgb_output'):
            ms=[row['ms_per_frame'] for value in values for row in value['timings'][mode]]
            round_ms=[statistics.median(r['ms_per_frame'] for r in value['timings'][mode]) for value in values]
            modes[mode]=dict(median_ms_per_frame=statistics.median(ms),median_fps=1000/statistics.median(ms),
                             all_ms_per_frame=ms,process_round_medians=round_ms)
        report=dict(batch=batch,model='video',frames_per_pass=len(selected),timings=modes,
            quality=compare(gt,teacher,single,actual,selected),rgb_repeatable=True,
            sampled_device_peak_mib=[v['device_used_sampled_peak_mib'] for v in values],
            engine_context_mib=values[0]['engine_context_bytes']/1024**2,
            engine=values[0]['engine'],engine_build_in_round1=values[0]['engine']['built'])
        reports.append(report)
    for report in reports:
        for mode in ('generator','rgb_output'):
            report['timings'][mode]['speedup_vs_batch1']=reports[0]['timings'][mode]['median_ms_per_frame']/report['timings'][mode]['median_ms_per_frame']
    public=HERE/'evidence_20261031';public.mkdir(exist_ok=True)
    with (public/'timings.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(runs[0]));writer.writeheader();writer.writerows(runs)
    for report in reports:
        frames=report['quality'].pop('frames')
        with (public/f'batch{report["batch"]}_quality.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(frames[0]));writer.writeheader();writer.writerows(frames)
    summary=dict(complete=True,reports=reports,capture={k:v for k,v in capture.items() if k!='code_sha256'},
        batch1_matches_stage17_rgb=True,unique_real_latents=unique_latents,
        scope='P generator only, RTX4060 Laptop, 640x360 q0, 20 selected real P latents; not full decoder FPS',
        timing_scope='GPU-resident inputs. Both modes include real batch gathering and private output cloning. rgb_output also includes finite check, FP32 pixel conversion and CPU RGB transfer. Build/warmup/memory sampling excluded.',
        memory_scope='CUDA mem_get_info usage proxy sampled every 20ms in separate untimed passes. Under Windows WDDM it must not be equated with nvidia-smi physical residency or an exact device/process peak. Engine context bytes are reported separately.',
        source_stream_reencoded=False,decoder_interface_changed=False,web_changed=False,training_steps=0,Orin_tested=False)
    save_json(TRIAL/'summary.json',summary);save_json(public/'summary.json',summary)
    save_json(public/'execution_manifest.json',dict(raw_directory=str(TRIAL),producer_code_sha256=code_hashes(),
        capture_result_sha256=file_sha(TRIAL/'capture/result.json'),raw_result_receipts=receipts))
    for report in reports:
        core=report['timings']['generator'];rgb=report['timings']['rgb_output']
        print(f'B{report["batch"]}: generator {core["median_ms_per_frame"]:.3f}ms/frame ({core["speedup_vs_batch1"]:.3f}x), RGB {rgb["median_ms_per_frame"]:.3f}ms/frame ({rgb["speedup_vs_batch1"]:.3f}x), extra PSNR {report["quality"]["additional_psnr_delta_vs_batch1_db"]:+.6f}dB',flush=True)


if __name__=='__main__':main()
