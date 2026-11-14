"""Read-back verification for the fixed-parameter known-video transfer round."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from runtime_speed_common import STAGE, FFMPEG, load_json, save_json, file_sha, check_inputs
from run_known_video_transfer import CASES, HISTORY, SOURCES
import structure_guide_stream as guide


def guard_probe(run,case,role):
    from runtime_chain_worker import ReadGuard
    root=run/case
    allowed=[root/'send/base.glcsnd']
    if role=='post':allowed += [root/'guide/guide.glcguide',root/'receive/decoded_rgb.npy',root/'receive/receipt.json']
    denied=[SOURCES/(case+'.mp4'),root/'input/source.npy',HISTORY/case/'source.npy',
        HISTORY/case/'flow/to_anchor.npy',HISTORY/case/'flow/from_anchor.npy',
        root/'guide/send.json',root/'send/receipt.json']
    guard=ReadGuard(role);guard.authorize(run/'verification/guard_scratch',allowed)
    blocked=[]
    for path in denied:
        try:path.read_bytes()
        except RuntimeError:blocked.append(str(path))
        else:raise ValueError('Receiver source boundary failed: '+str(path))
    for path in allowed:
        with path.open('rb') as handle:handle.read(1)
    print(json.dumps(dict(case=case,role=role,blocked=blocked,allowed=[str(p) for p in allowed]),ensure_ascii=False))


def verify(run):
    plan=load_json(run/'execution_plan.json');check_inputs(plan['input_sha256'])
    out=run/'verification';out.mkdir(exist_ok=False)
    code={name:file_sha(STAGE/name)==digest for name,digest in plan['code_sha256'].items()}
    assert all(code.values())
    cases={};guards=[]
    for case in CASES:
        root=run/case;chain=load_json(root/'chain_result.json');evaluation=load_json(root/'evaluation/summary.json')
        assert chain['complete'] and evaluation['complete']
        files={};arrays={}
        for mode in ('post','uncached_replay'):
            receipt=load_json(root/mode/'receipt.json')
            assert receipt['source_reads_blocked'] and receipt['zero_weight_unchanged']
            for name,digest in receipt['outputs_sha256'].items():
                path=root/mode/(name+'.npy');actual=file_sha(path)
                assert actual==digest
                files[str(path)]=actual
                array=np.load(path,mmap_mode='r',allow_pickle=False)
                if name=='rgb':assert array.dtype==np.uint8 and array.shape==(150,288,512,3)
                else:assert array.dtype==np.float32 and array.shape==((150 if name=='alpha' else 30),288,512)
                assert np.isfinite(array).all()
                arrays[(mode,name)]=array
        for name in ('rgb','alpha','shared'):assert np.array_equal(arrays[('post',name)],arrays[('uncached_replay',name)])
        base=root/'send/base.glcsnd';raw_path=root/'receive/decoded_rgb.npy'
        base_receipt=load_json(root/'receive/receipt.json')
        assert base_receipt['reference_verified'] and base_receipt['source_media_reads_forbidden']
        assert file_sha(base)==base_receipt['stream_sha256'] and file_sha(raw_path)==base_receipt['decoded_rgb_sha256']
        raw=np.load(raw_path,mmap_mode='r',allow_pickle=False);alpha=arrays[('post','alpha')]
        assert np.array_equal(arrays[('post','rgb')][alpha==0],raw[alpha==0])
        packet=(root/'guide/guide.glcguide').read_bytes();header,payload=guide.unpack(packet,file_sha(base))
        decoded_header,low,full=guide.decode(packet,file_sha(base),FFMPEG)
        assert low.shape==(150,144,256,3) and full.shape==(150,288,512,3) and decoded_header==header
        assert len(packet)==chain['guide_bytes'] and base.stat().st_size==chain['base_bytes']
        h265=evaluation['matched_hevc'];hevc=root/'evaluation/matched_hevc/h265_matched.mp4'
        assert file_sha(hevc)==h265['sha256'] and file_sha(hevc.with_name('rgb.npy'))==h265['decoded_rgb_file_sha256']
        assert h265['bytes']==hevc.stat().st_size and h265['target_bytes']==chain['total_bytes']
        assert 0<=h265['relative_shortfall']<=.02
        media={}
        videos=[root/'post/video.mp4',root/'uncached_replay/video.mp4',hevc,
            *sorted((root/'evaluation').glob(case+'*.mp4'))]
        for path in videos:
            stream=json.loads(subprocess.check_output([str(FFMPEG.with_name('ffprobe.exe')),'-v','error',
                '-select_streams','v:0','-count_frames','-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration',
                '-of','json',str(path)]))['streams'][0]
            assert int(stream['nb_read_frames'])==150 and Fraction(stream['avg_frame_rate'])==15 and float(stream['duration'])==10
            media[str(path)]=dict(sha256=file_sha(path),probe=stream)
        for role in ('receiver','post'):
            result=subprocess.check_output([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),
                '--run',str(run),'--probe',role,'--case',case],text=True,encoding='utf-8')
            item=json.loads(result.strip());assert len(item['blocked'])==7;guards.append(item)
        cases[case]=dict(saved_arrays=files,exact_cached_uncached=True,zero_alpha_unchanged=True,
            guide_frame_count_and_binding=True,base_reference_verified=True,media=media,
            actual_stream_bytes=dict(base=base.stat().st_size,guide=len(packet)),
            historical_baseline_metrics_reproduced=evaluation['historical_raw_K_scores_exact'])
    result=dict(complete=True,input_hashes_unchanged=True,code_matches_pretrial_snapshot=code,
        cases=cases,current_case_read_guard_probes=guards,guard_note='Python read audit for reproducibility, not an OS security sandbox',
        verifier_sha256=file_sha(__file__))
    save_json(out/'summary.json',result)
    print(json.dumps(dict(complete=True,cases=list(cases),verified_media=sum(len(v['media']) for v in cases.values()),
        blocked_current_source_reads=sum(len(v['blocked']) for v in guards)),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--probe',choices=('receiver','post'));p.add_argument('--case',choices=CASES)
    a=p.parse_args()
    if a.probe:guard_probe(a.run.resolve(),a.case,a.probe)
    else:verify(a.run.resolve())


if __name__=='__main__':main()
