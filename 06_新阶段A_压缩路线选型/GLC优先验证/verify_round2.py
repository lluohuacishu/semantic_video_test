"""Audit complete experimental evidence, actual bytes, frames and metric samples."""
from pathlib import Path
import argparse
import hashlib
import json
import math

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
sha=lambda path:hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
rows=[];glc_frames=0;sample_sets={}
for clip,expected in [('short',13),('long108',6),('long91',3),('long24',3)]:
    base=a.run/clip;data=read(base/'results.json');n=data['input']['frames']
    assert len(data['results'])==expected
    assert len({r['run'] for r in data['results']})==expected
    validation=data['metric_validation']
    assert abs(validation['identity']['dists'])<1e-4
    assert abs(validation['identity']['ms_ssim']-1)<1e-4
    for r in data['results']:
        folder=base/r['run'];cfg=r['config'];is_glc=cfg['codec']=='glc'
        stream=folder/('stream.glcv' if is_glc else 'selected.mp4')
        assert stream.stat().st_size==r['file_bytes']==cfg['file_bytes' if is_glc else 'bytes']
        assert sha(stream)==cfg['sha256' if is_glc else 'selected_sha256']
        assert len(list((folder/'receiver/frames').glob('im*.png')))==n
        assert [x['index'] for x in r['per_frame']]==list(range(n))
        assert [x['source_frame'] for x in r['per_frame']]==list(range(data['input']['source_start'],data['input']['source_start']+n))
        assert r['dists_frames']==[x['index'] for x in r['per_frame'] if x['dists'] is not None]
        sample_sets.setdefault(clip,r['dists_frames']);assert sample_sets[clip]==r['dists_frames']
        assert all(math.isfinite(r[k]) for k in ['global_psnr_db','ms_ssim_mean','dists_mean','temporal_error_delta_mae_mean'])
        assert 0<=r['dists_mean']<=1 and 0<=r['ms_ssim_mean']<=1
        assert r['fish_pixels']==sum(x.get('fish_pixels',0) for x in data['input']['records'])
        assert (r['fish_psnr_db'] is None)==(r['fish_pixels']==0)
        for rate,seconds in r['payload_wait_seconds'].items():assert abs(seconds-8*r['file_bytes']/(1000*int(rate)))<1e-10
        if is_glc:
            assert cfg['same_input_frames']==cfg['same_decoded_frames']==n and not cfg['source_media_reads']
            dec=read(folder/'receiver/decode.json');assert dec['all_frame_checksums_match'] and not dec['source_media_reads']
            glc_frames+=n
        else:
            assert cfg['key_frame_indices']==list(range(0,n,cfg['gop']))
            delta=(r['file_bytes']-cfg['target_bytes'])/cfg['target_bytes']
            assert abs(delta-cfg['relative_error'])<1e-12
            assert cfg['within_2_percent']==(abs(delta)<=.02)
        rows.append({'clip':clip,'run':r['run'],'frames':n,'bytes':r['file_bytes'],'sha256':sha(stream),'dists_frames':len(r['dists_frames']),'budget_within_2_percent':cfg.get('within_2_percent')})
assert len(rows)==25 and glc_frames==1150
out={'status':'PASS','total_runs':len(rows),'glc_runs':7,'glc_independent_decode_frames':glc_frames,'evaluated_frames':sum(x['frames'] for x in rows),'budget_misses':[x for x in rows if x['budget_within_2_percent'] is False],'runs':rows}
(a.run/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='runs'},ensure_ascii=False),flush=True)
