"""CPU-only rate/quality evaluation of independently decoded outputs."""
from __future__ import annotations
import argparse
import math
from experiment_common import *


def psnr(mse):
    return 10*math.log10(255**2/max(mse,1e-12))


def quality(source,baseline,actual,mask=None):
    rows=[]
    previous=None
    for i,(g,b,a) in enumerate(zip(source,baseline,actual)):
        d=a.astype('float64')-g;db=a.astype('float64')-b
        r=dict(index=i,kind='I' if i%64==0 else 'P',mse=float(np.mean(d*d)),
               baseline_mse=float(np.mean((b.astype('float64')-g)**2)),baseline_difference_mse=float(np.mean(db*db)),
               baseline_difference_max_abs=float(np.abs(db).max()),changed_rgb_values=int(np.count_nonzero(db)),
               error_delta_mae=float(np.mean(np.abs(d-previous))) if previous is not None else None,
               fish_pixels=0,fish_error_sum=0.,fish_baseline_error_sum=0.)
        if mask is not None:
            roi=mask[i]>0;r['fish_pixels']=int(roi.sum())
            r['fish_error_sum']=float(np.sum(d[roi]**2))
            r['fish_baseline_error_sum']=float(np.sum((b.astype('float64')-g)[roi]**2))
        rows.append(r);previous=d
    groups={}
    for name,rr in [('all',rows),('P',[r for r in rows if r['kind']=='P']),('I',[r for r in rows if r['kind']=='I'])]:
        m=float(np.mean([r['mse'] for r in rr]));bm=float(np.mean([r['baseline_mse'] for r in rr]));n=sum(r['fish_pixels'] for r in rr)
        groups[name]=dict(frames=len(rr),psnr_db=psnr(m),baseline_psnr_db=psnr(bm),psnr_delta_db=psnr(m)-psnr(bm),
            baseline_difference_mse=float(np.mean([r['baseline_difference_mse'] for r in rr])),
            changed_rgb_values=sum(r['changed_rgb_values'] for r in rr),
            fish_pixels=n,fish_psnr_db=psnr(sum(r['fish_error_sum'] for r in rr)/(3*n)) if n else None,
            fish_baseline_psnr_db=psnr(sum(r['fish_baseline_error_sum'] for r in rr)/(3*n)) if n else None)
    return dict(groups=groups,frames=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;plan=load_json(out/'plan.json');summary=load_json(out/'full_summary.json')
    results=[]
    for item in summary['reports']:
        clip=next(c for c in plan['clips'] if c['id']==item['clip'])
        folder=out/'full'/clip['id']/item['variant'];baseline=out/'full'/clip['id']/'eager'
        receiver=item['receiver'];path=folder/'receiver/decoded_rgb.npy'
        if file_sha(path)!=receiver['decoded_rgb_sha256']:raise ValueError('Decoded array changed')
        old=load_json(baseline/'receiver/result.json');ref=baseline/'receiver/decoded_rgb.npy'
        if file_sha(ref)!=old['decoded_rgb_sha256']:raise ValueError('Baseline RGB changed')
        if file_sha(clip['source'])!=clip['source_sha256']:raise ValueError('Source changed')
        source=np.load(clip['source'],mmap_mode='r');a=np.load(path,mmap_mode='r');b=np.load(ref,mmap_mode='r')
        mask=None;mask_path=None
        if clip['id'].startswith(('sf007','sf025')):
            name='v007' if clip['id'].startswith('sf007') else 'v025'
            mask_path=WORKSPACE/'02_实验数据/优化_GLC蒸馏/pilot_20261029_01'/name/'fish_mask.npy'
            mask=np.load(mask_path,mmap_mode='r')
        q=quality(source,b,a,mask)
        base_sender=next(x for x in summary['reports'] if x['clip']==clip['id'] and x['variant']=='eager')
        sender=item['sender_repeats'][0]
        entry=dict(clip=clip['id'],variant=item['variant'],quality=q,stream_bytes=sender['stream_bytes'],
            payload_bytes=sender['payload_bytes'],bpp=sender['stream_bytes']*8/(clip['frames']*clip['width']*clip['height']),
            bitrate_ratio_vs_eager=sender['stream_bytes']/base_sender['sender_repeats'][0]['stream_bytes'],
            payload_ratio_vs_eager=sender['payload_bytes']/base_sender['sender_repeats'][0]['payload_bytes'],
            container_overhead_bytes=sender['stream_bytes']-sender['payload_bytes'],
            stream_exact_vs_eager=sender['stream_sha256']==base_sender['sender_repeats'][0]['stream_sha256'],
            payload_and_reference_exact_vs_eager=all(
                all(a[k]==b[k] for k in ['latent_sha256','payload_sha256','entropy'])
                for a,b in zip(sender['records'],base_sender['sender_repeats'][0]['records'])),
            rgb_exact_vs_eager=receiver['decoded_rgb_sha256']==old['decoded_rgb_sha256'],
            source_sha256=clip['source_sha256'],mask_sha256=file_sha(mask_path) if mask_path else None,
            sender_core_median_seconds=item['sender_core_median_seconds'],
            paired_speedup_vs_eager=item['paired_speedup_vs_eager'],independent_reference_and_symbols_verified=True)
        results.append(entry)
        print(f'quality {clip["id"]} {item["variant"]}: P PSNR delta {q["groups"]["P"]["psnr_delta_db"]:.4f}dB, rate ratio {entry["bitrate_ratio_vs_eager"]:.5f}',flush=True)
    save_json(out/'quality.json',dict(reports=results,complete=True,
        psnr_policy='Aggregate RGB MSE; I/P separated. No averaging of PSNR dB. MSE floor 1e-12 only for finite dB display; equality reported separately.',
        temporal_policy='error-difference frame-index proxy only; no perceptual flicker or motion-compensated claim',
        mask_policy='Existing SFishTrack annotation-union masks only, no inferred UV masks; raw mask hashes recorded',
        stream_identity_policy='Candidate stream headers identify input encoder implementation; compare payload/state separately from whole-file identity',
        source_independence_established=False,Orin_tested=False,training_steps=0))


if __name__=='__main__':main()
