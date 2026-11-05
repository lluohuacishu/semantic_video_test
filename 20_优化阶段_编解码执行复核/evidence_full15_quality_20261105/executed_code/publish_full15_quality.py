"""Verify full FP32 quality comparisons, publish compact evidence and frame curves."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

HERE=Path(__file__).resolve().parent
REPO=HERE.parent


def load(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,d):
    Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw',type=Path,required=True)
    ap.add_argument('--destination',type=Path,required=True)
    args=ap.parse_args()
    plan=load(args.raw/'plan.json')
    evaluated=load(args.raw/'evaluation/summary.json')
    if not evaluated['complete'] or evaluated['quality_records']!=7434:raise ValueError('Evaluation incomplete')
    dest=args.destination;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'verification.json').exists():raise FileExistsError('Publication is already complete')
    receipts={};plots=[]
    for clip,result in zip(plan['clips'],evaluated['clips']):
        width=clip['width'];folder=args.raw/f'evaluation/width{width}'
        if result['width']!=width or result['frames']!=1239 or not result['metric_validation']['passed']:
            raise ValueError('Metric geometry or validation differs')
        metrics=list(csv.DictReader((folder/'per_frame.csv').open(encoding='utf-8-sig')))
        pixels=list(csv.DictReader((folder/'pixel_differences.csv').open(encoding='utf-8-sig')))
        if len(metrics)!=3717 or len(pixels)!=3717:raise ValueError('Incomplete record count')
        for name in ('fp32','encoder_only','accelerated'):
            rows=[r for r in metrics if r['name']==name]
            if sorted(int(r['index']) for r in rows)!=list(range(1239)):raise ValueError('Quality frame ordering differs')
            for r in rows:
                if int(r['source_index'])!=2*int(r['index']):raise ValueError('Source sampling differs')
            for key in ('lpips_alex','dists'):
                if abs(statistics.mean(float(r[key]) for r in rows)-result['summaries'][name][key]['mean'])>1e-10:
                    raise ValueError('Reported quality mean differs')
            m=statistics.mean(float(r['mse']) for r in rows)
            if abs(10*math.log10(255**2/m)-result['summaries'][name]['psnr_db'])>1e-9:
                raise ValueError('Global-MSE PSNR differs')
            expected=result['provenance']['reconstructions'][name]
            if sha(expected['path'])!=expected['sha256']:raise ValueError('Reconstruction RGB changed')
        for label in ('encoder_only_vs_fp32','decoder_only','total_vs_fp32'):
            rows=[r for r in pixels if r['comparison']==label]
            if sorted(int(r['index']) for r in rows)!=list(range(1239)):raise ValueError('Pixel frame ordering differs')
            m=statistics.mean(float(r['mse']) for r in rows)
            if abs(m-result['pixel_differences'][label]['mse'])>1e-9:raise ValueError('Pixel MSE differs')
        results={name:load(path) for name,path in clip['receiver_results'].items()}
        for name in ('fp32','encoder_only'):
            r=results[name]
            if r['backend']!='eager' or r['header']['arithmetic']['tf32']:
                raise ValueError('FP32 renderer definition differs')
        for r in results.values():
            if not r['reference_verified'] or not r['source_media_reads_forbidden']:
                raise ValueError('Independent reference check missing')
        for key in ('stream_sha256','reference_checksums','entropy_symbols_and_cdf_sha256'):
            if results['encoder_only'][key]!=results['accelerated'][key]:
                raise ValueError('Decoder-only comparison changes reference/entropy')
        for r in results.values():
            for rel,h in r.get('code_sha256',r.get('frozen_code_sha256',{})).items():
                if sha(REPO/rel)!=h:raise ValueError('Frozen reference source changed')
        for filename in ('summary.json','per_frame.csv','pixel_differences.csv'):
            p=folder/filename
            receipts[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
            target=dest/f'width{width}_{filename}'
            if target.exists():
                if sha(target)!=sha(p):raise ValueError('Existing partial publication differs')
            else:shutil.copy2(p,target)
        plots.append((width,metrics))
    # Scientific plot: no temporal filtering; raw per-frame quality deltas.
    # Use the already installed Pillow, without changing the pinned environment.
    from PIL import Image,ImageDraw,ImageFont
    canvas=Image.new('RGB',(1800,1000),'white')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    large=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',27)
    draw.text((30,16),'GLC 15 FPS: per-frame quality change vs original FP32',font=large,fill='#202020')
    draw.text((30,56),'Positive = greater distance from source. All 1239 frames; no temporal smoothing.',font=font,fill='#444444')
    draw.line([(1120,26),(1160,26)],fill='#ce7b20',width=3)
    draw.text((1170,12),'Encoder acceleration',font=small,fill='#333333')
    draw.line([(1120,58),(1160,58)],fill='#4058a6',width=3)
    draw.text((1170,44),'Encoder + decoder acceleration',font=small,fill='#333333')
    for row,(width,metrics) in enumerate(plots):
        indexed={name:{int(r['index']):r for r in metrics if r['name']==name}
                 for name in ('fp32','encoder_only','accelerated')}
        times=[i/15 for i in range(1239)]
        for col,key in enumerate(('lpips_alex','dists')):
            values={name:[float(indexed[name][i][key])-float(indexed['fp32'][i][key]) for i in range(1239)]
                    for name in ('encoder_only','accelerated')}
            low=min(0,min(min(v) for v in values.values()));high=max(0,max(max(v) for v in values.values()))
            pad=max((high-low)*.09,1e-5);low-=pad;high+=pad
            left=95+col*875;top=158+row*415;pw=730;ph=270
            draw.text((left,top-43),f'{width} x {width*9//16}: {key} change',font=font,fill='#222222')
            def point(t,y):return (left+t/82.6*pw,top+(high-y)/(high-low)*ph)
            for j in range(5):
                value=low+(high-low)*j/4;y=point(0,value)[1]
                draw.line([(left,y),(left+pw,y)],fill='#dedede',width=1)
                draw.text((left-83,y-10),f'{value:.3f}',font=small,fill='#555555')
            for t in (0,20,40,60,80):
                x=point(t,0)[0];draw.line([(x,top),(x,top+ph)],fill='#eeeeee',width=1)
                draw.text((x-9,top+ph+8),str(t),font=small,fill='#555555')
            draw.rectangle((left,top,left+pw,top+ph),outline='#777777',width=1)
            y0=point(0,0)[1];draw.line([(left,y0),(left+pw,y0)],fill='#555555',width=2)
            for name,color in [('encoder_only','#ce7b20'),('accelerated','#4058a6')]:
                draw.line([point(t,y) for t,y in zip(times,values[name])],fill=color,width=1)
            draw.text((left+235,top+ph+38),'Video time (seconds)',font=small,fill='#333333')
    canvas.save(dest/'quality_delta_vs_fp32.png')
    examples=args.raw/'evaluation/fp32_comparison_examples.png'
    if examples.is_file():shutil.copy2(examples,dest/examples.name)
    save(dest/'summary.json',evaluated)
    save(dest/'verification.json',dict(passed=True,quality_records=7434,pixel_comparison_records=7434,
        all_frames=True,frames_per_path=1239,paths_per_resolution=3,resolutions=2,
        metric_means_and_global_psnr_recomputed_from_csv=True,
        decoder_only_reference_and_entropy_identical=True,reconstruction_sha_verified=True,
        reference='lossless same-resolution source RGB; numerical pixel differences separately reference FP32 reconstruction',
        raw_directory=str(args.raw),receipts=receipts,plan_sha256=sha(args.raw/'plan.json'),
        publisher_sha256=sha(Path(__file__)),metric_script_sha256=evaluated['producer_sha256']))
    snapshot=dest/'executed_code';snapshot.mkdir()
    for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    print('PASS all 7434 quality and 7434 pixel records, reference/entropy identity and immutable RGB verified')


if __name__=='__main__':
    main()
