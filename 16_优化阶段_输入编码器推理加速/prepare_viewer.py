"""Create four synchronized viewing proxies from verified results, without GPU inference."""
from __future__ import annotations
import argparse
import statistics
import subprocess
from PIL import Image,ImageDraw,ImageFont
from experiment_common import *

VARIANTS=['eager','trt_fp32_stream','trt_fp16_stream']
TITLES={'long108_q0_640':('108 · q0','108：鱼与背景纹理'), 'sf007_q12_512':('007 · q12','007：鱼体与海草'),
        'sf025_q32_512':('025 · q32','025：不同水下场景'), 'uv096_q63_512_255':('096 · q63 · 255帧','096：连续255帧与GOP切换')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    plan=load_json(args.experiment/'plan.json');quality=load_json(args.experiment/'quality.json');compact=load_json(args.experiment/'compact_transport.json')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);items=[]
    for cid,(tab,title) in TITLES.items():
        c=next(c for c in plan['clips'] if c['id']==cid)
        if file_sha(c['source'])!=c['source_sha256']:raise ValueError('Source changed')
        arrays=[np.load(c['source'],mmap_mode='r')];methods=[]
        for variant in VARIANTS:
            folder=args.experiment/'full'/cid/variant/'receiver';r=load_json(folder/'result.json');f=folder/'decoded_rgb.npy'
            if file_sha(f)!=r['decoded_rgb_sha256']:raise ValueError('Decoded RGB changed')
            arrays.append(np.load(f,mmap_mode='r'))
            q=next(x for x in quality['reports'] if x['clip']==cid and x['variant']==variant)
            transfer=next(x for x in compact['reports'] if x['clip']==cid and x['variant']==variant)
            if file_sha(transfer['path'])!=transfer['sha256']:raise ValueError('Transport stream changed')
            pg=q['quality']['groups']['P']
            methods.append(dict(label={'eager':'原GLC','trt_fp32_stream':'TensorRT FP32','trt_fp16_stream':'TensorRT FP16'}[variant],
                core_seconds=q['sender_core_median_seconds'],speedup=statistics.median(q['paired_speedup_vs_eager']),
                p_psnr=pg['psnr_db'],p_delta=pg['psnr_delta_db'],fish_psnr=pg['fish_psnr_db'],
                stream_bytes=transfer['bytes'],file_delta=None if variant=='eager' else (transfer['file_ratio_vs_baseline']-1)*100))
        target=out/(cid+'.mp4');identity=dict(source_sha256=c['source_sha256'],
            decoded_rgb_sha256=[file_sha(args.experiment/'full'/cid/v/'receiver/decoded_rgb.npy') for v in VARIANTS])
        cache=out/(cid+'.cache.json');reuse=False
        if target.exists() and cache.exists():
            old=load_json(cache);reuse=old['inputs']==identity and old['sha256']==file_sha(target)
        if not reuse:
            if target.exists():raise FileExistsError('Preserve existing mismatched viewing proxy: '+str(target))
            w,h=c['width'],c['height'];height=h+34
            command=[str(load_settings()['runtime']['ffmpeg']),'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                '-s',f'{w*2}x{height*2}','-r',str(c['fps']),'-i','pipe:0','-an','-c:v','libx264','-threads','4',
                '-preset','fast','-crf','14','-g','25','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
            proc=subprocess.Popen(command,stdin=subprocess.PIPE)
            try:
                for i in range(c['frames']):
                    canvas=Image.new('RGB',(w*2,height*2),'#101b24');draw=ImageDraw.Draw(canvas)
                    for j,(label,a) in enumerate(zip(['原始画面','原GLC','TensorRT FP32','TensorRT FP16'],arrays)):
                        x=(j%2)*w;y=(j//2)*height;canvas.paste(Image.fromarray(a[i]),(x,y+34))
                        draw.text((x+8,y+3),f'{label} · {i+1:03d}',font=font,fill='white')
                    proc.stdin.write(canvas.tobytes())
                proc.stdin.close()
                if proc.wait()!=0:raise RuntimeError('Viewing proxy encoding failed')
            except BaseException:
                proc.kill();proc.wait();raise
            save_json(cache,dict(inputs=identity,sha256=file_sha(target),frames=c['frames']))
        items.append(dict(id=cid,tab=tab,title=title,width=c['width'],height=c['height'],frames=c['frames'],fps=c['fps'],q=c['q'],
                          video=target.name,video_sha256=file_sha(target),methods=methods))
        print('Viewer ready:',cid,flush=True)
    (out/'index.html').write_bytes((HERE/'viewer.html').read_bytes())
    (out/'live.js').write_bytes((HERE/'live.js').read_bytes())
    save_json(out/'summary.json',dict(schema='glc-inference-results-viewer-v1',clips=items,experiment=str(args.experiment.resolve()),
        viewing_proxies_only=True,new_neural_inference=False,metrics_source_sha256=file_sha(args.experiment/'quality.json')))


if __name__=='__main__':main()
