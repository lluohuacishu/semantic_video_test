"""CPU-rendered comparison from verified independent decoder outputs, after timing."""
from __future__ import annotations
import argparse
import subprocess
from PIL import Image,ImageDraw,ImageFont
from experiment_common import *


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;plan=load_json(out/'plan.json');quality=load_json(out/'quality.json')
    destination=out/'previews';destination.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    rendered={}
    for cid in ['long108_q0_640','sf025_q32_512']:
        c=next(c for c in plan['clips'] if c['id']==cid)
        arrays={'原始帧':np.load(c['source'],mmap_mode='r')}
        for key,variant in [('原执行方式','eager'),('TensorRT FP16','trt_fp16_stream')]:
            f=out/'full'/cid/variant/'receiver/decoded_rgb.npy'
            result=load_json(f.parent/'result.json')
            if file_sha(f)!=result['decoded_rgb_sha256']:raise ValueError('Preview source changed')
            arrays[key]=np.load(f,mmap_mode='r')
        index=33;w,h=c['width'],c['height'];canvas=Image.new('RGB',(w*2,(h+34)*2),'#18202a')
        delta=np.clip(np.abs(arrays['TensorRT FP16'][index].astype('int16')-arrays['原执行方式'][index].astype('int16'))*8,0,255).astype('uint8')
        for i,(title,rgb) in enumerate([*( (k,v[index]) for k,v in arrays.items()),('RGB差异×8：对原执行方式',delta)]):
            x=(i%2)*w;y=(i//2)*(h+34)
            canvas.paste(Image.fromarray(rgb),(x,y+34));ImageDraw.Draw(canvas).text((x+8,y+5),title,font=font,fill='white')
        path=destination/f'{cid}_frame034.png';canvas.save(path);rendered[path.name]=dict(sha256=file_sha(path),zero_based_index=index)
        if cid.startswith('long108'):
            path=destination/'long108_frame_order_comparison.mp4'
            cmd=[str(load_settings()['runtime']['ffmpeg']),'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                 '-s',f'{w*3}x{h+34}','-r',str(c['fps']),'-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','16','-pix_fmt','yuv420p',str(path)]
            process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
            try:
                for frame in range(c['frames']):
                    image=Image.new('RGB',(w*3,h+34),'#18202a');draw=ImageDraw.Draw(image)
                    for i,(name,a) in enumerate(arrays.items()):
                        image.paste(Image.fromarray(a[frame]),(i*w,34));draw.text((i*w+8,5),f'{name} · 帧序{frame+1:03d}',font=font,fill='white')
                    process.stdin.write(image.tobytes())
                process.stdin.close()
                if process.wait()!=0:raise RuntimeError('Comparison FFmpeg failed')
            except BaseException:
                process.kill();process.wait();raise
            rendered[path.name]=dict(sha256=file_sha(path),fps=c['fps'],viewing_proxy='H264 CRF16; metrics use raw RGB')
    save_json(destination/'manifest.json',rendered)


if __name__=='__main__':main()
