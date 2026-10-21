"""Create local D1 diagnostic sheets and correctly timed overlay previews."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from d1_protection import seed_masks


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def fit(image, width, height):
    result=Image.new('RGB',(width,height),(17,24,34))
    image=image.copy()
    image.thumbnail((width,height),Image.Resampling.LANCZOS)
    result.paste(image,((width-image.width)//2,(height-image.height)//2))
    return result


def outline(mask):
    return ndimage.maximum_filter(mask,size=3) ^ ndimage.minimum_filter(mask,size=3)


def overlay(image,gt,seed,hard):
    rgb=np.asarray(image).copy()
    rgb[hard]=(rgb[hard]*.80+np.array([20,190,255])*.20).astype(np.uint8)
    rgb[outline(hard)]=[30,190,255]
    rgb[outline(seed)]=[250,130,40]
    rgb[outline(gt)]=[90,255,90]
    return Image.fromarray(rgb)


def run(args):
    plan=read_json(args.run/'plan.json')
    frozen=read_json(args.run/'frozen_protection.json')
    inputs=read_json(args.run/'inputs.json')['inputs']
    metrics=read_json(args.run/'frame_metrics.json')
    manifest=read_json(args.corpus/'ready_manifest.json')
    suim={r['id']:r for r in manifest['suim']['images']}
    fish={f'sfishtrack_{s["video_id"]:03d}/{f["original_file_name"]}':f for s in manifest['fish_sequences'] if s['role']=='development' for f in s['frames']}
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    large=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',24)
    def frame_images(i):
        source=inputs[i]
        with Image.open(source['rgb_path']) as original:
            image=original.convert('RGB')
        if source.get('video_id'):
            gt=np.asarray(Image.open(args.corpus/fish[source['id']]['mask_path']))!=0
        else:
            indexed=np.asarray(Image.open(suim[source['id'][5:]]['eight_class_mask_path']))
            gt=(indexed==1)|(indexed==6)
        packed=np.asarray(Image.open(args.run/'packed'/f'{i:04d}.png'))
        f,d=seed_masks(packed,plan,frozen['config'])
        hard=np.asarray(Image.open(args.run/'selected_hard'/f'{i:04d}.png'))!=0
        return image,gt,f|d,hard
    def legend(draw,top):
        draw.text((18,top),'Green: labeled target   Orange: raw prediction   Blue: protection area',font=font,fill=(230,235,245))
    # Fixed frames, including misses; no hand-picked successful overview.
    sheet=Image.new('RGB',(1440,960),(17,24,34))
    draw=ImageDraw.Draw(sheet)
    draw.text((18,12),'D1 fish development: selected protection, not reconstructed video',font=large,fill='white')
    legend(draw,46)
    for row,video_id in enumerate((1,2,3)):
        for col,frame_index in enumerate((0,31,63)):
            i=(video_id-1)*64+frame_index
            image,gt,seed,hard=frame_images(i)
            x,y=col*480,82+row*290
            sheet.paste(fit(overlay(image,gt,seed,hard),480,250),(x,y+30))
            m=metrics[i]['candidates'][frozen['selected_name']]['protection']
            draw.text((x+10,y),f'Video {video_id:03d} / frame {frame_index}: coverage {m["recall"]:.1%}',font=font,fill='white')
    sheet.save(args.run/'fish_overview.jpg',quality=94)
    # One fixed and one lowest-coverage frame per clip, target-centered crops.
    examples=[]
    for video_id in (1,2,3):
        indices=list(range((video_id-1)*64,video_id*64))
        worst=min(indices,key=lambda i:metrics[i]['candidates'][frozen['selected_name']]['protection']['recall'])
        examples.append(indices[31])
        if worst!=indices[31]: examples.append(worst)
    sheet=Image.new('RGB',(1280,85+len(examples)*240),(17,24,34))
    draw=ImageDraw.Draw(sheet)
    draw.text((18,12),'D1 target crops: fixed middle frame + lowest coverage per clip',font=large,fill='white')
    for col,title in enumerate(('Original RGB','GT target','Predicted seeds','Protected region')):
        draw.text((col*320+12,52),title,font=font,fill='white')
    for row,i in enumerate(examples):
        image,gt,seed,hard=frame_images(i)
        yy,xx=np.where(gt)
        box=(max(0,int(xx.min())-64),max(0,int(yy.min())-64),min(image.width,int(xx.max())+65),min(image.height,int(yy.max())+65))
        blank=np.zeros_like(gt)
        panels=(image,overlay(image,gt,blank,blank),overlay(image,blank,seed,blank),overlay(image,gt,seed,hard))
        for col,panel in enumerate(panels):
            sheet.paste(fit(panel.crop(box),320,205),(col*320,85+row*240))
        rec=metrics[i]
        coverage=rec['candidates'][frozen['selected_name']]['protection']['recall']
        draw.text((12,294+row*240),f'Video {rec["video_id"]:03d}, frame {rec["index"]}: coverage {coverage:.1%}',font=font,fill='white')
    sheet.save(args.run/'fish_target_crops.jpg',quality=95)
    sheet=Image.new('RGB',(1280,82+6*265),(17,24,34))
    draw=ImageDraw.Draw(sheet)
    draw.text((18,12),'D1 SUIM: 20 diver images + 4 labeled no-target controls',font=large,fill='white')
    legend(draw,46)
    for j,i in enumerate(range(192,216)):
        image,gt,seed,hard=frame_images(i)
        x,y=(j%4)*320,82+(j//4)*265
        sheet.paste(fit(overlay(image,gt,seed,hard),320,230),(x,y+30))
        coverage=metrics[i]['candidates'][frozen['selected_name']]['protection']['recall']
        title=f'SUIM {j:02d}: '+(f'diver coverage {coverage:.1%}' if coverage is not None else 'labeled no-target')
        draw.text((x+8,y),title,font=font,fill='white')
    sheet.save(args.run/'suim_overview.jpg',quality=94)
    preview_paths=[]
    if args.ffmpeg:
        for video_id in (1,2,3):
            destination=args.run/f'video_{video_id:03d}_overlay.mp4'
            command=[str(args.ffmpeg),'-hide_banner','-loglevel','error','-n',
                     '-f','rawvideo','-pix_fmt','rgb24','-video_size','1280x432',
                     '-framerate','30000/1001','-i','pipe:0','-an','-c:v','libx264',
                     '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            try:
                for i in range((video_id-1)*64,video_id*64):
                    image,gt,seed,hard=frame_images(i)
                    canvas=Image.new('RGB',(1280,432),(17,24,34)); drawing=ImageDraw.Draw(canvas)
                    canvas.paste(fit(image,640,360),(0,72))
                    canvas.paste(fit(overlay(image,gt,seed,hard),640,360),(640,72))
                    rec=metrics[i]; coverage=rec['candidates'][frozen['selected_name']]['protection']['recall']
                    drawing.text((14,6),f'Original | video {video_id:03d} frame {rec["index"]}',font=large,fill='white')
                    drawing.text((654,6),f'D1 protection | GT coverage {coverage:.1%}',font=large,fill='white')
                    legend(drawing,40)
                    process.stdin.write(canvas.tobytes())
                process.stdin.close()
                error=process.stderr.read().decode('utf-8',errors='replace')
                if process.wait()!=0: raise RuntimeError(error)
            except BaseException:
                process.kill();process.wait();raise
            preview_paths.append(destination.name)
            print('Saved '+destination.name,flush=True)
    record={'sheets':['fish_overview.jpg','fish_target_crops.jpg','suim_overview.jpg'],
            'fish_examples':examples,'videos':preview_paths,'video_frame_rate':'30000/1001',
            'video_frame_count_each':64,'purpose':'Localization visualization only; lossy preview is never used for quality metrics.',
            'no_visual_claim_of_flicker_elimination':True}
    (args.run/'visualizations.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Diagnostic sheets complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--ffmpeg',type=Path)
    run(parser.parse_args())
