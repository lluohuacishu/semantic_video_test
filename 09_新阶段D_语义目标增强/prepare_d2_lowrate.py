"""Prepare only two larger-fish windows, original timing, and a fixed resize ladder."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import numpy as np
from PIL import Image, ImageFilter
from pycocotools import mask as coco_mask
from prepare_d2 import read_json, save_json
from prepare_d0 import resolve_rgb


def run(args):
    root=args.run.resolve(); plan=read_json(args.plan);root.mkdir(parents=True,exist_ok=True)
    if (root/'inputs.json').exists(): raise FileExistsError('Prepared inputs already exist')
    save_json(root/'plan.json',plan)
    ffmpeg=Path(args.ffmpeg); clips=[]
    (root/'sources').mkdir(exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        names=set(archive.namelist())
        for spec in plan['clips']:
            vid=spec['video_id']; first=spec['first_filename_number']; count=spec['frames']
            data=json.loads(archive.read(f'SFISHTRACK/Annotations/video{vid:03d}.json'))
            source_video=root/'sources'/f'video_{vid:03d}.mp4'
            if not source_video.exists():
                with archive.open(f'SFISHTRACK/Videos/video_{vid:03d}.mp4') as source,source_video.open('xb') as target:
                    shutil.copyfileobj(source,target)
            probe_path=root/'sources'/f'probe_{vid:03d}.json'
            if not probe_path.exists():
                probe=json.loads(subprocess.check_output([str(ffmpeg.with_name('ffprobe.exe')),'-v','error','-select_streams','v:0',
                    '-show_streams','-show_frames','-show_entries','stream=width,height,time_base,avg_frame_rate,r_frame_rate,nb_frames,duration:frame=best_effort_timestamp,duration',
                    '-of','json',str(source_video)]))
                save_json(probe_path,probe)
            frames=sorted([f for f in data['images'] if first<=int(Path(f['file_name']).stem)<first+count],key=lambda f:int(Path(f['file_name']).stem))
            if len(frames)!=count: raise ValueError('Selected annotated window has gaps')
            annotations=defaultdict(list)
            for a in data['annotations']: annotations[a['image_id']].append(a)
            folder=root/'inputs'/f'v{vid:03d}'; folder.mkdir(parents=True,exist_ok=False)
            (folder/'original').mkdir(); (folder/'gt_original').mkdir()
            for w,h in plan['delivery']['resolution_ladder']:
                (folder/f'{w}x{h}'/'A').mkdir(parents=True); (folder/f'{w}x{h}'/'gt').mkdir()
            probe=read_json(root/'sources'/f'probe_{vid:03d}.json'); stream=probe['streams'][0]
            tb=Fraction(stream['time_base']); timestamps=[int(f['best_effort_timestamp']) for f in probe['frames']]
            steps={b-a for a,b in zip(timestamps,timestamps[1:])}
            if len(steps)!=1 or min(steps)<=0: raise ValueError('This pilot requires true CFR; do not relabel VFR')
            fps=1/(next(iter(steps))*tb)
            raw=subprocess.check_output([str(ffmpeg),'-v','error','-threads','2','-i',str(root/'sources'/f'video_{vid:03d}.mp4'),'-an','-vf','scale=240:135:flags=bilinear','-fps_mode','passthrough','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
            video=np.frombuffer(raw,dtype=np.uint8).reshape(-1,135,240,3)
            if len(video)!=len(timestamps): raise ValueError('Decoded timeline differs from source PTS')
            rows=[]; matches=[]; track_area=defaultdict(int); all_instances=[]
            for index,f in enumerate(frames):
                member=resolve_rgb(names,vid,f['file_name']); original=folder/'original'/Path(member).name
                original.write_bytes(archive.read(member)); rgb=Image.open(original).convert('RGB')
                gt=np.zeros((f['height'],f['width']),dtype=bool); instances=annotations[f['id']]
                for a in instances:
                    gt |= coco_mask.decode(a['segmentation']).astype(bool)
                    track_area[a['track_id']]+=a['area']
                gt_im=Image.fromarray(gt.astype(np.uint8)*255); gp=folder/'gt_original'/f'{index:04d}.png';gt_im.save(gp)
                reference=np.asarray(rgb.resize((240,135),Image.Resampling.BILINEAR),dtype=np.float32)
                roi=np.asarray(gt_im.resize((240,135),Image.Resampling.NEAREST).filter(ImageFilter.MaxFilter(5)))>0
                if not roi.any(): raise ValueError('Selected large-fish frame has no GT target')
                errors=[]
                for start in range(0,len(video),32):
                    delta=video[start:start+32,roi].astype(np.float32)-reference[roi]
                    errors.extend(np.square(delta).mean(axis=(1,2)).tolist())
                order=np.argsort(errors); best,second=map(int,order[:2])
                mse=float(np.square(video[best].astype(np.float32)-reference).mean())
                psnr=float(10*np.log10(255**2/max(mse,1e-12))); ratio=errors[second]/max(errors[best],1e-12)
                if psnr<30 or ratio<1.03: raise ValueError(f'Ambiguous source timing {vid}/{f["file_name"]}: {psnr}, {ratio}')
                matches.append({'filename':f['file_name'],'source_index':best,'pts':timestamps[best],'global_psnr_db':psnr,'runner_up_error_ratio':ratio})
                resized={}
                for w,h in plan['delivery']['resolution_ladder']:
                    rp=folder/f'{w}x{h}'/'A'/f'{index:04d}.png'; mp=folder/f'{w}x{h}'/'gt'/f'{index:04d}.png'
                    rgb.resize((w,h),Image.Resampling.LANCZOS).save(rp,compress_level=1)
                    gt_im.resize((w,h),Image.Resampling.NEAREST).save(mp)
                    resized[f'{w}x{h}']={'rgb':str(rp),'gt':str(mp)}
                rows.append({'index':index,'filename':f['file_name'],'original_rgb':str(original),'original_gt':str(gp),'source_index':best,'source_pts':timestamps[best],'fish_fraction':float(gt.mean()),'largest_instance_fraction':max(a['area'] for a in instances)/gt.size,'resized':resized})
                all_instances.extend(instances)
            indices=[m['source_index'] for m in matches]
            if any(b!=a+1 for a,b in zip(indices,indices[1:])): raise ValueError('Source matches are not a consecutive original-rate sequence')
            track=max(track_area,key=track_area.get); boxes=[a['bbox'] for a in all_instances if a['track_id']==track]
            crop=[max(0,int(min(b[0] for b in boxes))-48),max(0,int(min(b[1] for b in boxes))-48),min(1920,int(max(b[0]+b[2]+1 for b in boxes))+48),min(1080,int(max(b[1]+b[3]+1 for b in boxes))+48)]
            duration=Fraction(count,1)/fps
            clip={'video_id':vid,'scene':spec['scene'],'role':'development_by_user_requested_reselection','fps':str(fps),'duration_exact':str(duration),'source_size':[1920,1080],'frames':rows,'fish_crop_original':crop,'crop_track':track,'rate_cap_bytes':int(40000*duration/8),'mean_fish_fraction':float(np.mean([x['fish_fraction'] for x in rows])),'mean_largest_instance_fraction':float(np.mean([x['largest_instance_fraction'] for x in rows]))}
            clips.append(clip)
            save_json(folder/'timing.json',{'fps':str(fps),'time_base':str(tb),'matches':matches,'consecutive_original_frames':True,'integrity_rescan':False})
            print(f'Prepared {vid:03d}: fish {clip["mean_fish_fraction"]:.2%}, fps {fps}, cap {clip["rate_cap_bytes"]} B',flush=True)
    save_json(root/'inputs.json',{'clips':clips,'integrity_rescan':False,'source_timeline_preserved':True})
    save_json(root/'preparation.json',{'state':'completed','source_frames':sum(len(c['frames']) for c in clips),'training':False,'model_inference':False,'integrity_rescan':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--plan',type=Path,default=Path(__file__).with_name('d2_lowrate_plan_v1.json'))
    p.add_argument('--ffmpeg',required=True);run(p.parse_args())
