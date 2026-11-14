"""Freeze five Q9 cases and locate local adverse corrections for visual diagnosis."""
import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from runtime_speed_common import STAGE,DATA,FFMPEG,file_sha,load_json,save_json,check_inputs
from receiver_structure import KEYS


def prepare(run):
    run.mkdir(exist_ok=False)
    previous=DATA/'generalization_reliability_20261112_02';long=DATA/'q9_long_transfer_20261112_03'
    old=load_json(previous/'execution_plan.json');cases={};frozen={}
    for case in ('video1','video2','video3','reef_spotted_fish','seagrass_spotted_fish'):
        if case.startswith('video'):
            s=old['cases'][case];cache=previous/case/'Q9/receive';rec=load_json(cache/'receipt.json')
            outputs=rec['variants']['R25']['outputs_sha256'];baseline=cache/'R25'
            packet=previous/case/'Q9/guide.glcguide';send=previous/case/'Q9/send.json'
            spec={k:s[k] for k in ('source','raw','base','base_receipt')};expected_source=old['input_sha256'][s['source']]
            hashes=dict(rec['flow_sha256'],guide_rgb=rec['guide_rgb_sha256'])
        else:
            root=long/'archive_transfer'/case;cache=root/'post';rec=load_json(cache/'receipt.json');baseline=cache
            packet=root/'guide/guide.glcguide';send=root/'guide/send.json';outputs=rec['outputs_sha256'];hashes=rec['local_cache_sha256']
            spec=dict(source=str(root/'source.npy'),raw=str(root/'base_receive/decoded_rgb.npy'),base=str(root/'send/base.glcsnd'),base_receipt=str(root/'base_receive/receipt.json'))
            expected_source=load_json(root/'source_receipt.json')['source_sha256']
        assert file_sha(spec['source'])==expected_source
        base_rec=load_json(spec['base_receipt']);assert base_rec['reference_verified'] and base_rec['source_media_reads_forbidden']
        assert file_sha(spec['raw'])==base_rec['decoded_rgb_sha256'] and file_sha(spec['base'])==rec['base_sha256']==base_rec['stream_sha256']
        assert file_sha(packet)==rec.get('auxiliary_sha256',rec.get('guide_sha256'))
        for name,h in hashes.items():assert file_sha(cache/(name+'.npy'))==h
        for name,h in outputs.items():assert file_sha(baseline/(name+'.npy'))==h
        spec.update(cache=str(cache),baseline=str(baseline),packet=str(packet),send_receipt=str(send),cache_receipt=str(cache/'receipt.json'),
                    cache_sha256=hashes,baseline_sha256=outputs,frames=150,fps=15,reef=old['cases'].get(case,{}).get('reef'))
        cases[case]=spec
        for p in [*(Path(spec[k]) for k in ('source','raw','base','base_receipt','packet','send_receipt','cache_receipt')),
                  *(cache/(n+'.npy') for n in hashes),*(baseline/(n+'.npy') for n in outputs)]:frozen[str(p)]=file_sha(p)
    cfg=STAGE/'当前采用配置.json';frozen[str(cfg)]=file_sha(cfg)
    save_json(run/'case_registration.json',dict(date='2026-11-12',authorization='那继续吧',reference='Q9_R25',
        cases=cases,input_sha256=frozen,scope='Five historical 10-second cases, not heldout; long video excluded from candidate tuning.',
        diagnosis='Two highest 48x48 local positive increases in sigma3 source error under Q9; times separated by >=16 frames. Candidate-independent review locations, not semantic fish labels.',
        source_use='Evaluation/diagnosis only. No source mask or chosen ROI is passed to receiver.',
        feedback=str(long/'user_feedback_ghosting_20261112.json'),diagnosis_code_sha256=file_sha(__file__)))
    snap=run/'diagnosis_source';snap.mkdir();shutil.copy2(__file__,snap/Path(__file__).name)
    print('REGISTERED',list(cases),flush=True)


def diagnosis(run):
    reg=load_json(run/'case_registration.json');out=run/'diagnosis';out.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15);events={}
    for case,s in reg['cases'].items():
        seq={n:np.load(s[k],mmap_mode='r') for n,k in [('source','source'),('raw','raw')]}
        seq['Q9']=np.load(Path(s['baseline'])/'rgb.npy',mmap_mode='r');seq['guide']=np.load(Path(s['cache'])/'guide_rgb.npy',mmap_mode='r')
        alpha=np.load(Path(s['baseline'])/'alpha.npy',mmap_mode='r');rank=[]
        for i in range(1,150):
            a,b,c=[cv2.GaussianBlur(seq[n][i].astype(np.float32),(0,0),3) for n in ('source','raw','Q9')]
            harm=np.maximum(np.abs(c-a).mean(2)-np.abs(b-a).mean(2),0)
            harm*=((alpha[i]>0)&(np.max(np.abs(c-b),axis=2)>=4))
            local=cv2.boxFilter(harm,-1,(48,48),normalize=True,borderType=cv2.BORDER_CONSTANT)
            local[:24]=0;local[-24:]=0;local[:,:24]=0;local[:,-24:]=0
            y,x=np.unravel_index(local.argmax(),local.shape)
            rank.append(dict(frame=i,score=float(local[y,x]),center=[int(x),int(y)]))
        chosen=[]
        for r in sorted(rank,key=lambda x:x['score'],reverse=True):
            if all(abs(r['frame']-p['frame'])>=16 for p in chosen):chosen.append(r)
            if len(chosen)==2:break
        events[case]=chosen
        full=Image.new('RGB',(1536,314*len(chosen)),(24,35,48));draw=ImageDraw.Draw(full)
        for row,e in enumerate(chosen):
            i=e['frame'];cx,cy=e['center'];x0=max(0,min(cx-48,416));y0=max(0,min(cy-48,192));box=[x0,y0,x0+96,y0+96];e['box']=box
            key=KEYS[np.searchsorted(KEYS,i,side='right')-1];e['anchor']=key
            for col,n in enumerate(('source','raw','Q9')):
                image=seq[n][i].copy();cv2.rectangle(image,(x0,y0),(x0+96,y0+96),(255,80,30),1)
                full.paste(Image.fromarray(image),(col*512,row*314+26));draw.text((col*512+5,row*314+4),f'{case} {n} f{i} / {i/15:.2f}s',font=font,fill='white')
            ids=sorted(set([key,max(key,i-8),max(key,i-4),i,min(i+4,149)]));e['review_frames']=list(map(int,ids))
            sheet=Image.new('RGB',(5*240,len(ids)*264),(24,35,48));d=ImageDraw.Draw(sheet)
            for row2,t in enumerate(ids):
                a=alpha[t,y0:y0+96,x0:x0+96,None]
                overlay=np.rint(seq['raw'][t,y0:y0+96,x0:x0+96]*(1-.5*a)+np.array([0,255,60])*.5*a).astype(np.uint8)
                imgs=[seq[n][t,y0:y0+96,x0:x0+96] for n in ('source','raw','Q9','guide')]+[overlay]
                for col2,(label,img) in enumerate(zip(('source','raw','Q9','guide','alpha'),imgs)):
                    d.text((col2*240+4,row2*264+3),f'{label} f{t}',font=font,fill='white')
                    sheet.paste(Image.fromarray(img).resize((240,240)),(col2*240,row2*264+24))
            sheet.save(out/f'{case}_event{len([r for r in chosen if r["frame"]>=i])}_f{i}.png')
        full.save(out/(case+'_locations.png'));print('DIAGNOSIS',case,chosen,flush=True)
    save_json(out/'events.json',dict(cases=events,selection_uses_baseline_only=True,not_fish_segmentation=True,requires_visual_interpretation=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','diagnose'));p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    (prepare if a.action=='prepare' else diagnosis)(a.run)
