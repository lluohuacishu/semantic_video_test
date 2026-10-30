"""Reuse verified arrays; add SF025 q32 and all 255 consecutive UV096 q63 frames."""
from __future__ import annotations
import argparse
from PIL import Image
import numpy as np
from experiment_common import *


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    old=load_json(REPO/'14_优化阶段_部署精简/evidence_20261029/plan.json')
    clips=[dict(x) for x in old['clips']]
    sf=WORKSPACE/'02_实验数据/优化_GLC蒸馏/pilot_20261029_01/v025/source.npy'
    a=np.load(sf,mmap_mode='r')
    clips.append(dict(id='sf025_q32_512',source=str(sf),source_sha256=file_sha(sf),width=512,height=288,
                      frames=len(a),fps=30.,q=32,timebase='frame order; nominal FPS is not capture timing'))
    uv=Path('C:/Users/lenovo/Desktop/works/work/semantic-video-mlvc/data/phase_c_20261013')
    meta=next(x for x in load_json(uv/'manifest.json')['videos'] if x['video']=='video96')
    files=sorted(Path(meta['sequence_dir']).glob('*.jpg'),key=lambda x:int(x.stem))
    indexes=[int(x.stem) for x in files]
    if indexes != list(range(255)):raise ValueError('UV096 sequence is not the expected consecutive 255 frames')
    path=out/'uv096_255_512.npy'
    target=np.lib.format.open_memmap(path,mode='w+',dtype='uint8',shape=(255,288,512,3))
    for i,f in enumerate(files):
        with Image.open(f) as im:target[i]=np.asarray(im.convert('RGB').resize((512,288),Image.Resampling.BICUBIC))
    target.flush();del target
    clips.append(dict(id='uv096_q63_512_255',source=str(path),source_sha256=file_sha(path),width=512,height=288,
                      frames=255,fps=25.,q=63,source_sequence_sha256={f.name:file_sha(f) for f in files},
                      timebase='paper-declared nominal 25 fps, not independently verified'))
    paths=[*HERE.glob('*.py'),REPO/'14_优化阶段_部署精简/role_codec.py',
           REPO/'15_优化阶段_算子精确排查/profile_worker.py',
           REPO/'04_外部依赖快照/semantic-video-training/scripts/glc_video_codec.py']
    plan=dict(schema='glc-encoder-inference-v1',date='2026-10-30',assets=old['assets'],
              assets_manifest_sha256=old['assets_manifest_sha256'],clips=clips,warmup_frames=8,
              fill_pairs=2,encoder_warmup=8,encoder_repeats=3,encoder_samples=[0,2,33],
              encoder_iterations=12,code_sha256={f.relative_to(REPO).as_posix():file_sha(f) for f in paths},
              scope='same RTX4060; no training; defaults unchanged; two paired repeats per expanded case',
              source_independence_established=False,
              sources_note='UV and SFishTrack already used in project; no new claim of capture-source independence',
              long_sequence_note='255 consecutive frames, four GOPs; not a long-duration deployment soak test')
    save_json(out/'plan.json',plan);save_json(out/'plan_identity.json',dict(sha256=file_sha(out/'plan.json')))
    print('Frozen 5 clips,',sum(c['frames'] for c in clips),'frames, q=0/12/17/32/63',flush=True)


if __name__=='__main__':main()
