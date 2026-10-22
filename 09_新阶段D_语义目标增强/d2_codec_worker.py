"""Single-role D2 worker; reuse C's neural arithmetic, native entropy and decoder."""
from __future__ import annotations
import argparse
from fractions import Fraction
import os
from pathlib import Path
import sys
import time
import traceback

from prepare_d2 import read_json, save_json

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'08_新阶段C_编解码加速与网页展示'))


def run(args):
    config=read_json(args.config)
    role=config['role']
    if role not in ('encode','decode'):
        raise ValueError('Worker has one fixed role')
    started=time.perf_counter()
    root=Path(config['run_root'])
    def progress(**values):
        save_json(root/f'{role}_progress.json',dict(role=role,pid=os.getpid(),elapsed_seconds=time.perf_counter()-started,**values))
    progress(state='loading_codec')
    from c1_runtime import ResidentRunner, receiver_audit, ref
    from c2_native import NativeGaussianRans
    runtime=config['runtime']
    def factory(*a,**kw):
        model=ref.NeuralCodec(*a,**kw)
        model.entropy=NativeGaussianRans(runtime['native_library'],model.entropy)
        return model
    runner=ResidentRunner(role,runtime['repo'],runtime['weights'],runtime['ffmpeg'],codec_factory=factory)
    codec=runner.acquire()
    ref.torch.cuda.reset_peak_memory_stats()
    loaded=time.perf_counter()-started
    completed=[]
    anchors={}
    for task in config['tasks']:
        if (root/'STOP').exists():
            raise RuntimeError('Stopped by explicit STOP file')
        folder=Path(task['output'])
        folder.mkdir(parents=True,exist_ok=False)
        runner.reset()
        progress(state='running',task=task['id'],completed_tasks=len(completed),total_tasks=len(config['tasks']),frame=0)
        begin=time.perf_counter()
        if role=='encode':
            header={'width':task['width'],'height':task['height'],'frames':len(task['frames']),
                    'fps':float(Fraction(task['fps'])),'q':task['q'],'intra_period':64,'reset_interval':32,
                    'revision':ref.REVISION,'entropy':ref.ENTROPY_ID,'table_sha256':codec.entropy.table_sha256,
                    'arithmetic':codec.profile,'weights':{k:v['sha256'] for k,v in ref.ASSETS.items()},
                    'container_version':2}
            writer=ref.StreamWriter(folder/'stream.glcv',header,version=2)
            records=[]
            try:
                for index,source_path in enumerate(task['frames']):
                    if (root/'STOP').exists(): raise RuntimeError('Stopped at frame boundary')
                    source=ref.np.asarray(ref.Image.open(source_path).convert('RGB'))
                    tensor=ref.torch.from_numpy(source.copy()).to('cuda').permute(2,0,1).unsqueeze(0).float().div(255)
                    tensor=ref.F.pad(tensor,(0,(-header['width'])%64,0,(-header['height'])%64),mode='replicate').mul(2).sub(1)
                    tick=time.perf_counter()
                    rgb,packet=codec.frame(header,index,x=tensor)
                    seconds=time.perf_counter()-tick
                    size=writer.add(packet)
                    records.append({'index':index,'seconds':seconds,'record_bytes':size,
                                    'z_bytes':len(packet['z']),'y_bytes':sum(map(len,packet['y'])),
                                    'rgb_sha256':packet['rgb_sha256'],'latent_sha256':packet['latent_sha256']})
                    progress(state='running',task=task['id'],completed_tasks=len(completed),total_tasks=len(config['tasks']),frame=index+1)
                    if (index+1)%8==0 or index==0: print(f'{task["id"]} encode {index+1}/64 {seconds:.2f}s',flush=True)
                writer.finish()
            finally:
                writer.close()
            result={'task':task['id'],'header':header,'file_bytes':(folder/'stream.glcv').stat().st_size,
                    'pid':os.getpid(),'worker_phase':config['phase'],'worker_run_root':str(root),
                    'file_sha256':ref.sha256_file(folder/'stream.glcv'),'frames':records,
                    'seconds':time.perf_counter()-begin,'model_load_seconds':loaded if not completed else 0,
                    'native_entropy':True,'input_transform':'RGB/255, replicate right/bottom to multiples of64, *2-1',
                    'peak_allocated_mib':ref.torch.cuda.max_memory_allocated()/2**20,
                    'preprocessing_not_part_of_codec':True}
            save_json(folder/'encode.json',result)
            if config.get('record_legacy_q32_anchor',True) and task['method']=='A' and task['q']==32:
                anchors[str(task['video_id'])]={'task':task['id'],'S0':result['file_bytes'],
                       'B10':11*result['file_bytes']//10,'file_sha256':result['file_sha256'],
                       'frozen_before_non_A_encoding':True}
                save_json(root/'budget_anchors.json',anchors)
        else:
            # This config contains no original image, mask or corpus paths.
            options=argparse.Namespace(input=Path(task['input']),output=folder/'receiver',
                    output_mode='research',png_level=1,compatibility_policy='strict')
            previews=[]; marks={}
            def check_cancel():
                if (root/'STOP').exists(): raise RuntimeError('Stopped at frame boundary')
            original_frame=codec.frame
            def frame(header,index,*a,**kw):
                check_cancel()
                value=original_frame(header,index,*a,**kw)
                progress(state='running',task=task['id'],completed_tasks=len(completed),total_tasks=len(config['tasks']),frame=index+1)
                return value
            codec.frame=frame
            try:
                with receiver_audit() as reads:
                    result=runner._decode(options,codec,previews,check_cancel,marks,begin)
                for preview in previews: preview.finish(check_cancel)
                result.update(source_media_reads=reads,pid=os.getpid(),seconds=time.perf_counter()-begin,
                              model_load_seconds=loaded if not completed else 0,native_entropy=True,
                              worker_phase=config['phase'],worker_run_root=str(root))
                save_json(folder/'decode.json',result)
            except BaseException:
                for preview in previews: preview.abort()
                raise
            finally:
                codec.frame=original_frame
        completed.append(task['id'])
        print(f'D2_DONE {role} {task["id"]} {time.perf_counter()-begin:.1f}s',flush=True)
    progress(state='completed',completed_tasks=len(completed),total_tasks=len(config['tasks']))
    save_json(root/f'{config["phase"]}_summary.json',{'state':'completed','pid':os.getpid(),'tasks':completed,
               'seconds':time.perf_counter()-started,'model_load_seconds':loaded,
               'peak_allocated_mib':ref.torch.cuda.max_memory_allocated()/2**20})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    options=parser.parse_args()
    try:
        run(options)
    except BaseException as exc:
        config=read_json(options.config)
        save_json(Path(config['run_root'])/(config['role']+'_failure.json'),
                  {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
