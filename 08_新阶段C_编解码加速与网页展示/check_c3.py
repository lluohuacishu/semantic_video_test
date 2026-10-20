"""Read-only C3 integration audit. Never encodes media or starts a GPU worker."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from c3_catalog import Catalog, HERE, REPO, WORKSPACE, read_json, sha256
from c3_server import verify_result
from c2_watch import inspect, write, utc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--download',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    cat=Catalog()
    enc,dec=verify_result(args.job)
    result=read_json(args.job/'result.json')
    if len(enc['frames'])!=8 or enc['selected_preset']!='compact': raise ValueError('Expected finite eight-frame compact integration job')
    gold=WORKSPACE/'02_实验数据/C_编解码加速与网页展示/C2_combined_gate8_20261020_r2/encode_1_native/stream.glcv'
    if sha256(gold)!=sha256(args.job/'received.glcv'): raise ValueError('Stream differs from C2 eight-frame reference')
    reference=cat.json_asset('c2.compact.encode_result')
    if enc['input']['rgb_sha256']!=reference['input']['rgb_sha256'][:8]: raise ValueError('Input RGB differs from frozen main clip')
    for actual,expected in zip(enc['frames'],reference['frames']):
        if any(actual[k]!=expected[k] for k in ('rgb_sha256','latent_sha256')): raise ValueError('RGB or latent differs from C2')
    processes={role:inspect(args.job/(role+'_process')) for role in ('encode','decode')}
    for status in processes.values():
        if status['observed_state']!='completed' or status['recorded']['returncode']!=0: raise ValueError('Worker not cleanly complete')
    ffprobe=Path('C:/Users/lenovo/Desktop/works/Python/video-baseline/tools/ffmpeg-9.0.1-essentials_build/bin/ffprobe.exe')
    previews={}
    for name,path in {'source':args.job/'encode/encoder/source_preview.mp4','decoded':args.job/'decode/receiver/reconstruction_preview.mp4'}.items():
        completed=subprocess.run([str(ffprobe),'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(path)],capture_output=True,text=True,check=True)
        stream=json.loads(completed.stdout)['streams'][0]
        if (stream['width'],stream['height'],stream['avg_frame_rate'],int(stream['nb_read_frames']))!=(640,360,'25/2',8): raise ValueError('Preview metadata mismatch')
        previews[name]={'sha256':sha256(path),**stream}
    if sha256(args.download)!=sha256(args.job/'received.glcv'): raise ValueError('Browser downloaded different stream')
    browser=read_json(args.validation/'browser_checks.json')
    if browser['console_errors'] or browser['mobile']['overflow']: raise ValueError('Browser checks failed')
    regression=[]
    for suite in ('c','b'):
        path=args.validation/(suite+'_regression_final.log')
        text=path.read_text(encoding='utf-8-sig')
        count=re.search(r'Ran (\d+) tests',text)
        if not count or not text.rstrip().endswith('OK'): raise ValueError('Regression not complete: '+suite)
        regression.append({'suite':suite,'tests':int(count[1]),'path':str(path),'sha256':sha256(path)})
    files=list(HERE.glob('c3_*.py'))+list((HERE/'c3_console').glob('*'))+[HERE/'check_c3.py',HERE/'test_c3_web.py']
    outputs={str(p.relative_to(args.job)):sha256(p) for p in args.job.rglob('*.json')}
    summary={'created_utc':utc(),'stage':'C3','status':'complete_with_stated_limits',
             'base_commit':'7e1da1dc99e4d8b04807f6d0d6f1b8161b4a00c4','web_url':'http://127.0.0.1:8773/',
             'catalog':cat.payload['asset_check'],'regression':regression,
             'regression_total':sum(r['tests'] for r in regression),
             'integration':{'job':str(args.job),'result':result,'previews':previews,'processes':processes,
                            'matches_c2_gate8_stream':True,'matches_c2_source_and_frame_hashes':True,
                            'browser_download_matches':True,'browser_timings':read_json(args.job/'browser.json'),
                            'json_sha256':outputs},
             'browser':browser,'validation_root':str(args.validation),
             'source_sha256':{str(p.relative_to(REPO)):sha256(p) for p in files},
             'retained_failure':{'job_id':'80b31d9b1d5e44eb8ccd1f4a829bda9e','cause':'C3 pre-created process directory owned by C2 supervise; WinError 183 before worker launch; fixed and regression added'},
             'limits':['One eight-frame integration job, not a full video speed benchmark.',
                       'New web jobs use separate fresh processes; historical resident performance is replay only.',
                       'Browser currentTime synchronization is not strict frame lock.',
                       'Active cancellation and queue recovery tested with controlled CPU fixtures; no extra real-GPU cancellation experiment.',
                       'Browser smoke checks on this machine only; no cross-browser/device or long-running stability claim.',
                       'No real network/acoustic transfer, no training, no C2 matrix rerun. C4 remains unstarted.']}
    write(args.output,summary)
    print(json.dumps({'status':summary['status'],'tests':summary['regression_total'],'stream_bytes':enc['file_bytes'],'service_seconds':result['service_seconds']},ensure_ascii=False))


if __name__=='__main__': main()
