"""Explicit four-frame web acceptance: encode once, decode three ways on the shared queue."""
from __future__ import annotations
import json
import subprocess
import time
from urllib.request import Request,build_opener,ProxyHandler
from decoder_common import *


def main():
    base='http://127.0.0.1:8773';opener=build_opener(ProxyHandler({}))
    def api(path,body=None):
        request=Request(base+'/inference/api/'+path,data=json.dumps(body).encode() if body is not None else None,
                        headers={'Content-Type':'application/json'})
        with opener.open(request,timeout=10) as response:return json.load(response)
    config=api('config')
    if not config['ready'] or len(config['decoder_modes'])!=3:raise ValueError('Decoder options are not ready')
    if any(j['status'] not in ('done','error','cancelled','interrupted','staged') for j in api('jobs')):
        raise ValueError('A user task is active; do not enqueue acceptance now')
    options=dict(sample='long108_q0_640',modes=['eager'],decoder_modes=list(BACKENDS),
                 range='first_frames',frames=4,size=640,q=0,fps_mode='original')
    job=api('jobs',options);identifier=job['id'];api('jobs/'+identifier+'/start',{})
    print('Started explicit four-frame web acceptance: '+identifier,flush=True)
    started=time.monotonic();last=''
    while True:
        job=api('jobs/'+identifier)
        if job['phase']!=last:print(job['phase'],flush=True);last=job['phase']
        if job['status'] in ('done','error','cancelled','interrupted'):break
        if time.monotonic()-started>600:raise TimeoutError('Web acceptance exceeded ten minutes')
        time.sleep(.5)
    output=TRIAL/'web_acceptance';save_json(output/'job.json',job)
    if job['status']!='done' or len(job['results'])!=3:raise ValueError('Web decoder acceptance failed: '+str(job.get('error')))
    directory=WORKSPACE/'02_实验数据/C_编解码加速与网页展示/C5_loopback/sender/inference_lab'/identifier
    send=load_json(directory/'eager_sender/result.json')
    probes=[];ranges=[]
    for row in job['results']:
        receive=load_json(directory/row['receiver_folder']/'result.json')
        if (row['frames']!=4 or not receive['reference_verified'] or not receive['source_media_reads_forbidden'] or
                receive['reference_checksums']!=send['reference_checksums'] or receive['stream_sha256']!=send['stream_sha256']):
            raise ValueError('Receiver does not independently verify the shared stream')
        if row['decoder_mode']!='eager' and any(info['built'] for info in receive['display_setup']['generators'].values()):
            raise ValueError('Expected a validated engine cache hit')
        ffprobe=Path(load_settings()['runtime']['ffmpeg']).with_name('ffprobe.exe')
        probe=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-select_streams','v:0','-count_frames',
            '-show_entries','stream=width,height,nb_read_frames,avg_frame_rate','-of','json',
            str(directory/row['receiver_folder']/'video.mp4')],encoding='utf-8'))['streams'][0]
        if int(probe['nb_read_frames'])!=4:raise ValueError('Web preview lost frames')
        probes.append(dict(decoder=row['decoder_mode'],**probe))
        for url in (row['video'],row['stream']):
            with opener.open(Request(base+url,headers={'Range':'bytes=0-31'}),timeout=10) as response:
                if response.status!=206 or len(response.read())!=32:raise ValueError('Playback/download Range failed')
            ranges.append(url)
    if len(list(directory.glob('*_sender.config.json')))!=1:raise ValueError('Decoder comparisons unnecessarily reencoded the input')
    if len({r['stream_sha256'] for r in job['results']})!=1 or len({r['video'] for r in job['results']})!=3:
        raise ValueError('Bitstream identity or preview routing differs')
    before=load_json(output/'prior_jobs.json');after={j['id']:j['status'] for j in api('jobs')}
    if any(after.get(j['id'])!=j['status'] for j in before):raise ValueError('Existing job history changed')
    receipt=dict(passed=True,job=identifier,frames=4,encodes=1,independent_decodes=3,engine_cache_hits=True,
                 unchanged_old_jobs=len(before),same_bitstream=True,reference_chains_verified=True,
                 range_requests=len(ranges),probes=probes,results=job['results'],
                 note='Functional web acceptance only; these four-frame timings are not the performance report')
    save_json(output/'acceptance.json',receipt)
    save_json(HERE/'evidence_trt_20261031/web_acceptance.json',receipt)
    print('PASS web: one encode, three independent decoders, same stream, all previews/downloads verified',flush=True)


if __name__=='__main__':main()
