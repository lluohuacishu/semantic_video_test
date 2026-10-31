"""Wait for an explicitly started full-video job and verify EOF, playback and bitrate."""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from urllib.request import Request,build_opener,ProxyHandler
from experiment_common import *


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prior-job')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base='http://127.0.0.1:8773';opener=build_opener(ProxyHandler({}))
    root=load_settings()['data_dir'].parent/'C5_loopback/sender/inference_lab'
    path=root/args.job;last='';reported=0;began=time.monotonic()
    while True:
        with opener.open(base+'/inference/api/jobs/'+args.job,timeout=15) as response:job=json.load(response)
        phase=job['phase'];progress=job.get('progress',{})
        if phase!=last or time.monotonic()-reported>=30:
            print(json.dumps(dict(status=job['status'],phase=phase,progress=progress),ensure_ascii=False),flush=True)
            last=phase;reported=time.monotonic()
        if job['status'] in ['done','error','cancelled','interrupted']:break
        if time.monotonic()-began>7200:raise TimeoutError('Full-video task did not finish in two hours')
        time.sleep(5)
    save_json(args.output/'full_job.json',job)
    if job['status']!='done':raise ValueError('Full-video job failed: '+str(job.get('error')))
    source=job['source'];count=source['frames'];duration=count/source['fps'];opts=job['options']
    if opts['range']!='full' or opts['frames'] is not None or not source['read_until_eof']:raise ValueError('Not an EOF run')
    if count<=66:raise ValueError('Use a genuine full-length video for acceptance')
    prepared=load_json(path/'prepare/result.json')
    config=load_json(path/'prepare.config.json');original=Path(config['input'])
    ffprobe=Path(load_settings()['runtime']['ffmpeg']).with_name('ffprobe.exe')
    def probe(file,expected=count):
        result=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-select_streams','v:0',
            '-count_frames','-show_entries','stream=width,height,nb_read_frames,avg_frame_rate,duration',
            '-of','json',str(file)],text=True,encoding='utf-8'))['streams'][0]
        if expected is not None and int(result['nb_read_frames'])!=expected:raise ValueError('Playable video frame count is incomplete: '+str(file))
        return result
    original_probe=probe(original,None);original_count=int(original_probe['nb_read_frames']);stride=source.get('frame_stride',1)
    if count!=(original_count+stride-1)//stride:raise ValueError('Output does not cover the full input at the selected frame stride')
    probes={'input':original_probe,'source_preview':probe(path/'prepare/source.mp4')}
    ranges=[];rows=[];prior=root/args.prior_job if args.prior_job else None
    for row in job['results']:
        mode=row['mode'];send=load_json(path/(mode+'_sender/result.json'));receive=load_json(path/(mode+'_receiver/result.json'))
        if row['frames']!=count or send['frames']!=count or receive['frames']!=count:raise ValueError('A mode truncated the video')
        if not receive['reference_verified'] or send['reference_checksums']!=receive['reference_checksums']:raise ValueError('Reference verification failed')
        if not receive['source_media_reads_forbidden']:raise ValueError('Receiver isolation missing')
        stream=path/(mode+'_sender')/send['stream_file'];size=stream.stat().st_size
        if size!=row['stream_bytes'] or file_sha(stream)!=row['stream_sha256'] or send['stream_sha256']!=receive['stream_sha256']:
            raise ValueError('Transport size or checksum mismatch')
        if abs(row['bitrate_kbps']-size*8/duration/1000)>1e-9 or abs(row['bpp']-size*8/(count*source['width']*source['height']))>1e-12:
            raise ValueError('Incorrect bitrate or bpp denominator')
        probes[mode]=probe(path/(mode+'_receiver/video.mp4'))
        if prior:
            old=load_json(prior/(mode+'_sender/result.json'))
            if len(send['records'])<len(old['records']) or any(a['latent_sha256']!=b['latent_sha256'] for a,b in zip(send['records'],old['records'])):
                raise ValueError('Prefix reference states changed from the previous short job')
        for url in [row['video'],row['stream']]:
            with opener.open(Request(base+url,headers={'Range':'bytes=0-31'}),timeout=15) as response:
                if response.status!=206 or len(response.read())!=32:raise ValueError('Playback/download Range request failed')
            ranges.append(url)
        rows.append({k:row[k] for k in ['mode','frames','encode_core_seconds','decode_core_seconds','encode_fps','decode_fps',
            'stream_bytes','bitrate_kbps','bpp','psnr_db','segments','reference_verified']})
    if {r['mode'] for r in rows}!=set(opts['modes']):raise ValueError('Requested mode did not finish')
    save_json(args.output/'full_acceptance.json',dict(passed=True,job=args.job,frames=count,duration_seconds=duration,
        width=source['width'],height=source['height'],fps=source['fps'],q=opts['q'],read_until_eof=True,
        original_frames_equal_decoded_frames=count==original_count,decoded_frames_match_sampling=True,
        source_frames=original_count,frame_stride=stride,independent_reference_checks=True,prefix_matches_prior=bool(prior),
        transport_bitrate_verified=True,range_requests_verified=len(ranges),probes=probes,modes=rows,
        source_video_sha256=file_sha(original),prepared_video_sha256=prepared['source_video_sha256']))
    print('PASS: full input and all selected decoded videos contain '+str(count)+' frames; bitrate, streams and independent references verified',flush=True)


if __name__=='__main__':main()
