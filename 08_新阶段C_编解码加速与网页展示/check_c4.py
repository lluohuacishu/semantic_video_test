"""Offline C4 acceptance audit; validates recorded runs, never starts inference."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

from c3_catalog import Catalog, HERE, REPO, read_json, sha256
from c3_server import verify_result
from c4_config import load_settings
from c2_watch import inspect, write, utc


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_hashes(path, info):
    # Read every source frame, independently of MemoryFrames' grab/skip iterator.
    import cv2
    cap = cv2.VideoCapture(str(path))
    hashes = []
    selected = set(info['sampled_source_frame_indices'])
    try:
        require(cap.isOpened(), 'Cannot audit source media')
        for index in range(max(selected) + 1):
            ok, bgr = cap.read()
            require(ok, 'Source EOF before expected sampling index')
            if index in selected:
                if bgr.shape[:2] != (info['height'], info['width']):
                    bgr = cv2.resize(bgr, (info['width'], info['height']), interpolation=cv2.INTER_AREA)
                hashes.append(hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest())
    finally:
        cap.release()
    return hashes


def preview(path, ffprobe):
    result = subprocess.run([str(ffprobe), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,duration', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True)
    return {**json.loads(result.stdout)['streams'][0], 'sha256':sha256(path)}


def pixels(path, ffmpeg):
    result = subprocess.run([str(ffmpeg), '-v', 'error', '-i', str(path), '-map', '0:v:0', '-f', 'hash',
        '-hash', 'sha256', '-'], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def complete(folder, frames, ffmpeg):
    enc, dec = verify_result(folder)
    job, result = read_json(folder/'job.json'), read_json(folder/'result.json')
    require(job['status']=='done' and result['verified'], 'Job not complete')
    require(len(enc['frames'])==frames, 'Unexpected frame count')
    processes = {role:inspect(folder/(role+'_process')) for role in ('encode','decode')}
    require(all(p['observed_state']=='completed' and p['recorded']['returncode']==0 for p in processes.values()), 'Worker not cleanly exited')
    meta = enc['input']
    manifest=read_json(folder/'manifest.json')
    source=Path(manifest['source'])
    require(sha256(source)==manifest['source_sha256'], 'Source changed')
    require(source_hashes(source,meta)==meta['rgb_sha256'], 'Selected source RGB mismatch')
    ffprobe=ffmpeg.with_name('ffprobe'+ffmpeg.suffix)
    paths={'source':folder/'encode/encoder/source_preview.mp4',
           'sender_reconstruction':folder/'encode/encoder/reconstruction_preview.mp4',
           'receiver_reconstruction':folder/'decode/receiver/reconstruction_preview.mp4'}
    previews={key:preview(path,ffprobe) for key,path in paths.items()}
    for value in previews.values():
        require(int(value['nb_read_frames'])==frames, 'Preview frame count mismatch')
        require((value['width'],value['height'])==(meta['width'],meta['height']), 'Preview dimensions mismatch')
        numerator,denominator=map(int,value['avg_frame_rate'].split('/'))
        require(abs(numerator/denominator-meta['fps'])<1e-4, 'Preview frame rate mismatch')
        require(abs(float(value['duration'])-meta['duration_seconds'])<0.02, 'Preview duration mismatch')
    hashes={key:pixels(path,ffmpeg) for key,path in paths.items() if key!='source'}
    require(len(set(hashes.values()))==1, 'Sender/receiver MP4 decoded pixels differ')
    browser=read_json(folder/'browser.json')
    initial=[v for v in browser if v['dispatch_to_preview_seconds'] is not None and not v['after_refresh']]
    require(bool(initial), 'No original browser click-to-preview observation')
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text('utf-8').splitlines()]
    require(events[-1]['status']=='done', 'No durable done event')
    return {'job_id':folder.name, 'source_sha256':manifest['source_sha256'],
        'source_bytes':manifest['source_bytes'], 'input':{k:meta[k] for k in
        ('source_width','source_height','width','height','fps','processed_frames','frame_stride','duration_seconds','reported_source_frames')},
        'stream_bytes':enc['file_bytes'], 'stream_sha256':enc['file_sha256'],
        'service_seconds':result['service_seconds'], 'queue_seconds':result['queue_seconds'],
        'encode_to_preview_seconds':result['timings']['encode']['task_seconds']['offline_view_ready_seconds'],
        'decode_to_preview_seconds':result['timings']['decode']['task_seconds']['offline_view_ready_seconds'],
        'browser_first_observation':initial[0], 'preview_metadata':previews,
        'preview_decoded_pixel_sha256':hashes, 'source_sampling_rgb_verified':True,
        'all_rgb_and_latent_equal':True, 'receiver_source_reads':dec['source_media_reads'],
        'worker_returncodes':{role:p['recorded']['returncode'] for role,p in processes.items()},
        'events':events, 'code_sha256_at_dispatch':manifest['code_sha256']}


def cancellation(folder, role, before, next_folder, queue_before, base_url):
    job=read_json(folder/'job.json')
    require(before['id']==folder.name and before['status']=='running' and before['stage']==role, 'Not observed in requested inference stage')
    progress=before['heartbeat']['last_progress']
    require(0<progress['frame_completed']<progress['frames'], 'No active frame progress observed')
    require(job['status']=='cancelled' and job['stage']==role, 'Cancellation was not terminal')
    config=read_json(folder/(role+'_config.json'))
    require(config['device']=='cuda', 'Cancellation did not use CUDA')
    process=inspect(folder/(role+'_process'))
    require(process['observed_state']=='cancelled' and process['recorded']['returncode']==2, 'Worker did not cooperatively exit')
    require(not process['recorded'].get('forced_termination'), 'Unexpected forced termination')
    beat=read_json(folder/(role+'_process')/'heartbeat.json')
    require(beat['type']=='TaskCancelled', 'Worker did not acknowledge cancellation')
    next_job=read_json(next_folder/'job.json')
    require(any(j['id']==next_folder.name and j['status']=='queued' for j in queue_before), 'Successor was not queued')
    require(next_job['status']=='done', 'Successor not complete')
    require(datetime.fromisoformat(next_job['queued_at'])<datetime.fromisoformat(job['cancel_requested_at']), 'Successor was not waiting before cancellation')
    require(datetime.fromisoformat(next_job['started_at'])>=datetime.fromisoformat(job['finished_at']), 'Overlapping worker dispatch')
    opener=build_opener(ProxyHandler({}))
    try:
        opener.open(base_url+'/jobs/'+folder.name+'/stream.glcv',timeout=5)
        raise ValueError('Cancelled stream exposed as successful download')
    except HTTPError as exc:
        require(exc.code==404, 'Unexpected incomplete download response')
    return {'job_id':folder.name,'role':role,'observed_progress_before_cancel':progress,
            'last_progress_at_exit':beat['last_progress'],'cancel_requested_at':job['cancel_requested_at'],
            'finished_at':job['finished_at'], 'cancel_to_finished_seconds':(
                datetime.fromisoformat(job['finished_at'])-datetime.fromisoformat(job['cancel_requested_at'])).total_seconds(),
            'observed_worker_state':process['observed_state'],'returncode':2,'forced_termination':False,
            'launcher_alive':process['current_process']['alive'],'worker_alive':process['current_worker']['alive'],
            'successor_job_id':next_folder.name,'successor_started_at':next_job['started_at'],
            'successor_done':True,'incomplete_download_http_status':404}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(), 'Refuse to overwrite prior evidence')
    acceptance=json.loads(args.acceptance.read_text('utf-8-sig'))
    validation=args.acceptance.parent
    settings=load_settings()
    root=Path(acceptance['job_store'])
    jobs=acceptance['jobs']
    cat=Catalog(workspace=settings['workspace'])
    completed={name:complete(root/jobs[name],count,settings['runtime']['ffmpeg']) for name,count in
               [('main_full',125),('new_source',125),('post_decode_cancel',8)]}
    main_result=completed['main_full']
    reference=cat.json_asset('c2.compact.encode_result')
    require(main_result['stream_sha256']==reference['file_sha256'], 'Full main stream differs from frozen reference')
    require(main_result['input']['duration_seconds']==10, 'Full main clip is not ten seconds')
    gate=settings['workspace']/'02_实验数据/C_编解码加速与网页展示/C2_combined_gate8_20261020_r2/encode_1_native/stream.glcv'
    require(completed['post_decode_cancel']['stream_sha256']==sha256(gate),'Post-cancel recovery changed reference output')
    require(completed['new_source']['source_sha256'] not in [sha256(cat.path(k)) for k in ('source.main','source.second')], 'New source repeats old media')
    downloaded={}
    for key,filename in acceptance['downloads'].items():
        expected=root/jobs['new_source']/'decode/receiver/reconstruction_preview.mp4' if key=='new_preview' else root/jobs[key]/'received.glcv'
        require(sha256(filename)==sha256(expected),'Browser download mismatch: '+key)
        downloaded[key]={'path':filename,'sha256':sha256(filename),'bytes':Path(filename).stat().st_size}
    cancelled={}
    for role,next_name in [('encode','new_source'),('decode','post_decode_cancel')]:
        before=json.loads((validation/(role+'_cancel_before.json')).read_text('utf-8-sig'))
        queue=json.loads((validation/(role+'_queue_before.json')).read_text('utf-8-sig'))
        cancelled[role]=cancellation(root/jobs[role+'_cancel'],role,before,root/jobs[next_name],queue,acceptance['base_url'])
    suites=[]
    for name in ('c','b'):
        path=validation/(name+'_regression_final.log')
        text=path.read_text('utf-8-sig')
        found=re.search(r'Ran (\d+) tests',text)
        require(found and text.rstrip().endswith('OK'),'Regression did not pass: '+name)
        suites.append({'suite':name,'tests':int(found[1]),'sha256':sha256(path)})
    browser=json.loads((validation/'browser_checks.json').read_text('utf-8-sig'))
    require(not browser['console_errors'] and all(not x['overflow'] for x in browser['layouts']),'Browser check failed')
    require(browser['per_tab_refresh_restored_expected_job'],'Per-tab recovery check failed')
    portable=json.loads((validation/'portable_smoke/result.json').read_text('utf-8-sig'))
    require(portable['cli_path_check_passed'] and portable['upload_only_http_smoke_passed'] and portable['launcher_config_plus_environment_override_passed'],'Portable configuration smoke failed')
    source_files=[HERE/n for n in ('c3_server.py','c3_catalog.py','c4_config.py','check_c4.py','test_c4_config.py','launch_c4.ps1','runtime.example.json')]
    source_files+=list((HERE/'c3_console').glob('*'))
    summary={'stage':'C4','status':'complete_with_stated_limits','created_utc':utc(),'base_commit':acceptance['base_commit'],
        'validation_root':str(validation),'completed_jobs':completed,'gpu_cancellation':cancelled,
        'actual_browser_downloads':downloaded,'browser':browser,'portable_configuration':portable,'regression':suites,'regression_total':sum(s['tests'] for s in suites),
        'provenance':json.loads((validation/'source/provenance.json').read_text('utf-8-sig')),
        'source_sha256':{str(p.relative_to(REPO)):sha256(p) for p in source_files},
        'limits':['Single full-run observations, not a new speed benchmark or C2 matrix.',
                  'New NOAA input is unmodified 29.97 fps MP4; compact samples 125 frames covering 8.341667 seconds, not its entire 27.627604 seconds.',
                  'Independent receiver uses a local copy of the complete stream; no network/acoustic transfer.',
                  'Configuration relocation validated on this Windows machine; no second-PC or cross-GPU execution claim.',
                  'Cancellation tested once during encode and once during decode, followed by successful queued tasks; not a stability guarantee.',
                  'C2 strict 2x challenge and semantic ROI equal-budget benefit remain unproven. C5 not started.']}
    write(args.output,summary)
    print(json.dumps({'status':summary['status'],'tests':summary['regression_total'],
          'main_browser_seconds':main_result['browser_first_observation']['dispatch_to_preview_seconds'],
          'cancel_seconds':{r:v['cancel_to_finished_seconds'] for r,v in cancelled.items()}},ensure_ascii=False))


if __name__=='__main__':
    main()
