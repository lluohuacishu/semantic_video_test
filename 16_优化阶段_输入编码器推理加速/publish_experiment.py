"""Verify raw evidence, export compact public tables, bind every file by SHA256."""
from __future__ import annotations
import argparse
import copy
import csv
import shutil
from experiment_common import *
from run_experiment import compare


def brief(r):
    return {k:v for k,v in r.items() if k!='records'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);args=p.parse_args();out=args.output;dest=args.destination
    dest.mkdir(parents=True,exist_ok=False);plan=load_json(out/'plan.json');frozen(plan)
    raw={};success=0
    for path in sorted([*out.glob('fill/*/*/*/result.json'),*out.glob('full/*/*/*/result.json'),*out.glob('encoder/*/*/result.json')]):
        r=load_json(path);raw[path.relative_to(out).as_posix()]=dict(sha256=file_sha(path),bytes=path.stat().st_size)
        if path.parts[-5]=='fill' or 'full' in path.parts:
            stream=path.parent/'stream.glcsnd' if r['role']=='sender' else None
            if stream is not None and file_sha(stream)!=r['stream_sha256']:raise ValueError('Stream changed')
            if r.get('decoded_rgb_sha256') and file_sha(path.parent/'decoded_rgb.npy')!=r['decoded_rgb_sha256']:
                raise ValueError('Decoded output changed')
            if len(r['records'])!=r['header']['frames']:raise ValueError('Incomplete run')
            success+=1
    fill=load_json(out/'fill_summary.json');encoder=load_json(out/'encoder_summary.json');full=load_json(out/'full_summary.json')
    if not all(x['complete'] for x in [fill,encoder,full]):raise ValueError('Campaign incomplete')
    fill_export=copy.deepcopy(fill)
    for item in fill_export['reports']:
        for pair in item['pairs']:
            compare(pair['on'],pair['off'],rgb=item['role']=='receiver')
            for key in ['on','off']:pair[key]=brief(pair[key])
    full_export=copy.deepcopy(full)
    for item in full_export['reports']:
        compare(item['sender_repeats'][0],item['sender_repeats'][1])
        compare(item['sender_repeats'][0],item['receiver'])
        item['sender_repeats']=[brief(x) for x in item['sender_repeats']];item['receiver']=brief(item['receiver'])
    for name,value in [('fill_results.json',fill_export),('encoder_results.json',encoder),('full_results.json',full_export),
                       ('quality.json',load_json(out/'quality.json')),('plan.json',plan)]:save_json(dest/name,value)
    # Full per-frame exactness evidence in a flat table; entropy subrecords have their own digest.
    rows=[]
    for path in sorted([*out.glob('fill/*/*/*/result.json'),*out.glob('full/*/*/*/result.json')]):
        r=load_json(path)
        for record in r['records']:
            x=dict(run=path.parent.relative_to(out).as_posix(),clip=r['clip'],role=r['role'],variant=r['variant'],fill=r['fill'])
            x.update({k:v for k,v in record.items() if k!='entropy'})
            x['entropy_sha256']=hashlib.sha256(__import__('json').dumps(record['entropy'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            rows.append(x)
    with (dest/'frame_checks.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    for name in ['validation_tests.log','inference_overview.png','inference_overview.svg']:
        source=out/name
        if name.endswith(('.log','.svg')):
            text=source.read_text(encoding='utf-8-sig');(dest/name).write_text('\n'.join(x.rstrip() for x in text.splitlines())+'\n',encoding='utf-8',newline='\n')
        else:shutil.copyfile(source,dest/name)
    for f in (out/'previews').glob('*.png'):shutil.copyfile(f,dest/f.name)
    save_json(dest/'selection.json',load_json(out/'selection.json'))
    save_json(dest/'engine_builds.json',{f.relative_to(out).as_posix():load_json(f) for f in out.glob('engines/*/*/*_build.json')})
    compact=load_json(out/'compact_transport.json')
    for r in compact['reports']:
        if file_sha(r['path'])!=r['sha256']:raise ValueError('Compact stream changed')
    save_json(dest/'compact_transport.json',compact)
    failures=[]
    for item in encoder.get('failures',[])+full.get('failures',[]):
        log=Path(item['folder']).with_suffix('.log')
        if log.exists():
            name='failure_'+str(len(failures)+1)+'.log'
            (dest/name).write_text(log.read_text(encoding='utf-8',errors='replace'),encoding='utf-8',newline='\n')
            failures.append(dict(**item,public_log=name))
    libraries={}
    for f in PACKAGES.rglob('nvinfer*.dll'):libraries[f.relative_to(PACKAGES).as_posix()]=dict(sha256=file_sha(f),bytes=f.stat().st_size)
    for f in out.glob('playback/*/result.json'):
        r=load_json(f)
        if not r['reference_verified'] or file_sha(f.parent/'video.mp4')!=r['video_sha256']:raise ValueError('Playback video changed')
    save_json(dest/'execution_manifest.json',dict(raw_directory=str(out.resolve()),raw_results=raw,
        successful_codec_runs=success,encoder_jobs=len(encoder['reports']),frame_rows=len(rows),
        capture_code_sha256=plan['code_sha256'],final_code_sha256={f.relative_to(REPO).as_posix():file_sha(f) for f in HERE.glob('*.py')},
        followup_job_code_manifests={f.relative_to(out).as_posix():load_json(f) for f in out.rglob('*.code.json')},
        failures=failures,engine_libraries=libraries,install_log_sha256=file_sha(PACKAGES.parent/'engine_packages_1050_install.log'),
        previews=load_json(out/'previews/manifest.json'),
        real_cli_playback={f.parent.name:load_json(f) for f in out.glob('playback/*/result.json')},
        source_independence_established=False,production_defaults_modified=False,training_steps=0,Orin_tested=False,
        cpu_tests=9,public_artifacts={f.name:dict(sha256=file_sha(f),bytes=f.stat().st_size) for f in dest.iterdir() if f.is_file()},
        note='Raw streams, RGB, ONNX and RTX engines remain local. PyTorch allocator statistics omit TensorRT allocations.'))
    print('Published',success,'full codec runs and',len(encoder['reports']),'encoder jobs;',len(failures),'failures retained',flush=True)


if __name__=='__main__':main()
