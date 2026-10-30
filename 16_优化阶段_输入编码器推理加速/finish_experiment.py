"""After all GPU timings: quality, unchanged real-file receiver, media and publication."""
from __future__ import annotations
import argparse
import subprocess
from experiment_common import *


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();summary=load_json(out/'full_summary.json')
    if not summary['complete'] or summary['failures']:raise ValueError('Resolve full-codec failures before finalizing')
    python=str(load_settings()['runtime']['python'])
    def call(script,arguments,log,executable=python):
        with log.open('w',encoding='utf-8') as handle:
            subprocess.run([executable,'-X','utf8',str(script),*map(str,arguments)],stdout=handle,stderr=subprocess.STDOUT,check=True)
    if not (out/'quality.json').exists():call(HERE/'evaluate_experiment.py',['--output',out],out/'quality.log')
    print('Full quality complete',flush=True)
    if not (out/'compact_transport.json').exists():call(HERE/'compact_transport.py',['--output',out],out/'compact.log')
    for name,variant,kind in [(v,v,'full') for v in ['trt_fp32_stream','trt_fp16_stream']]+[('compact_'+v,v,'compact') for v in ['trt_fp32_stream','trt_fp16_stream']]:
        folder=out/'playback'/name
        if not (folder/'result.json').exists():
            folder.parent.mkdir(exist_ok=True)
            call(REPO/'14_优化阶段_部署精简/deploy_codec.py',
                ['decode','--assets',load_json(out/'plan.json')['assets'],
                 '--input',out/kind/'long108_q0_640'/variant/('sender1/stream.glcsnd' if kind=='full' else 'stream.glcsnd'),
                 '--output',folder],out/f'playback_{name}.log')
        r=load_json(folder/'result.json')
        if not r['reference_verified'] or file_sha(folder/'video.mp4')!=r['video_sha256']:
            raise ValueError('Real CLI video not verified')
        print('Unchanged real receiver published video:',name,flush=True)
    if not (out/'previews/manifest.json').exists():
        call(HERE/'make_previews.py',['--output',out],out/'previews.log')
    cpu_python='C:/Users/lenovo/Desktop/works/work/semantic-video-mlvc/.venv/Scripts/python.exe'
    call(HERE/'plot_experiment.py',['--output',out],out/'plot.log',cpu_python)
    call(HERE/'write_report.py',['--output',out],out/'report.log')
    call(HERE/'publish_experiment.py',['--output',out,'--destination',HERE/'evidence_20261030'],out/'publish.log')
    print('Final report and public evidence verified',flush=True)


if __name__=='__main__':main()
