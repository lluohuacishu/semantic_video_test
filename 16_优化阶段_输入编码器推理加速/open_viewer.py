"""Start or reopen the C5 inference comparison page; opening does not start GPU work."""
from __future__ import annotations
import argparse
import os
import subprocess
import time
import webbrowser
from urllib.request import urlopen
from experiment_common import *


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8773)
    p.add_argument('--no-browser',action='store_true');args=p.parse_args()
    trial=WORKSPACE/'02_实验数据/优化_输入编码器推理/inference_20261030_01';root=trial/'viewer'
    url=f'http://127.0.0.1:{args.port}/inference/'
    def ready():
        try:
            with urlopen(url+'api/config',timeout=1) as r:return __import__('json').load(r).get('app')=='glc-inference-lab-v1'
        except (OSError,ValueError):return False
    if not ready():
        subprocess.run([str(load_settings()['runtime']['python']),'-X','utf8',str(HERE/'prepare_viewer.py'),
                        '--experiment',str(trial),'--output',str(root)],check=True)
        command=[str(load_settings()['runtime']['python']),'-X','utf8','-u',str(REPO/'08_新阶段C_编解码加速与网页展示/c5_loopback.py'),
                 '--sender-port',str(args.port)]
        with (root/'viewer.log').open('ab') as log:
            child=subprocess.Popen(command,stdout=log,stderr=log,stdin=subprocess.DEVNULL,cwd=REPO,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
        deadline=time.monotonic()+15
        while not ready():
            if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('入口启动失败，请检查viewer.log或端口冲突')
            time.sleep(.15)
    print(url,flush=True)
    if not args.no_browser:webbrowser.open(url)


if __name__=='__main__':main()
