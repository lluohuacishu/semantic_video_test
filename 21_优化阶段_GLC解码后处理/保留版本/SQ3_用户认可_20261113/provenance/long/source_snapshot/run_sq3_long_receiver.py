"""Explicit sender-metadata deny list for historical directories outside DATA.

The frozen receiver's numerical code is unchanged. Its shared ReadGuard covers
NPY/source video globally but JSON only under the postprocessing DATA root.
The historical full-video source metadata is under another experiment root.
"""
import argparse
import os
from pathlib import Path
import sys
import cv2
from run_sq3_long import receive,OLD
from runtime_speed_common import STAGE,load_json,file_sha

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    plan=load_json(a.run/'execution_plan.json');extension=load_json(a.run/'receiver_guard_extension.json')
    assert file_sha(__file__)==extension['wrapper_sha256']
    assert file_sha(a.run/'execution_plan.json')==extension['original_plan_sha256']
    forbidden={Path(plan['source']).resolve(),Path(plan['source_metadata']).resolve(),
               (OLD/'sender_receipt.json').resolve(),(a.run/'user_request.json').resolve()}
    def deny_source_metadata(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0])).resolve()
            if path in forbidden:raise RuntimeError('Explicit long receiver source/metadata exclusion: '+str(path))
    sys.addaudithook(deny_source_metadata)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    receive(a.run)
