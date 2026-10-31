"""Natural-GOP segments and a deterministic single-file transport bundle for long videos."""
from __future__ import annotations
import json
from pathlib import Path
import re
import shutil
import zipfile
from experiment_common import file_sha

SEGMENT_FRAMES=4096


def segment_ranges(frames,limit=SEGMENT_FRAMES):
    if frames<1 or limit<64 or limit%64 or limit>10000:raise ValueError('Invalid GOP segment boundary')
    return [(i,min(frames,i+limit)) for i in range(0,frames,limit)]


def package(folder,segments,metadata):
    folder=Path(folder)
    if len(segments)==1:return Path(segments[0]['path'])
    manifest=dict(schema='glc-segment-bundle-v1',**metadata,
        segments=[dict(file=Path(s['path']).name,frames=s['frames'],sha256=file_sha(s['path'])) for s in segments])
    path=folder/'stream.zip'
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for s in segments:
            info=zipfile.ZipInfo(Path(s['path']).name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED
            with Path(s['path']).open('rb') as source,archive.open(info,'w') as dest:shutil.copyfileobj(source,dest,2**20)
        info=zipfile.ZipInfo('bundle.json',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED
        archive.writestr(info,json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
    return path


def unpack(path,output):
    path=Path(path)
    if path.suffix!='.zip':return [path],None
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    paths=[]
    with zipfile.ZipFile(path) as archive:
        meta=json.loads(archive.read('bundle.json'))
        if meta.get('schema')!='glc-segment-bundle-v1' or sum(s['frames'] for s in meta['segments'])!=meta['frames']:
            raise ValueError('Invalid segment bundle')
        for s in meta['segments']:
            name=s['file']
            if not re.fullmatch(r'segment_\d{6}\.glcsnd',name) or not 1<=s['frames']<=10000:raise ValueError('Invalid segment entry')
            target=(root/name).resolve()
            if not target.is_relative_to(root):raise ValueError('Segment path escapes output')
            with archive.open(name) as source,target.open('xb') as dest:shutil.copyfileobj(source,dest,2**20)
            if file_sha(target)!=s['sha256']:raise ValueError('Segment checksum mismatch')
            paths.append(target)
    return paths,meta
