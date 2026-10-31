"""Bounded RGB storage for full-length videos; no whole-video RAM allocation."""
from __future__ import annotations
from bisect import bisect_right
from pathlib import Path
import numpy as np
from experiment_common import file_sha

CHUNK=64


def even_rgb(rgb):
    """H.264/yuv420 preview needs even dimensions; preserve all input pixels."""
    h,w=rgb.shape[:2]
    return np.pad(rgb,((0,h%2),(0,w%2),(0,0)),mode='edge') if h%2 or w%2 else rgb


class FrameWriter:
    def __init__(self,folder,width,height):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.width,self.height=width,height;self.array=None;self.count=0;self.chunks=[]

    def _finish_chunk(self):
        if self.array is None:return
        self.array.flush();self.array=None
        self.chunks[-1]['sha256']=file_sha(self.chunks[-1]['path'])

    def add(self,rgb):
        if tuple(rgb.shape)!=(self.height,self.width,3) or rgb.dtype!=np.uint8:raise ValueError('Frame shape changed')
        if self.count%CHUNK==0:
            self._finish_chunk();path=self.folder/f'{len(self.chunks):06d}.npy'
            self.array=np.lib.format.open_memmap(path,mode='w+',dtype='uint8',shape=(CHUNK,self.height,self.width,3))
            self.chunks.append(dict(path=str(path.resolve()),start=self.count,frames=0))
        self.array[self.count%CHUNK]=rgb;self.chunks[-1]['frames']+=1;self.count+=1

    def finish(self):
        self._finish_chunk()
        return dict(storage='chunked-rgb-v1',frames=self.count,width=self.width,height=self.height,chunks=self.chunks,chunk_frames=CHUNK)


class Frames:
    def __init__(self,meta,verify=True):
        self.meta=meta;self.count=meta['frames'];self.cached=None;self.key=None
        if meta.get('storage')=='chunked-rgb-v1':
            self.chunks=meta['chunks'];self.starts=[x['start'] for x in self.chunks]
            n=0
            for c in self.chunks:
                if c['start']!=n or not 0<c['frames']<=CHUNK:raise ValueError('Noncontiguous frame cache')
                if verify and file_sha(c['path'])!=c['sha256']:raise ValueError('Frame cache changed')
                n+=c['frames']
            if n!=self.count:raise ValueError('Frame cache is incomplete')
        else:
            self.chunks=None
            if verify and file_sha(meta['source'])!=meta['source_sha256']:raise ValueError('Source array changed')
            self.cached=np.load(meta['source'],mmap_mode='r')

    def __getitem__(self,index):
        if not 0<=index<self.count:raise IndexError(index)
        if self.chunks is None:return self.cached[index]
        key=bisect_right(self.starts,index)-1
        if self.key!=key:
            self.cached=np.load(self.chunks[key]['path'],mmap_mode='r');self.key=key
        return self.cached[index-self.starts[key]]

    def close(self):self.cached=None


def discard_working_frames(meta,owned_root):
    """Delete only this job's derived chunk files after metrics; retain input/videos/streams."""
    root=Path(owned_root).resolve()
    for item in meta.get('chunks',[]):
        f=Path(item['path']).resolve()
        if not f.is_relative_to(root) or f.suffix!='.npy':raise ValueError('Working frame is outside this job')
        f.unlink(missing_ok=True)
