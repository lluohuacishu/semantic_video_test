"""Read-only workspace census. Never follow reparse points; never delete or move files."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    if root==Path(root.anchor) or not root.is_dir():raise ValueError('Require an explicitly named workspace')
    out.mkdir(parents=True,exist_ok=True)
    totals=defaultdict(lambda:[0,0]);extensions=defaultdict(lambda:[0,0]);skipped=[];errors=[];largest=[]
    began=time.time();stack=[root];count=0;size=0
    with (out/'files.jsonl').open('w',encoding='utf-8') as index:
        while stack:
            folder=stack.pop()
            try:
                with os.scandir(folder) as entries:
                    for entry in entries:
                        path=Path(entry.path)
                        try:
                            info=entry.stat(follow_symlinks=False)
                            if getattr(info,'st_file_attributes',0)&0x400 or entry.is_symlink():
                                skipped.append(dict(path=str(path),reason='reparse point; not traversed'));continue
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name=='.git':skipped.append(dict(path=str(path),reason='protected git metadata; not traversed'))
                                else:stack.append(path)
                                continue
                            if not entry.is_file(follow_symlinks=False):continue
                            relative=path.relative_to(root)
                            record=dict(path=str(relative),bytes=info.st_size,mtime_ns=info.st_mtime_ns,
                                        device=info.st_dev,file_id=info.st_ino,hardlinks=info.st_nlink)
                            index.write(json.dumps(record,ensure_ascii=False)+'\n')
                            count+=1;size+=info.st_size
                            for depth in range(1,len(relative.parts)):
                                key='/'.join(relative.parts[:depth]);totals[key][0]+=info.st_size;totals[key][1]+=1
                            suffix=path.suffix.lower() or '(no suffix)';extensions[suffix][0]+=info.st_size;extensions[suffix][1]+=1
                            if info.st_size>=64*1024**2:largest.append(record)
                        except OSError as error:errors.append(dict(path=str(path),error=str(error)))
            except OSError as error:errors.append(dict(path=str(folder),error=str(error)))
    with (out/'directories.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['relative_directory','bytes','files'])
        writer.writerows((name,*value) for name,value in sorted(totals.items(),key=lambda item:item[1][0],reverse=True))
    disk=shutil.disk_usage(root)
    summary=dict(workspace=str(root),scanned_files=count,logical_bytes=size,seconds=time.time()-began,
                 disk_total_bytes=disk.total,disk_free_bytes=disk.free,skipped=skipped,errors=errors,
                 extensions={name:dict(bytes=value[0],files=value[1]) for name,value in extensions.items()},
                 largest_files=sorted(largest,key=lambda row:row['bytes'],reverse=True)[:100],
                 top_directories=[dict(path=name,bytes=value[0],files=value[1]) for name,value in sorted(totals.items(),key=lambda item:item[1][0],reverse=True)
                                  if name.count('/')<=2][:100],
                 deletion_performed=False,note='Logical sizes only; hardlinks and NTFS allocation may change physical reclaim. Git metadata and junction targets not counted twice.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=count,logical_gib=size/1024**3,seconds=time.time()-began,errors=len(errors),skipped=len(skipped)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
