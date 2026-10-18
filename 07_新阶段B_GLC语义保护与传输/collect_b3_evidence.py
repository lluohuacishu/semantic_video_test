"""Archive small B3 evidence byte-for-byte; media stay in their local run."""
import argparse
import hashlib
from pathlib import Path
import shutil
from b2_common import HERE, git, read, require, sha, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run=args.run.resolve()
    require(read(run/"evaluation/completion.json")["status"]=="B3_development_diagnostics_complete","Evaluation incomplete")
    args.output.mkdir(parents=True,exist_ok=False)
    copied=[]
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json",".csv",".log",".txt"}:
            continue
        require(path.stat().st_size<=5*1024*1024,f"Unexpected large evidence: {path}")
        relative=path.relative_to(run)
        target=args.output/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        digest=sha(path)
        require(sha(target)==digest,"Archive copy changed bytes")
        copied.append({"path":relative.as_posix(),"bytes":path.stat().st_size,"sha256":digest})
    media=[]
    for folder in (run/"glc_search",run/"geometry_search",run/"geometry_refinement",run/"evaluation"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".glcv",".scv",".mp4",".jpg"}:
                media.append({"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path),"archived_in_git":False})
    write(args.output/"local_media_index.json",media)
    copied.append({"path":"local_media_index.json","bytes":(args.output/"local_media_index.json").stat().st_size,"sha256":sha(args.output/"local_media_index.json")})
    write(args.output/"archive_manifest.json",{"run":str(run),"archived_at_code_commit":git("rev-parse","HEAD"),"files":copied,
          "bytes":sum(r["bytes"] for r in copied),"human_review":"pending","raw_bytes_preserved":True})
    print(f"Archived {len(copied)} small evidence files, {sum(r['bytes'] for r in copied):,} bytes; {len(media)} local media indexed")


if __name__=="__main__":main()
