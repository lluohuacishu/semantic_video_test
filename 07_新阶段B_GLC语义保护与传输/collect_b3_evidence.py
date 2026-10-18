"""Archive small B3 evidence byte-for-byte; media stay in their local run."""
import argparse
import gzip
import hashlib
from io import BytesIO
from pathlib import Path
import shutil
from b2_common import HERE, git, read, require, sha, write


def archive_payload(relative, payload):
    if len(payload) < 65536:
        return relative.as_posix(), payload, "identity"
    buffer = BytesIO()
    with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", mtime=0) as stream:
        stream.write(payload)
    packed = buffer.getvalue()
    require(gzip.decompress(packed) == payload, "Compressed archive changed source bytes")
    return relative.as_posix()+".gz", packed, "gzip"


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
        payload=path.read_bytes()
        archive_path, stored, compression=archive_payload(relative,payload)
        target=args.output/archive_path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(stored)
        original_digest=hashlib.sha256(payload).hexdigest()
        stored_digest=hashlib.sha256(stored).hexdigest()
        restored=target.read_bytes() if compression=="identity" else gzip.decompress(target.read_bytes())
        require(hashlib.sha256(restored).hexdigest()==original_digest,"Archive copy changed source bytes")
        copied.append({"source_path":relative.as_posix(),"path":archive_path,"compression":compression,
                       "original_bytes":len(payload),"original_sha256":original_digest,
                       "stored_bytes":len(stored),"stored_sha256":stored_digest})
    media=[]
    for folder in (run/"glc_search",run/"geometry_search",run/"geometry_refinement",run/"evaluation"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".glcv",".scv",".mp4",".jpg"}:
                media.append({"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path),"archived_in_git":False})
    write(args.output/"local_media_index.json",media)
    index_payload=(args.output/"local_media_index.json").read_bytes()
    copied.append({"source_path":"local_media_index.json","path":"local_media_index.json","compression":"identity",
                   "original_bytes":len(index_payload),"original_sha256":hashlib.sha256(index_payload).hexdigest(),
                   "stored_bytes":len(index_payload),"stored_sha256":hashlib.sha256(index_payload).hexdigest()})
    write(args.output/"archive_manifest.json",{"run":str(run),"archived_at_code_commit":git("rev-parse","HEAD"),
          "collector_script":str(Path(__file__).resolve()),"collector_script_sha256":sha(__file__),"files":copied,
          "original_bytes":sum(r["original_bytes"] for r in copied),"stored_bytes":sum(r["stored_bytes"] for r in copied),
          "human_review":"pending","raw_bytes_recoverable_byte_exact":True})
    print(f"Archived {len(copied)} evidence records: {sum(r['original_bytes'] for r in copied):,} source bytes, "
          f"{sum(r['stored_bytes'] for r in copied):,} stored bytes; {len(media)} local media indexed")


if __name__=="__main__":main()
