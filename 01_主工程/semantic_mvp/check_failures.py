"""Small protocol checks: corrupt/truncated data, forbidden reference, no target."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import zipfile
from runtime import ROOT,write_json,fresh
from system import unpack_packet
from transport import receive_connection

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);args=parser.parse_args()
    output=fresh(args.output)
    sample=ROOT/"runs/semantic_mvp_20261014_fish/sender/message.scv"
    with zipfile.ZipFile(sample) as archive:
        members={name:archive.read(name) for name in archive.namelist()}
    broken=output/"corrupt.scv"
    changed=dict(members);changed["context.mp4"]=changed["context.mp4"][:-1]+bytes([changed["context.mp4"][-1]^1])
    with zipfile.ZipFile(broken,"x") as archive:
        for name,data in changed.items():archive.writestr(name,data)
    try:unpack_packet(broken,fresh(output/"corrupt_result"))
    except ValueError as exc:corruption=str(exc)
    else:raise AssertionError("Corrupt content was accepted")
    # A complete header declaring more bytes than the closed connection contains.
    a,b=socket.socketpair()
    with a,b:
        header=json.dumps({"bytes":8,"sha256":"0"*64}).encode()
        a.sendall(struct.pack("!I",len(header))+header+b"short");a.shutdown(socket.SHUT_WR)
        try:receive_connection(b,output/"truncated.scv")
        except ConnectionError as exc:truncation=str(exc)
        else:raise AssertionError("Truncated transfer was accepted")
    private=sample.parent/"private_frames/00000.png"
    guard_code="from system import install_read_guard; from pathlib import Path; install_read_guard(Path(r'"+str(output)+"')); open(r'"+str(private)+"','rb')"
    denied=subprocess.run([sys.executable,"-c",guard_code],cwd=Path(__file__).parent,capture_output=True,text=True)
    assert denied.returncode!=0 and "PermissionError" in denied.stderr
    # Zero target is a valid task result, not a fabricated all-background GT label.
    manifest=json.loads(members["manifest.json"]);sem=json.loads(gzip.decompress(members["semantics.json.gz"]))
    sem["frames"]=[{"frame":i,"pixels":0,"regions":[]} for i in range(5)]
    data=gzip.compress(json.dumps(sem,separators=(",",":"),ensure_ascii=False).encode(),mtime=0)
    manifest.update({"mode":"information","context":None,"patches":[],"frames":5,"files":{"semantics.json.gz":{"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}}})
    empty=output/"no_target.scv"
    with zipfile.ZipFile(empty,"x") as archive:
        archive.writestr("manifest.json",json.dumps(manifest));archive.writestr("semantics.json.gz",data)
    result=subprocess.run([sys.executable,str(Path(__file__).with_name("system.py")),"decode","--input",str(empty),"--output",str(output/"no_target_receiver")],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    decoded=json.loads((output/"no_target_receiver/decode.json").read_text("utf-8"))
    assert decoded["frames"]==5 and decoded["target_present_frames"]==0
    write_json(output/"checks.json",{"success":True,"corrupt_member_rejected":corruption,"truncated_tcp_rejected":truncation,"external_sender_rgb_read_rejected":True,"zero_target_information_decoded":True})
    print("PROTOCOL_CHECKS_PASSED",flush=True)

if __name__=="__main__":main()
