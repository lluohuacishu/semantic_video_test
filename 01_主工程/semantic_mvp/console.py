"""Small local launcher for the semantic communication MVP; stdlib HTTP only."""
from __future__ import annotations
import argparse
import json
import mimetypes
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import unquote,urlsplit
import webbrowser
from runtime import ROOT

STATIC=Path(__file__).with_name("console.html")
JOBS={}
LOCK=threading.Lock()
TOKEN=secrets.token_hex(16)
PRESETS={"fish":"semantic_mvp_20261014_fish","diver":"semantic_mvp_20261014_diver","information":"semantic_mvp_20261014_fish_information"}

class LocalServer(ThreadingHTTPServer):
    allow_reuse_address=False
    def server_bind(self):
        if hasattr(socket,"SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass
    def send_json(self,value,status=200):
        self.send_bytes(json.dumps(value,ensure_ascii=False).encode("utf-8"),"application/json; charset=utf-8",status)
    def send_bytes(self,payload,mime,status=200):
        self.send_response(status);self.send_header("Content-Type",mime)
        self.send_header("Content-Length",str(len(payload)))
        self.send_header("Cache-Control","no-cache");self.end_headers()
        self.wfile.write(payload)
    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if path=="/":
            self.send_bytes(STATIC.read_text("utf-8").replace("__TOKEN__",TOKEN).encode("utf-8"),"text/html; charset=utf-8")
            return
        if path=="/api/status":
            with LOCK:
                values=[]
                for key,item in JOBS.items():
                    proc=item["process"];code=proc.poll()
                    output=Path(item["output"])
                    if code is not None:
                        item["log_handle"].flush()
                    lines=Path(item["log"]).read_text("utf-8",errors="replace").splitlines()[-8:]
                    if code==0 and (output/"end_to_end.json").exists():
                        done=json.loads((output/"end_to_end.json").read_text("utf-8"))
                        lines=["发送与接收已完成。",f"通信包：{done['file_bytes']:,} 字节",f"传输时间：{done['transfer_seconds']:.2f} 秒",f"记录帧数：{done['frames']}",f"检出目标的帧：{done['target_present_frames']}","收发文件校验一致，可打开接收结果。"]
                    values.append({"id":key,"status":"running" if code is None else ("done" if code==0 else "error"),"log":lines,"page":f"/results/{output.name}/receiver/查看结果.html" if code==0 else None})
            self.send_json({"jobs":values});return
        if path=="/api/health":
            self.send_json({"app":"semantic-video-mvp","ready":True});return
        if path.startswith("/results/"):
            relative=path[len("/results/"):]
            file=(ROOT/"runs"/relative).resolve()
            if not file.is_relative_to((ROOT/"runs").resolve()) or not file.relative_to(ROOT/"runs").parts[0].startswith("semantic_mvp_") or not file.is_file():
                self.send_error(404);return
            mime=mimetypes.guess_type(str(file))[0] or "application/octet-stream"
            if file.suffix==".html":mime="text/html; charset=utf-8"
            if file.suffix==".json":mime="application/json; charset=utf-8"
            # Byte ranges allow the browser to seek through local MP4 previews.
            size=file.stat().st_size
            start,end=0,size-1
            header=self.headers.get("Range","")
            if header.startswith("bytes=") and "," not in header:
                try:
                    a,b=header[6:].split("-",1)
                    if a:start=int(a);end=min(size-1,int(b)) if b else size-1
                    else:start=max(0,size-int(b))
                    if not 0<=start<=end<size:raise ValueError()
                except ValueError:
                    self.send_error(416);return
            self.send_response(206 if header else 200)
            self.send_header("Content-Type",mime);self.send_header("Accept-Ranges","bytes")
            self.send_header("Content-Length",str(end-start+1))
            if header:self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
            self.end_headers()
            with file.open("rb") as handle:
                handle.seek(start);remaining=end-start+1
                while remaining:
                    data=handle.read(min(remaining,65536));self.wfile.write(data);remaining-=len(data)
            return
        self.send_error(404)
    def do_POST(self):
        if self.path!="/api/run":self.send_error(404);return
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<=4096:raise ValueError("请求长度无效")
            data=json.loads(self.rfile.read(length))
            if data.get("token")!=TOKEN:raise ValueError("请从本机界面启动")
            target=data["target"];mode=data["mode"]
            frames=int(data.get("frames",50));kbps=float(data.get("kbps",20))
            if target not in {"fish","diver"} or mode not in {"information","video"} or not 1<=frames<=300 or not 1<=kbps<=100000:
                raise ValueError("参数不在支持范围内")
            source=str(data.get("input","")).strip().strip('"')
            if source and not Path(source).is_file():raise ValueError("找不到视频文件，请填写本机完整路径")
            with LOCK:
                if any(item["process"].poll() is None for item in JOBS.values()):
                    self.send_json({"error":"当前任务正在运行，请等待完成。"},409);return
                key=time.strftime("%Y%m%d_%H%M%S")+"_"+secrets.token_hex(3)
                output=ROOT/"runs"/f"semantic_mvp_ui_{key}"
                logfile=ROOT/"runs"/f"semantic_mvp_ui_{key}.log"
                handle=logfile.open("w",encoding="utf-8")
                command=[sys.executable,str(Path(__file__).with_name("system.py")),"demo","--target",target,"--mode",mode,"--frames",str(frames),"--kbps",str(kbps),"--output",str(output)]
                if source:command += ["--input",source]
                env=dict(__import__("os").environ);env["PYTHONUTF8"]="1";env["PYTHONUNBUFFERED"]="1"
                process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
                JOBS[key]={"process":process,"output":str(output),"log":str(logfile),"log_handle":handle}
            self.send_json({"id":key})
        except Exception as exc:
            self.send_json({"error":str(exc)},400)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--port",type=int,default=8776);parser.add_argument("--open",action="store_true")
    args=parser.parse_args()
    address=f"http://127.0.0.1:{args.port}/"
    from urllib.request import build_opener,ProxyHandler
    from urllib.error import URLError
    try:
        existing=json.loads(build_opener(ProxyHandler({})).open(address+"api/health",timeout=2).read())
    except (URLError,TimeoutError,ConnectionError):
        existing=None
    if existing is not None:
        if existing.get("app")!="semantic-video-mvp":raise RuntimeError("端口已被其他服务使用，请更换--port")
        print("语义通信界面已运行："+address,flush=True)
        if args.open:webbrowser.open(address)
        return
    try:
        server=LocalServer(("127.0.0.1",args.port),Handler)
    except OSError:
        existing=json.loads(build_opener(ProxyHandler({})).open(address+"api/health",timeout=2).read())
        if existing.get("app")!="semantic-video-mvp":raise
        print("语义通信界面已运行："+address,flush=True)
        if args.open:webbrowser.open(address)
        return
    print(address,flush=True)
    if args.open:webbrowser.open(address)
    try:server.serve_forever()
    finally:
        server.server_close()
        for item in JOBS.values():item["log_handle"].close()

if __name__=="__main__":main()
