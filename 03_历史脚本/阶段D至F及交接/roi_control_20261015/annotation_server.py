"""Local human-review interface. Unknown frames stay unknown until a person confirms."""
import argparse
from datetime import datetime,timezone
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import socket
import sys
import threading
from urllib.parse import urlsplit
from control_common import OUT,read,write_json

KIT=OUT/"连续人工ROI核对"
PAGE=Path(__file__).with_name("annotation.html")
TOKEN=secrets.token_hex(16)
LOCK=threading.Lock()

def status(document):
    confirmed=sum(x["status"] in {"publisher_verified","human_reviewed"} for x in document["frames"])
    return {"confirmed":confirmed,"unknown":50-confirmed,"oracle_ready":confirmed==50}

def validate(document):
    original=read(KIT/"publisher_template.json")
    expected={x["source_frame"]:x for x in original["frames"]}
    rows=document.get("frames")
    if not isinstance(rows,list) or len(rows)!=50 or [x.get("source_frame") for x in rows]!=list(range(420,470)):
        raise ValueError("必须保留真实连续420–469帧")
    for row in rows:
        reference=expected[row["source_frame"]]
        if row.get("image")!=reference["image"] or row.get("width")!=1280 or row.get("height")!=720:
            raise ValueError("图像和坐标系不能更改")
        if row.get("status") not in {"unreviewed","publisher_verified","human_reviewed"}:
            raise ValueError("核对状态无效")
        boxes=row.get("boxes")
        if not isinstance(boxes,list) or len(boxes)>100:raise ValueError("框数量无效")
        for box in boxes:
            if len(box)!=4 or any(type(v) is not int for v in box):raise ValueError("框需要整数[x,y,w,h]")
            x,y,w,h=box
            if min(x,y)<0 or min(w,h)<1 or x+w>1280 or y+h>720:raise ValueError("目标框超出图像")
        if row["status"]=="publisher_verified":
            if reference["status"]!="publisher_verified" or boxes!=reference["boxes"] or row.get("no_fish")!=reference["no_fish"]:
                raise ValueError("修改发布者框后须明确进行人工复核")
            row["source"]=reference["source"]
        elif row["status"]=="human_reviewed":
            if row.get("no_fish") is not (not boxes):raise ValueError("无鱼声明与框不一致")
            row["source"]="human-reviewed bounding boxes via local interface; not pixel segmentation GT"
        else:
            row["no_fish"]=None
            row["source"]="unreviewed draft; not usable GT"
    document.update({"dataset":original["dataset"],"purpose":original["purpose"],"write_time_utc":datetime.now(timezone.utc).isoformat(),"review_interface":"local human ROI annotation","status_summary":status(document)})
    return document

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(KIT),**kwargs)
    def log_message(self,*args):pass
    def json_response(self,value,code=200):
        payload=json.dumps(value,ensure_ascii=False).encode("utf-8")
        self.send_response(code);self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(payload)));self.send_header("Cache-Control","no-store");self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        route=urlsplit(self.path).path
        if route=="/health":self.json_response({"app":"roi-annotation-20261015"});return
        if route=="/api/annotations":
            value=read(KIT/"annotations.json")
            self.json_response({"data":value,"summary":status(value),"token":TOKEN});return
        if route=="/":
            payload=PAGE.read_bytes()
            self.send_response(200);self.send_header("Content-Type","text/html; charset=utf-8");self.send_header("Content-Length",str(len(payload)));self.end_headers();self.wfile.write(payload);return
        if not route.startswith("/frames/"):
            self.send_error(404);return
        super().do_GET()
    def do_POST(self):
        if self.path!="/api/save":self.send_error(404);return
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<256000:raise ValueError("请求大小无效")
            payload=json.loads(self.rfile.read(length))
            if payload.get("token")!=TOKEN:raise ValueError("请刷新本机核对页面")
            document=validate(payload["data"])
            with LOCK:
                versions=KIT/"versions";versions.mkdir(exist_ok=True)
                filename=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+".json"
                write_json(versions/filename,document)
                temporary=KIT/"annotations.tmp.json";write_json(temporary,document);temporary.replace(KIT/"annotations.json")
            self.json_response({"saved":True,"summary":status(document)})
        except Exception as exc:self.json_response({"error":str(exc)},400)

class Server(ThreadingHTTPServer):
    allow_reuse_address=False
    def server_bind(self):
        if hasattr(socket,"SO_EXCLUSIVEADDRUSE"):self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--port",type=int,default=8777);args=parser.parse_args()
    server=Server(("127.0.0.1",args.port),Handler)
    print(f"人工ROI核对：http://127.0.0.1:{args.port}/",flush=True)
    server.serve_forever()

if __name__=="__main__":main()
