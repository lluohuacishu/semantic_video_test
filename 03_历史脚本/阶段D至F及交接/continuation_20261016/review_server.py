"""Loopback review collector. Public tasks omit methods, GT answers and real frame IDs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import random
import re
import secrets
import threading
import time
from urllib.parse import parse_qs, urlsplit
from PIL import Image, ImageDraw
from probability_diagnosis import RUN

HUMAN=RUN/"human_review"
LOCK=threading.RLock()


def read(path):return json.loads(Path(path).read_text("utf-8"))


def durable(path,value):
    payload=json.dumps(value,ensure_ascii=False,indent=2)+"\n"
    temporary=path.with_suffix(".tmp")
    temporary.write_text(payload,"utf-8")
    temporary.replace(path)


def stamp():return datetime.now(timezone.utc).isoformat()


class ReviewApp:
    def __init__(self,qa=False):
        self.qa=qa
        self.key=read(HUMAN/"private_key.json")
        self.secret=Path(self.key["secret_path"]).read_text("ascii")
        self.answers=HUMAN/("qa_answers" if qa else "answers")
        self.answers.mkdir(exist_ok=True)
        self.page=Path(__file__).with_name("review.html").read_bytes()
        self.placeholders={}

    def path(self,session):
        if not re.fullmatch(r"[a-f0-9]{32}",str(session)):raise ValueError("无效核对编号")
        return self.answers/f"{session}.json"

    def load(self,session):return read(self.path(session))

    def persist(self,value):
        value["updated_utc"]=stamp()
        durable(self.path(value["session"]),value)

    def start(self,alias,exposure):
        alias=str(alias).strip()
        if not 1<=len(alias)<=32:raise ValueError("请填写1—32字的观察者代号")
        if exposure not in {"no","yes","unsure"}:raise ValueError("请选择是否看过本批图像或目标位置")
        with LOCK:
            existing=[read(path) for path in sorted(self.answers.glob("*.json"))]
            for session in existing:
                if session["alias"]==alias:return self.public(session)
            ident=secrets.token_hex(16)
            seed=int(hmac.new(self.secret.encode(),ident.encode(),hashlib.sha256).hexdigest(),16)
            rng=random.Random(seed)
            block=len(existing)%3
            condition_order=["original","E","H265"]
            global_seed=int(hashlib.sha256(self.secret.encode()).hexdigest(),16)
            random.Random(global_seed).shuffle(condition_order)
            discovery=[]
            for index,frame in enumerate(self.key["discovery_frames"]):
                condition=condition_order[(index+block)%3]
                asset=frame["variants"][condition]
                discovery.append({"id":secrets.token_hex(12),"phase":"discovery","asset":asset})
            rng.shuffle(discovery)
            recognition=[{"id":secrets.token_hex(12),"phase":"recognition","asset":asset}
                         for asset in self.key["recognition_assets"]]
            rng.shuffle(recognition)
            session={
                "session":ident,"alias":alias,"prior_exposure":exposure,"cohort":block,
                "qa_mode":self.qa,"created_utc":stamp(),"study_date":"2026-10-16",
                "protocol_sha256":self.key["protocol_sha256"],
                "tasks":discovery+recognition,"answers":[],"draft":None,
            }
            self.persist(session)
            return self.public(session)

    def public(self,session):
        index=len(session["answers"]);tasks=session["tasks"]
        result={
            "session":session["session"],"alias":session["alias"],
            "prior_exposure":session["prior_exposure"],"qa_mode":self.qa,
            "done":index,"total":len(tasks),"discovery_total":7,"recognition_total":30,
            "complete":index==len(tasks),"draft":session.get("draft"),
        }
        if index<len(tasks):
            task=tasks[index];asset=self.key["assets"][task["asset"]]
            public={k:asset[k] for k in ("natural_width","natural_height","display_width","display_height")}
            public.update({"id":task["id"],"phase":task["phase"],
                "phase_index":index+1 if index<7 else index-7+1,
                "phase_total":7 if index<7 else 30,
                "image_url":f"/asset/{task['asset']}.png?session={session['session']}"})
            if task["phase"]=="recognition":
                public["cue_box"]=[60,50,40,30] if self.qa else asset["cue_box"]
            result["task"]=public
        return result

    def validate(self,session,payload,commit=False):
        index=len(session["answers"])
        if index>=len(session["tasks"]):raise ValueError("本轮已经全部保存")
        task=session["tasks"][index]
        if payload.get("task_id")!=task["id"]:raise ValueError("画面已经切换，请刷新后继续")
        choice=payload.get("choice")
        valid={"found","none","uncertain"} if task["phase"]=="discovery" else {"recognizable","uncertain","not_recognizable"}
        if choice is not None and choice not in valid:raise ValueError("无效选项")
        if commit and choice not in valid:raise ValueError("请先选择判断")
        points=payload.get("points",[])
        if not isinstance(points,list) or len(points)>30:raise ValueError("标记数量无效")
        checked=[]
        for point in points:
            if not isinstance(point,list) or len(point)!=2:raise ValueError("无效坐标")
            x,y=point
            if type(x) not in (int,float) or type(y) not in (int,float) or not (0<=x<1280 and 0<=y<720):
                raise ValueError("坐标超出画面")
            checked.append([float(x),float(y)])
        if task["phase"]=="recognition" and points:raise ValueError("局部辨认无需标记")
        if commit and task["phase"]=="discovery":
            if choice=="found" and not checked:raise ValueError("发现鱼后，请至少点击一个位置")
            if choice=="none" and checked:raise ValueError("未发现鱼时请清除标记")
        geometry=payload.get("display",{})
        if not isinstance(geometry,dict):raise ValueError("显示信息无效")
        geometry={key:value for key,value in geometry.items()
                  if key in {"image_css_width","image_css_height","device_pixel_ratio","viewport_width","viewport_height"}
                  and type(value) in (int,float) and 0<value<100000}
        elapsed=payload.get("elapsed_ms",0)
        if type(elapsed) not in (int,float) or not (0<=elapsed<10**12):elapsed=0
        return {
            "task_id":task["id"],"phase":task["phase"],"choice":choice,"points":checked,
            "display":geometry,"elapsed_ms":elapsed,"saved_utc":stamp(),
        }

    def answer(self,payload,commit):
        with LOCK:
            session=self.load(payload.get("session"))
            answer=self.validate(session,payload,commit)
            if commit:
                session["answers"].append(answer);session["draft"]=None
            else:session["draft"]=answer
            self.persist(session)
            return self.public(session) if commit else {"saved":True,"is_draft":True}

    def asset(self,token,session_id):
        session=self.load(session_id)
        limit=min(len(session["answers"])+1,len(session["tasks"]))
        allowed={task["asset"] for task in session["tasks"][:limit]}
        if token not in allowed:raise PermissionError("尚未开放的核对画面")
        asset=self.key["assets"].get(token)
        if asset is None:raise ValueError("画面不存在")
        if self.qa:
            size=(asset["natural_width"],asset["natural_height"])
            if size not in self.placeholders:
                image=Image.new("RGB",size,"#d8e3e9");pen=ImageDraw.Draw(image)
                pen.rectangle((size[0]//4,size[1]//4,size[0]//2,size[1]//2),fill="#739ab2")
                pen.text((20,20),"QA - SYNTHETIC FIXTURE",fill="black")
                buffer=io.BytesIO();image.save(buffer,format="PNG");self.placeholders[size]=buffer.getvalue()
            return self.placeholders[size]
        return Path(asset["path"]).read_bytes()


class Handler(BaseHTTPRequestHandler):
    server_version="LocalReview/1"
    def log_message(self,*args):return
    @property
    def app(self):return self.server.review_app

    def reply(self,payload,status=200,kind="application/json; charset=utf-8"):
        if not isinstance(payload,bytes):payload=json.dumps(payload,ensure_ascii=False).encode("utf-8")
        self.send_response(status);self.send_header("Content-Type",kind)
        self.send_header("Content-Length",str(len(payload)))
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.end_headers();self.wfile.write(payload)

    def do_GET(self):
        request=urlsplit(self.path);query=parse_qs(request.query)
        try:
            if request.path=="/":return self.reply(self.app.page,kind="text/html; charset=utf-8")
            if request.path=="/health":return self.reply({"ok":True,"qa_mode":self.app.qa})
            if request.path=="/api/state":
                session=self.app.load(query.get("session",[""])[0])
                return self.reply(self.app.public(session))
            if request.path=="/api/export":
                session=self.app.load(query.get("session",[""])[0])
                return self.reply({key:session[key] for key in (
                    "session","alias","prior_exposure","qa_mode","created_utc","protocol_sha256","answers","draft")})
            if request.path.startswith("/asset/") and re.fullmatch(r"/asset/[a-f0-9]{32}\.png",request.path):
                token=request.path.split("/")[-1][:-4]
                return self.reply(self.app.asset(token,query.get("session",[""])[0]),kind="image/png")
            return self.reply({"error":"未提供此内容"},404)
        except PermissionError as error:return self.reply({"error":str(error)},403)
        except (ValueError,KeyError,FileNotFoundError) as error:return self.reply({"error":str(error)},400)

    def do_POST(self):
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<131072:raise ValueError("请求大小无效")
            payload=json.loads(self.rfile.read(length))
            if self.path=="/api/start":
                return self.reply(self.app.start(payload.get("alias",""),payload.get("prior_exposure")))
            if self.path=="/api/draft":return self.reply(self.app.answer(payload,False))
            if self.path=="/api/answer":return self.reply(self.app.answer(payload,True))
            return self.reply({"error":"未提供此操作"},404)
        except (ValueError,KeyError,FileNotFoundError) as error:return self.reply({"error":str(error)},400)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--port",type=int,default=8780)
    parser.add_argument("--qa",action="store_true");parser.add_argument("--ready",type=Path,required=True)
    args=parser.parse_args()
    app=ReviewApp(args.qa)
    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
    server.review_app=app
    args.ready.parent.mkdir(parents=True,exist_ok=True)
    import os
    durable(args.ready,{"host":"127.0.0.1","port":server.server_address[1],"pid":os.getpid(),"qa_mode":args.qa})
    print("REVIEW_READY",server.server_address[1],"QA" if args.qa else "HUMAN",flush=True)
    server.serve_forever()


if __name__=="__main__":main()
