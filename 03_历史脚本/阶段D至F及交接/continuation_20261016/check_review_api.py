"""Integration checks against QA-only collector; no actual human answer is created."""
from __future__ import annotations
import json
from pathlib import Path
import secrets
import urllib.error
import urllib.request
from probability_diagnosis import RUN,OUT,read

BASE="http://127.0.0.1:8781"


def call(path,payload=None,expected=200):
    request=urllib.request.Request(BASE+path,data=json.dumps(payload).encode() if payload is not None else None,
                                   headers={"Content-Type":"application/json"} if payload is not None else {})
    try:
        with urllib.request.urlopen(request,timeout=10) as response:
            raw=response.read();status=response.status
            value=json.loads(raw) if "application/json" in response.headers.get("Content-Type","") else raw
    except urllib.error.HTTPError as response:
        status=response.code;value=json.loads(response.read())
    assert status==expected,(path,status,value)
    return value


def main():
    assert call("/health")["qa_mode"] is True
    real_answers=RUN/"human_review/answers"
    before={p.name:p.read_bytes() for p in real_answers.glob("*.json")}
    alias="QA_API_"+secrets.token_hex(3)
    state=call("/api/start",{"alias":alias,"prior_exposure":"unsure"})
    session=state["session"]
    assert state["done"]==0 and state["qa_mode"] and "condition" not in str(state)
    key=read(RUN/"human_review/private_key.json")
    future=key["recognition_assets"][0]
    call("/asset/"+future+".png?session="+session,expected=403)
    call("/private_key.json",expected=404)
    old_id=state["task"]["id"]
    payload={"session":session,"task_id":old_id,"choice":"found","points":[[123,234]],
             "display":{"image_css_width":960,"image_css_height":540},"elapsed_ms":100}
    call("/api/draft",payload)
    resumed=call("/api/state?session="+session)
    assert resumed["done"]==0 and resumed["draft"]["points"]==[[123.,234.]]
    disk=read(RUN/f"human_review/qa_answers/{session}.json")
    assert disk["draft"]["choice"]=="found" and len(disk["answers"])==0
    bad={**payload,"points":[]}
    call("/api/answer",bad,expected=400)
    for index in range(37):
        assert state["done"]==index
        phase="discovery" if index<7 else "recognition"
        assert state["task"]["phase"]==phase
        state=call("/api/answer",{
            "session":session,"task_id":state["task"]["id"],"choice":"none" if index<7 else "uncertain",
            "points":[],"display":{"image_css_width":960 if index<7 else 480,"image_css_height":540 if index<7 else 360},
            "elapsed_ms":100})
        if index==0:call("/api/answer",payload,expected=400)
    assert state["complete"] and state["done"]==37 and "task" not in state
    export=call("/api/export?session="+session)
    assert len(export["answers"])==37 and "tasks" not in export and export["qa_mode"]
    # Counterbalancing across three distinct QA sessions covers each method at each frame.
    matrix={}
    for group in range(3):
        result=call("/api/start",{"alias":alias+"_balance"+str(group),"prior_exposure":"no"})
        stored=read(RUN/f"human_review/qa_answers/{result['session']}.json")
        for task in stored["tasks"][:7]:
            asset=key["assets"][task["asset"]]
            matrix.setdefault(asset["source_frame"],set()).add(asset["condition"])
    assert all(values=={"original","E","H265"} for values in matrix.values())
    after={p.name:p.read_bytes() for p in real_answers.glob("*.json")}
    assert before==after
    result={
        "qa_mode_enforced":True,"real_human_answer_files_unchanged":True,
        "draft_persisted_without_confirm":True,"all37_confirmations_complete_without_final_save":True,
        "future_prompted_assets_blocked_during_discovery":True,"private_key_route_not_served":True,
        "public_tasks_hide_method_and_real_frame":True,"committed_task_cannot_be_overwritten":True,
        "three_cohort_counterbalance_verified":True,"human_observations_generated":0,
    }
    (OUT/"人工核对接口核验.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n","utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
