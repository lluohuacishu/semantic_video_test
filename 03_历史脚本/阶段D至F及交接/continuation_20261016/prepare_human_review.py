"""Prepare anonymous, fixed-scale review assets without creating observer answers."""
from __future__ import annotations
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sys
import numpy as np
from PIL import Image
from probability_diagnosis import WORKSPACE, PROJECT, RUN, OUT, F_OUT, read
sys.path.insert(0,str(PROJECT/"semantic_mvp"))
from runtime import sha256, write_json

HUMAN = RUN/"human_review"
SERVER = Path(__file__).with_name("review_server.py")
PAGE = Path(__file__).with_name("review.html")


def main():
    HUMAN.mkdir(parents=True,exist_ok=True)
    secret_path = HUMAN/"randomization_secret.txt"
    if not secret_path.exists(): secret_path.write_text(secrets.token_hex(32),"ascii")
    secret = secret_path.read_text("ascii")
    records = read(OUT/"逐帧逐区域概率.json")
    lookup = {(r["source_frame"],r["condition"]):r for r in records}
    components = read(F_OUT/"positive_case/含鱼GT保护与逐连通域核对.json")["gt_components"]
    protocol = {
        "date":"2026-10-16","purpose":"human readability follow-up; no answers prefilled or AI ratings",
        "all_cases":"seven previously fixed labeled frames; ten GT8-connected regions, not ten verified fish",
        "discovery":{
            "frames":[0,15,30,45,60,75,90],"tasks_per_observer":7,
            "one_version_per_frame_per_observer":True,
            "assignment":"balanced rotating3-cohort assignment; three observers cover three conditions at every frame; separate shuffled order per observer",
            "limitation":"one observer is an exploratory unpaired comparison; repeated frame versions deliberately avoided in discovery",
            "display":"960x540 CSS pixels for1280x720 originals; no extra image zoom; record viewport and devicePixelRatio",
            "response":"found with source-coordinate click points / none / uncertain",
            "scoring_if_done":"maximum one-to-one matching to GT connected-component bounding boxes expanded8 source pixels; geometry proxy, not pixel recall or verified fish-instance recall",
        },
        "recognition":{
            "tasks_per_observer":30,"all_methods_all_components":True,
            "display":"same240x180 source crop,480x360 CSS pixels; cue is a bounding rectangle, no GT silhouette",
            "response":["recognizable as fish with basic outline","uncertain","not recognizable"],
            "known_location_not_unprompted_discovery":True,
        },
        "order":"all7 discovery tasks confirmed before any prompted crop; committed discovery answers cannot be rewritten after seeing cues",
        "blinding":"no method, real frame number, probability, PSNR, source path, or GT feedback in public tasks",
        "exposure":"record prior familiarity with these images/GT; prior-exposed discovery is not naive",
        "saving":"current draft autosaved; each confirm is durable; no final global-save button",
        "reviewer_count":"one is exploratory; independent extra observers optional; do not infer population significance",
        "QA":"separate QA answer directory and visible QA mode; never mixed into actual human results",
        "unstarted_human_results":True,
        "asset_builder_sha256":sha256(__file__),"server_sha256":sha256(SERVER),"page_sha256":sha256(PAGE),
        "source_probability_protocol_sha256":sha256(OUT/"概率诊断冻结协议.json"),
        "randomization_secret_sha256":sha256(secret_path),
    }
    target = HUMAN/"protocol.json"
    revised=False
    if target.exists() and read(target)!=protocol:
        assert not list((HUMAN/"answers").glob("*.json")),"Human answers exist; do not revise their protocol"
        history=HUMAN/"protocol_history";history.mkdir(exist_ok=True)
        old_hash=sha256(target)
        (history/f"{old_hash}.json").write_bytes(target.read_bytes())
        if (HUMAN/"private_key.json").exists():
            (history/f"{old_hash}_private_key.json").write_bytes((HUMAN/"private_key.json").read_bytes())
        write_json(target,protocol);revised=True
    elif not target.exists():write_json(target,protocol)
    write_json(OUT/"人工核对预先约定.json",protocol)
    asset_dir = HUMAN/"crop_assets";asset_dir.mkdir(exist_ok=True)
    assets = {}; discovery_frames = []
    def token(text):
        return hmac.new(secret.encode(),text.encode(),hashlib.sha256).hexdigest()[:32]
    for real in (0,15,30,45,60,75,90):
        variants = {}
        for condition in ("original","E","H265"):
            record = lookup[(real,condition)]
            path = Path(record["image"]);assert sha256(path) == record["image_file_sha256"]
            ident = token(f"full:{real}:{condition}")
            assets[ident] = {
                "path":str(path),"file_sha256":sha256(path),"phase":"discovery",
                "condition":condition,"source_frame":real,"natural_width":1280,"natural_height":720,
                "display_width":960,"display_height":540,
            }
            variants[condition] = ident
        discovery_frames.append({"source_frame":real,"variants":variants})
    recognition = []
    for region in components:
        x,y,w,h = region["box"]; real=region["source_frame"]; component=region["component"]
        assert w+64 <= 240 and h+64 <= 180
        left=int(np.clip((2*x+w-240)//2,0,1280-240))
        top=int(np.clip((2*y+h-180)//2,0,720-180))
        for condition in ("original","E","H265"):
            image = Image.open(lookup[(real,condition)]["image"]).convert("RGB")
            ident = token(f"crop:{real}:{component}:{condition}")
            path=asset_dir/f"{ident}.png"
            crop=image.crop((left,top,left+240,top+180))
            if not path.exists(): crop.save(path)
            assert np.array_equal(np.asarray(Image.open(path)),np.asarray(crop))
            assets[ident]={
                "path":str(path),"file_sha256":sha256(path),"phase":"recognition",
                "condition":condition,"source_frame":real,"component":component,
                "natural_width":240,"natural_height":180,"display_width":480,"display_height":360,
                "source_crop":[left,top,240,180],"cue_box":[x-left,y-top,w,h],
            }
            recognition.append(ident)
    private = {
        "version":1,"protocol_sha256":sha256(target),"assets":assets,
        "discovery_frames":discovery_frames,"recognition_assets":recognition,
        "secret_path":str(secret_path),
    }
    key=HUMAN/"private_key.json"
    if key.exists() and not revised: assert read(key) == private
    else: write_json(key,private)
    for qa in (False,True):
        (HUMAN/("qa_answers" if qa else "answers")).mkdir(exist_ok=True)
    write_json(OUT/"人工材料核验.json",{
        "anonymous_full_images":21,"anonymous_prompted_crops":30,"source_pixels_and_crops_verified":True,
        "tasks_per_observer":37,"real_answer_files":len(list((HUMAN/"answers").glob("*.json"))),
        "private_key_not_served_by_public_API":True,
        "human_ratings_created":0,"test_results_are_not_human_observations":True,
    })
    print("REVIEW_MATERIALS_READY",json.dumps({"full_images":21,"crops":30,"tasks_per_observer":37}),flush=True)


if __name__=="__main__":main()
