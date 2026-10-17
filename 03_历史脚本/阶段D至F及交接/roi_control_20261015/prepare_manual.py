"""Freeze the user's saved continuous ROI review for diagnostic C/D controls."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
import shutil
from control_common import *
from annotation_server import validate

SUPPLEMENT=OUT/"人工ROI补充"

def main():
    SUPPLEMENT.mkdir(exist_ok=True)
    kit=OUT/"连续人工ROI核对"
    source=kit/"annotations.json"
    confirmed=read(source)
    validate(copy.deepcopy(confirmed))
    draft=read(kit/"用户确认前的目标框草稿_20261015.json")
    assert all(row["status"] in {"human_reviewed","publisher_verified"} for row in confirmed["frames"])
    assert all(row["boxes"]==old["boxes"] for row,old in zip(confirmed["frames"],draft["frames"]))
    assert all(row["no_fish"] is (not row["boxes"]) for row in confirmed["frames"])
    snapshot=RUN/"manual_roi_snapshot.json"
    if snapshot.exists():assert sha256(snapshot)==sha256(source),"Saved labels changed since C/D inputs were frozen"
    else:shutil.copy2(source,snapshot)
    base=read(RUN/"protocol.json")
    assert sha256(DIRECTION)==base["direction_sha256"]
    _,semantics,_,_,labels=frozen()
    assert sha256(PROJECT/"runs/phase_c_20261013/segformer_adaptation_preserve8/best.pt")==base["teacher"]["model_sha256"]
    for label in labels:
        expected=next(x for x in base["gt_frames"] if x["frame"]==label["frame_index"])
        assert sha256(label["fish_mask"])==expected["sha256"]
    records=[{"frame":row["local_frame"],"regions":[{"box":box} for box in row["boxes"]]} for row in confirmed["frames"]]
    known=read(kit/"publisher_template.json")
    changes=[{"frame":row["source_frame"],"publisher_boxes":old["boxes"],"user_boxes":row["boxes"]} for row,old in zip(confirmed["frames"],known["frames"]) if old["status"]=="publisher_verified" and row["boxes"]!=old["boxes"]]
    protocol={"base_protocol_sha256":sha256(RUN/"protocol.json"),"direction_sha256":sha256(DIRECTION),"prepared_at_utc":datetime.now(timezone.utc).isoformat(),"annotation_source":str(source),"annotation_snapshot":str(snapshot),"annotation_snapshot_sha256":sha256(snapshot),"reviewed_by":"human user; assistant applied confirmation flags and saved per user's explicit statement","user_attestation_file":str(kit/"人工核对完成声明_20261015.json"),"confirmed_frames":50,"positive_frames":sum(bool(x["boxes"]) for x in confirmed["frames"]),"no_fish_frames":sum(x["no_fish"] for x in confirmed["frames"]),"boxes":sum(len(x["boxes"]) for x in confirmed["frames"]),"boxes_unchanged_after_user_statement":True,"publisher_roi_box_differences":changes,"pixel_gt":"unchanged four publisher fish masks only; manual boxes are not segmentation GT","semantic_messages":"unchanged frozen model prediction messages for every layered group; C/D human ROI coordinates are charged in manifest","target_complete_bytes":100000,"tolerance":.02,"groups":{"C":{"window":32,"source":"continuous reviewed human/publisher ROI; diagnostic only"},"D":{"window":4,"source":"same reviewed ROI; diagnostic only"}},"profile":"same fixed background CRF42 and ROI P-only medium encoding as A/B","no_new_training":True,"input_is_known_training_clip":True,"interpretation":"human ROI diagnosis; not deployed performance and not a strict oracle upper bound"}
    write_json(RUN/"manual_protocol.json",protocol)
    write_json(SUPPLEMENT/"人工ROI冻结协议.json",protocol)
    write_json(SUPPLEMENT/"人工ROI几何复核.json",[geometry(records,w,labels) for w in (32,4)])
    print(json.dumps({k:protocol[k] for k in ("confirmed_frames","positive_frames","no_fish_frames","boxes","annotation_snapshot_sha256","publisher_roi_box_differences")},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
