"""Meaningful geometry, decoder, and packet-validation checks for stage E."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
from run_stage_e import RUN,OUT,CONTROL,PROJECT,read,write_json,sha256
from roi_layout import plan_halves,MULTI_PROFILE
from system import unpack_packet

def unit():
    cases=[(1280,720,[{"regions":[{"box":[620,300,50,40]},{"box":[0,0,10,10]},{"box":[1278,718,2,2]}]}]),(66,34,[{"regions":[{"box":[31,16,4,5]},{"box":[65,33,1,1]}]}])]
    for width,height,records in cases:
        patches=plan_halves(records,width,height,32,16)
        coverage=np.zeros((height,width),dtype=bool)
        for p in patches:
            x,y,w,h=p["box"]
            assert min(w,h)>=16 and all(k%2==0 for k in (x,y,w,h))
            assert 0<=x<x+w<=width and 0<=y<y+h<=height
            assert not coverage[y:y+h,x:x+w].any()
            coverage[y:y+h,x:x+w]=True
        for r in records:
            for box in r["regions"]:
                x,y,w,h=box["box"];assert coverage[y:y+h,x:x+w].all()
    assert plan_halves([{"regions":[]}],1280,720)==[]
    OUT.mkdir(exist_ok=True,parents=True)
    write_json(OUT/"几何单元检查.json",{"cross_midline_and_borders_preserve_all_predictions":True,"even_geometry_and_no_overlap":True,"narrow_even_frame_geometry":True,"empty_roi_plan":True})
    print("GEOMETRY_CHECKS_PASS",flush=True)

def rewrite_packet(source,target,transform):
    with zipfile.ZipFile(source) as z:
        m=json.loads(z.read("manifest.json"));transform(m)
        with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_STORED) as result:
            result.writestr("manifest.json",json.dumps(m))
            for name in m["files"]:result.writestr(name,z.read(name))

def full():
    selected=read(RUN/"selection.json");packet=Path(selected["file"])
    decoder=read(RUN/"network/receiver/decode.json")
    assert decoder["profile"]==MULTI_PROFILE and decoder["frames"]==50
    assert sha256(RUN/"network/receiver/message.scv")==selected["sha256"]
    m=read(Path(selected["file"]).parent/"manifest.json")
    receiver=RUN/"network/receiver"
    double=0
    for frame in range(50):
        rgb=np.asarray(Image.open(receiver/f"rgb/{frame:05d}.png").convert("RGB"))
        context=np.asarray(Image.open(receiver/f"context_rgb/{frame:05d}.png").convert("RGB"))
        covered=np.zeros((720,1280),dtype=bool);active=0
        for index,patch in enumerate(m["patches"]):
            if not patch["start"]<=frame<patch["start"]+patch["frames"]:continue
            x,y,w,h=patch["box"];local=frame-patch["start"]
            decoded=np.asarray(Image.open(receiver/f"roi_rgb_{index:04d}/{local:05d}.png").convert("RGB"))
            assert np.array_equal(rgb[y:y+h,x:x+w],decoded),"A ROI was overwritten or omitted during compositing"
            assert not covered[y:y+h,x:x+w].any();covered[y:y+h,x:x+w]=True;active+=1
        assert np.array_equal(rgb[~covered],context[~covered]),"Background altered outside ROI"
        double+=active==2
    assert double>0,"This check must exercise two simultaneous ROI layers"
    folder=RUN/"protocol_checks";folder.mkdir()
    tests=[]
    def overlap(m):m["patches"].append(copy.deepcopy(m["patches"][0]))
    def crossing(m):m["patches"][0]["box"]=[630,0,32,32]
    def unknown(m):m["profile"]="unsupported-future-profile"
    cases=[("spatial_overlap",packet,overlap),("cross_partition",packet,crossing),("unknown_version",packet,unknown),("legacy_temporal_overlap",Path(read(CONTROL/"A/selection.json")["file"]),overlap)]
    for name,source,transform in cases:
        changed=folder/f"{name}.scv";rewrite_packet(source,changed,transform)
        output=folder/name;output.mkdir()
        try:unpack_packet(changed,output)
        except ValueError as error:tests.append({"case":name,"rejected":True,"reason":str(error)})
        else:raise AssertionError(f"Invalid packet accepted: {name}")
    def empty(m):
        m["patches"]=[];m["files"]={k:v for k,v in m["files"].items() if not k.startswith("roi_")}
    empty_packet=folder/"no_roi.scv";rewrite_packet(packet,empty_packet,empty)
    empty_output=folder/"no_roi";empty_output.mkdir()
    empty_manifest,_,_=unpack_packet(empty_packet,empty_output);assert empty_manifest["patches"]==[]
    result={"all_50_composited_frames_checked":True,"two_simultaneous_roi_frames":double,"each_roi_pixel_equals_received_decoded_patch":True,"non_roi_pixels_equal_received_context":True,"no_roi_v2_packet_accepted":True,"invalid_packets":tests,"legacy_exact_rgb_check":"see old-package CLI validation"}
    write_json(OUT/"多区域恢复与协议检查.json",result)
    print("MULTI_ROI_RECONSTRUCTION_AND_PROTOCOL_CHECKS_PASS",double,flush=True)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["unit","full"]);args=p.parse_args()
    if args.mode=="unit":unit()
    else:full()
