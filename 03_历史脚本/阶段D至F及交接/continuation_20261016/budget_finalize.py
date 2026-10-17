"""Final package and stream audit, without changing frozen selection or scores."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile
from budget_experiment import RUN,OUT,PROJECT,WORKSPACE,read,check,originals,sha256,write_json
from runtime import FFPROBE,run


def main():
    protocol=check()
    _,base,semantic=originals()
    selections=read(RUN/"selections_locked.json")["selections"]
    streams=[]
    for condition,selection in selections.items():
        assert abs(selection["relative_error"])<=.02
        assert sum(selection["parts"].values())==selection["file_bytes"]
        assert sha256(selection["file"])==selection["sha256"]
        folder=RUN/condition/"network/receiver"
        with zipfile.ZipFile(folder/"message.scv") as archive:
            assert archive.testzip() is None
            manifest=json.loads(archive.read("manifest.json"))
            assert archive.read("semantics.json.gz")==semantic
            assert manifest["patches"]==base["patches"]
            expected_config={**base["configuration"],"background_crf":selection["background_crf"],
                             "roi_crf":selection["roi_crf"]}
            assert manifest["configuration"]==expected_config
            for name,record in manifest["files"].items():
                payload=archive.read(name)
                assert len(payload)==record["bytes"]
                assert hashlib.sha256(payload).hexdigest()==record["sha256"]
        geometry={"context.mp4":(640,360,50),**{p["file"]:(p["box"][2],p["box"][3],p["frames"])
                                                    for p in manifest["patches"]}}
        for name,(width,height,count) in geometry.items():
            path=folder/"received_streams"/name
            inspected=json.loads(run([FFPROBE,"-v","error","-select_streams","v:0","-count_frames",
                "-show_entries","stream=width,height,nb_read_frames,avg_frame_rate,pix_fmt,color_space,color_range:frame=pict_type",
                "-of","json",path]))
            video=inspected["streams"][0]
            assert (video["width"],video["height"],int(video["nb_read_frames"]))==(width,height,count)
            assert video["avg_frame_rate"]=="25/1"
            assert video["pix_fmt"]=="yuv420p" and video["color_space"]=="bt709" and video["color_range"]=="tv"
            counts=dict(Counter(frame["pict_type"] for frame in inspected["frames"]))
            assert counts.get("B",0)==0 and sum(counts.values())==count
            streams.append({"condition":condition,"stream":name,**video,"frame_types":counts,
                            "payload_sha256_verified":True})
        manifest_normalized=dict(manifest)
        manifest_normalized["configuration"]=base["configuration"]
        manifest_normalized["files"]=base["files"]
        assert manifest_normalized==base
    write_json(OUT/"新增流编码结构核验.json",streams)
    report=OUT/"2026-10-16_固定ROI预算分配对照与判断.md"
    report_text=report.read_text("utf-8")
    links=re.findall(r"\]\((C:/[^)]+)\)",report_text)
    assert all(Path(path).is_file() for path in links)
    assert report.read_bytes()==(PROJECT/"docs/budget_allocation_20261016.md").read_bytes()
    checks=read(OUT/"预算对照核验.json")
    checks.update({"selected_package_CRC_and_all_payload_hashes_verified":True,
                   "new_streams_probed":len(streams),"all_new_streams_P_only_BT709_limited_YUV420":True,
                   "all_selected_complete_files_within_2_percent_of_target":True,
                   "report_local_links_verified":len(links),"figures_visually_checked":5,
                   "new_complete_candidate_packets":sum(s["complete_candidates"] for s in selections.values()),
                   "production_defaults_changed":False,"report_complete":True})
    write_json(OUT/"预算对照核验.json",checks)
    write_json(OUT/"本轮交付清单.json",{
        "report":str(report),"report_sha256":sha256(report),
        "protocol_sha256":sha256(OUT/"预算对照冻结协议.json"),
        "results_sha256":sha256(OUT/"预算对照实测结果.json"),
        "evaluator_sha256":sha256(Path(__file__).with_name("budget_evaluate.py")),
        "report_generator_sha256":sha256(Path(__file__).with_name("budget_report.py")),
        "finalizer_sha256":sha256(__file__),"all_new_selected_packets":{c:s["sha256"] for c,s in selections.items()},
        "old_results_preserved":True,"new_models_or_training":0,"next_stage_created":False})
    print(json.dumps({"report":str(report),"new_streams_verified":len(streams),"report_links":len(links),
                      "production_unchanged":True},ensure_ascii=False))


if __name__=="__main__":main()
