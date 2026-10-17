"""Bounded, fixed-layout allocation experiment using previously frozen predictions."""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from PIL import Image

WORKSPACE = Path(r"C:\Users\lenovo\Documents\ChatGPT\语义通信")
PROJECT = Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
OLD_RUN = PROJECT/"runs/semantic_mvp_stage_f_20261015/step3/positive_case"
OLD_OUT = WORKSPACE/"outputs/semantic_roi_phase_f/step3_frozen_external/positive_case"
DATA = PROJECT/"data/phase_f_external_20261015/positive_case/manifest.json"
RUN = PROJECT/"runs/semantic_mvp_budget_20261016"
OUT = WORKSPACE/"outputs/semantic_budget_20261016"
SOURCE = OLD_RUN/"initial_crf32/private_frames"
WORK = Path(__file__).parent
sys.path.insert(0,str(PROJECT/"semantic_mvp"))
from runtime import encode_h265, sha256, write_json, FFMPEG, run
from system import deliver

VARIANTS = {"BG38":38., "BG34":34.}


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def originals():
    selection = read(OLD_RUN/"E/selection.json")
    assert sha256(selection["file"]) == selection["sha256"]
    with zipfile.ZipFile(selection["file"]) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        semantic = archive.read("semantics.json.gz")
    return selection, manifest, semantic


def freeze():
    RUN.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    selected, manifest, semantic = originals()
    previous = read(WORKSPACE/"outputs/semantic_review_20261016/概率诊断冻结协议.json")
    for path, expected in previous["file_sha256"].items():
        assert sha256(path) == expected, path
    data = read(DATA)
    sources = sorted(SOURCE.glob("*.png"))
    assert len(sources) == 50
    for index, path in enumerate(sources):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == data["source_rgb_sha256"][index]
    assert manifest["configuration"]["background_crf"] == 42
    assert manifest["configuration"]["roi_crf"] == 31
    files = [Path(__file__), WORK/"budget_evaluate.py", DATA, OLD_RUN/"E/selection.json",
             OLD_RUN/"H265/selection.json", WORKSPACE/"outputs/semantic_review_20261016/单人核对汇总.json",
             *[Path(p) for p in previous["file_sha256"]]]
    protocol = {
        "date":"2026-10-16", "scope":"one bounded allocation follow-up, no new stage or architecture",
        "video":"video6 source50..99, one development video, not independent validation",
        "GT_scope":"three fixed publisher labels60/75/90; remaining47 frames unlabeled, not background",
        "target_complete_file_bytes":selected["file_bytes"],
        "baseline":"reuse exact previously received E42/ROI31 packet and H265 B4 MP4; no new baseline transfer",
        "new_variants":VARIANTS,
        "fixed_configuration":manifest["configuration"],
        "fixed_patches":manifest["patches"],
        "fixed_teacher":manifest["teacher"],
        "fixed_semantic_gzip_sha256":hashlib.sha256(semantic).hexdigest(),
        "change_only":"background CRF and byte-matched ROI CRF; stream hashes/bytes updated accordingly",
        "byte_selection":{
            "start_ROI_CRF":31., "range":[0.,51.], "preferred_relative_error":.01,
            "valid_comparison_max_relative_error":.02, "max_complete_packets_per_variant":10,
            "algorithm":"initial31, high51 floor check if oversized, up to8 bisections; choose closest bytes only",
            "no_padding":True, "no_quality_GT_probability_or_human_scores_used":True,
            "infeasible_policy":"keep closest feasible candidate and label unmatched, do not change spatial/temporal configuration"
        },
        "context_geometry":"fixed640x360 reconstructed to1280x720; more bytes do not restore original background resolution",
        "selection_gate":"finish both byte-only selections before decoding/evaluating either new variant",
        "network":"both selected new packets20 decimal kbps TCP loopback to separate receiver processes",
        "metrics":["all50 pooled RGB PSNR","outside geometric ROI all50 PSNR (not GT background)",
                   "GT fish and GT background3-frame PSNR","GT background ring32px PSNR",
                   "fixed0.9 TP/FP/FN","GT probability mean/quantiles/max","3-label pixel AP"],
        "ring_definition":"65x65 maximum-filter dilation of binary GT, subtract all GT fish, three labeled frames only",
        "human_ratings":"existing ratings apply only to old E/H265; no ratings inferred for new variants",
        "causality_limit":"two fixed background settings; cannot separate all context/ROI-loss effects or establish generalization",
        "selection_file_hashes":{str(path):sha256(path) for path in dict.fromkeys(files)},
        "source_png_sha256":{str(path):sha256(path) for path in sources},
        "ffmpeg_version":run([FFMPEG,"-version"]).splitlines()[0],
    }
    target = RUN/"protocol.json"
    if target.exists():
        assert read(target) == protocol, "Frozen protocol mismatch"
    else:
        write_json(target,protocol)
    write_json(OUT/"预算对照冻结协议.json",protocol)
    print("BUDGET_PROTOCOL_FROZEN", selected["file_bytes"], VARIANTS, flush=True)
    return protocol


def check():
    protocol = read(RUN/"protocol.json")
    for path, expected in {**protocol["selection_file_hashes"], **protocol["source_png_sha256"]}.items():
        assert sha256(path) == expected, path
    return protocol


def context(variant):
    folder = RUN/variant
    folder.mkdir(exist_ok=True)
    path = folder/"context.mp4"
    record = folder/"context.json"
    if record.exists():
        assert sha256(path) == read(record)["sha256"]
        return path
    started = time.perf_counter()
    command = encode_h265(SOURCE,path,"25/1",VARIANTS[variant],scale=(640,360),count=50)
    write_json(record,{"file":str(path),"bytes":path.stat().st_size,"sha256":sha256(path),
                       "background_crf":VARIANTS[variant],"command":command,
                       "seconds":time.perf_counter()-started})
    print("CONTEXT",variant,path.stat().st_size,flush=True)
    return path


def candidate(variant, crf, protocol):
    folder = RUN/variant/"candidates"/f"roi_crf_{crf:.3f}"
    record = folder/"candidate.json"
    if record.exists():
        value = read(record)
        assert sha256(value["file"]) == value["sha256"]
        return value
    folder.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    _, base, semantic = originals()
    streams = folder/"streams"
    streams.mkdir()
    commands = []
    for patch in base["patches"]:
        commands.append(encode_h265(SOURCE,streams/patch["file"],base["fps"],crf,
                                    bbox=patch["box"],start=patch["start"],count=patch["frames"]))
    payloads = {"context.mp4":context(variant).read_bytes(),"semantics.json.gz":semantic}
    payloads.update({patch["file"]:(streams/patch["file"]).read_bytes() for patch in base["patches"]})
    manifest = copy.deepcopy(base)
    manifest["configuration"].update({"background_crf":VARIANTS[variant],"roi_crf":crf})
    manifest["files"] = {name:{"bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest()}
                         for name,payload in sorted(payloads.items())}
    packet = folder/"message.scv"
    with zipfile.ZipFile(packet,"x",compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json",json.dumps(manifest,ensure_ascii=False,separators=(",",":")))
        for name,payload in sorted(payloads.items()):
            archive.writestr(name,payload)
    # Inspect the real package; only the two quality fields and payload bookkeeping may differ.
    restored = copy.deepcopy(manifest)
    restored["files"] = base["files"]
    restored["configuration"] = base["configuration"]
    assert restored == base
    assert hashlib.sha256(semantic).hexdigest() == protocol["fixed_semantic_gzip_sha256"]
    parts = {"context":len(payloads["context.mp4"]),"roi":sum(len(payloads[p["file"]]) for p in base["patches"]),
             "semantic":len(semantic),"manifest_zip":packet.stat().st_size-sum(map(len,payloads.values()))}
    value = {"variant":variant,"background_crf":VARIANTS[variant],"roi_crf":crf,
             "file":str(packet),"file_bytes":packet.stat().st_size,"sha256":sha256(packet),
             "parts":parts,"target_bytes":protocol["target_complete_file_bytes"],
             "relative_error":(packet.stat().st_size/protocol["target_complete_file_bytes"])-1,
             "commands":commands,"encode_seconds":time.perf_counter()-started,
             "patches_and_semantic_bytes_unchanged":True}
    write_json(folder/"manifest.json",manifest)
    write_json(record,value)
    print("BUDGET_CANDIDATE",variant,crf,value["file_bytes"],flush=True)
    return value


def select(variant, protocol):
    destination = RUN/variant/"selection.json"
    if destination.exists():
        value=read(destination)
        assert sha256(value["file"]) == value["sha256"]
        return value
    target = protocol["target_complete_file_bytes"]
    values = [candidate(variant,31.,protocol)]
    def best(): return min(values,key=lambda value:abs(value["file_bytes"]-target))
    if abs(best()["relative_error"]) > .01:
        if values[0]["file_bytes"] > target:
            values.append(candidate(variant,51.,protocol))
            low, high = 31.,51.
            infeasible = values[-1]["file_bytes"] > target*1.02
        else:
            low, high = 0.,31.
            infeasible = False
        if not infeasible:
            for _ in range(8):
                if abs(best()["relative_error"]) <= .01:
                    break
                crf = round((low+high)/2,3)
                row = candidate(variant,crf,protocol)
                values.append(row)
                if row["file_bytes"] > target:
                    low=crf
                else:
                    high=crf
                write_json(RUN/variant/"search.json",values)
    assert len(values) <= 10
    chosen = {**best(),"matched_within_2_percent":abs(best()["relative_error"]) <= .02,
              "selection_basis":"complete package bytes only, no quality or task score",
              "complete_candidates":len(values)}
    write_json(destination,chosen)
    write_json(RUN/variant/"search.json",values)
    print("BUDGET_SELECTED",variant,chosen["file_bytes"],chosen["roi_crf"],flush=True)
    return chosen


def select_all():
    protocol=check()
    chosen={variant:select(variant,protocol) for variant in VARIANTS}
    write_json(RUN/"selections_locked.json",{
        "selections":chosen,"locked_before_any_new_decoded_scores":True,
        "utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
    })
    write_json(OUT/"完整文件匹配选择.json",chosen)
    check()


def transfer_all():
    check()
    chosen=read(RUN/"selections_locked.json")["selections"]
    for variant in VARIANTS:
        folder=RUN/variant/"network"
        folder.mkdir(exist_ok=True)
        if not (folder/"end_to_end.json").exists():
            deliver(Path(chosen[variant]["file"]),folder,20)
        decoded=read(folder/"receiver/decode.json")
        assert decoded["frames"] == 50
        assert decoded["file_sha256"] == chosen[variant]["sha256"]
        assert read(folder/"end_to_end.json")["separate_sender_receiver_processes"]
        audit=read(folder/"receiver/receiver_read_audit.json")
        assert all(row["allowed"] for row in audit["python_reads"])
    write_json(OUT/"新增两组真实收发.json",{v:read(RUN/v/"network/end_to_end.json") for v in VARIANTS})
    check()
    print("BUDGET_TRANSFERS_COMPLETE",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("freeze","select","transfer"))
    args=parser.parse_args()
    if args.command == "freeze":freeze()
    elif args.command == "select":select_all()
    else:transfer_all()
