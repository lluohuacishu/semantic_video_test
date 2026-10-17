"""A declared positive-GT supplement, same video and frozen method, no score selection."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import zlib
import numpy as np
from PIL import Image
import external_preflight as pre
import external_experiment as engine
import external_evaluate as evaluator
from runtime import run, sha256

BASE_RUN, BASE_OUT, BASE_DATA = pre.RUN, pre.OUT, pre.DATA
POS_RUN = BASE_RUN/"positive_case"
POS_OUT = BASE_OUT/"positive_case"
POS_DATA = BASE_DATA/"positive_case"


def prepare_data():
    available = pre.read(BASE_OUT/"同视频正样本可用性检查.json")
    assert available["first_eligible"] is not None
    anchor = available["first_eligible"]["source_frame"]
    start = min(609, max(50, anchor-24))
    indices = list(range(start, start+50))
    assert not set(indices) & set(range(50))
    pre.write(POS_RUN/"positive_clip_selection.json", {
        "rule": available["selection_rule"], "selected_first_positive_anchor": anchor,
        "selected_real_frame_indices": indices, "same_video_not_new_independent_video": True,
        "selection_used_model_or_codec_performance": False, "script_sha256": sha256(__file__),
    })
    labels = [r for r in available["gt_masks"] if r["source_frame"] in indices]
    names_by_group = {
        "sequence": [f"video6/sequence/{i:05d}.jpg" for i in indices],
        "images": [f"video6/images/{r['source_frame']:05d}.jpg" for r in labels],
    }
    file = next(f for f in pre.read(BASE_DATA/"figshare_metadata_live.json")["files"]
                if f["name"] == "video7.zip")
    reader = pre.AuditedRange(file["download_url"], file["size"])
    downloads = []
    with zipfile.ZipFile(reader) as remote:
        infos = sorted(remote.infolist(), key=lambda z:z.header_offset)
        next_offsets = {a.filename:b.header_offset for a,b in zip(infos,infos[1:])}
        for group, names in names_by_group.items():
            missing = [n for n in names if not (BASE_DATA/"raw"/n).exists()]
            if missing:
                first = min(remote.getinfo(n).header_offset for n in missing)
                last = max((remote.getinfo(n) for n in missing), key=lambda z:z.header_offset)
                assert next_offsets[last.filename]-first < 32*1024**2
                print("POSITIVE_PREFETCH", group, next_offsets[last.filename]-first, flush=True)
                reader.seek(first); reader.read(next_offsets[last.filename]-first)
            for name in names:
                path = BASE_DATA/"raw"/name
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(remote.read(name))
                info = remote.getinfo(name); payload = path.read_bytes()
                assert len(payload) == info.file_size and zlib.crc32(payload) == info.CRC
                downloads.append({"entry": name, "sha256": sha256(path), "crc_verified": True})
            pre.write(POS_RUN/"download_audit.json", {
                "entries": downloads, "http_bytes": reader.transferred, "ranges": reader.log,
                "full_archive_checksum_checked": False})
    prior = pre.read(BASE_DATA/"manifest.json")
    earlier_hashes = {h for v in pre.read(pre.PROJECT/"data/phase_c_20261013/manifest.json")["videos"]
                      for h in v["source_frame_rgb_hashes"]}
    earlier_hashes |= set(prior["source_rgb_sha256"])
    sequence = POS_DATA/"sequence"; sequence.mkdir(parents=True, exist_ok=True)
    hashes = []
    for number in indices:
        path = BASE_DATA/f"raw/video6/sequence/{number:05d}.jpg"
        target = sequence/path.name
        if not target.exists(): shutil.copy2(path, target)
        assert sha256(path) == sha256(target)
        rgb = np.asarray(Image.open(target).convert("RGB"))
        assert rgb.shape == (720,1280,3)
        value = hashlib.sha256(rgb.tobytes()).hexdigest()
        assert value not in earlier_hashes
        hashes.append(value)
    sparse = []
    for label in labels:
        number = label["source_frame"]
        rgb = np.asarray(Image.open(sequence/f"{number:05d}.jpg").convert("RGB"))
        annotated = np.asarray(Image.open(BASE_DATA/f"raw/video6/images/{number:05d}.jpg").convert("RGB"))
        assert np.array_equal(rgb, annotated)
        published = Path(label["published_mask"]); assert sha256(published) == label["mask_sha256"]
        gt = np.all(np.asarray(Image.open(published).convert("RGB")) == [0,0,128], axis=2)
        assert gt.shape == rgb.shape[:2] and int(gt.sum()) == label["fish_pixels"]
        target = POS_DATA/f"fish_masks/{number:05d}.png"; target.parent.mkdir(exist_ok=True)
        Image.fromarray(gt.astype(np.uint8)*255).save(target)
        sparse.append({
            "local_frame": number-start, "source_frame": number,
            "published_mask": str(published), "published_mask_sha256": sha256(published),
            "fish_mask": str(target), "fish_mask_sha256": sha256(target),
            "gt_available": True, "fish_pixels": int(gt.sum()),
            "annotated_rgb_equals_sequence": True,
        })
    manifest = {
        **prior, "sequence_dir": str(sequence), "source_frame_indices": indices,
        "source_rgb_sha256": hashes, "sparse_gt": sparse,
        "selection": "first positive publisher-GT anchor >=50; 50-frame disjoint consecutive supplement",
        "positive_anchor": anchor, "http_bytes_this_run": reader.transferred,
        "same_video_as_initial_case": True,
        "exact_rgb_duplicates_with_all_5_previous_experiment_videos_and_initial_case": 0,
        "conditional_GT_selection_not_model_or_method_score_selection": True,
    }
    pre.write(POS_DATA/"manifest.json", manifest); pre.write(POS_OUT/"新视频数据核验.json", manifest)
    print("POSITIVE_DATA", {"indices": [indices[0],indices[-1]],
                           "gt": [(r["source_frame"], r["fish_pixels"]) for r in sparse]}, flush=True)
    return manifest


def initial_encode(data):
    if (engine.INITIAL/"encode.json").exists(): return
    command = [
        sys.executable, engine.SYSTEM, "encode", "--input", data["sequence_dir"],
        "--output", engine.INITIAL, "--target", "fish", "--mode", "video",
        "--first", "0", "--frames", "50", "--fps", "25", "--roi-layout", "halves",
        "--background-scale", "0.5", "--background-crf", "42", "--roi-crf", "32",
        "--segment-frames", "32", "--padding", "16", "--min-pixels", "64",
    ]
    pre.write(POS_RUN/"initial_encode_command.json", [str(v) for v in command])
    started = time.perf_counter(); output = run(command)
    (POS_RUN/"initial_encode.log").write_text(output, "utf-8")
    pre.write(POS_RUN/"initial_encode_elapsed.json", {
        "seconds": time.perf_counter()-started,
        "includes": "fresh source preparation, perception, CRF32 encode; excludes rate search and transmission"})
    source = pre.read(engine.INITIAL/"source_audit.json")
    assert source["source_frame_indices"] == data["source_frame_indices"]
    assert source["rgb_sha256"] == data["source_rgb_sha256"]
    print(output, flush=True)


def main():
    POS_RUN.mkdir(parents=True, exist_ok=True); POS_OUT.mkdir(parents=True, exist_ok=True)
    data = prepare_data()
    engine.RUN, engine.OUT, engine.DATA = POS_RUN, POS_OUT, POS_DATA
    engine.INITIAL = POS_RUN/"initial_crf32"; engine.SOURCE = engine.INITIAL/"private_frames"
    protocol = copy.deepcopy(pre.read(BASE_RUN/"experiment_protocol.json"))
    protocol.update({
        "step": "3b", "source": data,
        "scope": "disjoint positive-GT supplement 50..99 from the same holdout video6; not a second independent video",
        "empty_gt": "positive publisher GT anchors are available; sparse unlabeled frames remain unknown",
        "conditional_source_selection": pre.read(POS_RUN/"positive_clip_selection.json"),
    })
    for path in (Path(__file__), POS_DATA/"manifest.json", Path(evaluator.__file__)):
        protocol["input_sha256"][str(path)] = sha256(path)
    path = POS_RUN/"experiment_protocol.json"
    if path.exists(): assert pre.read(path) == protocol
    else: pre.write(path, protocol)
    pre.write(POS_OUT/"步骤3b冻结实验协议.json", protocol)
    initial_encode(data)
    selected_e = engine.select("E", 100000)
    selected_h = engine.select("H265", selected_e["file_bytes"])
    pre.write(POS_OUT/"新视频完整文件选择.json", {"E": selected_e, "H265": selected_h})
    network = POS_RUN/"E/network"; network.mkdir(parents=True, exist_ok=True)
    if not (network/"end_to_end.json").exists():
        engine.deliver(Path(selected_e["file"]), network, 20)
    engine.transfer_h265(selected_h); engine.audit()
    pre.write(POS_OUT/"真实收发核验.json", {
        method: pre.read(POS_RUN/method/"network/end_to_end.json") for method in ("E","H265")})
    evaluator.RUN, evaluator.OUT, evaluator.DATA = POS_RUN, POS_OUT, POS_DATA
    evaluator.main()
    assert all(sha256(p) == value for p,value in protocol["input_sha256"].items())
    print("POSITIVE_CASE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
