"""Fetch publisher samples, verify continuous RGB / sparse GT, preserve official roles."""
from __future__ import annotations

import concurrent.futures
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import time
import urllib.request
import zipfile
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

PROJECT = Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
DATA = PROJECT / "data/phase_c_20261013"
OUT = Path(r"C:\Users\lenovo\Documents\Codex\2026-10-12\wo-d\outputs\semantic_roi_phase_c")
USER_DATA = PROJECT / "data/uvmulti_downloads"
COLORS = np.array([[0,0,0],[128,0,0],[128,128,0],[128,0,128],[0,128,128],[0,0,128],[0,128,0]], dtype=np.uint8)
ROLES = {24:"train",131:"train",91:"train",96:"development_validation",108:"official_validation"}
ARCHIVES = {24:"video25.zip",108:"video25.zip::video109.zip",131:"video131.zip",91:"video91.zip",96:"video96.zip"}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, headers=None, limit=None):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent":"semantic-video-research/1.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                data = r.read() if limit is None else r.read(limit+1)
                return data, r.status, dict(r.headers)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(min(2**attempt, 4))


class CachedRange(io.RawIOBase):
    """Read validated, aligned 2 MiB HTTP ranges with a bounded local cache."""
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.cache = OrderedDict()
        self.bytes_received = 0
        self.ranges = []
        self.block = 2*1024*1024

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if not 0 <= self.position <= self.size: raise ValueError("Invalid ZIP seek")
        return self.position

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        chunks = []
        while size:
            start = self.position//self.block*self.block
            if start not in self.cache:
                end = min(start+self.block, self.size)-1
                payload, status, headers = fetch(self.url, {"Range":f"bytes={start}-{end}", "Cache-Control":"no-cache"}, end-start+1)
                content_range = next((v for k,v in headers.items() if k.lower()=="content-range"), None)
                if status != 206 or content_range != f"bytes {start}-{end}/{self.size}" or len(payload) != end-start+1:
                    raise RuntimeError(f"Invalid partial response: {status}, {content_range}")
                self.cache[start] = payload
                self.bytes_received += len(payload)
                self.ranges.append([start,end])
                if len(self.cache)>8: self.cache.popitem(last=False)
            self.cache.move_to_end(start)
            offset = self.position-start
            take = min(size, len(self.cache[start])-offset)
            chunks.append(self.cache[start][offset:offset+take])
            self.position += take
            size -= take
        return b"".join(chunks)


class SliceRange(io.RawIOBase):
    def __init__(self,parent,start,size):self.parent,self.start,self.size,self.position=parent,start,size,0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if not 0<=self.position<=self.size:raise ValueError("Invalid nested ZIP seek")
        return self.position
    def read(self,size=-1):
        size=self.size-self.position if size<0 else min(size,self.size-self.position)
        self.parent.seek(self.start+self.position);data=self.parent.read(size);self.position+=len(data);return data


def fetch_video(video, file):
    root = USER_DATA if (USER_DATA/f"video{video}"/"sequence").is_dir() else DATA / "raw"
    remote = CachedRange(file["download_url"], file["size"])
    records = []
    nested_source=None
    if "::" in ARCHIVES[video]:
        nested_name=ARCHIVES[video].split("::")[1]
        with zipfile.ZipFile(remote) as parent_archive: nested_info=parent_archive.getinfo(nested_name)
        if nested_info.compress_type!=zipfile.ZIP_STORED:raise ValueError("Nested ZIP not stored; explicit decompression required")
        remote.seek(nested_info.header_offset);header=remote.read(30)
        name_length,extra_length=struct.unpack_from("<HH",header,26)
        nested_start=nested_info.header_offset+30+name_length+extra_length
        archive_reader=SliceRange(remote,nested_start,nested_info.file_size)
        nested_source={"entry":nested_name,"bytes":nested_info.file_size,"expected_crc32":f"{nested_info.CRC:08x}","whole_nested_crc_checked":False,"inner_selected_entries_crc_checked":True}
    else:archive_reader=remote
    with zipfile.ZipFile(archive_reader) as archive:
        infos = [z for z in archive.infolist() if not z.is_dir()]
        inventory = [{"name":z.filename,"bytes":z.file_size,"compressed_bytes":z.compress_size,"crc32":f"{z.CRC:08x}"} for z in infos]
        save(DATA / "indexes" / f"video{video}.json", inventory)
        selected = [z for z in infos if re.fullmatch(rf"video{video}/(sequence|images|mask)/\d+\.(jpg|png)",z.filename) or z.filename==f"video{video}/time.txt"]
        selected.sort(key=lambda z:z.header_offset)
        print(f"DOWNLOAD video{video}: {len(selected)} RGB/GT/source entries", flush=True)
        for i,info in enumerate(selected):
            target = (root / info.filename).resolve()
            if not target.is_relative_to(root.resolve()): raise ValueError("ZIP path traversal")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                payload = target.read_bytes()
                if len(payload)!=info.file_size or zlib.crc32(payload)!=info.CRC: raise ValueError(f"Cached entry mismatch: {target}")
            else:
                payload = archive.read(info)
                target.write_bytes(payload)
            records.append({"entry":info.filename,"bytes":len(payload),"crc32_verified":True,"sha256":digest(payload)})
            if i%100==0: print(f"DOWNLOADED video{video}: {i+1}/{len(selected)}; received {remote.bytes_received/2**20:.1f} MiB",flush=True)
    save(DATA / "downloads" / f"video{video}.json",{"url":file["download_url"],"local_data_root":str(root),"publisher_archive_name":file["name"],"nested_source":nested_source,"canonical_video_directory":f"video{video}","archive_published_md5":file["supplied_md5"],"full_archive_md5_verified":False,"entry_checks_verified":True,"http_bytes_received":remote.bytes_received,"ranges":remote.ranges,"records":records})
    print(f"DOWNLOAD_OK video{video}: {remote.bytes_received/2**20:.1f} MiB",flush=True)


def inspect_video(video, role):
    base = USER_DATA/f"video{video}" if (USER_DATA/f"video{video}"/"sequence").is_dir() else DATA / "raw" / f"video{video}"
    images = {int(p.stem):p for p in (base/"sequence").glob("*.jpg")}
    ids = sorted(images)
    if not ids or ids != list(range(ids[0],ids[-1]+1)): raise ValueError(f"Non-continuous source video{video}")
    masks = sorted((base/"mask").glob("*.png"))
    mask_dir = DATA / "fish_masks" / f"video{video}"
    mask_dir.mkdir(parents=True, exist_ok=True)
    labels, areas, components, same_rgb, annotated_zero, hashes = [], [], [], 0, 0, []
    for p in masks:
        if not p.stem.isdigit(): continue
        idx = int(p.stem)
        matched = base / "images" / f"{idx:05d}.jpg"
        if idx not in images or not matched.exists(): raise ValueError(f"Missing RGB for GT: {p}")
        source_rgb = np.asarray(Image.open(images[idx]).convert("RGB"))
        annotated_rgb = np.asarray(Image.open(matched).convert("RGB"))
        if source_rgb.shape != annotated_rgb.shape or not np.array_equal(source_rgb,annotated_rgb):
            raise ValueError(f"Annotated RGB is not identical to sequence RGB: video{video}/{idx}")
        gt = np.asarray(Image.open(p).convert("RGB"))
        if gt.shape != source_rgb.shape: raise ValueError("Mask geometry mismatch")
        known = np.zeros(gt.shape[:2],bool)
        for color in COLORS: known |= np.all(gt==color,axis=2)
        if not known.all(): raise ValueError(f"Unknown/void mask color requires explicit handling: {p}")
        fish = np.all(gt==[0,0,128],axis=2)
        n = int(fish.sum())
        binary_path = mask_dir / f"{idx:05d}.png"
        Image.fromarray(fish.astype(np.uint8)).save(binary_path)
        cc, count = ndimage.label(fish,np.ones((3,3),bool))
        cc_areas = np.bincount(cc.ravel())[1:].astype(int).tolist()
        components.extend(cc_areas)
        areas.append(n/fish.size)
        annotated_zero += n==0
        same_rgb += 1
        labels.append({"frame_index":idx,"rgb":str(images[idx]),"annotated_rgb":str(matched),"published_mask":str(p),"fish_mask":str(binary_path),"gt_available":True,"fish_pixels":n,"fish_fraction":n/fish.size,"component_areas":cc_areas,"rgb_sha256":digest(source_rgb.tobytes()),"mask_sha256":digest(p.read_bytes()),"previous_frame_index":idx-1 if idx>ids[0] else None,"previous_rgb":str(images[idx-1]) if idx>ids[0] else None,"target_and_previous_index_gap":1 if idx>ids[0] else None})
    if not labels: raise ValueError(f"No valid segmentation pairs video{video}")
    for idx in ids:
        rgb = np.asarray(Image.open(images[idx]).convert("RGB"))
        hashes.append(digest(rgb.tobytes()))
        if rgb.shape != (720,1280,3): raise ValueError("Unexpected source dimensions")
    time_text = (base/"time.txt").read_text("utf-8") if (base/"time.txt").exists() else None
    source_path = re.search(r"原始视频路径\s*[:：]\s*(.+)",time_text) if time_text else None
    start_time = re.search(r"提取起始时间\s*[:：]\s*([\d.]+)\s*秒",time_text) if time_text else None
    end_time = re.search(r"提取终止时间\s*[:：]\s*([\d.]+)\s*秒",time_text) if time_text else None
    interval = float(end_time[1])-float(start_time[1]) if start_time and end_time else None
    inferred_fps = len(ids)/interval if interval and interval>0 else None
    splits=json.loads((DATA/"protocol.json").read_text("utf-8"))["official_splits"]
    official_split=next(s for s,values in splits.items() if f"video{video}" in values)
    summary = {"video":f"video{video}","publisher_archive_name":ARCHIVES[video],"official_split":official_split,"experiment_role":role,"frames":len(ids),"first_index":ids[0],"last_index":ids[-1],"frame_index_gaps":[],"paper_declared_preprocessed_fps":25,"source_time_inferred_fps":inferred_fps,"evaluation_fps":inferred_fps or 25,"fps_basis":"source time.txt duration and extracted count" if inferred_fps else "paper-declared 25 fps, not independently verified for this clip","fps_metadata_disagrees_with_paper":inferred_fps is not None and abs(inferred_fps-25)>0.1,"annotated_frames":len(labels),"annotated_frames_with_fish":len(labels)-annotated_zero,"annotated_frames_without_fish":annotated_zero,"unannotated_frames":len(ids)-len(labels),"all_annotated_images_equal_sequence_pixels":same_rgb==len(labels),"fish_fraction_quantiles":np.quantile(areas,[0,.25,.5,.75,1]).tolist(),"fish_components":len(components),"small_component_definition":"8-connected fish components with area <=1024 pixels at 1280x720; components are not necessarily fish instances","small_components":sum(a<=1024 for a in components),"component_area_quantiles":np.quantile(components,[0,.25,.5,.75,1]).tolist() if components else None,"source_time_txt":time_text,"source_recording_path":source_path[1].strip() if source_path else None,"source_start_seconds":float(start_time[1]) if start_time else None,"source_end_seconds":float(end_time[1]) if end_time else None,"capture_group_verified":source_path is not None,"capture_group_reason":"source recording path published in time.txt" if source_path else "sample archive lacks per-clip source recording identity; independence not fully verified","source_frame_rgb_hashes":hashes,"labels":labels,"sequence_dir":str(base/"sequence")}
    save(DATA / "manifests" / f"video{video}.json",summary)
    return summary


def contact_sheet(videos):
    canvas = Image.new("RGB",(len(videos)*480,3*295),"white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc",18)
    for column,v in enumerate(videos):
        selected = [v["labels"][i] for i in sorted({0,len(v["labels"])//2,len(v["labels"])-1})]
        for row,entry in enumerate(selected):
            rgb = np.asarray(Image.open(entry["rgb"]).convert("RGB")).copy()
            mask = np.asarray(Image.open(entry["fish_mask"]))>0
            rgb[mask] = (rgb[mask]*.45+np.array([40,255,100])*.55).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb).resize((480,270)),(column*480,row*295+25))
            draw.text((column*480+4,row*295+3),f'{v["video"]} / {entry["frame_index"]} / fish {entry["fish_fraction"]*100:.3f}%',font=font,fill="black")
    OUT.mkdir(parents=True,exist_ok=True)
    canvas.save(OUT / "RGB与鱼类GT核对.png")


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    payload,_,_ = fetch("https://api.figshare.com/v2/articles/30105151")
    meta = json.loads(payload)
    save(DATA / "figshare_metadata.json",meta)
    files = {x["name"]:x for x in meta["files"]}
    list_file = files["video_list.txt"]
    split_raw,_,_ = fetch(list_file["download_url"])
    if hashlib.md5(split_raw).hexdigest()!=list_file["supplied_md5"]: raise ValueError("Official split checksum mismatch")
    (DATA/"video_list.txt").write_bytes(split_raw)
    splits = {}; current = None
    for line in split_raw.decode().splitlines():
        line = line.strip()
        if line in ("train","val","test"): current=line; splits[current]=[]
        elif line: splits[current].append(line)
    assert [len(splits[x]) for x in ("train","val","test")]==[70,10,20]
    assert len(set(sum(splits.values(),[])))==100
    for video,role in ROLES.items(): assert f"video{video}" in splits["val" if role=="official_validation" else "train"]
    protocol = {"created_before_training":True,"source":"https://figshare.com/articles/dataset/30105151","official_repository":"https://github.com/pingchuan/UVMulti","official_splits":splits,"experiment_roles":{f"video{k}":v for k,v in ROLES.items()},"development_validation_is_official_validation":False,"final_test_used":False,"reserved_available_test":"video7","selection_basis":"final roles fixed before any model update or semantic scoring: official train24/131/91 train; official train96 whole-video development holdout; actual nested108 official validation","archive_name_correction":"video25.zip contains video24/ and video109.zip, whose actual directory is video108/ (official val), plus nested video91.zip","official_validation_status":"actual video108 found inside the misleadingly named nested video109.zip","capture_level_independence_verified":False,"data_license":meta["license"],"training_semantics":"GT is loss supervision only; encoder inference gets no ROI/mask input","unannotated_target_policy":"ordinary rate-distortion only; missing GT is never synthesized as an all-background label"}
    if (DATA/"protocol.json").exists():
        previous=(DATA/"protocol.json").read_bytes()
        (DATA/f"protocol_before_archive_correction_{digest(previous)[:10]}.json").write_bytes(previous)
    save(DATA / "protocol.json",protocol)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_video,v,files[ARCHIVES[v].split("::")[0]]) for v in ROLES]
        for f in concurrent.futures.as_completed(futures): f.result()
    videos = [inspect_video(v,role) for v,role in ROLES.items()]
    train_hashes = set(h for v in videos if v["experiment_role"]=="train" for h in v["source_frame_rgb_hashes"])
    dev_hashes = set(h for v in videos if v["experiment_role"]!="train" for h in v["source_frame_rgb_hashes"])
    duplicate = train_hashes & dev_hashes
    if duplicate: raise ValueError("Exact train/development RGB duplicate")
    manifest = {"protocol":protocol,"protocol_sha256":digest((DATA/"protocol.json").read_bytes()),"train_development_exact_rgb_duplicates":len(duplicate),"videos":videos,"data_root":str(DATA)}
    save(DATA / "manifest.json",manifest)
    save(OUT / "数据核验统计.json",{"protocol":protocol,"videos":[{k:v for k,v in s.items() if k not in ("labels","source_frame_rgb_hashes")} for s in videos],"train_development_exact_rgb_duplicates":0})
    contact_sheet(videos)
    print("DATA_PREPARATION_OK",flush=True)
    for v in videos: print({k:v[k] for k in ("video","experiment_role","frames","annotated_frames","annotated_frames_with_fish","small_components")},flush=True)


if __name__=="__main__": main()
