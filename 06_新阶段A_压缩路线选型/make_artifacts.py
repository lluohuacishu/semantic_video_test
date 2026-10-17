"""Export small, auditable tables and all annotated fish crops; no raw video copies."""
from pathlib import Path
import argparse
import csv
import json
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--project", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
fig = a.output / "figures"
fig.mkdir(exist_ok=True)
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save(name, value):
    (a.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def components(mask):
    remaining = {tuple(x) for x in np.argwhere(mask)}
    parts = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack = [first]
        coords = []
        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nxt = y + dy, x + dx
                    if nxt in remaining:
                        remaining.remove(nxt)
                        stack.append(nxt)
        points = np.array(coords)
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0) + 1
        parts.append({"bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)], "mask_pixels": len(coords)})
    return sorted(parts, key=lambda x: (x["bbox_xyxy"][1], x["bbox_xyxy"][0]))

metrics = read(a.run / "metrics.json")
save("metrics.json", metrics)
save("input_manifest.json", read(a.run / "input_manifest.json"))
save("selection_search.json", read(a.run / "selections.json"))
save("protocol_files.json", read(a.run / "protocol_files.json"))
save("execution_commands.json", {str(p.relative_to(a.run)).replace("\\", "/"):read(p) for p in sorted(a.run.rglob("*.log.json"))})
ffprobe = Path(read(a.run / "protocol_files.json")["ffmpeg"]).with_name("ffprobe.exe")
stream_checks = {}
for q in [0,63]:
    for codec in ["h265", "av1"]:
        name = f"{codec}_match_q{q:02d}"
        info = json.loads(subprocess.check_output([str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,color_range,color_space,color_transfer,color_primaries:frame=key_frame,pict_type", "-of", "json", str(a.run/name/"selected.mp4")]))
        assert len(info["frames"]) == 50
        stream_checks[name] = {"streams": info["streams"], "decoded_frames": len(info["frames"]), "key_frame_indices_0_based": [i for i,f in enumerate(info["frames"]) if f["key_frame"]]}
save("standard_stream_checks.json", stream_checks)
rows = [{k:v for k,v in r.items() if k != "per_frame"} for r in metrics["results"]]
fields = ["codec", "glc_budget_q", "file_bytes", "relative_budget_error", "global_rgb_psnr_db", "gt_fish_psnr_db", "gt_background_psnr_db", "rgb_ms_ssim_mean", "gt_fish_pixels", "encode_process_seconds", "decode_process_seconds", "file_sha256"]
with (a.output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields + ["payload_seconds_20kbps", "payload_seconds_30kbps", "payload_seconds_40kbps"])
    w.writeheader()
    for r in rows:
        w.writerow({**{k:r[k] for k in fields}, **{f"payload_seconds_{k}kbps":r["payload_only_seconds"][str(k)] for k in [20,30,40]}})

verification = {}
for q in [0,63]:
    run = a.run / f"glc_q{q:02d}"
    enc, dec = read(run/"sender/encode.json"), read(run/"receiver/decode.json")
    verification[str(q)] = {
        **read(run/"verification.json"),
        "header": enc["header"],
        "container_overhead_bytes": enc["container_overhead_bytes"],
        "encode_frame_processing_seconds": enc["frame_processing_seconds"],
        "decode_frame_processing_seconds": dec["frame_processing_seconds"],
        "encode_peak_allocated_mib": enc["peak_allocated_mib"],
        "decode_peak_allocated_mib": dec["peak_allocated_mib"],
        "source_media_reads": dec["source_media_reads"],
        "all_frame_checksums_match": dec["all_frame_checksums_match"],
    }
save("glc_verification.json", verification)

annotations = {}
for n in [60,75,90]:
    mask = np.asarray(Image.open(a.project / "data/phase_f_external_20261015/positive_case/fish_masks" / f"{n:05d}.png")) > 0
    annotations[str(n)] = components(mask)
save("crop_manifest.json", {"selection":"all 8-connected components in all three published GT masks; none omitted", "frames":annotations})

for q in [0,63]:
    names = ["Original", f"GLC q{q}", "H.265 matched", "AV1 matched"]
    dirs = [a.run/"source", a.run/f"glc_q{q:02d}/receiver/frames", a.run/f"h265_match_q{q:02d}/receiver/frames", a.run/f"av1_match_q{q:02d}/receiver/frames"]
    overview = Image.new("RGB", (1280, 3*220+38), "white")
    od = ImageDraw.Draw(overview)
    od.text((10,8), f"All GT frames | GLC q{q} byte budget | overview only, not metric input", fill="black", font=font)
    for fi,n in enumerate([60,75,90]):
        ims = [Image.open(d/f"im{n-49:05d}.png").convert("RGB") for d in dirs]
        for col,(im,name) in enumerate(zip(ims,names)):
            y = 38 + fi*220
            od.text((col*320+8,y+2), f"{name} - frame {n}", fill="black", font=small)
            overview.paste(im.resize((320,180), Image.Resampling.LANCZOS), (col*320,y+28))
        parts = annotations[str(n)]
        sheet = Image.new("RGB", (1024, 64+len(parts)*244), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8,5), f"Frame {n} | q{q} byte budget | all GT components | 2x nearest-neighbor", fill="black", font=font)
        for col,name in enumerate(names):
            draw.text((col*256+8,33), name, fill="black", font=font)
        for idx,part in enumerate(parts):
            x0,y0,x1,y1 = part["bbox_xyxy"]
            # Identical 112x96 source-pixel crop for every method; centered on GT.
            cx,cy=(x0+x1)//2,(y0+y1)//2
            left=max(0,min(1280-112,cx-56)); top=max(0,min(720-96,cy-48))
            part.setdefault("visual_crop_xyxy", [left,top,left+112,top+96])
            for col,im in enumerate(ims):
                y = 64+idx*244
                draw.text((col*256+8,y), f"ID {idx+1}; GT {part['mask_pixels']} px", fill="black", font=small)
                crop = im.crop((left,top,left+112,top+96)).resize((224,192), Image.Resampling.NEAREST)
                sheet.paste(crop,(col*256+8,y+22))
                # The outline marks the same GT extent on every reconstruction.
                draw.rectangle((col*256+8+(x0-left)*2,y+22+(y0-top)*2,col*256+8+(x1-left)*2-1,y+22+(y1-top)*2-1), outline=(255,100,80), width=1)
        sheet.save(fig / f"fish_q{q:02d}_frame{n}.png")
        for im in ims: im.close()
    overview.save(fig / f"overview_q{q:02d}.png")
save("crop_manifest.json", {"selection":"all 8-connected components in all three published GT masks; none omitted", "frames":annotations})
print(json.dumps({"metric_rows":len(rows),"gt_components":{k:len(v) for k,v in annotations.items()},"figure_count":len(list(fig.glob('*.png')))},ensure_ascii=False))
