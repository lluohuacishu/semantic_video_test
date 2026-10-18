"""Compare four accepted received reconstructions on identical source regions."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
from b2_common import HERE, Journal, git, read, require, sha, write
from evaluate_b2 import rgb, mask, score, font, panel, movie


def rect_mask(patches, index, height=720, width=1280):
    result = np.zeros((height,width), dtype=bool)
    for patch in patches:
        if patch["start"] <= index < patch["start"]+patch["frames"]:
            x,y,w,h = patch["box"]
            result[y:y+h,x:x+w] = True
    return result


def edges(error, patches, index):
    result = []
    height,width = error.shape[:2]
    for patch in patches:
        if not patch["start"] <= index < patch["start"]+patch["frames"]:
            continue
        x,y,w,h = patch["box"]
        if x > 0: result.append(np.abs(error[y:y+h,x]-error[y:y+h,x-1]).ravel())
        if x+w < width: result.append(np.abs(error[y:y+h,x+w-1]-error[y:y+h,x+w]).ravel())
        if y > 0: result.append(np.abs(error[y,x:x+w]-error[y-1,x:x+w]).ravel())
        if y+h < height: result.append(np.abs(error[y+h-1,x:x+w]-error[y+h,x:x+w]).ravel())
    return float(np.concatenate(result).mean(dtype=np.float64)) if result else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--wait-for-network", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.wait_for_network:
        deadline = time.monotonic()+14400
        while not (run/"network/summary.json").exists():
            require(not (run/"network/failure.json").exists(), "Network failed; do not evaluate partial runs")
            require(time.monotonic()<deadline, "Network wait expired")
            time.sleep(1)
    network = read(run/"network/summary.json")
    require(network["status"] == "B3_development_controls_engineering_passed", "Controls not accepted")
    saved = read(run/"glc_search/config.json")
    config, baseline = saved["b3"], saved["baseline"]
    b2 = Path(config["b2_run"])
    accepted = read(b2/"accepted_network.json")
    summary_file = b2/accepted["directory"]/"summary.json"
    require(sha(summary_file) == accepted["summary_sha256"], "Accepted B2 index changed")
    b2_summary = read(summary_file)
    auto = Path(b2_summary["result_directories"]["40"])/"receiver"
    glc = run/"network/glc/rate_40kbps/receiver"
    geometry = run/"network/geometry/rate_40kbps/receiver"
    directories = {"GLC_base": (auto/"base_decoded/frames", "im{:05d}.png", 1), "auto_ROI": (auto/"rgb", "{:05d}.png", 0),
                   "full_GLC_control": (glc/"decoded/frames", "im{:05d}.png", 1), "fixed_geometry": (geometry/"rgb", "{:05d}.png", 0)}
    source = b2/"sender/private_frames"
    audit = read(b2/"sender/source_audit.json")
    patches = {"auto": read(auto/"received_manifest.json")["patches"], "geometry": read(geometry/"received_manifest.json")["patches"]}
    hashes = {"GLC_base": read(auto/"decode.json")["base_rgb_sha256"], "auto_ROI": read(auto/"decode.json")["enhanced_rgb_sha256"],
              "full_GLC_control": [r["rgb_sha256"] for r in read(glc/"decoded/decode.json")["frames"]],
              "fixed_geometry": read(geometry/"decode.json")["enhanced_rgb_sha256"]}
    output = run/"evaluation"
    output.mkdir()
    journal = Journal(output)
    invocation = journal.begin([sys.executable,*sys.argv], code_commit=git("rev-parse","HEAD"), evaluator_sha256=sha(__file__))
    started = time.perf_counter()
    aggregates, temporal, edge_values, previous = {}, {}, {}, {}
    rows, coverage = [], []
    x1,y1,x2,y2 = config["fixed_diagnostic_crop_xyxy"]
    first,last = config["slow_view_source_frame_range"]
    for method in directories:
        aggregates[method] = {}
        temporal[method] = {key: [] for key in ("all","window_boundaries","non_boundaries","fixed_stripe_crop")}
        edge_values[method] = {key: [] for key in patches}
    for index in range(250):
        original = rgb(source/f"{index:05d}.png")
        require(hashlib.sha256(original.tobytes()).hexdigest() == audit["mapping"][index]["rgb_sha256"], "Evaluation source changed")
        rects = {name: rect_mask(plan,index) for name,plan in patches.items()}
        focus = np.zeros((720,1280),dtype=bool)
        if first <= index <= last: focus[y1:y2,x1:x2] = True
        masks = {"full": np.ones((720,1280),dtype=bool), "fixed_stripe_crop": focus,
                 "inside_auto_ROI": rects["auto"], "outside_auto_ROI": ~rects["auto"],
                 "inside_fixed_geometry": rects["geometry"], "outside_fixed_geometry": ~rects["geometry"]}
        for target in ("fish","diver"):
            for kind in ("raw","kept"):
                masks[f"{target}_{kind}_prediction"] = mask(b2/"sender/private_masks"/target/kind/f"{index:05d}.png")
        cov = {"frame": index}
        for kind in ("raw","kept"):
            m = masks[f"fish_{kind}_prediction"]
            cov[f"fish_{kind}_pixels"] = int(m.sum())
            for plan in rects: cov[f"{plan}_fish_{kind}_covered"] = int((m & rects[plan]).sum())
        coverage.append(cov)
        row = {"frame": index, "window_boundary": index>0 and index%32==0}
        for method,(directory,pattern,offset) in directories.items():
            image = rgb(directory/pattern.format(index+offset))
            require(hashlib.sha256(image.tobytes()).hexdigest() == hashes[method][index], "Accepted RGB changed")
            error = image.astype(np.float32)-original.astype(np.float32)
            squared = np.square(error)
            for name,selected in masks.items():
                total, count = float(squared[selected].sum(dtype=np.float64)), int(selected.sum())*3
                totals = aggregates[method].setdefault(name,[0.0,0])
                totals[0] += total
                totals[1] += count
                row[f"{method}_{name}_psnr_db"] = score(total,count)
            if method in previous:
                change = np.abs(error-previous[method])
                value = float(change.mean(dtype=np.float64))
                temporal[method]["all"].append(value)
                temporal[method]["window_boundaries" if row["window_boundary"] else "non_boundaries"].append(value)
                row[f"{method}_error_change_mae"] = value
                if first < index <= last: temporal[method]["fixed_stripe_crop"].append(float(change[y1:y2,x1:x2].mean(dtype=np.float64)))
            previous[method] = error
            for name,plan in patches.items():
                value = edges(error,plan,index)
                if value is not None: edge_values[method][name].append(value)
                row[f"{method}_on_{name}_edges_mae"] = value
        rows.append(row)
        if (index+1)%50 == 0: print(f"B3 metrics {index+1}/250",flush=True)
    methods = {}
    for method in directories:
        methods[method] = {"rgb_psnr_db": {name:score(*v) for name,v in aggregates[method].items()},
                           "sample_counts_RGB_values": {name:v[1] for name,v in aggregates[method].items()},
                           "error_change_mae": {name:float(np.mean(v)) if v else None for name,v in temporal[method].items()},
                           "edge_error_difference_mae": {name:float(np.mean(v)) if v else None for name,v in edge_values[method].items()}}
    regressions = {}
    for comparison in ("GLC_base","full_GLC_control","fixed_geometry"):
        for region in ("full","fish_raw_prediction","fixed_stripe_crop"):
            key = f"auto_ROI_lower_PSNR_than_{comparison}_{region}"
            regressions[key] = [r["frame"] for r in rows if r[f"auto_ROI_{region}_psnr_db"] is not None and r[f"{comparison}_{region}_psnr_db"] is not None and r[f"auto_ROI_{region}_psnr_db"] < r[f"{comparison}_{region}_psnr_db"]]
    selections = {k:read(run/f"{k}_search/selection.json") for k in ("glc","geometry")}
    byte_sizes = {"GLC_base": config["base_bytes"], "auto_ROI": config["target_file_bytes"],
                  "full_GLC_control": selections["glc"]["selected"]["file_bytes"], "fixed_geometry": selections["geometry"]["selected"]["file_bytes"]}
    result = {"role": "previously_used_video91_development_clip", "frames":250,"methods":methods, "file_bytes":byte_sizes,
              "budget_selections": selections, "regression_frame_indices": regressions, "human_review": None,
              "crop_xyxy":config["fixed_diagnostic_crop_xyxy"], "crop_frames":[first,last],
              "limitations":["Source predictions are not GT and are reused from the automatic ROI selector", "Temporal/edge errors are diagnostics, not validated perceptual flicker", "No diver prediction pixels; no diver score", "Budget tolerances applied without rounding", "No semantic-net-benefit conclusion without independent evidence and human review"]}
    write(output/"metrics.json",result)
    write(output/"per_frame_metrics.json",rows)
    write(output/"prediction_coverage.json",{"frames":coverage,"summary":{k:sum(r[k] for r in coverage) for k in coverage[0] if k!="frame"}})
    labels = {"GLC_base":"GLC基础q0", "auto_ROI":"自动语义ROI", "full_GLC_control":f"全画面GLC q{selections['glc']['selected']['q']}", "fixed_geometry":"固定中央带"}
    def frame(index,crop=False):
        names = ["source",*directories]
        size = (384,216) if crop else (640,360)
        canvas = Image.new("RGB",(1920,252 if crop else 792),"#101e2a")
        for i,name in enumerate(names):
            path = source/f"{index:05d}.png" if name=="source" else directories[name][0]/directories[name][1].format(index+directories[name][2])
            with Image.open(path) as im:
                im=im.convert("RGB")
                if crop: im=im.crop((x1,y1,x2,y2))
                im=im.resize(size,Image.Resampling.NEAREST if crop else Image.Resampling.LANCZOS)
            label="源图" if name=="source" else labels[name]
            title=f"{label} · 帧{index}" if crop or name=="source" else f"{label} · {byte_sizes[name]:,} B · 帧{index}"
            position=(i*384,0) if crop else ((i%3)*640,(i//3)*396)
            canvas.paste(panel(im,title,font(17 if crop else 19)),position)
        if not crop:
            text="B3开发片预算控制\n所有重建来自40 kbps实际收包\n指标使用接收PNG，视频为观看副本\n预算未匹配项不判等预算胜出\n用户观看：待评；预测掩膜不是GT"
            ImageDraw.Draw(canvas).multiline_text((1300,440),text,font=font(23),fill="white",spacing=16)
        return canvas
    ffmpeg=baseline["runtime_dependencies"]["ffmpeg"]
    views=[movie(ffmpeg,output/"video91_B3_五路同步对比.mp4",list(range(250)),15,lambda i:frame(i),journal,"Four received methods plus source, normal speed, budget differences labeled"),
           movie(ffmpeg,output/"video91_B3_固定鱼纹五路半速.mp4",list(range(first,last+1)),7.5,lambda i:frame(i,True),journal,"Previously frozen crop and frame range, half speed")]
    frame(224).save(output/"固定边界帧224.jpg",quality=95)
    crop_canvas=Image.new("RGB",(1920,504))
    for i,index in enumerate((223,239)): crop_canvas.paste(frame(index,True),(0,i*252))
    crop_canvas.save(output/"固定鱼纹帧223_239.jpg",quality=95)
    write(output/"viewing.json",{"videos":views,"human_review":"pending; user has not watched B2 yet", "static_frames":[224,223,239]})
    write(output/"completion.json",{"status":"B3_development_diagnostics_complete","script_sha256":sha(__file__),"network_summary_sha256":sha(run/"network/summary.json")})
    journal.finish(invocation,0,time.perf_counter()-started)
    print("B3 received-RGB metrics and viewing videos complete",flush=True)


if __name__ == "__main__":
    main()
