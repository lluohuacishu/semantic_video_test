"""Keep supervised fish accuracy and source-prediction consistency separate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from common import ROOT, configure, save_json, sha256, environment, torch
from semantic_teacher import CHECKPOINT, EXPECTED_SHA, LABELS, load_teacher, predict, confusion, class_metrics, overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, help="Diagnostic input resolution; default is checkpoint training size")
    args = parser.parse_args()
    configure()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(args.manifest).read_text("utf-8"))
    model, state = load_teacher()
    inference_size = args.size or state["size"]
    if inference_size<32 or inference_size%32: raise ValueError("Teacher size must be a positive multiple of 32")
    all_results = []
    with torch.inference_mode():
        for sequence in manifest["sequences"]:
            seq_dir = output / sequence["name"]
            seq_dir.mkdir()
            refs = sorted(Path(sequence["reference_dir"]).glob("*.png"))
            if len(refs) != sequence["frame_count"]: raise ValueError("Reference count mismatch")
            gt_paths = sorted(Path(sequence["ground_truth_dir"]).glob("*.png")) if sequence.get("ground_truth_dir") else None
            if gt_paths and len(gt_paths) != len(refs): raise ValueError("GT count mismatch")
            valid = np.ones((sequence["height"],sequence["width"]), dtype=bool)
            for x0,y0,x1,y1 in sequence.get("exclude_rectangles",[]): valid[y0:y1,x0:x1] = False
            ref_predictions, teacher_rois, gt_masks = [], [], []
            streams = [{"name":"reference","frames":sequence["reference_dir"],"file":None}, *sequence["streams"]]
            results = []
            for stream in streams:
                stream_dir = seq_dir / stream["name"]
                (stream_dir / "prediction_index").mkdir(parents=True)
                (stream_dir / "overlays").mkdir()
                if stream["name"] == "reference": (stream_dir / "roi_confidence_070").mkdir()
                frames = sorted(Path(stream["frames"]).glob("*.png"))
                if len(frames) != len(refs): raise ValueError("Decoded/reference count differs")
                cm_full = np.zeros((8,8),dtype=np.int64)
                cm_valid = np.zeros((8,8),dtype=np.int64)
                cm_gt = np.zeros((2,2),dtype=np.int64)
                frame_results = []
                change_disagreements, gt_change_errors = [], []
                roi_squared_error, roi_samples = 0.,0
                gt_squared_error, gt_samples = 0.,0
                prev_pred = None
                preview_indices = set(np.linspace(0,len(frames)-1,min(5,len(frames)),dtype=int).tolist())
                start = time.perf_counter()
                for idx, path in enumerate(frames):
                    pred, confidence, geometry = predict(model,path,inference_size)
                    if pred.shape != valid.shape: raise ValueError("Decoded spatial shape differs")
                    if stream["name"] == "reference":
                        ref_predictions.append(pred)
                        roi = np.isin(pred,[1,6]) & (confidence>=0.7) & valid
                        teacher_rois.append(roi)
                        Image.fromarray(roi.astype(np.uint8)*255).save(stream_dir / "roi_confidence_070" / f"{idx:05d}.png")
                        if gt_paths:
                            gt = np.asarray(Image.open(gt_paths[idx]))
                            if gt.shape != pred.shape or not np.isin(gt,[0,1]).all(): raise ValueError("Invalid binary fish GT")
                            gt_masks.append(gt.astype(bool))
                    cm_full += confusion(pred,ref_predictions[idx])
                    per_frame_agreement = class_metrics(confusion(pred,ref_predictions[idx],valid=valid))
                    cm_valid += np.asarray(per_frame_agreement["confusion_matrix"])
                    rgb_ref = np.asarray(Image.open(refs[idx]).convert("RGB"),dtype=np.float64)
                    rgb_dec = np.asarray(Image.open(path).convert("RGB"),dtype=np.float64)
                    errors = (rgb_ref-rgb_dec)**2
                    roi_squared_error += float(errors[teacher_rois[idx]].sum())
                    roi_samples += int(teacher_rois[idx].sum())*3
                    item = {"frame":idx,"source_prediction_agreement_hd_iou":per_frame_agreement["per_class"][1]["iou"],
                            "source_prediction_agreement_fv_iou":per_frame_agreement["per_class"][6]["iou"],
                            "predicted_hd_pixels":int((pred==1).sum()),"predicted_fv_pixels":int((pred==6).sum()),
                            "mean_confidence":float(confidence[valid].mean())}
                    if gt_masks:
                        gt = gt_masks[idx]
                        matrix = confusion((pred==6).astype(np.uint8),gt.astype(np.uint8),classes=2)
                        cm_gt += matrix
                        per_gt = class_metrics(matrix,("non-fish","fish"))["per_class"][1]
                        item.update({"fish_gt_iou":per_gt["iou"],"fish_gt_f1":per_gt["f1"],"ground_truth_fish_pixels":int(gt.sum())})
                        gt_squared_error += float(errors[gt].sum())
                        gt_samples += int(gt.sum())*3
                        if prev_pred is not None:
                            gt_change_errors.append(float(np.logical_xor((pred==6)!=(prev_pred==6), gt!=gt_masks[idx-1]).mean()))
                    if prev_pred is not None:
                        change_disagreements.append(float(np.logical_xor(pred!=prev_pred,ref_predictions[idx]!=ref_predictions[idx-1])[valid].mean()))
                    prev_pred = pred
                    Image.fromarray(pred).save(stream_dir / "prediction_index" / f"{idx:05d}.png")
                    if idx in preview_indices:
                        overlay(path,pred).resize((640,360),Image.Resampling.LANCZOS).save(stream_dir / "overlays" / f"{idx:05d}.png")
                    frame_results.append(item)
                result = {"name":stream["name"],"frames":len(frames),"input_frames_dir":str(Path(stream["frames"]).resolve()),
                          "complete_file_bytes":Path(stream["file"]).stat().st_size if stream["file"] else None,
                          "complete_file_sha256":sha256(stream["file"]) if stream["file"] else None,
                          "source_prediction_agreement_full_frame":class_metrics(cm_full),
                          "source_prediction_agreement_exclusions":class_metrics(cm_valid),
                          "teacher_roi_rgb_psnr_db":float(10*np.log10(255**2/(roi_squared_error/roi_samples))) if roi_samples and roi_squared_error else None,
                          "teacher_roi_valid_pixels":roi_samples//3,
                          "mean_change_map_disagreement_vs_source_prediction":float(np.mean(change_disagreements)) if change_disagreements else None,
                          "seconds":time.perf_counter()-start,"per_frame":frame_results}
                if gt_paths:
                    binary = class_metrics(cm_gt,("non-fish","fish"))
                    result.update({"fish_ground_truth":binary,"fish_gt_iou":binary["per_class"][1]["iou"],"fish_gt_f1":binary["per_class"][1]["f1"],
                                   "ground_truth_fish_rgb_psnr_db":float(10*np.log10(255**2/(gt_squared_error/gt_samples))) if gt_samples and gt_squared_error else None,
                                   "mean_fish_change_map_error_vs_gt":float(np.mean(gt_change_errors)) if gt_change_errors else None})
                fps_a,fps_b = map(int,sequence["fps"].split("/"))
                duration = len(frames)*fps_b/fps_a
                result["duration_seconds"] = duration
                result["video_kbps"] = result["complete_file_bytes"]*8/duration/1000 if result["complete_file_bytes"] else None
                result["theoretical_transfer_seconds_at_20kbps"] = result["complete_file_bytes"]*8/20000 if result["complete_file_bytes"] else None
                save_json(stream_dir / "result.json",result)
                results.append(result)
                print(f"{sequence['name']} / {stream['name']}: frames {len(frames)}, GT fish IoU {result.get('fish_gt_iou')}, HD source consistency {result['source_prediction_agreement_exclusions']['per_class'][1]['iou']}",flush=True)
            all_results.append({"name":sequence["name"],"ground_truth_kind":"fish_binary" if gt_paths else None,
                                "evaluation_role":sequence.get("role","engineering_diagnostic"),"exclude_rectangles":sequence.get("exclude_rectangles",[]),"streams":results})
            save_json(seq_dir / "summary.json",all_results[-1])
    summary = {"teacher_checkpoint":str(CHECKPOINT),"teacher_sha256":EXPECTED_SHA,"teacher_frozen":True,"training_performed":False,
               "input_size":inference_size,"model_training_size":state["size"],"preprocessing":state["preprocessing"],"prediction_grid":"crop padding from logits, bilinear resize logits to source RGB dimensions, argmax",
               "classes":state["classes"],"manifest_sha256":sha256(args.manifest),"environment":environment(),"sequences":all_results,
               "limitations":["Source prediction agreement is not ground-truth accuracy","Fish GT only, no eight-class video mIoU or true diver accuracy","Change-map disagreement is unwarped and includes motion; not a complete temporal stability measure","Rates not yet exactly matched","Evaluation sequences not used to update any weights"]}
    save_json(output / "summary.json",summary)


if __name__ == "__main__": main()
