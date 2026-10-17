"""Resolution ablation: no weights changed, unchanged validation split, native GT grid."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from common import ROOT, configure, save_json, sha256, torch
from semantic_teacher import LEGACY, load_teacher, predict, confusion, class_metrics

configure()
model,state = load_teacher()
manifest_file = LEGACY / "data/SUIM/manifest.json"
if sha256(manifest_file)!=state["manifest_sha256"]: raise ValueError("SUIM manifest changed")
data=json.loads(manifest_file.read_text("utf-8"))
data_root=manifest_file.parent / data.get("root",".")
output=ROOT / "runs/phase_b_resolution_validation"
output.mkdir(exist_ok=False)
results=[]
with torch.inference_mode():
    for size in [256,512]:
        matrix=np.zeros((8,8),dtype=np.int64)
        start=time.perf_counter()
        for row in data["splits"]["validation"]:
            prediction,_,_=predict(model,data_root / row["image"],size)
            target=np.asarray(Image.open(data_root / row["mask_index"]))
            matrix+=confusion(prediction,target)
        metrics=class_metrics(matrix)
        result={"input_size":size,"scoring_grid":"original image/mask resolution; common GT pixels for both sizes",
                "images":len(data["splits"]["validation"]),"training_performed":False,"seconds":time.perf_counter()-start,**metrics}
        results.append(result)
        save_json(output / f"size{size}.json",result)
        print(f"SUIM validation native-grid: input{size} mIoU {metrics['mean_iou_present_union']:.6f}, FV {metrics['per_class'][6]['iou']:.6f}, HD {metrics['per_class'][1]['iou']:.6f}",flush=True)
save_json(output / "summary.json",{"split":"validation","manifest_sha256":sha256(manifest_file),"checkpoint_training_size":state["size"],
         "protocol":"Diagnostic comparison at a common original-resolution GT grid; not directly comparable to historical padded 256-grid mIoU", "results":results})
