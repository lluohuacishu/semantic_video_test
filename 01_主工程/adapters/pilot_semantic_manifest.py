import json
from pathlib import Path
from common import ROOT, save_json

comparison = json.loads((ROOT / "runs/comparison_native/comparison.json").read_text("utf-8"))
streams = []
for row in comparison["rows"]:
    file = Path(row["complete_file"])
    if not file.is_absolute(): file = ROOT / file
    if row["codec"] == "GLC": folder = file.parent / "receiver/frames"
    elif row["codec"] == "HEVC/libx265": folder = file.parent / "rgb"
    else: folder = file.parent / "receiver/rgb"
    prefix = "glc" if row["codec"] == "GLC" else "hevc" if row["codec"] == "HEVC/libx265" else "mlvc"
    streams.append({"name":f"{prefix}_{row['setting']}","frames":str(folder),"file":str(file)})
save_json(ROOT / "configs/phase_b_pilot.json",{"sequences":[{"name":"legacy_diver_720p","role":"engineering_diagnostic_not_independent_gt", "reference_dir":str(ROOT / "runs/baseline_720p_dmci_q0/reference"),
          "width":1280,"height":720,"fps":"5/1","frame_count":50,"ground_truth_dir":None,"exclude_rectangles":[[0,0,320,144]],"streams":streams}],
          "protocol":"No weight updates; teacher agreement reported separately from accuracy. Fixed watermark exclusion chosen before predictions. Full-frame agreement retained as secondary."})
