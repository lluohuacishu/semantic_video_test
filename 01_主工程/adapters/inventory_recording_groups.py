"""Inventory capture groups before selecting future training/validation videos."""
import json
from pathlib import Path
import subprocess
import sys
from collections import defaultdict
from common import ROOT, save_json

entries=[f"SFISHTRACK/metadata/video_{idx:03d}.json" for idx in range(1,55)]
subprocess.run([sys.executable,str(ROOT / "adapters/fetch_sfishtrack.py"),"--reuse","--entry",*entries],cwd=ROOT,check=True)
folder=ROOT / "data/sfishtrack_official/selected/SFISHTRACK/metadata"
groups=defaultdict(list)
for path in sorted(folder.glob("*.json")):
    metadata=json.loads(path.read_text("utf-8"))
    groups[metadata["Date of recording"]].append({"video_id":int(path.stem.split("_")[-1]),"local_time":metadata.get("local_time"),
                                                "camera":metadata.get("camera model"),"location":metadata.get("approximate location")})
reserved_dates={"03/06/2025","26/07/2022","09/05/2023"}
inventory={"date_groups":dict(groups),"reserved_evaluation_dates":sorted(reserved_dates),
           "candidate_dates_outside_evaluation":sorted(set(groups)-reserved_dates),
           "selection_status":"Inventory only; no training set/validation set selected or model updates performed", 
           "policy":"Prefer date/camera/site isolation, preserve original recording groups; metadata uncertainty must be reviewed before training"}
save_json(ROOT / "data/recording_groups_inventory.json",inventory)
print(json.dumps(inventory,ensure_ascii=False,indent=2))
