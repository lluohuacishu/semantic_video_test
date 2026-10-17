"""Do not treat metadata spacing, invalid dates or month/day swaps as new data."""
import datetime as dt
import json
from pathlib import Path
import re
from collections import defaultdict
from common import ROOT, save_json

folder=ROOT / "data/sfishtrack_official/selected/SFISHTRACK/metadata"
groups=defaultdict(list)
invalid=[]
for path in sorted(folder.glob("*.json")):
    item=json.loads(path.read_text("utf-8"))
    raw=item["Date of recording"]
    normalized=re.sub(r"\s+","",raw)
    row={"video_id":int(path.stem.split("_")[-1]),"raw_date":raw,"local_time":item.get("local_time"),"camera":item.get("camera model"),"site":item.get("approximate location")}
    try: date=dt.datetime.strptime(normalized,"%d/%m/%Y").date().isoformat()
    except ValueError:
        invalid.append(row)
        continue
    groups[date].append(row)
reserved={"2025-06-03","2022-07-26","2023-05-09"}
candidate=set(groups)-reserved
review=[]
for date in sorted(candidate):
    parsed=dt.date.fromisoformat(date)
    if parsed.day<=12:
        swapped=dt.date(parsed.year,parsed.day,parsed.month).isoformat()
        if swapped in reserved:
            review.append({"date":date,"possible_day_month_swap_with_reserved":swapped,"video_ids":[r["video_id"] for r in groups[date]],
                           "status":"Quarantine as prospective training data until source footage/group independence verified; no assumption of a genuinely new recording day"})
result={"normalized_groups":dict(groups),"date_interpretation":"DD/MM/YYYY assumed for bookkeeping; source ambiguities retained", "invalid_date_records":invalid,
        "reserved_evaluation_dates":sorted(reserved),"candidate_dates_needing_review":review,
        "safe_training_dates_verified":sorted(candidate-{r["date"] for r in review}),
        "training_set_created":False,"validation_set_created":False,
        "recommendation":"Add an independently verified annotated video source, or resolve recording metadata before selecting train/validation data; keep current evaluation examples out of optimization"}
save_json(ROOT / "data/recording_groups_review.json",result)
print(json.dumps({"normalized_groups":{k:len(v) for k,v in groups.items()},"invalid_records":len(invalid),"safe_training_dates_verified":result["safe_training_dates_verified"]},indent=2))
