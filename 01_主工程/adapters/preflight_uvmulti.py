"""Independent prospective training source: official Figshare labels and ZIP index."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
from common import ROOT, save_json
from fetch_sfishtrack import RangeFile

folder=ROOT / "data/uvmulti_preflight"
folder.mkdir(exist_ok=False)
metadata=json.loads((ROOT / "audit/uvmulti_figshare_metadata.json").read_text("utf-8"))
files={f["name"]:f for f in metadata["files"]}
file=files["seg_label.txt"]
with urllib.request.urlopen(file["download_url"],timeout=30) as response: labels=response.read()
if hashlib.md5(labels).hexdigest()!=file["supplied_md5"]: raise ValueError("Published label table checksum differs")
(folder / "seg_label.txt").write_bytes(labels)
inventory=[]
for name in ["video91.zip","video96.zip"]:
    file=files[name]
    reader=RangeFile(file["download_url"],file["size"])
    with zipfile.ZipFile(reader) as archive:
        entries=[{"name":z.filename,"bytes":z.file_size,"compressed_bytes":z.compress_size,"crc32":format(z.CRC,"08x")} for z in archive.infolist() if not z.is_dir()]
    save_json(folder / f"{Path(name).stem}_index.json",entries)
    inventory.append({"archive":name,"bytes":file["size"],"http_bytes_for_index":reader.bytes_received,"entries":len(entries),"examples":entries[:10]})
save_json(folder / "provenance.json",{"source":"https://figshare.com/articles/dataset/30105151","publication":"https://doi.org/10.1038/s41597-026-07402-0",
              "official_repository":"https://github.com/pingchuan/UVMulti","data_license":metadata["license"],"label_table_md5_checked":True,
              "archive_full_checksums_checked":False,"training_data_downloaded":False,"training_performed":False,"indexes":inventory,
              "purpose":"Prospective independent training/validation source; SFISHTRACK evaluation dates remain reserved"})
print(labels.decode("utf-8"),flush=True)
print(json.dumps(inventory,ensure_ascii=False,indent=2),flush=True)
