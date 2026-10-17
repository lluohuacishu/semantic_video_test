"""Handle the last branch independently, ahead of the serial cold-decoder loop."""
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from experiment_common import *
from evaluate_roi_experiment import invoke,I_WEIGHT


def main():
    rows=json.loads((RUN/"file_selections.json").read_text("utf-8"))
    python=str(PROJECT/".venv/Scripts/python.exe")
    fast=str(Path(__file__).with_name("cold_decode_fast_png.py"))
    native=str(PROJECT/"adapters/mlvc_file.py")
    for row in rows:
        if row["branch"]!="M_PRE":continue
        folder=Path(row["run"]);receiver=folder/"receiver"
        if not (receiver/"decode.json").exists():invoke([python,fast,"decode","--file",str(folder/"video.mlvcv"),"--output",str(receiver),"--weight",row["weight"],"--i-weight",str(I_WEIGHT)],folder/"cold_decode.log")
        if not (folder/"verification.json").exists():invoke([python,native,"verify","--run",str(folder),"--decoded",str(receiver)],folder/"verification.log")
        enc=json.loads((folder/"encode.json").read_text("utf-8"))
        for frame in enc["frames"]:
            pixels=np.asarray(Image.open(receiver/"rgb"/f'{frame["index"]:05d}.png').convert("RGB"))
            import hashlib
            if hashlib.sha256(pixels.tobytes()).hexdigest()!=frame["rgb_sha256"]:raise AssertionError("Saved PNG pixels differ from encoder reconstruction")
        print("PARALLEL_COLD_DECODE_OK",row["video"],row["q_index"],flush=True)


if __name__=="__main__":main()
