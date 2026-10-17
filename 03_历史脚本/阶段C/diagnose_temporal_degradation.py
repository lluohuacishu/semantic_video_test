"""Localize degradation using existing independently verified reconstructions."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from experiment_common import *

def main():
    rows=json.loads((RUN/"task_quality_results.json").read_text("utf-8"))["rows"]
    matched=[]
    for r in rows:
        if r["branch"]!="M_RD":continue
        pre=next(x for x in rows if x["branch"]=="M_PRE" and x["video"]==r["video"] and x["anchor"]==r["anchor"])
        difference=(r["file_bytes"]-pre["file_bytes"])/pre["file_bytes"]
        if abs(difference)<=.02:matched.append({"video":r["video"],"anchor":r["anchor"],"M_PRE_bytes":pre["file_bytes"],"M_RD_bytes":r["file_bytes"],"file_difference_percent":difference*100,"M_PRE_PSNR":pre["global_rgb_psnr_db"],"M_RD_PSNR":r["global_rgb_psnr_db"],"delta_PSNR_dB":r["global_rgb_psnr_db"]-pre["global_rgb_psnr_db"],"delta_fish_PSNR_dB":r["gt_fish_rgb_psnr_db"]-pre["gt_fish_rgb_psnr_db"]})
    folder=RUN/"codecs/video96"
    pre=json.loads((folder/"M_PRE/q08/verification.json").read_text("utf-8"))["frames"]
    tuned=json.loads((folder/"M_RD/q08/verification.json").read_text("utf-8"))["frames"]
    positions=[]
    for p in range(32):
        a=np.mean([x["rgb_mse"] for i,x in enumerate(pre) if i%32==p]);b=np.mean([x["rgb_mse"] for i,x in enumerate(tuned) if i%32==p])
        positions.append({"gop_position":p,"pre_group_PSNR_dB":float(-10*np.log10(a)),"M_RD_group_PSNR_dB":float(-10*np.log10(b)),"delta_group_PSNR_dB":float(10*np.log10(a/b))})
    result={"matched_pretrained_comparisons":matched,"video96_q8_by_GOP_position":positions,"inference":"First P already somewhat worse; degradation increases along GOP. This is consistent with temporal accumulation and reference feature distribution mismatch, not proof of the sole cause.","model_weights_changed_by_diagnostic":False}
    save_json(RUN/"temporal_degradation_diagnostic.json",result);save_json(OUT/"时序退化诊断.json",result)
    font=FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc");plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False})
    fig,ax=plt.subplots(figsize=(9,4.3),constrained_layout=True)
    ax.plot(range(32),[r["pre_group_PSNR_dB"] for r in positions],label="预训练 M_PRE",color="#777777")
    ax.plot(range(32),[r["M_RD_group_PSNR_dB"] for r in positions],label="普通微调 M_RD",color="#245c96")
    ax.set_xlabel("GOP 内帧位置（0 为 I 帧）");ax.set_ylabel("按累计 MSE 计算的 RGB PSNR（dB）")
    ax.set_title("video96，q8：I 帧相同，微调模型在后续 P 帧逐步退化")
    ax.legend();ax.grid(alpha=.25);fig.savefig(OUT/"GOP内重建质量退化.png",dpi=160);plt.close(fig)
    print(json.dumps(matched,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
