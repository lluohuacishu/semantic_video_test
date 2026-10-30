"""Standalone scientific figure; original uninstrumented inference timings only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output
    load=lambda name:json.loads((out/name).read_text(encoding='utf-8-sig'))
    encoder=load('encoder_summary.json');full=load('full_summary.json');fill=load('fill_summary.json')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'svg.hashsalt':'glc-encoder-20261030'})
    fig,axes=plt.subplots(1,3,figsize=(17,5.4),gridspec_kw={'width_ratios':[1.3,1,1.2]})
    rows=[r for r in encoder['reports'] if r['model']=='video'];names=[r['variant'] for r in rows]
    y=np.arange(len(rows));times=[r['candidate_wall_median_ms'] for r in rows]
    exact=[all(d['bit_exact'] for d in r['changes']) for r in rows]
    axes[0].barh(y,times,color=['#357d69' if e else '#4a74aa' for e in exact])
    axes[0].set_yticks(y,names);axes[0].invert_yaxis();axes[0].set_xlabel('单模块同步墙钟中位数（ms）')
    axes[0].set_title('P帧输入编码器 · 640×384')
    for i,v in enumerate(times):axes[0].text(v+.5,i,f'{v:.2f}',va='center',fontsize=9)
    axes[0].set_xlim(0,max(times)*1.22)
    clips=list(dict.fromkeys(r['clip'] for r in fill['reports']));x=np.arange(len(clips))
    for role,shift,color,label in [('sender',-.18,'#4a74aa','编码'),('receiver',.18,'#357d69','解码')]:
        values=[next(r for r in fill['reports'] if r['clip']==c and r['role']==role)['paired_speedup_median'] for c in clips]
        axes[1].bar(x+shift,values,width=.35,color=color,label=label)
    axes[1].axhline(1,color='#777',lw=1);axes[1].set_ylim(.95,1.15)
    axes[1].set_xticks(x,[c.split('_q')[0]+'\nq'+c.split('_q')[1].split('_')[0] for c in clips],rotation=30)
    axes[1].set_ylabel('逐对加速比的中位数');axes[1].set_title('填充关闭 · 每端每组两对');axes[1].legend()
    rr=[r for r in full['reports'] if r['clip']=='long108_q0_640']
    xx=np.arange(len(rr));seconds=[r['sender_core_median_seconds'] for r in rr]
    axes[2].bar(xx,seconds,color='#4a74aa')
    axes[2].set_xticks(xx,[r['variant'] for r in rr],rotation=35,ha='right')
    axes[2].set_ylabel('66帧编码核心（秒）');axes[2].set_title('完整编码 · 相同fill-off条件')
    for i,v in enumerate(seconds):axes[2].text(i,v+.05,f'{v:.3f}',ha='center',fontsize=9)
    axes[2].set_ylim(0,max(seconds)*1.18)
    fig.suptitle('GLC 输入编码器推理对照 · RTX4060 Laptop · 无训练',fontsize=15)
    fig.text(.5,.012,'单模块绿色仅表示本次采样潜变量逐位一致；完整码流、画质与独立解码另行验收。未测试Orin或能耗。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.94]);fig.savefig(out/'inference_overview.png',dpi=160);fig.savefig(out/'inference_overview.svg')


if __name__=='__main__':main()
