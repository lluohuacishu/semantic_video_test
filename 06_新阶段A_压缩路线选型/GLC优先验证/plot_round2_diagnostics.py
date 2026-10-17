"""Plot predeclared per-frame diagnostics; no result-dependent frame selection."""
from pathlib import Path
import argparse
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
cn=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
series=[('glc_540_g64','GLC GOP64','#bc3440'),('h265_540_for_540_g64','H.265 GOP64','#2575b3'),('av1_540_for_540_g64','AV1 GOP64','#2a9370')]
diagnostics=[]
for clip in ['long108','long91','long24']:
    data=json.loads((a.run/clip/'results.json').read_text(encoding='utf-8'))
    byname={x['run']:x for x in data['results']}
    for result in data['results']:
        rows=result['per_frame'][1:]
        reset=[r for r in rows if r['index']%32==1]
        rest=[r for r in rows if r['index']%32!=1]
        diagnostics.append({'clip':clip,'run':result['run'],'gop':result['config']['gop'],'glc_reset_reference_indices':[r['index'] for r in reset],'mean_at_reference_reset_indices':statistics.mean(r['temporal_error_delta_mae_8bit'] for r in reset),'mean_at_other_indices':statistics.mean(r['temporal_error_delta_mae_8bit'] for r in rest),'top_five_error_change_frames':[{'index':r['index'],'error_change_mae':r['temporal_error_delta_mae_8bit']} for r in sorted(rows,key=lambda r:r['temporal_error_delta_mae_8bit'],reverse=True)[:5]]})
    current=series+([('glc_540_g128','GLC GOP128','#bd7b20')] if clip=='long108' else [])
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
    keys=[('psnr_db','RGB PSNR (dB)'),('dists','DISTS (lower is better)'),('temporal_error_delta_mae_8bit','Error-change MAE (8-bit)')]
    for ax,(key,label) in zip(axes,keys):
        for name,title,color in current:
            rows=[r for r in byname[name]['per_frame'] if r[key] is not None]
            ax.plot([r['index'] for r in rows],[r[key] for r in rows],color=color,label=title,lw=1.35,marker='.' if key=='dists' else None,markersize=3)
        for n in range(64,250,64):ax.axvline(n,color='#9199a5',linestyle='--',lw=.7,zorder=0)
        for n in range(1,250,32):ax.axvline(n,color='#d2d6df',linestyle=':',lw=.7,zorder=0)
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.18);ax.set_xlim(0,249)
    axes[0].legend(ncol=4,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False)
    axes[-1].set_xlabel('Source frame index (0-based)')
    fig.suptitle(f'{clip}：250帧诊断，540p工作分辨率 / 720p统一评价',fontproperties=cn,fontsize=17,y=.99)
    fig.text(.08,.015,'DISTS仅在预先固定的采样帧计算；虚线为GOP64边界，点线为reset候选位置。误差变化量不能直接当作感知闪烁评分。',fontproperties=cn,fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.96))
    fig.savefig(a.output/f'08_{clip}_逐帧诊断.png',dpi=140)
    plt.close(fig)
(a.output.parent/'temporal_boundary_diagnostics.json').write_text(json.dumps({'note':'Descriptive diagnostic only, not a perceptual flicker score or a causal claim. Standard codecs are evaluated at the same GLC reference positions.','results':diagnostics},ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved three diagnostic plots',flush=True)
