"""Plot measured D2 files against fish fidelity without interpolated rate claims."""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from prepare_d2 import read_json


def run(args):
    result=read_json(args.run/'evaluation.json')
    output=args.run/'figures'; output.mkdir(exist_ok=False)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'font.size':10,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors={'A':'#185d9e','B':'#777d85','C':'#008878','O':'#cc6d17'}
    markers={'A':'o','B':'s','C':'^','O':'D'}
    labels={'A':'A 原图 GLC','B':'B 全图低通','C':'C 预测保护','O':'O GT 保护'}
    fig,axes=plt.subplots(1,2,figsize=(13.2,5.5),sharey=True)
    for ax,video in zip(axes,(1,2)):
        points=[p for p in result['points'] if p['video_id']==video]
        anchor=result['budget_anchors'][str(video)]
        lo=min(p['file_bytes'] for p in points)/1000-5
        hi=max(p['file_bytes'] for p in points)/1000+8
        ax.axvspan(anchor['B10']/1000,hi,color='#f2ebe7',zorder=0)
        ax.axvline(anchor['S0']/1000,color='#185d9e',linestyle='--',linewidth=1)
        ax.axvline(anchor['B10']/1000,color='#9b5149',linestyle=':',linewidth=1.4)
        for method in 'ABCO':
            for p in [p for p in points if p['method']==method]:
                x=p['file_bytes']/1000; y=p['regions']['fish']['psnr_db']
                if y is None: continue
                ax.scatter(x,y,s=46,marker=markers[method],edgecolors=colors[method],
                           facecolors=colors[method] if p['within_B10'] else 'white',linewidths=1.2,zorder=4)
                dx,dy={'A':(5,3),'B':(-5,-12),'C':(5,-11),'O':(-5,5)}[method]
                ax.annotate(str(p['q']),(x,y),xytext=(dx,dy),textcoords='offset points',
                            fontsize=8,color=colors[method],ha='left' if dx>0 else 'right')
        ax.set(xlim=(lo,hi),xlabel='完整 glcv 文件（十进制 kB）',title=f'片段 {video:03d} · 64 帧')
        ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('全部 GT 鱼区 PSNR（dB）')
    handles=[Line2D([],[],linestyle='none',marker=markers[m],color=colors[m],label=labels[m]) for m in 'ABCO']
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.93),ncol=4,frameon=False)
    fig.suptitle('D2：实际文件大小与目标画质',fontsize=17,y=.99)
    fig.text(.5,.035,'每点为实际码流，数字为 q。蓝虚线：A@q32 基准；红点线：+10% 上限。空心点超预算，仅作诊断。',
             ha='center',fontsize=9,color='#4b5563')
    fig.subplots_adjust(top=.80,bottom=.18,wspace=.12)
    fig.savefig(output/'fish_psnr_vs_file_bytes.png',dpi=180)
    fig.savefig(output/'fish_psnr_vs_file_bytes.svg')
    svg=output/'fish_psnr_vs_file_bytes.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    print('Created measured rate/quality figure in '+str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args())
