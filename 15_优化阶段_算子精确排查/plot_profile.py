"""Standalone scientific plots from verified low-overhead CUPTI timings."""
from pathlib import Path
import argparse,json,sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output
    analysis=json.loads((output/'analysis.json').read_text('utf-8'))
    font=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})
    frames=[next(x for x in analysis['roles'][r]['frames'] if x['index']==2) for r in ['sender','receiver']]
    names=['aten::cudnn_convolution','aten::native_group_norm','aten::add_','aten::fill_','torchscript.swish','aten::add']
    labels=['卷积','GroupNorm','原位加法','填充','融合Swish','加法']
    colors=['#3266a8','#57a1cf','#7ca873','#e2b64f','#ae75a4','#91b9a8']
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),gridspec_kw={'width_ratios':[1.5,1,1]})
    left=np.zeros(2)
    for name,label,color in zip(names,labels,colors):
        values=[100*sum(x['fraction_of_kernel_sum'] for x in f['gpu_operators'] if x['name']==name) for f in frames]
        axes[0].barh(['编码','解码'],values,left=left,label=label,color=color,height=.5);left+=values
    axes[0].barh(['编码','解码'],100-left,left=left,label='其他',color='#c3cbd1',height=.5)
    axes[0].set_xlim(0,100);axes[0].set_xlabel('GPU 内核累计时间份额（%）');axes[0].set_title('普通 P 帧：算子级分解')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=4,frameon=False,fontsize=8)
    async_ms=[next(x['wall_ns']/1e6 for x in f['runtime_apis'] if x['name'].startswith('cudaMemcpyAsync')) for f in frames]
    copy_ms=[f['memcpy_ms'] for f in frames]
    sync_ms=[f['runtime_sync_ms'] for f in frames]
    x=np.arange(2);width=.24
    for off,vals,label,color in [(-width,async_ms,'Memcpy CPU调用','#d58b42'),(0,sync_ms,'同步 CPU调用','#8c659f'),(width,copy_ms,'GPU实际拷贝','#4d9d9b')]:
        bars=axes[1].bar(x+off,vals,width,label=label,color=color)
        for bar,value in zip(bars,vals):axes[1].text(bar.get_x()+bar.get_width()/2,value*1.12,f'{value:.2f}',ha='center',fontsize=8)
    axes[1].set_yscale('log');axes[1].set_ylim(.1,300);axes[1].set_xticks(x,['编码','解码']);axes[1].set_ylabel('毫秒（对数轴）')
    axes[1].set_title('调用等待 ≠ DMA 拷贝时间');axes[1].legend(fontsize=8,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.18))
    fill=analysis['fill_ablation'];on=[fill[r]['on_median_seconds'] for r in ['sender','receiver']];off=[fill[r]['off_median_seconds'] for r in ['sender','receiver']]
    for pos,values,label,color in [(-.17,on,'默认填充','#b9c7d8'),(.17,off,'实验关闭填充','#4f8a7d')]:
        bars=axes[2].bar(x+pos,values,.34,label=label,color=color)
        for bar,value in zip(bars,values):axes[2].text(bar.get_x()+bar.get_width()/2,value+.08,f'{value:.2f}',ha='center',fontsize=9)
    axes[2].set_xticks(x,['编码','解码']);axes[2].set_ylim(0,12);axes[2].set_ylabel('66 帧核心耗时（秒）');axes[2].set_title('额外开关对照：各三组配对');axes[2].legend(fontsize=8,frameon=False)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('GLC 精简版真实内核排查 · RTX 4060 · FP32 · 640×360 / q0',fontsize=14)
    fig.text(.5,.015,'GPU 标签经两次原生 NVTX 完整内核序列校验映射；CPU/GPU 区间重叠，不能相加。填充开关未改生产默认。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.92]);fig.savefig(output/'operator_overview.png',dpi=170);fig.savefig(output/'operator_overview.svg');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(14,6.5))
    for ax,role,title in zip(axes,['sender','receiver'],['编码：普通 P 帧','解码：普通 P 帧']):
        p=output/role/'P002_activity_trace.json';events=json.loads(p.read_text('utf-8'))['traceEvents']
        groups={'kernel':[],'copy':[],'async':[],'sync':[]}
        for row in events:
            if row['cat']=='gpu':key='kernel' if row['name'] not in ('memcpy','memset') else 'copy'
            elif 'MemcpyAsync' in row['name']:key='async'
            elif 'Synchronize' in row['name']:key='sync'
            else:continue
            groups[key].append((row['ts']/1000,row['dur']/1000))
        for i,(key,color) in enumerate([('copy','#4d9d9b'),('kernel','#3266a8'),('sync','#8c659f'),('async','#d58b42')]):
            ax.broken_barh(groups[key],(i-.28,.55),facecolors=color,edgecolors='none')
        ax.set_yticks(range(4),['GPU 拷贝/清零','GPU 内核','CPU 同步调用','CPU Memcpy调用']);ax.set_xlabel('相对采样窗口开始（毫秒）');ax.set_ylim(-.6,3.7)
        ax.set_title(title,loc='left');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('低干扰 CUPTI 时间线：主机等待与 GPU 活动重叠',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(output/'P002_timeline.png',dpi=170);fig.savefig(output/'P002_timeline.svg');plt.close(fig)
    for name in ['operator_overview.svg','P002_timeline.svg']:
        path=output/name;path.write_text('\n'.join(line.rstrip() for line in path.read_text('utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    print('Saved operator breakdown and actual CPU/GPU timeline plots')


if __name__=='__main__':main()
