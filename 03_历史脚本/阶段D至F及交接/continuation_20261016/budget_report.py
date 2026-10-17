"""Render the complete bounded allocation experiment, including negative results."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from budget_experiment import WORKSPACE,PROJECT,OLD_RUN,RUN,OUT,SOURCE,read,check,write_json,sha256


def link(path): return str(path).replace("\\", "/")


def main():
    protocol=check()
    result=read(OUT/"预算对照实测结果.json")
    rows={r["condition"]:r for r in result["rows"]}
    order=("E42","BG38","BG34","H265")
    names={"E42":"原 E：背景42", "BG38":"背景38", "BG34":"背景34", "H265":"普通 H.265"}
    colors=["#607d9d","#198e79","#c78b38","#9c6aad"]
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),constrained_layout=True)
    for ax,metric,title in zip(axes,("global_psnr_db","fish_psnr_db","ring_psnr_db"),
                              ("全部 50 帧 RGB","3 张 GT 的鱼像素","3 张 GT 的鱼周围背景环")):
        values=[rows[c][metric] for c in order]
        bars=ax.bar(range(4),values,color=colors,width=.65)
        ax.set_xticks(range(4),[names[c] for c in order],rotation=18)
        ax.set(ylabel="PSNR / dB",title=title,ylim=(0,max(values)*1.18))
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+.35,f"{value:.2f}",ha="center")
        ax.grid(axis="y",alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("同一 video6 开发片段，完整文件接近 99,269 B；两档新分配全部保留",fontsize=13)
    fig.savefig(OUT/"预算分配画质对照.png",dpi=170)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.5,4.2),constrained_layout=True)
    bottom=np.zeros(3)
    for part,color,label in (("context","#607d9d","背景流"),("roi","#198e79","ROI 流"),
                             ("semantic","#c78b38","语义消息"),("container","#aeb9c5","清单及 ZIP")):
        values=[]
        for c in order[:3]:
            parts=rows[c]["parts"]
            values.append(parts.get(part,parts.get("manifest_zip",0) if part=="container" else 0)/1000)
        ax.bar(range(3),values,bottom=bottom,color=color,label=label)
        bottom+=values
    ax.set_xticks(range(3),[names[c] for c in order[:3]])
    ax.set(ylabel="完整文件组成 / 十进制 KB",title="空间区域完全不变，只改变背景与 ROI 质量分配",ylim=(0,115))
    for i,c in enumerate(order[:3]):
        ax.text(i,rows[c]["complete_file_bytes"]/1000+1.7,f"{rows[c]['complete_file_bytes']:,} B",ha="center")
    ax.legend(ncol=4,loc="upper center",bbox_to_anchor=(.5,-.12),frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"完整文件预算组成.png",dpi=170)
    plt.close(fig)

    # All fixed GT components, identical source coordinates/context/display across methods.
    component_reference=read(WORKSPACE/"outputs/semantic_roi_phase_f/step3_frozen_external/positive_case/含鱼GT保护与逐连通域核对.json")["gt_components"]
    component_evaluation=read(OUT/"预算对照逐标注帧与连通域.json")
    component_lookup={(row["condition"],row["source_frame"],region["component"]):region
                      for row in component_evaluation for region in row["components"]}
    component_table=[]
    for region in component_reference:
        f,c=region["source_frame"],region["component"]
        values=[component_lookup[(condition,f,c)]["fish_psnr_db"] for condition in order]
        component_table.append(f"| {f}/c{c} | {region['fish_pixels']} | " +
                               " | ".join(f"{value:.2f}" for value in values)+" |")
    inputs={"original":SOURCE, **{c:Path(rows[c]["receiver_rgb"]) for c in order}}
    columns=("original",*order)
    display_names={"original":"原图",**names}
    figures=[]
    for frame in (60,75,90):
        selected=[r for r in component_reference if r["source_frame"]==frame]
        fig,axes=plt.subplots(len(selected),5,figsize=(12,len(selected)*1.9+.6),squeeze=False,constrained_layout=True)
        for r_index,region in enumerate(selected):
            x,y,w,h=region["box"]
            left=int(np.clip((2*x+w-240)//2,0,1040))
            top=int(np.clip((2*y+h-180)//2,0,540))
            for col,condition in enumerate(columns):
                image=np.asarray(Image.open(inputs[condition]/f"{region['local_frame']:05d}.png").convert("RGB"))
                ax=axes[r_index,col]
                ax.imshow(image[top:top+180,left:left+240],interpolation="nearest")
                ax.add_patch(plt.Rectangle((x-left,y-top),w,h,fill=False,edgecolor="#ff58b8",lw=.8))
                ax.set_xticks([]);ax.set_yticks([])
                if r_index==0:ax.set_title(display_names[condition])
                if col==0:ax.set_ylabel(f"c{region['component']}\n{region['fish_pixels']} px",rotation=0,labelpad=20,va="center")
        fig.suptitle(f"源帧 {frame}：全部 GT 连通域同坐标裁剪（提示性诊断，未人工评分）",fontsize=12)
        path=OUT/f"源帧{frame}_全部GT局部对照.png"
        fig.savefig(path,dpi=160)
        plt.close(fig)
        figures.append(path)
    summary_rows=[]
    for c in order:
        row=rows[c]; task=row["task_fixed_09_three_labels"]
        summary_rows.append(f"| {names[c]} | {row['complete_file_bytes']:,} | {100*row['relative_to_fixed_target']:+.2f}% | "
                            f"{row['global_psnr_db']:.2f} | {row['fish_psnr_db']:.2f} | {row['ring_psnr_db']:.2f} | "
                            f"{100*task['iou']:.2f}% | {row['GT_probability']['mean']:.4f} |")
    allocation=[]
    for c in order[:3]:
        r=rows[c];parts=r["parts"]
        allocation.append(f"| {names[c]} | {r['background_crf']:.0f} | {r['roi_crf']:.3f} | {parts['context']:,} | "
                          f"{parts['roi']:,} | {100*parts['context']/r['complete_file_bytes']:.2f}% |")
    changes=[]
    for c in ("BG38","BG34"):
        r=rows[c]; b=rows["E42"]
        changes.append(f"- **{names[c]}**：全局 PSNR {r['global_psnr_db']-b['global_psnr_db']:+.2f} dB；"
                       f"GT 鱼区 {r['fish_psnr_db']-b['fish_psnr_db']:+.2f} dB；"
                       f"鱼周围背景环 {r['ring_psnr_db']-b['ring_psnr_db']:+.2f} dB。")
    b34=rows["BG34"]
    h265=rows["H265"]
    measured_judgment=(f"背景34将原 E 的全局画质损失收窄：相对普通 H.265，全局 PSNR 仍低 "
                       f"{h265['global_psnr_db']-b34['global_psnr_db']:.2f} dB，GT 鱼区仍高 "
                       f"{b34['fish_psnr_db']-h265['fish_psnr_db']:.2f} dB。"
                       "这是一种可复核的像素质量取舍，尚未转化为已验证的识别或人工辨认收益。")
    all_zero=all(rows[c]["task_fixed_09_three_labels"]["tp"]==0 for c in order)
    task_conclusion=("所有分配及 H.265 在固定 0.9 下仍为 TP=0，鱼 IoU、召回率均为零；这轮没有得到分割任务收益。"
                     if all_zero else "固定 0.9 下出现非零 TP，见逐项结果；仅是开发片段观察，不能作为泛化优势或最终方案选择。")
    task_rows=[]
    for c in order:
        r=rows[c]; t=r["task_fixed_09_three_labels"];s=r["GT_probability"]
        task_rows.append(f"| {names[c]} | {t['tp']} / {t['fp']} / {t['fn']} | {s['p50']:.5f} | "
                         f"{s['max']:.4f} | {s['near_085_to_09_pixels']} | {100*r['pixel_AP_three_labels']:.4f}% |")
    transfers=read(OUT/"新增两组真实收发.json")
    transfer_text="\n".join(f"- {names[c]}：{transfers[c]['transfer_seconds']:.3f} 秒，"
                              f"应用层 {transfers[c]['application_transmit_bytes']:,} B，独立接收恢复 50 帧。"
                              for c in ("BG38","BG34"))
    search=read(OUT/"完整文件匹配选择.json")
    candidates=sum(s["complete_candidates"] for s in search.values())
    report=f"""# 阶段 F 后续：固定 ROI 的背景／目标预算分配小对照

日期：2026-10-16。本轮只在 video6 的源 50—99 帧完成一组有限对照，没有新增模型、训练或阶段名称。沿用已经分析过的开发视频；不是新的独立验证。

## 1. 做了什么

保留原 E（背景 CRF42、ROI CRF31）和普通 H.265 的既有真实接收结果。新增背景 CRF38、34 两档，固定预测消息、0.9 阈值、64 像素筛选、双区域布局、32 帧窗口、外扩、背景 640×360、ROI 原分辨率、色彩与时序编码配置。只改变背景 CRF，并按完整文件字节匹配 ROI CRF。

正式编码前已保存[冻结协议]({link(OUT/'预算对照冻结协议.json')})。目标为原 E 完整包 **99,269 B**，优先匹配 1%，最多容许相对目标 2%，每档最多 10 个完整候选，不补无效字节，不用 GT、画质、概率或人工评分挑选候选。两档均选完后才进行接收解码和评价。

两档实际误差为 +0.80%、−1.69%，均符合预设容差；最大和最小新包相差约 2.53%，所以这里称相近预算，不能当作字节完全相同。BG34 达到预设 10 个候选上限，保留最近结果，没有继续搜索。两档合计 {candidates} 个匹配候选、2 个新传输包，所有候选记录均保留。

## 2. 字节分配确实发生了变化

| 配置 | 背景 CRF | ROI CRF | 背景流 B | ROI 流 B | 背景占完整文件 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(allocation)}

语义消息三组均为 1,419 B，清单及 ZIP 开销约 2.26 KB，全部计入预算。几何 ROI 与 GT 覆盖保持不变：平均画面覆盖约 34.98%，GT 鱼覆盖仍为 1,940/2,010 像素，即 96.52%。这次没有修复此前漏保护的 70 像素区域。

![完整文件组成]({link(OUT/'完整文件预算组成.png')})

## 3. 实际接收结果

全局 PSNR 使用全部 50 帧；鱼区、背景与模型任务指标只用 60、75、90 三张发布者 GT。47 帧未标注，不视为背景。本片段没有无鱼 GT 对照，不能据此评价空场景误报。鱼周围背景环采用 GT 外扩 32 像素的方形邻域并排除全部 GT 鱼像素，作为上下文质量的局部代理。

| 配置 | 完整文件 B | 对目标误差 | 全局 PSNR dB | GT 鱼区 dB | 鱼周围背景环 dB | 鱼 IoU | GT 平均鱼概率 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary_rows)}

相对原 E：

{chr(10).join(changes)}

![画质对照]({link(OUT/'预算分配画质对照.png')})

{task_conclusion} 背景流仍是半分辨率，提高其编码质量不能恢复原始背景分辨率。背景画质、ROI 细节与整幅上下文之间的具体因果不能靠这两个配置完全分离。

{measured_judgment}

平均值不能保证逐目标改善。例如未进 ROI 的 75/c1（70 像素），PSNR 从原 E 的 37.81 dB 降至背景38的 32.61、背景34的 33.33 dB；其他连通域也并非随 CRF 单调变化。不能从整体背景变好就推断漏保护的小鱼也变好；局部 PSNR 的变化本身也不等于人工辨认变化。

| 帧/连通域 | GT 像素 | 原 E 鱼区 dB | 背景38 dB | 背景34 dB | H.265 dB |
|---|---:|---:|---:|---:|---:|
{chr(10).join(component_table)}

| 配置 | TP / FP / FN | GT 概率中位数 | GT 最大概率 | [0.85,0.9) GT 像素 | 三张 GT 像素 AP |
|---|---|---:|---:|---:|---:|
{chr(10).join(task_rows)}

本表 AP 使用三张含鱼 GT 的全部鱼与背景像素，区别于前一份报告纳入四张空 GT 的七帧 AP；两者不可直接横比。概率与扫分数曲线仅作诊断，主评价 0.9 没有变化。

误报像素从原 E 的 3,409 增至背景38的 4,373、背景34的 6,745；GT 平均概率也未随背景质量单调改善。新配置没有 GT 鱼像素落在 [0.85,0.9) 区间，仍不支持“只需略降阈值”这一解释。

## 4. 实际收发与复核

{transfer_text}

以上为 20 十进制 kbps 的本机 TCP 应用层限速，发送时间不含编码、参数搜索和恢复，也不是真实水声链路。两秒名义视频仍需约四十秒发送，当前系统为离线批量原型。

评价只用实际接收端 RGB，接收进程仅读取收到的包及解码目录。两组原 E／H.265 的 100 帧接收结果复用历史实测，不重复发送；两组新配置又各传输并独立恢复 50 帧。四组共 200 个接收帧哈希已复核；旧画质与固定阈值统计精确复现。生产代码、模型、GT、预测消息及 ROI 布局保持不变。

接收端视频（观看副本在收包后生成，不是评价输入）：

- [原 E]({link(rows['E42']['preview_video'])})
- [背景38]({link(rows['BG38']['preview_video'])})
- [背景34]({link(rows['BG34']['preview_video'])})
- [普通 H.265]({link(rows['H265']['preview_video'])})

## 5. 怎么解释，下一步做什么

本轮检验的是把既有字节从 ROI 转给背景是否有帮助，并未改变发送端找鱼能力。所有分配正负结果均保留，暂不替换演示默认参数，也不据 video6 宣称独立场景优势。

此前单人局部辨认的原图 8/10、原 E 6/10、H.265 4/10 仍只对应旧恢复图；**新配置尚无人工评分**。下列全部 GT 局部图仅为带位置提示的诊断展示，不能据此自动产生新的人工可辨认率，也不能只挑表现较好的目标。

{chr(10).join(f'![全部局部对照]({link(p)})' for p in figures)}

下一轮不继续扩大预算网格，也不因全局 PSNR 提高就直接把背景34设为正式默认值。优先用少量固定帧拆分失真来源：检查原 RGB 经颜色／YUV420 采样往返后的模型响应，再与原图、全幅 H.265 和 ROI 合成结果比较，判断模型首先在哪一步变得不稳定。这个诊断沿用现有网络与阈值，不立即重新训练。若另做压缩鲁棒性适配，应独立使用开发数据并保留原冻结模型结果，不为制造胜负更换口径。之后再固定一个方案到新保留视频验证。本轮不能代替独立验证，也尚未证明新配置的人工可辨识性收益。

## 6. 数据入口

- [完整实测表]({link(OUT/'预算对照实测结果.json')})
- [逐帧与逐连通域]({link(OUT/'预算对照逐标注帧与连通域.json')})
- [全部候选选择记录]({link(OUT/'完整文件匹配选择.json')})
- [收发核验]({link(OUT/'新增两组真实收发.json')})
- [最终核验]({link(OUT/'预算对照核验.json')})
"""
    destination=OUT/"2026-10-16_固定ROI预算分配对照与判断.md"
    destination.write_text(report,"utf-8")
    (PROJECT/"docs/budget_allocation_20261016.md").write_text(report,"utf-8")
    state_path=WORKSPACE/"outputs/semantic_review_20261016/本次推进状态.json"
    state=read(state_path)
    state.update({"budget_experiment":"complete_two_fixed_variants","budget_report":str(destination),
                  "new_training":0,"new_complete_packet_candidates":candidates,
                  "new_video_encodes":candidates*3+2,"new_h265_stream_encoder_calls":candidates*3+2,
                  "video_encode_count_definition":"H265 stream invocations including byte-matching candidates",
                  "new_TCP_transfers":2,
                  "new_independently_received_frames":100,"new_GT_inference_images":6,
                  "production_defaults_changed":False,"human_ratings_new_variants":0,
                  "next_action":"bounded color/subsampling/composition diagnostic with fixed model before new holdout validation"})
    write_json(state_path,state)
    index=WORKSPACE/"2026-10-16_继续推进记录.md"
    current=index.read_text("utf-8").replace("人工与模型响应存在差异，下一次做固定文件预算下的背景/ROI 质量分配小对照。", "人工与模型响应存在差异，随后已完成固定文件预算下的背景/ROI 质量分配小对照（见下文）。")
    current=current.replace("阶段 F 保持首轮完成，本次未新增训练、编码或通信测试。", "阶段 F 保持首轮完成；概率与人工检查之后新增两组预算对照收发，未训练新模型。")
    if link(destination) not in current:
        current += f"\n## 后续预算小对照已完成\n\n固定既有 ROI，新测背景 CRF38、34，并各完成真实接收恢复 50 帧。"
        current += "没有替换模型、阈值或演示默认配置。具体画质、任务与文件差异见下表，前述单人评分不适用于新配置。\n\n"
        current += f"[预算对照实测与判断]({link(destination)})\n\n"
        current += "| 配置 | 文件 B | 全局 PSNR | 鱼区 PSNR | 鱼 IoU |\n|---|---:|---:|---:|---:|\n"
        for c in order:
            r=rows[c]
            current+=f"| {names[c]} | {r['complete_file_bytes']:,} | {r['global_psnr_db']:.2f} | {r['fish_psnr_db']:.2f} | {100*r['task_fixed_09_three_labels']['iou']:.2f}% |\n"
        index.write_text(current,"utf-8")
    human_report=WORKSPACE/"outputs/semantic_review_20261016/2026-10-16_单人可辨识性核对与方向更新.md"
    human_note=f"> 同日后续：已完成下一步[固定 ROI 预算分配对照]({link(destination)})。本文结尾“尚未启动”的描述保留为人工核对刚结束时的历史状态。\n\n"
    human_text=human_report.read_text("utf-8")
    if human_note not in human_text:
        title,body=human_text.split("\n",1)
        human_report.write_text(title+"\n\n"+human_note+body.lstrip("\n"),"utf-8")
        (PROJECT/"docs/human_review_20261016.md").write_text(human_report.read_text("utf-8"),"utf-8")
    check()
    print("BUDGET_REPORT_COMPLETE",str(destination),flush=True)


if __name__=="__main__":main()
