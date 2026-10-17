"""Summarize all four ROI controls after the user's continuous review."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
from control_common import *

SUP=OUT/"人工ROI补充"
SCRIPT=Path(__file__).resolve().parent

def link(path,title):return f"[{title}]({Path(path).as_posix()})"

def main():
    old={x["group"]:x for x in read(OUT/"实际码流与GT任务评价.json")}
    new={x["group"]:x for x in read(SUP/"原图与人工ROI任务评价.json")}
    assert old["original"]["gt"]==new["original"]["gt"]
    assert old["original"]["teacher"]==new["original"]["teacher"]
    metrics={**old,"C":new["C"],"D":new["D"]}
    protocol=read(RUN/"manual_protocol.json")
    assert sha256(RUN/"manual_roi_snapshot.json")==protocol["annotation_snapshot_sha256"]
    assert sha256(DIRECTION)==protocol["direction_sha256"]
    geometry={"A":next(x for x in read(OUT/"逐帧与小目标覆盖复核.json") if x["window"]==32),"B":next(x for x in read(OUT/"逐帧与小目标覆盖复核.json") if x["window"]==4)}
    manual_geometry=read(SUP/"人工ROI几何复核.json")
    geometry.update({"C":manual_geometry[0],"D":manual_geometry[1]})
    selected={};receipts={};decoders={};coded={}
    _,_,semantic_bytes,_,_=frozen()
    for group in ("A","B","C","D","H265"):
        s=read(RUN/group/"selection.json");selected[group]=s
        packet=Path(s["file"])
        received=RUN/group/"network/receiver"/("received.mp4" if group=="H265" else "message.scv")
        receipt=read(RUN/group/"network/end_to_end.json");receipts[group]=receipt
        decoder=read(RUN/group/"network/receiver/decode.json");decoders[group]=decoder
        assert sha256(packet)==sha256(received)==s["sha256"]==decoder["file_sha256"]
        assert receipt["success"] and receipt["packet_sha256_equal"] and receipt["separate_sender_receiver_processes"]
        assert receipt["frames"]==decoder["frames"]==50
        assert len(list((RUN/group/"network/receiver/rgb").glob("*.png")))==50
        assert sum(s["parts"].values())==packet.stat().st_size==receipt["file_bytes"]
        assert abs(s["file_bytes"]-100000)<=2000
        audit=read(RUN/group/"network/receiver/receiver_read_audit.json")
        assert audit["guard_installed"] and all(x["allowed"] for x in audit["python_reads"])
        assert all(Path(x["path"]).resolve().is_relative_to((RUN/group/"network/receiver").resolve()) for x in audit["python_reads"])
        if group!="H265":
            with zipfile.ZipFile(packet) as archive:
                assert archive.testzip() is None
                manifest=json.loads(archive.read("manifest.json"))
                assert archive.read("semantics.json.gz")==semantic_bytes
                for name,record in manifest["files"].items():
                    payload=archive.read(name)
                    assert len(payload)==record["bytes"] and hashlib.sha256(payload).hexdigest()==record["sha256"]
                if group in {"C","D"}:assert manifest["experiment"]["annotation_snapshot_sha256"]==protocol["annotation_snapshot_sha256"]
        counts=Counter()
        for stream in read(RUN/group/"stream_audit.json")["streams"]:counts.update(stream["frames_by_type"])
        coded[group]=dict(counts)
    write_json(SUP/"四组与H265完整汇总.json",{"annotation_protocol":protocol,"task_metrics":metrics,"selections":selected,"network_receipts":receipts,"geometry":geometry})
    write_json(SUP/"人工ROI交付核验.json",{"reviewed_frames":50,"positive_frames":38,"no_fish_frames":12,"boxes":61,"box_coordinates_unchanged_during_confirmation":True,"annotation_snapshot_sha256":protocol["annotation_snapshot_sha256"],"all_five_groups_file_budget_matched":True,"all_five_groups_independently_received_and_restored_50_frames":True,"packet_crc_member_sizes_and_hashes_checked":True,"all_layered_semantic_messages_identical_frozen_predictions":True,"receiver_read_audits_stay_in_own_output":True,"guard_not_os_sandbox":True,"original_gt_counts_and_teacher_profile_reproduced":True,"publisher_pixel_gt_unchanged":True})
    names={"A":"预测ROI×32帧","B":"预测ROI×4帧","C":"人工ROI×32帧","D":"人工ROI×4帧","H265":"普通H.265，B帧"}
    rows=[]
    for g in ("A","B","C","D","H265"):
        m=metrics[g];s=selected[g]
        rows.append(f"| {g} {names[g]} | {s['file_bytes']:,} | {m['gt']['iou']*100:.3f} | {m['gt']['recall']*100:.3f} | {m['gt']['precision']*100:.3f} | {m['gt_fish_rgb_psnr_db']:.3f} | {m['gt_background_rgb_psnr_db']:.3f} | {m['global_rgb_psnr_db']:.3f} |")
    geo_rows=[]
    for g in ("A","B","C","D"):
        s=selected[g];geom=geometry[g];ft=coded[g]
        geo_rows.append(f"| {g} | {geom['mean_rectangle_fraction']*100:.3f}% | {geom['pooled_gt_pixel_coverage']*100:.2f}% | {round(geom['positive_frame_hit_fraction']*3)}/3 | {geom['segments']} | {s['fixed_crf32_diagnostic']['file_bytes']:,} | {s['crf']} | {ft.get('I',0)}/{ft.get('P',0)}/{ft.get('B',0)} |")
    task_rows=[]
    for g in ("original","A","B","C","D","H265"):
        m=metrics[g]
        task_rows.append(f"| {g} | {m['gt']['tp']} | {m['gt']['fp']} | {m['gt']['fn']} | {m['positive_frame_mean_iou']*100:.3f}% | {m['positive_frame_mean_recall']*100:.3f}% | {round(m['positive_frame_task_hit_fraction']*3)}/3 |")
    network_rows=[]
    for g in ("C","D"):
        s=selected[g];r=receipts[g];d=decoders[g];p=s["parts"]
        network_rows.append(f"| {g} | {p['context']:,} | {p['roi']:,} | {p['semantic']:,} | {p['container']:,} | {r['application_transmit_bytes']:,} | {r['transfer_seconds']:.3f} | {d['decode_and_preview_seconds']:.3f} |")
    c32=selected["C"]["fixed_crf32_diagnostic"]["file_bytes"];d32=selected["D"]["fixed_crf32_diagnostic"]["file_bytes"]
    delta_text=[]
    for manual,pred in (("C","A"),("D","B")):
        a,b=metrics[manual],metrics[pred]
        delta_text.append(f"- {manual}对{pred}：GT鱼区PSNR变化{a['gt_fish_rgb_psnr_db']-b['gt_fish_rgb_psnr_db']:+.3f}dB，IoU变化{100*(a['gt']['iou']-b['gt']['iou']):+.3f}个百分点，召回变化{100*(a['gt']['recall']-b['gt']['recall']):+.3f}个百分点。")
    first=OUT/"阶段D_ROI窗口对照实测_2026-10-15.md"
    report=SUP/"阶段D_连续人工ROI四组对照补充_2026-10-15.md"
    report.write_text(f"""# 阶段D：连续人工ROI确认与四组对照补充

日期：2026-10-15。本报告续接{link(first,'首轮预测ROI报告')}。先保存用户的人工框与完成声明，再启动此前等待标注的C/D；原方向文档及首轮结果保留。

## 1. 人工核对已确认保存

用户明确说明“人工核对完了，全框出来了，就是没点保存，你来确认吧”。助手先保存现有草稿，随后依据该人工完成声明补齐逐帧确认，未生成、插值、移动或删改目标框。

- 连续420–469帧全部确认，38帧有框、12帧无鱼，共61个框。421–431帧原本没有框，依据用户已检查且全部鱼已框出的声明确认无鱼；420帧沿用发布者无鱼标注。
- 保留3帧未修改的发布者ROI；其余47帧记录为人工确认。450帧在发布者原有框外有用户新增框，此差异单独留档，原来的4张发布者像素GT不改。
- 用户完成声明、确认前草稿、保存版本和固定快照均保留。该状态表示用户完成了人工核对，不是助手另行逐帧鉴定了真值。
- 本次快照SHA256：`{protocol['annotation_snapshot_sha256']}`。人工框控制ROI位置，不生成新增像素分割GT；部署时不能依赖这些人工标签。

{link(OUT/'连续人工ROI核对/annotations.json','已保存人工核对文件')}；{link(RUN/'manual_roi_snapshot.json','实验固定快照')}；{link(OUT/'连续人工ROI核对/人工核对完成声明_20261015.json','人工完成声明与操作来源')}。

## 2. 四组执行条件

A/B沿用之前已完成的预测ROI×32/4帧结果；C/D使用同一份连续人工/发布者框×32/4帧。输入仍为已见训练视频video131的连续50帧。背景CRF42、0.5倍宽高，ROI为medium/P-only/GOP32；普通H.265仍允许B帧。

所有组按100,000B完整文件预算选CRF，误差≤2%，不用GT任务分数调参。语义消息在A/B/C/D中保持同一份旧模型预测，人工差异只改变ROI裁剪及其manifest元数据；元数据全部收费。C/D是人工参与的诊断条件，不代表自动部署能力，也不是严格性能上界。

## 3. 几何与实际文件开销

| 组 | 平均ROI画面面积 | 汇总GT鱼像素覆盖 | 含鱼GT帧覆盖命中 | ROI段数 | 同ROI CRF32完整B | 约100KB选定CRF | 总I/P/B帧 |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(geo_rows)}

人工ROI使短窗对现有3张含鱼GT的覆盖由1/3变成3/3，但这仍只是几何覆盖。相同CRF32时，人工短窗D为{d32:,}B、人工长窗C为{c32:,}B，短窗完整文件反而增加{100*(d32/c32-1):.2f}%。不能把区域面积下降直接当作传输字节减少；独立小流的I帧、视频头、容器和外层元数据都有代价。CRF相同也不保证不同尺寸/时序片段的失真相同。

## 4. 同预算恢复质量与任务表现

IoU、召回、精确率单位为%；PSNR单位dB。全局评价使用全部50帧RGB；GT鱼区、背景和任务准确率仅使用4张已发布像素标注，3张含鱼，共1425个鱼像素。新增人工框不拿来计算像素IoU。

| 方案 | 完整字节 | 鱼IoU | 鱼召回 | 鱼精确率 | GT鱼区PSNR | GT帧背景PSNR | 全局PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

原始输入固定分割器IoU26.196%、召回33.053%。所有压缩结果均由接收端真实恢复RGB后重新推理，不用收到的目标框或叠加图计分。

{chr(10).join(delta_text)}

| 输入/组 | TP鱼像素 | FP像素 | FN鱼像素 | 含鱼帧IoU等权均值 | 含鱼帧召回等权均值 | 含鱼GT帧任务命中 |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(task_rows)}

“任务命中”仅要求该含鱼标注帧至少有1个TP像素，不等于完整检测到鱼。IoU变化须结合TP、FP和召回解读，不能把减少背景误检等同于鱼本身保留更好。450帧人工框与发布者像素标签存在差异，本轮维持原评分口径，不能据此声称新增框有像素级真值支持。

## 5. 实际独立收发与成本

| 组 | 背景B | ROI合计B | 冻结语义B | manifest/ZIP B | 应用层实际发送B | 20kbps发送秒 | 恢复及展示生成秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(network_rows)}

C/D均经真实本机TCP限速发送到独立接收进程，收发SHA256一致，恢复50帧。收到的包、全部视频成员的CRC、长度、SHA256及读取路径已核验。读取防护是程序级审计，不是OS沙箱；没有跨机器或真实水声链路实验。

识别、背景和原始输入有缓存，候选编码耗时不能作为端到端速度。上表传输秒数不含发送端识别和编码，接收秒数包含PNG与预览/叠加图生成，也不能作为纯解码速度。50帧名义25fps视频仍需约40秒传输，属于批次系统。

## 6. 技术判断与后续边界

四组诊断现已完成，能分别观察ROI来源与组织方式的作用。人工ROI使现有稀疏GT覆盖齐全，说明预测和筛选遗漏确实限制了短窗；但人工ROI条件下短窗的同CRF文件更大，也说明小片段编码成本不能忽略。实际任务收益必须依据上表，不能只用覆盖率或鱼区PSNR代替。

人工长窗C相对H.265，GT鱼区PSNR提高{metrics['C']['gt_fish_rgb_psnr_db']-metrics['H265']['gt_fish_rgb_psnr_db']:.3f}dB、IoU增加{100*(metrics['C']['gt']['iou']-metrics['H265']['gt']['iou']):.3f}个百分点、召回增加{100*(metrics['C']['gt']['recall']-metrics['H265']['gt']['recall']):.3f}个百分点，但全局PSNR下降{metrics['H265']['global_rgb_psnr_db']-metrics['C']['global_rgb_psnr_db']:.3f}dB。这为“可靠ROI值得保护”提供单片段诊断信号，尚不能作为自动系统优于传统编码的证据。

人工短窗D的GT鱼区PSNR比预测短窗B提高{metrics['D']['gt_fish_rgb_psnr_db']-metrics['B']['gt_fish_rgb_psnr_db']:.3f}dB，鱼类IoU仍很低。D虽在2/3张含鱼GT上有TP，但435帧只有3个TP，465帧只有49个TP；C则仅在465帧命中，却有699个TP。因此“命中帧更多”也不能替代目标保留质量。组织、预算分配及固定分割器对压缩输入的响应都仍需分开核查，不能认定只要修好检测器就解决了全部问题。

当前应继续围绕“完整视频恢复＋关键目标保护”迭代。先检查人工ROI所揭示的可取得收益及代价，再选择自动ROI提取/筛选和区域组织的单一改动；不得把人工组当成自动系统成绩。不因为一次诊断追加新网络或重训全部模型。

正式有效性仍缺独立连续场景和多预算验证，固定评价模型的跨场景可靠性也有限。人工框来自单人核对，尚无第二标注者一致性检查；现有4帧像素GT不足以验证50帧全部目标。本轮不形成显著性或泛化结论。

阶段C继续遵循零更新一致性→少量多帧普通RD→完整GOP验证，尚未在本轮开展，不阻塞阶段D。下一次自动ROI改动要保留本轮A/B作为旧基线，重新记录完整文件及帧级漏保护，而非只展示漂亮截图。

## 7. 证据

- {link(SUP/'人工ROI冻结协议.json','人工ROI协议')}；{link(SUP/'四组与H265完整汇总.json','四组与H.265汇总')}；{link(SUP/'人工ROI交付核验.json','实际交付核验')}。
- {link(SUP/'原图与人工ROI任务评价.json','逐GT帧及连通域指标')}；{link(SUP/'人工ROI几何复核.json','人工ROI几何复核')}。
- {link(RUN/'C/selection.json','C码流选择与全部命令')}；{link(RUN/'D/selection.json','D码流选择与全部命令')}。
- {link(SCRIPT/'manual_controls.log','编码及传输日志')}。接收端RGB在各组`network/receiver/rgb`，真实包在`selection.json`指向的位置。
""",encoding="utf-8")
    (PROJECT/"docs/phase_d_manual_roi_control_20261015.md").write_text(report.read_text("utf-8"),encoding="utf-8")
    write_json(OUT/"当前实验状态.json",{"A":"completed","B":"completed","C":"completed after human review","D":"completed after human review","H265":"completed","report":str(report),"annotation_sha256":protocol["annotation_snapshot_sha256"],"new_training":False,"initial_report_preserved":True})
    archive=SUP/"阶段D_人工ROI四组对照_报告与证据.zip"
    files=[(report,report.name),(DIRECTION,DIRECTION.name),(first,"首轮预测ROI报告.md")]
    files.extend((p,p.name) for p in SUP.glob("*.json") if p.name!="证据包指纹.json")
    for name in ("manual_roi_snapshot.json","manual_protocol.json"):
        files.append((RUN/name,"annotations/"+name))
    for name in ("人工核对完成声明_20261015.json","用户确认前的目标框草稿_20261015.json","publisher_template.json"):
        files.append((OUT/"连续人工ROI核对"/name,"annotations/"+name))
    for g in ("A","B","C","D","H265"):
        files.append((Path(selected[g]["file"]),f"actual_transmitted/{g}_"+Path(selected[g]["file"]).name))
        for name in ("selection.json","search.json","stream_audit.json"):
            files.append((RUN/g/name,f"evidence/{g}/"+name))
        for name in ("end_to_end.json","transfer.csv"):
            files.append((RUN/g/"network"/name,f"evidence/{g}/network/"+name))
        for name in ("decode.json","receiver_read_audit.json"):
            files.append((RUN/g/"network/receiver"/name,f"evidence/{g}/receiver/"+name))
    files.extend((p,"code/"+p.name) for p in SCRIPT.iterdir() if p.suffix in {".py",".html"})
    files.append((SCRIPT/"manual_controls.log","logs/manual_controls.log"))
    files.append((OUT/"人工ROI已确认保存.jpg","人工ROI已确认保存.jpg"))
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("阅读说明.txt","人工ROI为用户核对后的诊断条件，原始像素GT和评价模型保持不变。本包不包含完整原始数据与模型；代码依赖本机既有工程路径。实际传输文件与日志完整保存。\n")
        for path,name in files:
            if path.is_file():z.write(path,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.startswith("actual_transmitted/")])==5
    write_json(SUP/"证据包指纹.json",{"file":str(archive),"bytes":archive.stat().st_size,"sha256":sha256(archive),"zip_crc_valid":True})
    print(json.dumps({"report":str(report),"archive":str(archive),"bytes":archive.stat().st_size},ensure_ascii=False))

if __name__=="__main__":main()
