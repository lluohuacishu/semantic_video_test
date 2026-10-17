"""Short measured MVP report and a reproducible local delivery bundle."""
import gzip
import json
from pathlib import Path
import shutil
import sys
import zipfile

PROJECT=Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
WORKSPACE=Path(r"C:\Users\lenovo\Documents\Codex\2026-10-12\wo-d")
OUT=WORKSPACE/"outputs/semantic_communication_mvp"
sys.path.insert(0,str(PROJECT/"semantic_mvp"))
from runtime import write_json,sha256
from system import make_page
from PIL import Image

def read(path):return json.loads(Path(path).read_text("utf-8"))
def link(path,label):return f"[{label}]({str(Path(path).resolve()).replace(chr(92),'/')})"

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runs={"fish":PROJECT/"runs/semantic_mvp_20261014_fish","diver":PROJECT/"runs/semantic_mvp_20261014_diver","information":PROJECT/"runs/semantic_mvp_20261014_fish_information"}
    ui=sorted(p for p in (PROJECT/"runs").glob("semantic_mvp_ui_*") if p.is_dir() and (p/"end_to_end.json").exists())
    for root in [*runs.values(),*ui]:
        receiver=root/"receiver"
        semantics=read(receiver/"target_information.json")
        best=max(range(len(semantics["frames"])),key=lambda i:semantics["frames"][i]["pixels"])
        Image.open(receiver/"rgb"/f"{best:05d}.png").save(receiver/"reconstructed_poster.jpg",quality=93)
        make_page(receiver,read(receiver/"decode.json"),semantics)
    ends={name:read(root/"end_to_end.json") for name,root in runs.items()}
    validations={name:read(runs[name]/"validation/results.json") for name in ["fish","diver"]}
    checks=read(PROJECT/"runs/semantic_mvp_20261014_protocol_checks/checks.json")
    assert checks["success"]
    audit=[]
    for name,root in {**runs,**{p.name:p for p in ui}}.items():
        result=read(root/"end_to_end.json")
        received=read(root/"receiver/decode.json")
        assert result["success"] and result["separate_sender_receiver_processes"]
        assert result["file_bytes"]==(root/"receiver/message.scv").stat().st_size
        assert sha256(root/"receiver/message.scv")==received["file_sha256"]
        with zipfile.ZipFile(root/"receiver/message.scv") as archive:
            assert json.loads(gzip.decompress(archive.read("semantics.json.gz")))==read(root/"receiver/target_information.json")
        audit.append({"name":name,**result,"semantic_metadata_equal":True})
    fish={x["name"]:x for x in validations["fish"]["rows"]}
    diver={x["name"]:x for x in validations["diver"]["rows"]}
    base=validations["fish"]["baseline"]
    table="\n".join(f"| {label} | {ends[name]['file_bytes']:,} | {ends[name]['application_transmit_bytes']:,} | {ends[name]['transfer_seconds']:.3f} | {ends[name]['frames']} |" for name,label in [("information","鱼类：仅目标信息"),("fish","鱼类：视频＋目标"),("diver","潜水员：视频＋目标")])
    gt_table="\n".join(f"| {label} | {fish[key]['file_bytes'] or '—'} | {fish[key]['fish_gt']['iou']*100:.2f}% | {fish[key]['fish_gt']['recall']*100:.2f}% | {format(fish[key]['global_rgb_psnr_db'],'.2f') if fish[key]['global_rgb_psnr_db'] is not None else '—'} |" for key,label in [("original","原始输入"),("context_only","仅低清背景（不同预算）"),("semantic_video","语义分层完整包"),("matched_offline_h265","H.265 离线 B4，近似同预算")])
    report=f'''# 最小语义通信系统实测 · 2026-10-14

最低工程目标已经完成：本机可从视频提取鱼类或潜水员的目标信息，生成真实通信包，通过限速 TCP 发送给独立进程，接收端恢复目标信息；选择视频模式时还能恢复画面。提供双击启动界面，可输入自己的视频路径，不需要重新训练。

## 使用

双击 {link(PROJECT/"启动语义通信系统.cmd","启动语义通信系统.cmd")}，打开本机界面 `http://127.0.0.1:8776/`。先选择“只传目标信息”，使用默认示例即可。选择“视频与目标信息”可以查看完整恢复画面，但低速传输耗时更长。关闭启动服务的窗口可停止界面服务；本次已启动的后台服务可继续使用。

默认鱼类输入是用户下载的 video131 中真实连续的 00420–00469 帧，未移动或改写原数据。25 fps 是名义播放帧率，拍摄时间戳尚未确认。潜水员示例是已有的 720p、5 fps、50 帧视频，本身已经过重建，且没有人工 GT。

## 系统行为

- 信息模式：发送逐帧类别、连接区域位置、置信度和目标像素面积；接收端按坐标绘制目标分布，不恢复原始视频。
- 视频模式：在信息之外，发送低分辨率背景视频和高质量目标区域视频。目标区域由发送时的模型预测决定，每32帧取一个区域并外扩16像素。接收端独立解码并合成。
- 传输包包含协议头、语义信息、全部视频码流与各成员哈希。接收端无需原视频、GT、分割模型或发送端缓存。绿色框是收到的发送端预测，不计作接收端识别准确率。

本版本采用固定 SegFormer 与标准 H.265，暂未使用阶段 C 中发生长 GOP 退化的微调编码器。它是“显式语义消息＋可选 ROI 视频”的数字通信原型。神经编解码器仍可作为下一版模块加入；本次没有训练新模型，也没有声称实现端到端学习的联合信源信道编码或真实水声通信。

## 真实收发测量

| 模式 | 完整包 B | 发送方向应用层 B | 传输秒 | 帧记录数 |
| --- | ---: | ---: | ---: | ---: |
{table}

三项均使用真实 TCP 回环连接、20 kbps 十进制限速和不同发送/接收进程。收发 SHA256 相同，语义记录逐项一致。信息模式把已提取的50帧目标信息压缩到2,011 B；其0.844秒是传输耗时，不含此前识别耗时。鱼类视频示例提取与编码约39.23秒，随后传输133.48秒，因此当前是批次原型，不能声称20 kbps实时传送该视频。

最终界面另外从原始50帧完整执行了一次轻量模式，语义提取与打包约16.54秒，传输0.844秒，独立接收核验通过；没有依靠预先生成的目标消息跳过识别。首次视频模式的39.23秒还包含视频编码，两次耗时不作为并行负载下的严格速度比较。

应用层计量包括通信包及传输头，不含 TCP/IP、链路层包头及反向确认。接收端生成的 MP4 预览和日志不参与传输字节。信息模式只保留模型已识别的目标，模型漏检不会被通信过程修复。

独立读取审计与解码输入命令已保存。审计约束 Python 的外部媒体/模型读取，FFmpeg 只接收接收目录内的明确文件参数；它不是操作系统沙箱。损坏成员哈希、截断传输均被拒绝；读取发送端原图被拒绝；无目标的消息可正常恢复为“未检出”。

## 恢复画面的任务核验

固定评价模型对恢复画面重新推理，不使用传来的目标框计分。H.265 基线允许4个B帧，质量参数只按完整文件字节选择，未看 IoU 或 PSNR。鱼类文件差为{base['relative_complete_file_difference']*100:.3f}%。

鱼类片段共有4张发布者 GT（其中3张有鱼，共1,425个鱼像素）：

| 输入 | 完整文件 B | 鱼 GT IoU | 鱼 GT 召回 | 全局 PSNR dB |
| --- | ---: | ---: | ---: | ---: |
{gt_table}

这个诊断片段中，分层视频的鱼 IoU 比近似同预算 H.265 高{(fish['semantic_video']['fish_gt']['iou']-fish['matched_offline_h265']['fish_gt']['iou'])*100:.2f}个百分点，全局 PSNR 低{fish['matched_offline_h265']['global_rgb_psnr_db']-fish['semantic_video']['global_rgb_psnr_db']:.2f}dB。video131 已参与鱼类模型适配，仅4张稀疏 GT，不能作为独立泛化结论。分层恢复的 IoU 仍低于原始输入；误检明显，目标矩形平均覆盖64.37%画面，导致 ROI 视频较大。

潜水员片段没有 GT，只报告原图预测一致性：分层视频 IoU {diver['semantic_video']['source_prediction_consistency']['iou']*100:.2f}%，近似同预算 H.265 为{diver['matched_offline_h265']['source_prediction_consistency']['iou']*100:.2f}%。分层方案没有在该指标上胜过传统编码，不能把所有样例写成收益。

## 下一步

先让用户通过界面跑自己的短视频，确认关心的是目标位置汇报还是看清目标画面。随后优先改善独立鱼类视频上的漏检/误检，并缩小 ROI 保护范围、减少视频传输量。满足这些实际需求后，再把稳定的预训练 MLVC 接入视频层；多帧训练、可微任务损失和新模型均留在后续，不作为当前原型可用的前提。硬件接入时再验证实际水声链路。

## 证据与复现

{link(PROJECT/"semantic_mvp/README.md","本机操作说明与命令")}；{link(runs['fish']/"validation/results.json","鱼类任务实测JSON")}；{link(runs['diver']/"validation/results.json","潜水员实测JSON")}；{link(PROJECT/"runs/semantic_mvp_20261014_protocol_checks/checks.json","协议异常核验")}。

H.265 封装与 CRF 参数来自本机 FFmpeg；接口说明见 [FFmpeg 官方文档](https://ffmpeg.org/ffmpeg-all.html#libx265)。固定鱼类/潜水员权重 SHA256、全部配置、编码和独立解码命令保留在各次通信的 JSON 中。
'''
    (OUT/"最小语义通信系统_实测说明_2026-10-14.md").write_text(report,encoding="utf-8")
    readme='''# 本机语义通信原型

双击工程根目录的“启动语义通信系统.cmd”，打开 http://127.0.0.1:8776/ 。默认先传目标信息，可切换为视频与目标信息。填入自己的本机视频路径后点击“开始发送并接收”。当前按批次处理，默认前50帧，示例鱼类使用video131真实连续420–469帧。结果保存到runs/semantic_mvp_ui_日期_编号/receiver。

信息模式传类别、位置、置信度、像素面积，接收端展示目标分布；视频模式另传低清背景与高清目标区域，恢复视频。目标框是发送端预测，连接区域数量不等于真实物体数量。鱼类模型泛化仍有限。绿色框完整收到不代表识别正确。

本机环境已安装，不需要下载或训练：发送端复用工程.venv、原SUIM SegFormer以及阶段C固定鱼类适配权重。接收端只需Python、NumPy、Pillow、FFmpeg，不需要GPU或模型。FFmpeg位置可用FFMPEG_PATH环境变量指定。

```powershell
cd C:\\Users\\lenovo\\Desktop\\works\\work\\semantic-video-mlvc
# 默认鱼类轻量模式；运行完打开本地结果页
& .\\.venv\\Scripts\\python.exe semantic_mvp/system.py demo --target fish --mode information --open
# 已有潜水员示例，传视频与目标
& .\\.venv\\Scripts\\python.exe semantic_mvp/system.py demo --target diver --mode video --open
# 自己的视频；新输出目录必须不存在
& .\\.venv\\Scripts\\python.exe semantic_mvp/system.py demo --input 'C:\\视频\\水下.mp4' --target fish --mode video --frames 50 --kbps 20 --open
# 只从收到的通信包独立解码，不需要原图或模型
& .\\.venv\\Scripts\\python.exe semantic_mvp/system.py decode --input 'C:\\接收\\message.scv' --output 'C:\\接收\\恢复结果'
```

两端分开使用：接收端运行 `system.py listen --host 127.0.0.1 --port 8772 --output 新接收目录`；发送端先 `system.py encode --input 视频 --output 新发送目录 --target fish --mode video`，再 `system.py send --input 发送目录/message.scv --host 127.0.0.1 --port 8772 --kbps 20`。改为局域网时显式指定对应地址；当前只实测本机回环，不代表已完成跨机器或水声硬件测试。

通信包为ZIP_STORED格式.scv，包含manifest.json、gzip目标消息及可选H.265码流。全部封装、目标消息和高清区域都计算在完整文件大小中。TCP发送统计还包括发送方向的应用层传输头，不包括反向确认及TCP/IP链路开销。接收端只读取自己的目录，保存逐帧RGB哈希和读取审计；这是程序约束，不是OS沙箱。

每次输出：sender/message.scv、sender/encode.json、transfer.json、end_to_end.json，以及receiver/查看结果.html、视频预览、target_information.json、decode.json、receiver_read_audit.json。发送端私有源图和原始预测掩码不进入通信包。改ROI_CRf、背景比例或片段长度可在命令行操作，目前默认背景0.5/CRF42、ROI CRF32、每32帧一个矩形、外扩16像素。

已经实测：完整鱼类/潜水员视频各50帧，冷接收端独立恢复；鱼类消息2,011B在20kbps下约0.844秒传输。该时间不含识别模型耗时。允许B帧的近似同预算H.265对照与正负结果保留，详见Codex工作区outputs/semantic_communication_mvp。
'''
    (PROJECT/"semantic_mvp/README.md").write_text(readme,encoding="utf-8")
    write_json(OUT/"收发与协议核验.json",{"passed":True,"runs":audit,"protocol_checks":checks,"quality_validation_runs":[str(runs[n]/"validation/results.json") for n in validations]})
    for name in ["fish","diver"]:
        shutil.copy2(runs[name]/"validation/恢复画面对比.jpg",OUT/f"{name}_恢复画面对比.jpg")
    # Bundle code, exact sent files, configurations, results and previews; keep raw
    # training data and existing shared model weights in their original locations.
    files=[]
    files += [(p,f"工程/semantic_mvp/{p.name}") for p in (PROJECT/"semantic_mvp").iterdir() if p.is_file()]
    for name in ["启动语义通信系统.cmd","run_semantic_mvp.cmd","run_semantic_information.cmd"]:
        files.append((PROJECT/name,f"工程/{name}"))
    files.append((PROJECT/"README.md","工程/README.md"))
    files.append((PROJECT/"runs/phase_c_20261013/fixed_teacher.json","配置/固定鱼类模型.json"))
    for name,root in runs.items():
        for rel in ["end_to_end.json","transfer.json","transfer.csv","receiver_process.log","receiver/message.scv","receiver/decode.json","receiver/receiver_read_audit.json","receiver/target_information.json","receiver/received_manifest.json","receiver/transfer.json","receiver/reconstructed.mp4","receiver/semantic_overlay.mp4","receiver/查看结果.html","receiver/poster.jpg","receiver/reconstructed_poster.jpg"]:
            files.append((root/rel,f"实测/{name}/{rel}"))
        for rel in ["sender/encode.json","sender/source_audit.json","sender/manifest.json","validation/results.json","validation/matched_h265/selection.json","validation/恢复画面对比.jpg"]:
            if (root/rel).exists():files.append((root/rel,f"实测/{name}/{rel}"))
        if name in validations:
            selected=Path(validations[name]["baseline"]["file"])
            files.append((selected,f"实测/{name}/validation/近似同预算H265_B4.mp4"))
    for root in ui:
        for rel in ["end_to_end.json","sender/encode.json","sender/source_audit.json","receiver/message.scv","receiver/decode.json","receiver/receiver_read_audit.json","receiver/target_information.json","transfer.json","receiver/查看结果.html","receiver/semantic_overlay.mp4","receiver/poster.jpg"]:
            files.append((root/rel,f"实测/界面完整流程/{root.name}/{rel}"))
        logfile=root.with_suffix(".log")
        if logfile.exists():files.append((logfile,f"实测/日志/{logfile.name}"))
    files += [(p,"实测/日志/"+p.name) for p in (WORKSPACE/"work").glob("semantic_mvp*.log")]
    files.append((PROJECT/"runs/semantic_mvp_20261014_protocol_checks/checks.json","实测/协议异常核验.json"))
    files += [(p,"报告/"+p.name) for p in OUT.iterdir() if p.is_file() and p.suffix in {".md",".json",".jpg"}]
    archive=OUT/"语义通信原型_代码与实测.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for path,name in files:z.write(path,name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise AssertionError("Delivery ZIP integrity failed")
    print(json.dumps({"report":str(OUT/"最小语义通信系统_实测说明_2026-10-14.md"),"zip_bytes":archive.stat().st_size,"verified_runs":len(audit)},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
