"""Build a verified document/path handoff; no inference, encoding, downloads or transfers."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

W=Path(r"C:\Users\lenovo\Documents\ChatGPT\语义通信")
P=Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
OLD=Path(r"C:\Users\lenovo\Documents\Codex\2026-10-12\wo-d")
AB=Path(r"C:\Users\lenovo\Documents\Codex\2026-10-12\segformer-c-users-lenovo-desktop-works\outputs")
OUT=W/"outputs/session_handoff_20261016"
MAIN=W/"会话交接_工程现况与后续方向_2026-10-16.md"


def read(path):return json.loads(Path(path).read_text("utf-8"))
def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()
def link(path):return str(path).replace("\\","/")
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n","utf-8")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reports=[
        ("阶段 A：工程验证",AB/"阶段A_工程验证记录_2026-10-13.md",P/"docs/phase_a_results_20261013.md"),
        ("阶段 B：语义评价与 ROI 准备",AB/"阶段B_语义评价与ROI准备_2026-10-13.md",P/"docs/phase_b_results_20261013.md"),
        ("阶段 C：小规模 ROI 训练",OLD/"outputs/semantic_roi_phase_c/阶段C_小规模ROI对照实测_2026-10-13.md",P/"docs/phase_c_results_20261013.md"),
        ("阶段 D：最小通信原型",OLD/"outputs/semantic_communication_phase_d/阶段D_最小语义通信系统与收发实测_2026-10-14.md",P/"docs/phase_d_results_20261014.md"),
        ("阶段 D：窗口对照",W/"outputs/roi_control_20261015/阶段D_ROI窗口对照实测_2026-10-15.md",P/"docs/phase_d_roi_control_20261015.md"),
        ("阶段 D：连续人工 ROI 四组对照",W/"outputs/roi_control_20261015/人工ROI补充/阶段D_连续人工ROI四组对照补充_2026-10-15.md",P/"docs/phase_d_manual_roi_control_20261015.md"),
        ("阶段 E：自动双区域",W/"outputs/semantic_roi_phase_e/阶段E_自动双区域ROI视频与收发实测_2026-10-15.md",P/"docs/phase_e_results_20261015.md"),
        ("E 执行约定（历史协议）",W/"阶段E_执行约定_2026-10-15.md",P/"docs/phase_e_protocol_20261015.md"),
        ("E 后现况评价（历史方向）",W/"阶段E后_现况评价与后续方向_2026-10-15.md",P/"docs/phase_e_review_and_next_direction_20261015.md"),
        ("E 后联合意见",W/"阶段E后_联合意见与下一步共识_2026-10-15.md",P/"docs/phase_e_consensus_20261015.md"),
        ("D 后现状总结（历史方向）",W/"现状总结与后续方向_2026-10-15.md",P/"docs/current_direction_20261015.md"),
        ("F 步骤 1：漏保护与概率",W/"outputs/semantic_roi_phase_f/step1_probability_diagnosis/阶段F_步骤1_漏保护与概率诊断_2026-10-15.md",P/"docs/phase_f_step1_diagnosis_20261015.md"),
        ("F 步骤 2：61 框与码流开销",W/"outputs/semantic_roi_phase_f/step2_box_coverage/阶段F_步骤2_人工框覆盖与ROI开销核对_2026-10-15.md",P/"docs/phase_f_step2_coverage_20261015.md"),
        ("F 步骤 3：冻结方案新保留视频",W/"outputs/semantic_roi_phase_f/step3_frozen_external/阶段F_步骤3_冻结方案新保留视频对照_2026-10-15.md",P/"docs/phase_f_step3_holdout_20261015.md"),
        ("F 首轮总结（10 月 15 日历史截点）",W/"阶段F_首轮总结与后续方向_2026-10-15.md",P/"docs/phase_f_summary_20261015.md"),
        ("10 月 16 日：概率诊断",W/"outputs/semantic_review_20261016/2026-10-16_概率诊断与人工核对准备.md",P/"docs/probability_review_20261016.md"),
        ("10 月 16 日：单人可辨识性",W/"outputs/semantic_review_20261016/2026-10-16_单人可辨识性核对与方向更新.md",P/"docs/human_review_20261016.md"),
        ("10 月 16 日：预算分配（最新实验）",W/"outputs/semantic_budget_20261016/2026-10-16_固定ROI预算分配对照与判断.md",P/"docs/budget_allocation_20261016.md"),
    ]
    c_source,c_copy=reports[2][1:]
    if not c_copy.exists():shutil.copyfile(c_source,c_copy)
    assert sha(c_source)==sha(c_copy)
    catalog=[]
    for title,original,engineering in reports:
        assert original.is_file() and engineering.is_file(), title
        catalog.append({"title":title,"original":str(original),"original_bytes":original.stat().st_size,
                        "original_sha256":sha(original),"engineering_copy":str(engineering),
                        "engineering_copy_sha256":sha(engineering)})
    report_table="\n".join(f"| {title} | [原报告]({link(original)}) | [工程 docs]({link(engineering)}) |"
                           for title,original,engineering in reports)
    data=read(P/"data/phase_c_20261013/manifest.json")
    dataset_table="\n".join(f"| {v['video']} | {v['official_split']} | {v['experiment_role']} | {v['frames']} | "
                            f"{v['annotated_frames']} | {v['annotated_frames_with_fish']} |" for v in data["videos"])
    assets=[
        ("官方 MLVC-S",P/"checkpoints/mlvc-s-psnr-v1.ckpt"),
        ("DMCI I 帧镜像",P/"checkpoints/DMCI-6.0-mirror.pth.tar"),
        ("旧 SUIM 底模（工程外依赖）",Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-training\runs\suim_segformer_b0_20260915\segmentation_best.pt")),
        ("冻结鱼类适配权重",P/"runs/phase_c_20261013/segformer_adaptation_preserve8/best.pt"),
        *[(name+" 第512步",P/f"runs/phase_c_20261013/training/{name}/checkpoint_512.ckpt") for name in ("M_RD","M_ROI4","M_ROI16")],
    ]
    model_catalog=[]
    for title,path in assets:
        assert path.is_file(),path
        model_catalog.append({"title":title,"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path)})
    assert model_catalog[2]["sha256"]=="ff00cb439a4af290dc3efde8ae76d4acfad797784483b669ab6db79aaf5b1476"
    assert model_catalog[3]["sha256"]=="10e4783330112982e941b021afa9603313429d4f30e871d1ddfa059a258a6ca9"
    model_table="\n".join(f"| {r['title']} | [文件]({link(r['path'])}) | `{r['sha256']}` |" for r in model_catalog)
    body=Path(__file__).with_name("handoff_body.md").read_text("utf-8")
    body=body.replace("@REPORT_TABLE@",report_table).replace("@DATASET_TABLE@",dataset_table).replace("@MODEL_TABLE@",model_table)
    MAIN.write_text(body,"utf-8")
    (P/"docs/handoff_20261016.md").write_text(body,"utf-8")
    prompt=f"""# 新会话启动说明

可以把下面整段复制到新会话，也可以直接让新会话读取交接文件。

```text
继续我本机的水下视频语义通信项目。先完整读取交接文档：
{MAIN}

工作区：{W}
主工程：{P}

导师要求是接收端恢复可观看视频，语义用于保护鱼类、潜水员等关键目标。不要把信息模式替代视频目标。
截至2026-10-16，阶段F首轮、video6概率诊断、1名观察者37项核对、两档固定ROI预算对照均已完成。不是让你重新做这些工作，也暂不新增阶段名称。
先简要确认你理解当前结果和目录。等我明确说继续，再按交接文档的小范围失真来源诊断推进：固定现有模型、0.9阈值和既有材料，检查颜色转换/YUV420采样/ROI合成对模型响应的影响。先固定协议，再运行。
保持每天小步推进；先做简单可用的视频系统，不急着换模型、大规模下载或重训。今天原E和新预算配置的任务指标都不代表可靠目标保护已成功；单人评分只适用于旧E/H.265，新配置尚无人工评分。
```
"""
    prompt_path=W/"新会话启动说明_2026-10-16.md"
    prompt_path.write_text(prompt,"utf-8")
    directory_catalog=[W,P,OLD,AB.parent,P/"data",P/"runs",P/"semantic_mvp",P/"adapters",
                       W/"work/continuation_20261016",OLD/"work/phase_c",
                       P/"runs/semantic_mvp_probability_review_20261016",P/"runs/semantic_mvp_budget_20261016"]
    assert all(path.is_dir() for path in directory_catalog)
    answer_folder=P/"runs/semantic_mvp_probability_review_20261016/human_review/answers"
    actual=[read(path) for path in answer_folder.glob("*.json")]
    assert len(actual)==1 and not actual[0]["qa_mode"] and len(actual[0]["answers"])==37
    manual=read(P/"runs/semantic_mvp_roi_control_20261015/manual_roi_snapshot.json")
    confirmed=sum(r["status"] in {"human_reviewed","publisher_verified"} for r in manual["frames"])
    assert confirmed==50
    save(OUT/"工程与报告定位清单.json",{
        "date":"2026-10-16","primary_handoff":str(MAIN),"primary_handoff_sha256":sha(MAIN),
        "project_copy":str(P/"docs/handoff_20261016.md"),"new_conversation_prompt":str(prompt_path),
        "report_catalog":catalog,"model_catalog":model_catalog,
        "directories":[str(path) for path in directory_catalog],"actual_human_observers":1,"actual_confirmed_answers":37,
        "manual_roi_frames_confirmed":confirmed,"stage_c_report_copy_added":str(c_copy),
        "old_files_moved_or_deleted":False,"new_experiments_in_this_handoff":False,
        "generated_utc":datetime.now(timezone.utc).isoformat()})
    links=re.findall(r"\]\((C:/[^)]+)\)",body)
    assert all(Path(path).exists() for path in links),[path for path in links if not Path(path).exists()]
    assert MAIN.read_bytes()==(P/"docs/handoff_20261016.md").read_bytes()
    # Only the current progress pointer is updated; old frozen F status/protocols remain untouched.
    state_path=W/"outputs/semantic_review_20261016/本次推进状态.json"
    state=read(state_path)
    state.update({"handoff_document":str(MAIN),"handoff_ready":True,"user_requested_stop_after_summary":True,
                  "reason_for_next_step":"allocation tradeoffs measured, fixed-model response still fails; isolate transformation steps",
                  "work_state":"stopped_after_requested_summary; await new user instruction"})
    save(state_path,state)
    save(OUT/"交接文档核验.json",{
        "report_sources_verified":len(catalog),"report_engineering_copies_verified":len(catalog),
        "model_paths_and_hashes_verified":len(model_catalog),"local_links_verified":len(links),
        "primary_and_project_handoff_equal":True,"actual_human_answers_unchanged":True,
        "frozen_historical_reports_and_protocols_not_edited":True,"new_encodes_training_transfers":0})
    print(json.dumps({"handoff":str(MAIN),"prompt":str(prompt_path),"reports":len(catalog),
                      "models":len(model_catalog),"verified_links":len(links)},ensure_ascii=False))


if __name__=="__main__":main()
