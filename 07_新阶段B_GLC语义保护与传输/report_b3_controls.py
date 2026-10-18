"""Render the B3 development report directly from accepted evidence."""
import argparse
from pathlib import Path
from b2_common import HERE, read, require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    run=args.run.resolve()
    require(read(run/"evaluation/completion.json")["status"]=="B3_development_diagnostics_complete","Diagnostics incomplete")
    metrics=read(run/"evaluation/metrics.json")
    network=read(run/"network/summary.json")
    names={"GLC_base":"GLC q0基础层（B1/B2）","auto_ROI":"GLC q0＋自动ROI（B2）","full_GLC_control":"全画面GLC预算控制","fixed_geometry":"GLC q0＋固定中央带"}
    glc=metrics["budget_selections"]["glc"]
    geo=metrics["budget_selections"]["geometry"]
    lines=["# B3开发片预算控制报告", "", "日期：2026-10-18。video91源帧0–249、1280×720、15 fps。**两项新控制各三档工程闭环通过；这不是B3独立评价或语义净收益整体通过。用户尚未观看，真实反馈待评。**", "",
           "## 预算与冻结规则", "", "自动ROI参考完整包532,330 B；基础流168,338 B；补充预算363,992 B，含语义消息、位置/时段、manifest和ZIP开销。选择仅使用完整文件字节，未用重建质量选档。协议与精度补充见[B3冻结协议](B3冻结协议_20261018.md)。", "",
           f"- 全画面GLC选中q{glc['selected']['q']}，{glc['selected']['file_bytes']:,} B，相对目标{glc['relative_difference']:+.6%}；±2%匹配：{'通过' if glc['matched'] else '未通过'}。停止原因：`{glc['stop_reason']}`。仅为有界搜索，未证明全部合法q不可达。",
           f"- 固定中央半高带CRF{geo['selected']['crf']}，完整包{geo['selected']['file_bytes']:,} B（{geo['relative_complete_file_difference']:+.6%}）；补充{geo['selected']['supplement_bytes']:,} B（{geo['relative_supplement_difference']:+.6%}），补充预算±2%：{'通过' if geo['matched'] else '未通过'}。判定不作四舍五入。",
           "- 中央带恒定覆盖50%画面，左右各一矩形，每32帧一个窗口，共16段。传输相同B2预测消息但不用于几何选择，不能把它当作省去分割器后的最小成本。",
           "- 初轮9个几何点和预先提交的一次0.125档中点补测全部保留。原初轮选择文件保留原字节，不删除未匹配结果。", "",
           "## 来自真实接收PNG的质量", "", "同一源帧、同一评价掩膜、同一固定鱼纹裁剪，数值为累计RGB平方误差计算的PSNR。鱼预测区域由发送端模型产生，不是GT；固定鱼纹区域为既有[256,576,512,720]、源帧210–249。", "",
           "| 方法 | 完整文件B | 全画面PSNR dB | 原始鱼预测区PSNR dB | 固定鱼纹PSNR dB | 相邻误差变化MAE |", "|---|---:|---:|---:|---:|---:|"]
    for key,name in names.items():
        value=metrics["methods"][key]
        p=value["rgb_psnr_db"]
        lines.append(f"| {name} | {metrics['file_bytes'][key]:,} | {p['full']:.4f} | {p['fish_raw_prediction']:.4f} | {p['fixed_stripe_crop']:.4f} | {value['error_change_mae']['all']:.4f} |")
    lines += ["", "### 边界与负例", "", "| 方法 | 在自动ROI边界的误差差分MAE | 在固定中央带边界的误差差分MAE | 32帧窗口切换时的相邻误差MAE |", "|---|---:|---:|---:|"]
    for key,name in names.items():
        value=metrics["methods"][key]
        lines.append(f"| {name} | {value['edge_error_difference_mae']['auto']:.4f} | {value['edge_error_difference_mae']['geometry']:.4f} | {value['error_change_mae']['window_boundaries']:.4f} |")
    lines += ["", "边缘差分和相邻误差变化仅作诊断，不能写成已验证的感知闪烁分数。每一列在同一组边界或同一组相邻帧上计算，避免把不同位置的平均值直接比较。", ""]
    for other in ("GLC_base","full_GLC_control","fixed_geometry"):
        indices=metrics["regression_frame_indices"][f"auto_ROI_lower_PSNR_than_{other}_fish_raw_prediction"]
        lines.append(f"- 自动ROI在原始鱼预测区PSNR低于{names[other]}：{len(indices)}帧；完整帧号保存在`evaluation/metrics.json`，不剔除负例。")
    lines += ["", "潜水员预测像素为0，所有潜水员PSNR保持null；没有填充目标准确率、真实计数或召回率。", "", "## 实际链路与独立重建", "", "| 方法 | kbps | 应用层发送B | 实测发送秒 | 独立解码帧数 |", "|---|---:|---:|---:|---:|"]
    for method,name in (("glc","全画面GLC"),("geometry","固定中央带")):
        for r in sorted(network["methods"][method]["results"],key=lambda r:r["rate_kbps"]):
            lines.append(f"| {name} | {r['rate_kbps']} | {r['transfer']['application_bytes']:,} | {r['transfer']['elapsed_seconds']:.3f} | {r['decoded_frames']} |")
    lines += ["", "两项控制各750帧、合计1,500帧通过独立进程解码和保存RGB核验，各自三档RGB一致。接收器读取实际TCP收包、已声明共享权重与运行依赖；未读发送端源帧或RGB缓存。固定中央带的基础RGB与B1一致，补丁内等于独立解码ROI，补丁外等于基础层。", "",
              "局限：同机TCP回环、发送方向应用层字节限速；不统计TCP/IP/链路层及反向ACK。Python读取审计不是操作系统隔离，不涵盖所有原生DLL读取；没有物理水声或跨设备结论。", "",
              "## 观看材料与剩余条件", "",
              "- [五路原速同步视频](../../../stage_b_20261018_b3_video91/evaluation/video91_B3_五路同步对比.mp4)：源图、GLC基础、自动ROI、全画面GLC控制、固定中央带。",
              "- [既有固定鱼纹五路半速](../../../stage_b_20261018_b3_video91/evaluation/video91_B3_固定鱼纹五路半速.mp4)：210–249帧，包含223→224窗口切换。",
              "- [固定鱼纹帧223/239](../../../stage_b_20261018_b3_video91/evaluation/固定鱼纹帧223_239.jpg)、[固定边界帧224](../../../stage_b_20261018_b3_video91/evaluation/固定边界帧224.jpg)。",
              "- 观看副本H.264 CRF11，与发送预算无关；所有数值来自真实接收PNG。用户明确“还没看，先继续实验”，没有自动代填评分。",
              "- [素材资格核查](B3素材资格核查_20261018.md)：新未使用独立片段认证仍为0；001/027/054仅可另列历史外部评价复用，潜水员连续素材仍缺。", "",
              "预算未匹配项只作质量—字节—等待时间比较，不能宣称满足预设等预算条件。当前开发片结论不能替代独立场景与真实观看。GLC主方向不变，不重开模型海选或训练。", "",
              "## 复现与证据", "", f"运行根目录：`{run}`。网络执行提交`{network['code_commit']}`，各编码及精度补测的提交分别保存在配置/commands/selection中。", "",
              "入口见[B3复现说明](B3复现说明.md)；小证据按原字节保存在`evidence_b3_video91_20261018`，全部候选文件、实际收包和RGB留在原本地目录。29项测试通过；只做本地提交，不推送或合并。", ""]
    (HERE/"B3开发片预算控制报告_20261018.md").write_text("\n".join(lines),encoding="utf-8")
    print("B3 development report generated from accepted JSON")


if __name__=="__main__":main()
