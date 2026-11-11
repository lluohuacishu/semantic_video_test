"""Archive the bounded mixed-boundary experiment without promoting its candidates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from coral_reference_sidecar import file_sha, load_json, save_json
from run_support_factorial import timing


def release(workspace, run):
    stage = Path(__file__).resolve().parent
    evidence = stage/'evidence_20261111/mixed_boundary'
    evidence.mkdir(exist_ok=False)
    view = workspace/'05_成果与观看/GLC_混合边界探针_20261111'
    view.mkdir(exist_ok=True)
    final = load_json(run/'final_summary.json')
    assert final['complete'] and load_json(run/'verification.json')['complete']
    review = run/'visual_review.json'
    if not review.exists():
        save_json(review, dict(date='2026-11-11',status='pending',cases_reviewed=[],quote=None,
            shown_comparison='video1 left K, right M12 full frame at original speed',
            decision='Keep K as default. No claim of visible improvement or acceptance until explicit feedback.'))
    diagnosis = {}
    for case, result in final['cases'].items():
        scope = 'coral' if case == 'video1' else 'full'
        rows = list(csv.DictReader((run/case/'evaluation/temporal.csv').open(encoding='utf-8',newline='')))
        groups = {}
        for group in ('refresh_with_exit','group_update','group_interior'):
            groups[group] = {}
            for name in ('K','M03','M12'):
                rr = [r for r in rows if r['scope']==scope and r['method']==name and timing(int(r['frame']))==group]
                groups[group][name] = sum(float(r['coarse_sum']) for r in rr)/sum(int(r['pixels']) for r in rr)
        scores = result['scores'][scope]
        diagnosis[case] = dict(scope=scope,time_groups=groups,
            changes={n:dict(coarse_mae_percent=(scores[n]['coarse_mae']/scores['K']['coarse_mae']-1)*100,
                ncc_delta=scores[n]['fine_ncc']-scores['K']['fine_ncc'],
                psnr_delta=scores[n]['psnr_db']-scores['K']['psnr_db']) for n in ('M03','M12')},
            small_regressions={s:{n:(v[n]['coarse_mae']/v['K']['coarse_mae']-1)*100
                for n in ('M03','M12') if v[n]['coarse_mae']>v['K']['coarse_mae']}
                for s,v in result['scores'].items() if any(v[n]['coarse_mae']>v['K']['coarse_mae'] for n in ('M03','M12'))})
    save_json(run/'diagnosis.json',dict(complete=True,cases=diagnosis,
        source='Saved per-frame CSV and fixed-support scores; no new rendering'))
    aliases = [
        ('video1/evaluation/K_vs_M12_full.mp4','01_video1_左K_右M12_全画面.mp4'),
        ('video1/evaluation/K_vs_M12_coral.mp4','02_video1_左K_右M12_珊瑚放大.mp4'),
        ('video2/evaluation/K_vs_M12_full.mp4','03_video2_左K_右M12_全画面.mp4'),
        ('video3/evaluation/K_vs_M12_full.mp4','04_video3_左K_右M12_全画面.mp4'),
        ('video1/evaluation/K_vs_M03_full.mp4','05_video1_左K_右M03_全画面.mp4'),
        ('video1/evaluation/K_vs_M03_coral.mp4','06_video1_左K_右M03_珊瑚放大.mp4'),
        ('video2/evaluation/K_vs_M03_full.mp4','07_video2_左K_右M03_全画面.mp4'),
        ('video3/evaluation/K_vs_M03_full.mp4','08_video3_左K_右M03_全画面.mp4')]
    files = {}
    for source, name in aliases:
        shutil.copy2(run/source,view/name)
        assert file_sha(run/source)==file_sha(view/name)
        files[name] = dict(source=str(run/source),sha256=file_sha(view/name))
    prior = run.parent/'support_refinement_20261111_01'
    batch = run.parent/'source_dense_flow_three_clips_20261109_01'
    for case in ('video1','video2','video3'):
        for name in ('K','M03','M12'):
            dest=view/'实际传输文件'/case/name;dest.mkdir(parents=True,exist_ok=True)
            packet=(prior/case/'K/support.glccsp2') if name=='K' else (run/case/name/'support.glccsp3')
            for source in (batch/case/'base.glcsnd',packet):
                shutil.copy2(source,dest/source.name)
                assert file_sha(source)==file_sha(dest/source.name)
                files[str((dest/source.name).relative_to(view))]=dict(source=str(source),sha256=file_sha(source))
    save_json(view/'来源核验.json',dict(complete=True,files=files))
    save_json(run/'release_verification.json',dict(complete=True,
        frozen_inputs_unchanged=all(file_sha(Path(p))==h for p,h in load_json(run/'input_manifest.json').items()),
        preview_and_packet_copies_exact=True,previews=8,transmitted_file_copies=18,
        no_new_H265_claim=True,default_remains_K=True))
    if not load_json(run/'release_verification.json')['frozen_inputs_unchanged']:
        raise ValueError('Input changed')
    shutil.copy2(Path(__file__),run/'source_final'/Path(__file__).name)
    save_json(run/'source_final/manifest.json',{p.name:file_sha(p) for p in (run/'source_final').iterdir() if p.name!='manifest.json'})
    for path in run.rglob('*'):
        if not path.is_file() or path.suffix not in ('.json','.csv','.py'):
            continue
        relative=path.relative_to(run)
        dest=evidence/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    save_json(evidence/'manifest.json',{str(p.relative_to(evidence)):file_sha(p) for p in evidence.rglob('*') if p.is_file()})
    report = '''# 混合边界细化：有限字节探针与实际接收

2026-11-11。本轮承接 K 基线后的边界探针计划，已完成 12 份字节探针、6 份候选独立接收、三素材固定支持评分和 8 段原速对照。**当前保留 K；候选尚未获得新的原速观感验收，不能按指标自动升级。**

## 1. 冻结设置与选择规则

三素材各前 150 帧、15 FPS、512×288、10 秒。保留 K 的半像素运动、各素材既有网格、锚点、旧参考六帧过渡、五帧交集及 2px 向内羽化。不改 GLC 码流、解码 RGB 和潜变量参考链。

保留全部 K 的 4×4 安全块，只在与 K 块邻接、包含部分安全 2×2 子块的混合边界父块增加信息。候选新增区域始终是已核验 N 安全区的子集，不放宽源端安全判据。每个时间组统一选择排名最高的 1%/3%/6%/12% 混合父块；三素材使用同一规则，不按珊瑚手画 ROI。

排名使用已有解码画面：潜在 alpha 增量乘以当前帧与同一参考投影之间 sigma3 亮度差的组内均值。它是可用结构修正量的启发式，不是时序收益预测，不读取源 RGB 或评价指标。父块邻接不等于新增像素逐个连通。新旧参考分别应用相同规则。

实际辅助格式 GLCCSP03 增加“粗安全区＋稀疏子块码”，比较有限的坐标差分布局，并复用既有无损运动预测。先还原可靠区，再解必要运动节点，RGB 渲染代码不变。新格式仍为整段实验文件；不代表接入实时部署。接收器内部将支持区规范化后调用既有 GLCCSP02 校验与运动解码，因此存在额外 CPU 开销，尚未优化或测量时延。

## 2. 先看完整文件字节，再选两档渲染

| 素材 | K 辅助 B | M01 B | M03 B | M06 B | M12 B |
| --- | ---: | ---: | ---: | ---: | ---: |
| video1 | 24,649 | 25,108 | 25,588 | 26,221 | 27,414 |
| video2 | 23,036 | 23,575 | 24,213 | 25,066 | 26,610 |
| video3 | 19,412 | 19,979 | 20,591 | 21,408 | 22,888 |

完整封装、编码元数据和 CRC 均计费。根据字节、在产生候选 RGB 和评分前冻结 M03/M12 两档，分别代表少量追加与本次探针上端；没有根据评分挑选素材专用参数。12 份探针均通过支持区与必要运动精确往返，所有混合布局也逐项核验。

| 素材 | K 辅助 kbps | M03 辅助 kbps | M12 辅助 kbps | M12 增量 kbps | 本体 kbps |
| --- | ---: | ---: | ---: | ---: | ---: |
| video1 | 19.7192 | 20.4704 | 21.9312 | +2.2120 | 25.5704 |
| video2 | 18.4288 | 19.3704 | 21.2880 | +2.8592 | 25.4792 |
| video3 | 15.5296 | 16.4728 | 18.3104 | +2.7808 | 24.7960 |

M03 相对 K 辅助增加约 3.8%–6.1%，M12 增加约 11.2%–17.9%；均低于本体且低于 40 kbps。低于本体是优先偏好，不是新增硬限。这些是额外字节换取画质的候选，不能说成同预算改善。

## 3. 数值改善有限，没有恢复真实形状的证据

主评价仍使用同一源相邻对应与外部可靠纹理支持，不取候选掩码交集缩小分母。源对应只用于评价。原始 GLC、K、N 在所有评价区域的历史分数精确重现。

| 素材/区域 | K MAE↓ | M03 MAE↓ | M12 MAE↓ | M12 相对 K | K NCC↑ | M12 NCC↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| video1 黄珊瑚 | 2.677827 | 2.631910 | 2.544605 | −4.975% | 0.756759 | 0.771980 |
| video2 整帧 | 1.614944 | 1.610867 | 1.602715 | −0.757% | 0.629409 | 0.632343 |
| video3 整帧 | 1.283938 | 1.274825 | 1.256938 | −2.103% | 0.848415 | 0.853014 |

video1 珊瑚原始 GLC 为 4.558，旧局部 U 为 1.426，GP 为 1.972，统一 2×2 的 N 为 2.405。M12 未达到 1.8/0.88 筛选门槛，也没有接近 U。MAE 改善百分比不是肉眼改善百分比。

video1 非换锚组更新处 K/M12 为 4.062/3.763，组内为 2.468/2.338；换锚及退出为 2.470/2.422。组界与组内都有小幅改善，不能归因成仅修复某一种时间事件。

M12 实际改变三素材约 0.615%/0.773%/0.786% 的像素帧。video1 左下区和 video3 上方两区有微小 MAE 退步；不能声称所有区域改善。三段对应主区域的 PSNR 分别下降约 0.0060/0.0043/0.0027 dB，不能声称真实形状或保真度提高。静态抽帧不代替原速验收。

## 4. 接收与核验

6 份 M03/M12 均独立进程接收，读取白名单仅包含基础包、已独立核验的基础解码及其元数据、实际辅助包；源 RGB、源运动、外部掩码和发送端记录读取阻断。必要运动精确往返，省略节点前后 RGB/alpha 逐位一致；零权重位置原样输出，全部 K 支持保留且新增支持不超过 N。

通过 CRC、截断、错误基础绑定、超限、有效 CRC 错误头、错误网格/锚点/时间组/粒度/载荷长度/预测器/混合布局/数量负例。21 份输入文件哈希保持不变；8 段预览核验 150 帧、15 FPS、10 秒以及首中末面板身份。所有评分使用无损 RGB，MP4 仅供观看。

本轮用于筛选相对 K 的增量价值，没有新跑同总字节 H.265，也不引用旧 H.265 的分数或观感宣称新候选胜出。若候选得到明确可见收益并拟升级，仍需按其新总字节补对照。

## 5. 当前决定与接续

原速观看反馈见 `visual_review.json`；截至归档为待反馈。K 保持默认，M03/M12 仅为有额外字节的研究候选。

字节方面已经证明少量边界可以紧凑表示；质量方面目前只有有限数值收益。若用户仍看不出明确改善，应按上一轮停止条件结束这一边界候选，不继续扩大比例或堆排名参数。后续若继续，应把沿运动的可靠状态/参考权重连续性作为单独受控因素，先验证观感潜力、接收端可获得的信息及安全退出，再投入编码；当前到锚帧运动不能直接当相邻流。新素材、长片、实时与真实形状恢复仍未完成。

## 文件与复现

- 运行：`02_实验数据/优化_GLC后处理/mixed_boundary_20261111_01`。
- 观看：`05_成果与观看/GLC_混合边界探针_20261111`，01–04 为 M12，05–08 为 M03；均左 K、右候选。
- 实际传输文件：观看目录下三素材 K/M03/M12，各自基础文件和辅助文件；GLCCSP03 需要本轮接收器。
- 小证据：[evidence_20261111/mixed_boundary](evidence_20261111/mixed_boundary/README.md)，含冻结计划、全部字节探针、输入清单、独立回执、CSV、核验、代码快照和哈希。
- 入口：`probe_mixed_support.py`、`mixed_support_stream.py`、`finish_mixed_support.py`、`release_mixed_support.py`。字节探针与渲染选择分开保存；复现必须新建运行目录，保留现有输入和结果。
'''
    (stage/'混合边界细化字节探针_20261111.md').write_text(report,encoding='utf-8')
    (view/'README.md').write_text('''# 混合边界探针对照

2026-11-11。当前仍以 K 为默认；M03/M12 是增加辅助字节的研究候选，原速反馈待记录。

01–04：左 K、右 M12，依次为 video1 全画面、video1 珊瑚放大、video2、video3。
05–08：左 K、右 M03，顺序相同。

三素材 K 辅助 19.7192/18.4288/15.5296 kbps；M03 为 20.4704/19.3704/16.4728；M12 为 21.9312/21.2880/18.3104。
全部 150 帧、15 FPS、10 秒。标题为辅助码率，实际传输需再加本体 25.5704/25.4792/24.7960 kbps。观看 MP4 文件大小不是传输码率。

video1 珊瑚 MAE K/M03/M12 为 2.678/2.632/2.545；这不等于肉眼改善幅度。未接近局部 U，不能声称真实形状恢复。

实际传输文件包含三素材 K/M03/M12。K 用 GLCCSP02，候选用 GLCCSP03，需要本轮新接收代码。尚未接入通用实时部署。

[实验报告](../../01_项目/代码/semantic_video_test/21_优化阶段_GLC解码后处理/混合边界细化字节探针_20261111.md)
''',encoding='utf-8')
    (evidence/'README.md').write_text('''# 混合边界探针证据

运行 mixed_boundary_20261111_01 的原字节 JSON、CSV 和代码快照。大数组和视频保留本机。

plan.json 是执行前冻结计划；render_selection.json 是仅根据字节、在渲染评分之前冻结的 M03/M12。
summary.json 为 12 份字节探针；final_summary.json 为 6 份接收与评分；verification.json 为独立接收和负例。
visual_review.json 独立保存用户观感状态，不能根据数值自动升级。
zero_addition_control.json 单独记录格式零修正控制；未生成第三个画质候选。
manifest.json 记录归档文件原字节 SHA-256。
''',encoding='utf-8')
    save_json(evidence/'manifest.json',{str(p.relative_to(evidence)):file_sha(p) for p in evidence.rglob('*') if p.is_file() and p!=evidence/'manifest.json'})
    print('Archived bounded probe, six receivers, eight previews and exact evidence',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();release(a.workspace.resolve(),a.run.resolve())
