"""Documentation-only closeout. Never runs codecs, inference, or git writes."""
from pathlib import Path
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
WORK=REPO.parents[2]
STAGE=REPO/'21_优化阶段_GLC解码后处理'
DATA=WORK/'02_实验数据/优化_GLC后处理'
VIEW=WORK/'05_成果与观看'
STATE=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
KEEP=STAGE/'保留版本/SQ3_用户认可_20261113'
CONFIG='b5eeb844d0adcdfe97648351a0654a2575a98942e1627c98896ba302555f7b04'
HEAD='ae46c91c6820ffc49012714c44659ead0e1fc162'
SUMMARY=STAGE/'SQ3阶段总结与后续路线_20261113.md'
INDEX=STAGE/'运行入口与文件索引.md'
SQVIEW=VIEW/'GLC_SQ3长片四宫格_20261113'
UVIEW=VIEW/'GLC_U30长片四宫格_20261113'
DVIEW=VIEW/'GLC_SQ3保留与温和锐度实验_20261113'
SQMOV=SQVIEW/'82秒_四宫格_原片_H265_原GLC_SQ3.mp4'
UMOV=UVIEW/'82秒_四宫格_原片_U30_原GLC_SQ3.mp4'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def link(origin,label,target):return f'[{label}]({os.path.relpath(target,origin.parent).replace(chr(92),"/")})'

def main():
    assert not (HERE/'整理前清单.json').exists(),'Closeout already started; do not overwrite history'
    assert sha(STAGE/'当前采用配置.json')==CONFIG
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==HEAD
    state=load(STATE);feedback=load(HERE/'用户反馈.json');fallback=load(KEEP/'回退配置.json')
    longrun=DATA/'sq3_long_fourway_20261113_01';u30run=DATA/'sq3_u30_long_fourway_20261113_01'
    detail=DATA/'sq3_detail_20261113_02';short=DATA/'q9_video4_jitter_20261113_03'
    longplan=load(longrun/'execution_plan.json');longreceipt=load(longrun/'receive/receipt.json')
    prior_roots=[KEEP,STAGE/'evidence_20261113/video4_short_history_jitter',STAGE/'evidence_20261113/sq3_long_fourway',
        STAGE/'evidence_20261113/sq3_detail',STAGE/'evidence_20261113/u30_long_fourway']
    protected={str(STAGE/'当前采用配置.json'):CONFIG}
    for root in prior_roots:
        protected[str(root/'manifest.json')]=sha(root/'manifest.json')
    for r in fallback['outputs'].values():protected[r['original']]=r['sha256'];protected[r['snapshot']]=r['sha256']
    protected[str(UMOV)]=load(u30run/'previews/verification.json')['sha256']
    for p,h in load(detail/'execution_plan.json')['code_sha256'].items():protected[str(STAGE/p)]=h
    for p in (STAGE/'short_history_filter.py',STAGE/'received_structure_fusion.py',STAGE/'run_sq3_long.py',STAGE/'run_u30_long_fourway.py'):
        protected[str(p)]=sha(p)
    targets=[REPO/'交接文档.md',REPO/'当前项目状态与接续.md',REPO/'新会话接续.txt',REPO/'README.md',REPO/'项目总结与未来方向.md',
        STAGE/'README.md',STATE,REPO/'05_资料与索引/会话收束_20261110/文件索引.md',WORK/'02_实验数据/README.md',VIEW/'README.md',
        DATA/'README.md',SQVIEW/'README.md',UVIEW/'README.md',DVIEW/'README.md']
    originals={};backups={}
    for p in targets:
        if not p.exists():continue
        dest=HERE/'历史入口原文'/p.relative_to(WORK);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        assert sha(dest)==sha(p)
        backups[str(p)]=dict(archive=str(dest),sha256=sha(p),bytes=p.stat().st_size)
        originals[p]=p.read_text(encoding='utf-8-sig')
    save(HERE/'整理前清单.json',dict(complete=True,original_files=backups,protected_files=protected,large_files_moved=False))
    (HERE/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=REPO,text=True,encoding='utf-8'),encoding='utf-8')
    print('ARCHIVED original entry documents',len(backups),flush=True)
    written=[]
    def write(p,text):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.strip()+'\n',encoding='utf-8');written.append(p)

    # Canonical machine state retains the compatibility path but archives its
    # large historical payload. Frozen per-experiment records remain untouched.
    latest=copy.deepcopy(state['latest_experiment'])
    latest.update(subjective_review='user_reports_U30_and_SQ3_similar',feedback_file=str(HERE/'用户反馈.json'),selected=None)
    current={k:copy.deepcopy(state[k]) for k in ('timezone','scope','workspace','repository','experiment_branch','main_baseline_commit',
        'last_experiment_commit','current_branch_head_check','new_experiments_uncommitted','budget','current_accepted_profile','other_local_untracked_repo_paths','archived_scope') if k in state}
    paths={k:v for k,v in state['workspace_paths'].items() if k in ('source_clips','base_manifest','independent_deployment','current_profile')}
    paths.update(latest_viewing=str(UVIEW/'README.md'),accepted_SQ3_viewing=str(SQVIEW/'README.md'),
        sq3_archive=str(KEEP),short_run=str(short),SQ3_long_run=str(longrun),detail_run=str(detail),U30_long_run=str(u30run),
        handoff=str(REPO/'交接文档.md'),file_index=str(INDEX))
    current.update(schema='glc_handoff_current_v2',date='2026-11-13',status='session_closed_experiments_paused_SQ3_retained',
        user_latest_decision=feedback['quote'],latest_user_perceptual_feedback=feedback,
        current_session_work='Documentation and file-index closeout only. User sees U30 and SQ3 as similar; keep SQ3 and pause experiments for session change.',
        active_experiment_plan=None,latest_experiment_plan=None,
        experiment_execution_gate=dict(status='paused_by_user',currently_running=False,automatic_next_run=False,authorization_to_resume='New explicit user instruction after handoff'),
        next_experiment_plan=dict(status='paused_by_user',parameters_chosen=False,automatic_next_run=False),
        preferred_research_reference=dict(profile='SQ3',status='user_accepted_and_preserved',upstream='Q9_R25 -> GS25 -> SQ3',legacy_reference='Q9_R25'),
        preferred_fallback=state['preferred_fallback'],formal_deployment=dict(profile='S8I + CURRENT_I12',changed=False,
            config=str(STAGE/'当前采用配置.json'),sha256=CONFIG,note='Research baseline SQ3 has not been wired into the formal deployment default.'),
        latest_experiment=latest,latest_analysis_report=str(SUMMARY.relative_to(REPO)),new_experiment_report=str(SUMMARY.relative_to(REPO)),
        latest_diagnosis=dict(report=str(SUMMARY),SQ3_user_accepted=True,U30_subjective_difference='similar per user',
            numerical_failure_preserved=True,no_sharpening_upgrade_selected=True),
        current_fourway=str(UMOV),accepted_SQ3_fourway=str(SQMOV),workspace_paths=paths,
        handoff_repo_path='交接文档.md',stage_report_repo_path='21_优化阶段_GLC解码后处理/README.md',
        next_priority=['Read the handoff and current file index; do not rerun completed experiments',
            'Wait for user direction before any new experiment or deployment-default change'],
        future_routes_not_started=['Scene-cut and cross-block continuity','Frozen-profile independent ocean clips',
            'End-to-end runtime and total RAM after quality direction is fixed'],
        history_snapshot=backups[str(STATE)]['archive'],history_snapshot_sha256=backups[str(STATE)]['sha256'],
        session_closeout=str(HERE),observed_git_HEAD=HEAD,no_commit_or_push_this_closeout=True)
    save(STATE,current);written.append(STATE);save(HERE/'接续状态快照.json',current)

    p=REPO/'交接文档.md'
    write(p,f'''# GLC 项目交接（2026-11-13 收束）

**用户最新结论：U30 与 SQ3 长片看起来差不多，先这样，整理后换会话。当前暂停新实验，保留 SQ3 为研究基准和首选回退版本。**

用户原话：“{feedback['quote']}”

## 新会话先读什么

1. 本文件：当前决定、文件入口和接续边界。
2. {link(p,'当前状态',REPO/'当前项目状态与接续.md')} 与 {link(p,'SQ3 阶段总结、瓶颈和后续路线',SUMMARY)}。
3. 需要看代码或复现时，再读 {link(p,'运行入口与文件索引',INDEX)}。
4. 机器状态为 {link(p,'接续状态 JSON',STATE)}；目录名“20261110”仅为兼容历史脚本，内容已更新。

## 三种状态要分清

| 层级 | 方案 | 现在的含义 |
| --- | --- | --- |
| 当前研究基准 / 回退版本 | SQ3 | 用户认可抖动减轻，接受不够锐利；源码、参数、四片与完整长片均独立保存。 |
| 已完成的比较方案 | U30 | 用户看过完整长片，认为和 SQ3 差不多，没有要求替换。U15/U30 的原数值筛查失败照常保留。 |
| 历史正式采用配置 | S8I + CURRENT_I12 | 配置文件与部署默认未改；不要误以为正式部署已切到 SQ3。 |

{link(p,'SQ3 回退归档',KEEP/'README.md')} · {link(p,'正式配置',STAGE/'当前采用配置.json')} · {link(p,'本次用户反馈',HERE/'用户反馈.json')}

## 直接看结果

- {link(p,'最新 U30 长片四宫格',UMOV)}：左上原片、右上 U30、左下原 GLC、右下 SQ3。用户反馈两者差不多。
- {link(p,'已认可的 SQ3 长片四宫格',SQMOV)}：左上原片、右上 H.265、左下原 GLC、右下 SQ3。
- {link(p,'四短片 U15/U30 对照',DVIEW/'README.md')}：保留试验结果；当前无需再次验收来继续交接。

两条长片均完整 1239 帧 / 15 FPS / 82.6 秒；单路 512×288，四宫格 1024×688，原速且无音频。U30 与 SQ3 合计均约 37.7766 kbps；原 GLC 仅基础流约 26.2222 kbps。旧 H.265 对照约 37.4263 kbps，按完整文件整段平均预算匹配，不是逐秒相同码率。观看 MP4 的大小不是通信码率。

## 现在的算法与证据

- 链路为 Q9_R25 → GS25 → SQ3。GS25 用收到的 guide 作 25% 低频修正；SQ3 将当前和前两帧 GS25 以收到的 guide 双向光流对齐，名义权重 0.5 / 0.3 / 0.2，并连续衰减不可靠历史。
- 历史输入是未滤波的 GS25，不递归回灌 SQ3。SQ3 新增历史只到前两帧，但上游 Q9 仍含长参考，所以不能说所有残像已解决。
- SQ3 长片按150帧窗口处理，跨窗口保留前两帧；尾块39帧，输出已逐帧重放核验。没有新切镜检测。
- U30 是 SQ3 后的 0.30 亮度高通增强，最多±3级。无新传输或光流；长片只读取SQ3生成，源图用于评价/显示。用户要求扩展长片，不等于原数值筛查被改成通过。
- 已验证四个开发片段和这一条长片；不等于独立留出泛化、任意长片、全面优于 H.265 或实时。

## 用户优先级与尚存问题

先保住整体可看、主体存在与运动自然，再看锐度和指标。video2 只需适度改善；不要继续围绕礁石斑点鱼4.07秒旧影反复调参。video4 已提供并完成多轮实验，不再是“等用户提供”。

SQ3 已改善密集珊瑚抖动观感，但细节和动态幅度有代价。基础 GLC / guide 缺失的鱼形与真实细节未被保证恢复；切镜、遮挡和旧参考问题仍在。简单加锐没有形成值得替换 SQ3 的优势。

**当前仅收束与交接。** 后续若用户恢复研究，再选择切镜/分块连续性、独立海洋素材或工程开销中的一个方向；这些不是已开始任务。不重跑旧矩阵，不按每段素材挑参数后声称泛化，不自动改正式配置。

## 环境、版本和文件安全

- 工作区：`{WORK.as_posix()}`。
- 仓库：`{REPO.as_posix()}`。
- Python：`{(WORK/'01_项目/运行环境/GLC_Python环境/Scripts/python.exe').as_posix()}`，既有命令使用 `-B -u -X utf8`。
- FFmpeg：`{(WORK/'01_项目/运行环境/FFmpeg/ffmpeg.exe').as_posix()}`。
- 分支：`codex/glc-temporal-stability`；HEAD：`{HEAD}`。
- 多轮源码和文档尚有本地修改/未跟踪文件；本次未提交或推送。不要 reset/clean，不把旧变化当作本次新增。
- 数据、视频、光流和码包保留原路径；已有脚本含绝对路径。本次按索引整理，没有搬移或删除大文件。历史失败现场与旧比较口径保留。

## 历史与收束记录

旧入口原文已按原字节归档，不再叠在本页。{link(p,'本次收束包',HERE/'README.md')} 包含用户原话、整理前清单、文件目录、状态快照和核验；原文中的“当前/等待/最新”属于当时状态。
''')

    p=SUMMARY
    write(p,f'''# SQ3 阶段总结与后续路线（2026-11-13）

当前决定：保留用户认可的 SQ3，U30 仅作为完成的对照。用户看完整长片后说两者差不多，暂停继续优化并换会话。{link(p,'用户原话',HERE/'用户反馈.json')}

## 这轮推进到了哪里

| 阶段 | 结果与定位 |
| --- | --- |
| Q9_R25 | video1/2/3 的历史正面反馈保留；复杂素材出现旧参考、轮廓与密集珊瑚问题。 |
| GS25 | 已传 guide 的低频修正；无新增传输，仍有动态和细节代价。 |
| SQ3 | guide对齐三帧融合，四片完成；用户认可短片和82.6秒长片的稳定性，接受锐度下降。当前回退基准。 |
| U15 / U30 | 尝试增强SQ3现有对比度，两档均未通过四片全ROI预设数值筛查。没有证明真实细节恢复。 |
| U30 完整长片 | 用户明确要求后完成1239帧四宫格；用户反馈与SQ3观感接近，暂不升级。 |

{link(p,'SQ3 三帧实验',STAGE/'video4抖动_三帧短参考实验_20261113.md')} · {link(p,'SQ3 长片报告',STAGE/'SQ3长片四宫格_原片_H265_GLC_20261113.md')} · {link(p,'锐度实验',STAGE/'SQ3保留与温和锐度实验_20261113.md')} · {link(p,'U30 长片说明',STAGE/'U30长片四宫格_原片_U30_GLC_SQ3_20261113.md')}

## 优势

1. 有用户认可的可看基准：SQ3 对长片抖动的改善得到直接反馈，毋须回到从零搜索。
2. 延用已有基础流与guide预算。SQ3与U30新增后处理传输均0字节，但它们仍包含原有辅助流；不能与仅基础GLC混用码率口径。
3. 固定方案已在video1/2/3/4及一条完整长片运行，没有为每片另选参数。码包、数组、源码快照、核验和原速预览均可追溯。
4. SQ3新增融合不递归，跨处理窗口连续；U30更是逐帧确定性处理。已有精确重放证据可用于回归。

## 瓶颈与代价

| 问题 | 当前边界 |
| --- | --- |
| 稳定性与锐度/动态取舍 | SQ3更稳也更软；不能单靠低时序误差判断更好。U15/U30增强对比度时也提高误差，且用户未感到U30有明显优势。 |
| 原始结构缺失 | 鱼形、鳍尾和真实珊瑚纹理在基础重建/guide中可能已失真，锐化不能保证找回。 |
| 历史参考与切镜 | SQ3仅限制新增两帧历史，上游Q9仍含长锚点；没有新增切镜检测，残像与新显露区域仍是边界。 |
| 泛化证据有限 | video4已经作为开发失败案例参与分析，四片与同一长片均不能当作独立留出。 |
| 速度与总内存 | SQ3长片新增接收阶段约{longreceipt['seconds']:.1f}秒，含guide解码/光流/融合/I/O，但不含复用的基础GLC与Q9。不是端到端实时测速。 |

## 如果以后继续，怎样安排

以下只是后备路线，当前不执行：

1. 冻结SQ3并按原速定位剩余明显问题，分别标注真实切镜、同镜头分包边界和主体显露；避免只优化平均分。
2. 选一个接收端可靠性或切镜假设，维持统一设置，先做video1–4回归再完整长片。暂不继续单纯增加全局锐度。
3. 质量方向固定后，在未参与调参的海洋素材上检验；记录失败，不按各片挑最优档后宣称泛化。
4. 再做总RAM、首帧时延、吞吐和完整收发链路计时；基础与辅助的全部必要字节同时记录。
5. 只有确认需要时再讨论辅助表示或正式部署集成；当前部署默认仍是S8I+CURRENT_I12。

## 解释结果时遵守的口径

源像素只用于离线评价/展示/传统编码控制，不进入接收算法。细纹理幅度不等于真实细节，时序/二阶误差不是肉眼抖动百分比；数值筛查失败与用户说“差不多”可同时成立。旧selected=null、video3珊瑚+5.373553108170714%及各历史比较基线保留，不重写为追认成功。

{link(p,'当前交接',REPO/'交接文档.md')} · {link(p,'文件索引',INDEX)} · {link(p,'SQ3归档',KEEP/'README.md')}
''')

    runs=[('q9_received_structure_20261113_01','Q9/GS25与模糊控制，历史上游'),
        ('q9_video4_diagnosis_20261113_02','video4固定参数诊断'),
        ('q9_video4_jitter_20261113_03','四片SQ3原始结果，保留'),
        ('sq3_long_fourway_20261113_01','已认可的SQ3完整长片'),
        ('sq3_detail_20261113_01','首次舍入校验中止；部分数组无效，仅失败现场'),
        ('sq3_detail_20261113_02','U15/U30四片完整实验，未通过预设筛查'),
        ('sq3_u30_long_fourway_20261113_01','U30完整长片，用户反馈与SQ3接近')]
    inventory=[]
    for name,status in runs:
        root=DATA/name;assert root.is_dir()
        files=[f for f in root.rglob('*') if f.is_file()]
        inventory.append(dict(name=name,path=str(root),status=status,file_count=len(files),bytes=sum(f.stat().st_size for f in files),moved=False,deleted=False))
    save(HERE/'运行目录清单.json',dict(complete=True,scope='Recent relevant runs only; not a whole-workspace inventory',directories=inventory))
    p=INDEX
    lines=['# GLC 后处理运行入口与文件索引','', '2026-11-13收束：SQ3为研究基准，U30观感接近；新实验暂停。数据与视频原路径保留。','',
        '## 先找哪类文件','', '| 用途 | 入口 |','| --- | --- |',
        f'| 当前交接 | {link(p,"交接文档",REPO/"交接文档.md")} |',
        f'| 用户反馈与暂停决定 | {link(p,"用户反馈",HERE/"用户反馈.json")} |',
        f'| SQ3回退源码和数组 | {link(p,"归档说明",KEEP/"README.md")} / {link(p,"回退配置",KEEP/"回退配置.json")} |',
        f'| 最新原速长片 | {link(p,"原片 / U30 / 原GLC / SQ3",UMOV)} |',
        f'| 已认可长片 | {link(p,"原片 / H.265 / 原GLC / SQ3",SQMOV)} |',
        f'| 当前结论与路线 | {link(p,"SQ3阶段总结",SUMMARY)} |','',
        '## 实验目录（完成状态，不是待运行任务）','', '| 目录 | 定位 |','| --- | --- |']
    for name,status in runs:lines.append(f'| {link(p,name,DATA/name)} | {status} |')
    lines+=['','目录里的execution_plan.json、case_registration.json、receipt/verification/summary记录来源与核验。冻结的旧round_result里pending是当时状态；最新反馈以本次收束为准。','',
        '## 算法和脚本','', '| 文件 | 用途 |','| --- | --- |']
    for name,role in [('received_structure_fusion.py','GS25已传guide低频修正'),('short_history_filter.py','SQ3三帧对齐与连续置信度'),
        ('run_video4_jitter.py','四短片SQ3实验冻结/接收/评价'),('run_sq3_long.py','SQ3长片执行计划与收发控制'),
        ('run_sq3_long_receiver.py','长片接收禁读包装'),('finish_sq3_long.py','SQ3重放与四宫格'),
        ('sq3_detail_filter.py','U15/U30共享亮度校正；U30固定0.30'),('run_sq3_detail.py','历史锐度实验，常量固定到既有目录'),
        ('finish_sq3_detail.py','历史锐度对照与发布'),('run_u30_long_fourway.py','本次U30长片生成与四宫格'),
        ('preserve_sq3_accepted.py','只执行过一次的SQ3归档脚本，不要对同目录重跑')]:
        lines.append(f'| {link(p,name,STAGE/name)} | {role} |')
    lines+=['','这些是可追溯的历史入口，不是新会话应直接执行的命令。多数输出目录按exist_ok=False保护，发布脚本还会更新状态。若未来明确要新实验，先定新目录、保持SQ3参数和旧输出，不对已完成目录强行重跑。旧收束audit可能断言当时state.status，不能拿它判定今天的状态。','',
        '## 素材、输出和复现关系','',
        f'- 四片实际源路径与GS25/Q9输入：{link(p,"case_registration.json",short/"case_registration.json")}。video1/2/3各150帧，video4为104帧，均15FPS。',
        f'- 完整长片的原片、raw、base、guide等绝对路径：{link(p,"SQ3长片执行计划",longrun/"execution_plan.json")}。1239帧，512×288。',
        f'- U30长片参数和输入哈希：{link(p,"U30执行计划",u30run/"execution_plan.json")}；输出在receive/U30.npy。',
        '- SQ3独立归档的outputs包含video1–4、long_SQ3.npy及已认可四宫格；大型光流与上游包仍通过哈希引用原目录。',
        '- 长片光流缓存按150帧窗口带2帧历史，尾块实际39帧；复用时读取receipt.records，不能假设每块恰为150帧。','',
        '## 历史证据与环境','']
    for root in prior_roots:lines.append(f'- {link(p,root.name,root/"manifest.json")}')
    lines+=[f'- {link(p,"本次整理和完整历史入口原文",HERE/"README.md")}。文件统计只覆盖近期相关运行，不扫描或搬移全部工程资产。',
        f'- Python：`{(WORK/"01_项目/运行环境/GLC_Python环境/Scripts/python.exe").as_posix()}`；FFmpeg：`{(WORK/"01_项目/运行环境/FFmpeg/ffmpeg.exe").as_posix()}`。','']
    write(p,'\n'.join(lines))

    p=REPO/'当前项目状态与接续.md'
    write(p,f'''# 当前项目状态与接续

更新于2026-11-13；当前暂停新实验，准备换会话。

| 项目 | 状态 |
| --- | --- |
| 用户最新反馈 | U30与SQ3长片观感差不多，先这样。 |
| 研究基准 / 回退 | SQ3，已获用户认可并独立保存。 |
| U30 | 长片已生成并看过，保留为对照，无替换决定。 |
| 正式配置 | 历史S8I+CURRENT_I12，文件哈希不变。 |
| 活动实验 | 无；没有自动下一轮。 |
| Git | codex/glc-temporal-stability；HEAD {HEAD}；本次未提交/推送。 |

{link(p,'主交接',REPO/'交接文档.md')} · {link(p,'总结与路线',SUMMARY)} · {link(p,'文件索引',INDEX)} · {link(p,'用户反馈',HERE/'用户反馈.json')} · {link(p,'机器状态',STATE)}

新会话先阅读，不因旧文档中的“待反馈/下一轮”自动继续实验。用户恢复研究后再确定范围；保留SQ3，以主体、运动、稳定性和细节的实际观感决定是否升级。无需重做已完成的四片或82.6秒四宫格。
''')
    p=REPO/'新会话接续.txt'
    write(p,f'''请阅读：
{(REPO/'交接文档.md').as_posix()}
然后按需阅读：
{SUMMARY.as_posix()}
{INDEX.as_posix()}

当前截至2026-11-13：用户认可SQ3稳定性，接受略软；看完U30完整长片后认为两者差不多，先这样并换会话。
保留SQ3为研究基准和首选回退版本，U30只保留对照；正式部署配置仍为历史S8I+CURRENT_I12，没有切换。
四片与完整82.6秒长片均已完成，不需要重跑。video4已提供且已经实验，不再等待素材。
当前无活动实验，也没有自动下一轮。先了解现状，等我给出下一步要求；不要按旧文档自行续跑。
数据、视频、代码快照与失败现场保留原路径；旧入口已归档，当前信息以新的交接为准。
Git分支codex/glc-temporal-stability，HEAD {HEAD}。有既有本地修改，未提交/推送，不要重置。
''')
    p=REPO/'项目总结与未来方向.md'
    write(p,f'''# 项目总结与未来方向

2026-11-13收束。当前阶段已形成用户认可的SQ3长片基准；U30与SQ3观感接近，暂不升级。新实验暂停。

{link(p,'完整阶段总结、优势、瓶颈及后备路线',SUMMARY)}

下一次恢复研究时再选一条主线：切镜/分块连续性、独立海洋素材泛化、或质量固定后的完整链路性能。后备路线没有自动执行计划。

{link(p,'当前交接',REPO/'交接文档.md')} · {link(p,'历史项目总结原文',Path(backups[str(p)]['archive']))}
''')
    p=STAGE/'README.md'
    write(p,f'''# GLC 第21阶段：解码后处理

2026-11-13收束：SQ3是当前用户认可并独立保留的研究基准。U30完整长片已看过，用户认为与SQ3差不多；先暂停新实验。

- {link(p,'当前交接',REPO/'交接文档.md')}
- {link(p,'SQ3阶段总结与后续路线',SUMMARY)}
- {link(p,'算法、运行目录与文件索引',INDEX)}
- {link(p,'SQ3回退归档',KEEP/'README.md')}
- {link(p,'最新U30与SQ3长片四宫格',UMOV)}
- {link(p,'已认可SQ3与H.265长片四宫格',SQMOV)}
- {link(p,'用户反馈与本次收束',HERE/'README.md')}

正式配置仍为历史S8I+CURRENT_I12；研究基准与部署默认分别记录。旧实验报告、数值筛查、失败现场和证据清单保持原样。历史阶段入口已归档，不再把多轮旧“最新状态”叠在本页。
''')
    p=REPO/'README.md'
    oldcore=originals[p].split('## 当前能力',1)[1].split('## GLC 时序实验',1)[0] if '## 当前能力' in originals[p] else ''
    write(p,f'''# 海洋视频语义通信 / GLC 研究工程

**当前（2026-11-13）：SQ3已获用户认可并保留；U30长片与SQ3观感接近。用户决定暂时收束并换会话，新实验暂停。**

{link(p,'开始阅读交接',REPO/'交接文档.md')} · {link(p,'当前状态',REPO/'当前项目状态与接续.md')} · {link(p,'阶段总结',SUMMARY)} · {link(p,'文件索引',INDEX)} · {link(p,'新会话接续文字',REPO/'新会话接续.txt')}

{link(p,'最新U30/SQ3长片对照',UMOV)} · {link(p,'已认可SQ3长片',SQMOV)} · {link(p,'SQ3回退归档',KEEP/'README.md')}

当前研究链路Q9_R25→GS25→SQ3；正式部署配置仍为历史S8I+CURRENT_I12。当前无后台实验，不自动继续。旧多轮状态已归档到{link(p,'会话收束包',HERE/'README.md')}。

## 既有基础工程能力与实测（历史基线）

下列基础编解码测量属于原阶段，不能作为SQ3端到端实时性或独立泛化的证明。

{oldcore}
''')
    p=REPO/'05_资料与索引/会话收束_20261110/文件索引.md'
    write(p,f'''# 文件索引兼容入口

本文件保留在原路径，当前索引已经整理为{link(p,'GLC运行入口与文件索引',INDEX)}。

{link(p,'本次会话收束',HERE/'README.md')} · {link(p,'旧索引原文',Path(backups[str(p)]['archive']))} · {link(p,'当前交接',REPO/'交接文档.md')}
''')
    p=DATA/'README.md'
    rows=['# GLC后处理实验数据索引','', '当前保留SQ3，U30与SQ3长片观感接近；新实验暂停（2026-11-13）。','',
        link(p,'详细文件与代码索引',INDEX)+' · '+link(p,'当前交接',REPO/'交接文档.md'),'','| 运行目录 | 定位 |','| --- | --- |']
    rows += [f'| {link(p,name,DATA/name)} | {status} |' for name,status in runs]
    rows+=['','所有目录保持原位置，已完成的冻结结果不改写。首次失败的sq3_detail_20261113_01仅作失败现场，不能把部分数组当作通过结果。',
        link(p,'旧数据索引原文',Path(backups[str(p)]['archive'])) if str(p) in backups else '', '']
    write(p,'\n'.join(rows))
    p=WORK/'02_实验数据/README.md'
    tail=originals[p].split('## 历史实验数据与观看入口',1)[1] if '## 历史实验数据与观看入口' in originals[p] else ''
    write(p,f'''# 实验数据索引

当前（2026-11-13）：GLC后处理以SQ3保留版为研究基准，U30长片与其观感接近；用户暂停新实验并换会话。

{link(p,'当前GLC实验目录',DATA/'README.md')} · {link(p,'当前交接',REPO/'交接文档.md')} · {link(p,'文件索引',INDEX)} · {link(p,'本次整理记录',HERE/'README.md')}

## 历史实验数据与观看入口

以下保留B/C/D等既有工程索引；其中阶段结论采用当时口径。

{tail}
''')
    p=VIEW/'README.md'
    tail=originals[p].split('## 完整基线与既有汇报',1)[1] if '## 完整基线与既有汇报' in originals[p] else ''
    write(p,f'''# 成果与观看入口

截至2026-11-13：SQ3获认可并保留；用户看完整长片后认为U30与SQ3差不多，暂停进一步实验。

| 观看内容 | 说明 |
| --- | --- |
| {link(p,'最新：原片 / U30 / 原GLC / SQ3',UMOV)} | 82.6秒，左上/右上/左下/右下排列；已得到“两者差不多”反馈。 |
| {link(p,'已认可：原片 / H.265 / 原GLC / SQ3',SQMOV)} | SQ3完整长片，用户接受稳定性提升与略软的代价。 |
| {link(p,'四短片锐度实验',DVIEW/'README.md')} | U15/U30对照，不是已采用升级。 |
| {link(p,'SQ3原始四片三帧实验',VIEW/'GLC_video4抖动三帧对照_20261113/README.md')} | SQ3形成时的原速对照。 |

{link(p,'当前交接',REPO/'交接文档.md')} · {link(p,'SQ3独立保留版本',KEEP/'README.md')} · {link(p,'本次收束包',HERE/'README.md')}

近期SQ3/U30观看文件是独立副本；历史视频保持既有组织方式。MP4用于观看，通信码率按实际必要码包字节计。

## 完整基线与既有汇报（历史索引）

{tail}
''')
    p=UVIEW/'README.md'
    write(p,f'''# U30 长片四宫格

**2026-11-13反馈已记录：用户认为U30与SQ3差不多，先这样并换会话。继续保留SQ3，不把U30改为默认。**

{link(p,'播放完整82.6秒四宫格',UMOV)}

左上：原片　｜　右上：U30

左下：原始GLC　｜　右下：SQ3

1239帧、15FPS原速、无音频。U30与SQ3完整通信预算均约37.78kbps；原GLC基础约26.22kbps。U30的旧数值筛查结果保留，观感接近不改写其门槛结论。

{link(p,'生成说明与核验',STAGE/'U30长片四宫格_原片_U30_GLC_SQ3_20261113.md')} · {link(p,'用户反馈',HERE/'用户反馈.json')} · {link(p,'当前交接',REPO/'交接文档.md')}
''')
    p=SQVIEW/'README.md'
    write(p,f'''# 已认可的 SQ3 完整长片四宫格

用户已认可SQ3明显减轻抖动，接受不够锐利，已独立保留为首选回退版本。后续U30长片看起来与SQ3接近，当前继续保留SQ3并暂停实验。

{link(p,'播放完整82.6秒四宫格',SQMOV)}

左上：原片　｜　右上：H.265（约37.43kbps）

左下：原始GLC（约26.22kbps）　｜　右下：SQ3（约37.78kbps）

1239帧、15FPS原速、无音频。H.265按SQ3整段完整文件通信预算匹配，少0.927%，不是逐秒相同码率。观看MP4大小不作通信成本。

{link(p,'SQ3独立归档',KEEP/'README.md')} · {link(p,'原生成报告',STAGE/'SQ3长片四宫格_原片_H265_GLC_20261113.md')} · {link(p,'最新反馈与交接',REPO/'交接文档.md')}
''')
    p=DVIEW/'README.md'
    rows=['# SQ3 温和锐度实验（已收束）','', 'U15/U30四片试验均未通过预设数值筛查，原速对照保留。之后用户要求U30完整长片，并反馈与SQ3观感差不多；当前保留SQ3、暂停新实验。','',
        link(p,'U30完整长片',UMOV)+' · '+link(p,'最新交接',REPO/'交接文档.md'),'']
    for case in ('video4','video1','video2','video3'):
        rows.append(f'- {link(p,case+"四宫格",DVIEW/f"{case}_原片_SQ3_U15_U30.mp4")} · {link(p,"SQ3/U15并排",DVIEW/f"{case}_SQ3_vs_U15.mp4")}')
    rows+=['','短片四宫格左上原片、右上SQ3、左下U15、右下U30。15FPS原速；前三片10秒，video4为104帧约6.933秒。',link(p,'实验报告',STAGE/'SQ3保留与温和锐度实验_20261113.md'),'']
    write(p,'\n'.join(rows))

    p=HERE/'README.md'
    write(p,f'''# 会话收束：SQ3保留（2026-11-13）

用户已看U30与SQ3完整长片，认为差不多；当前暂停新实验并换会话。SQ3继续作为研究基准与首选回退，正式部署配置不变。

- {link(p,'最新交接',REPO/'交接文档.md')}
- {link(p,'新会话可复制文字',REPO/'新会话接续.txt')}
- {link(p,'用户原话与解释',HERE/'用户反馈.json')}
- {link(p,'当前状态快照',HERE/'接续状态快照.json')}
- {link(p,'文件与代码索引',INDEX)}
- {link(p,'近期运行目录清单',HERE/'运行目录清单.json')}
- {link(p,'整理前原文清单与哈希',HERE/'整理前清单.json')}
- {link(p,'整理核验',HERE/'整理核验.json')}

整理动作：把多轮叠加的旧入口保存为原字节副本，重写精简当前交接、状态、路线、观看与数据索引。源码、码包、光流、视频、SQ3保留版、正式配置和已有证据清单均保留原路径及内容。首次失败的锐度运行也保留。

历史入口位于 `历史入口原文/`，按工作区相对路径存储。它们是原字节历史文本，其中相对链接应按整理前原文件位置解释；当前可点击导航以上面的新索引为准。旧报告里的pending/待提供/待运行仅代表当时状态，不是当前任务。

本次只整理，不运行新实验，不提交或推送；新会话等待用户指定下一步。
''')
    save(HERE/'当前入口清单.json',dict(files=[str(p) for p in written],active_handoff=str(REPO/'交接文档.md')))
    print('REWROTE current entries',len(written),flush=True)
    verify(protected,backups,written,prior_roots)

def verify(protected,backups,written,roots):
    for p,h in protected.items():assert sha(p)==h,('protected file changed',p)
    manifests={}
    for root in roots:
        entries=load(root/'manifest.json')['files']
        for name,r in entries.items():assert sha(root/name)==r['sha256'],('evidence changed',root,name)
        manifests[str(root)]=len(entries)
    for p,r in backups.items():assert sha(r['archive'])==r['sha256'],('backup changed',p)
    broken=[];links=0
    for p in written:
        if p.suffix!='.md':continue
        for t in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if '://' in t or t.startswith('#'):continue
            target=(p.parent/t.split('#')[0]).resolve()
            if target==HERE/'整理核验.json':continue
            if not target.exists():broken.append(dict(document=str(p),target=t))
            links+=1
    # Historical lower sections may retain broken links; report separately,
    # but all freshly written navigation must resolve before closeout.
    assert not broken,broken
    s=load(STATE);assert s['status']=='session_closed_experiments_paused_SQ3_retained'
    assert s['active_experiment_plan'] is None and not s['experiment_execution_gate']['currently_running']
    assert s['latest_experiment']['subjective_review']=='user_reports_U30_and_SQ3_similar'
    assert s['preferred_fallback']['profile']=='SQ3' and s['formal_deployment']['sha256']==CONFIG
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==HEAD
    result=dict(complete=True,date='2026-11-13',current_entries=len(written),links_checked=links,
        original_documents_archived=len(backups),protected_files_verified=len(protected),prior_manifests_verified=manifests,
        accepted_SQ3_unchanged=True,U30_video_unchanged=True,formal_config_unchanged=True,
        code_and_existing_evidence_unchanged=True,large_files_moved=False,files_deleted=False,
        state_compacted_with_full_history_backup=True,new_experiments_run=False,committed=False,pushed=False,
        current_document_sha256={str(p):sha(p) for p in written})
    save(HERE/'整理核验.json',result)
    (HERE/'git_status_after.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=REPO,text=True,encoding='utf-8'),encoding='utf-8')
    files={str(f.relative_to(HERE)).replace('\\','/'):dict(sha256=sha(f),bytes=f.stat().st_size)
        for f in sorted(HERE.rglob('*')) if f.is_file() and f.name!='manifest.json'}
    save(HERE/'manifest.json',dict(complete=True,files=files,self_excluded=True))
    for name,r in files.items():assert sha(HERE/name)==r['sha256']
    print('CLOSEOUT VERIFIED',len(written),'entries',links,'links',len(files),'archive files',flush=True)

if __name__=='__main__':main()
