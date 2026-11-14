"""Record a documentation-only strategy summary without altering experiments."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

OUT=Path(__file__).resolve().parent
STAGE=OUT.parents[1]
REPO=STAGE.parent
WORKSPACE=REPO.parents[2]
DOC=STAGE/'现阶段瓶颈_优势劣势与后续路线_20261113.md'
STATE=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
EXPECTED='b5eeb844d0adcdfe97648351a0654a2575a98942e1627c98896ba302555f7b04'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def rel(p,base):
    return Path(os.path.relpath(p,base)).as_posix()

assert DOC.exists() and not (OUT/'receipt.json').exists()
assert sha(STAGE/'当前采用配置.json')==EXPECTED
state=read(STATE)
assert state['active_experiment_plan'] is None and not state['experiment_execution_gate']['currently_running']
original=copy.deepcopy(state)
entries=[REPO/'交接文档.md',REPO/'当前项目状态与接续.md',REPO/'项目总结与未来方向.md',
         STAGE/'当前方案总结与瓶颈路线_20261111.md',
         STAGE/'下一阶段_海洋素材与长视频泛化计划_20261111.md']
sources=[STAGE/n for n in ('当前采用配置.json','Q9已传引导结构融合与纹理对照_20261113.md',
         'Q9当前轮廓保留与运动范围限制_20261112.md','Q9跨素材与82秒长片验证_20261112.md',
         '接收链路计时与精确缓存提速_20261111.md','泛化第二轮_Q9观感反馈_20261112.md')]
source_hashes={str(p):sha(p) for p in sources}
old_manifests={str(p):sha(p) for folder in STAGE.glob('evidence_*') for p in folder.rglob('manifest.json')}
before={}
for p in entries+[STATE]:
    target=OUT/'before'/rel(p,WORKSPACE)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,target)
    before[str(p)]=sha(p)
save(OUT/'before_manifest.json',dict(files=before,existing_evidence_manifests=old_manifests))

for p in entries:
    old=p.read_text(encoding='utf-8-sig')
    title,body=old.split('\n',1)
    notice=f"""**路线总结（2026-11-13）：** 已整理[现阶段瓶颈、优势劣势与后续路线]({rel(DOC,p.parent)})。先保住video1/2/3的主体与运动，再登记video4密集珊瑚并验证局部结构修正，之后处理长片切镜/分块与独立留出，最后集中解决内存和速度。Q9_R25仍为研究参照，GS25仅供观看，正式采用配置不变。本次只总结，未新跑实验。

"""
    p.write_text(title+'\n\n'+notice+body.lstrip('\n'),encoding='utf-8',newline='\n')

state['previous_next_priority_before_strategy_summary_20261113']=copy.deepcopy(state.get('next_priority'))
state['latest_strategy_summary']=dict(date='2026-11-13',path=str(DOC),status='documented_proposed_not_executed',
    scope='Bottlenecks, evidence-backed strengths and limitations, prioritized roadmap.',
    new_experiments_started=False,formal_adoption_unchanged=True,
    sequence=['Review video1/2/3 Q9 vs GS25 at original speed',
              'Register user-supplied video4 and localize dense-coral failure',
              'Test a small receiver-only local-structure hypothesis with three-clip regression',
              'Isolate scene cuts and block boundaries, then frozen-profile long-video and holdout evaluation',
              'Bound CPU output memory, remeasure full pipeline, then optimize speed'])
state['next_priority']=state['latest_strategy_summary']['sequence']
state['next_experiment_plan']['strategy_document']=str(DOC)
save(STATE,state)
for key in ('status','active_experiment_plan','latest_experiment','latest_generalization_experiment',
            'latest_user_perceptual_feedback','preferred_research_reference','experiment_execution_gate'):
    assert state[key]==original[key],key
for p,h in source_hashes.items():assert sha(p)==h
for p,h in old_manifests.items():assert sha(p)==h
checked=0
for p in [DOC]+entries:
    text=p.read_text(encoding='utf-8')
    if p!=DOC:text=text.split('\n\n',2)[1]
    for href in re.findall(r'\]\(([^)]+)\)',text):
        assert (p.parent/href).resolve().exists(),(p,href)
        checked+=1
for p in entries+[STATE]:
    target=OUT/'after'/rel(p,WORKSPACE)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,target)
shutil.copy2(DOC,OUT/DOC.name)
receipt=dict(complete=True,date='2026-11-13',summary=str(DOC),summary_sha256=sha(DOC),
    source_sha256=source_hashes,entries_after_sha256={str(p):sha(p) for p in entries+[STATE]},
    links_checked=checked,experiment_results_unchanged=True,prior_evidence_manifests_unchanged=True,
    config_sha256=sha(STAGE/'当前采用配置.json'),new_experiments_started=False,
    no_active_experiment=True,committed=False,pushed=False)
save(OUT/'receipt.json',receipt)
files={rel(p,OUT):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()}
save(OUT/'manifest.json',dict(complete=True,files=files,self_excluded=True))
for n,v in files.items():assert sha(OUT/n)==v['sha256']
print(json.dumps(dict(complete=True,document=str(DOC),links_checked=checked,evidence_files=len(files)),ensure_ascii=False))

