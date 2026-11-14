"""Read-back final deliverables and confirm preserved SQ3 evidence."""
from pathlib import Path
import re
import subprocess
from runtime_speed_common import STAGE,REPO,file_sha,load_json,save_json
from preserve_sq3_accepted import DEST,CONFIG
from run_sq3_detail import RUN
from finish_sq3_detail import VIEW,REPORT

records={}
for directory in (DEST,STAGE/'evidence_20261113/video4_short_history_jitter',
        STAGE/'evidence_20261113/sq3_long_fourway',STAGE/'evidence_20261113/sq3_detail'):
    manifest=load_json(directory/'manifest.json')
    for name,r in manifest['files'].items():
        assert file_sha(directory/name)==r['sha256'],(directory,name)
    records[str(directory)]=len(manifest['files'])
fallback=load_json(DEST/'回退配置.json')
for name,r in fallback['outputs'].items():
    assert file_sha(r['original'])==r['sha256'],('original changed',name)
    assert file_sha(r['snapshot'])==r['sha256'],('snapshot changed',name)
for name in ('short_history_filter.py','received_structure_fusion.py','run_video4_jitter.py',
        'run_sq3_long.py','run_sq3_long_receiver.py'):
    assert file_sha(STAGE/name)==file_sha(DEST/'source'/name)
for name,r in load_json(VIEW/'media_manifest.json')['files'].items():
    assert file_sha(VIEW/name)==r['sha256']
links=0
for p in (REPORT,VIEW/'README.md'):
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
        assert (p.parent/target).resolve().exists(),(p,target)
        links+=1
state=load_json(REPO/'05_资料与索引/会话收束_20261110/接续状态.json')
assert state['status']=='SQ3_preserved_detail_round_complete'
assert state['active_experiment_plan'] is None and not state['experiment_execution_gate']['currently_running']
assert state['preferred_fallback']['profile']=='SQ3'
assert state['latest_experiment']['selected'] is None
assert file_sha(STAGE/'当前采用配置.json')==CONFIG
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
assert head=='ae46c91c6820ffc49012714c44659ead0e1fc162'
for p in (RUN/'verification.json',RUN/'summary.json',RUN/'publication.json',RUN/'previews/verification.json'):
    assert load_json(p)['complete']
save_json(RUN/'closeout_verification.json',dict(complete=True,manifests_verified=records,
    original_and_archived_SQ3_outputs_unchanged=True,core_SQ3_sources_unchanged=True,
    links_checked=links,formal_config_unchanged=True,git_HEAD_unchanged=head,
    state_complete=True,preferred_fallback='SQ3',no_candidate_adopted=True))
print('CLOSEOUT VERIFIED',records,'links',links,flush=True)
