"""Preserve the user-accepted SQ3 baseline without changing deployment."""
from pathlib import Path
import shutil
from runtime_speed_common import STAGE, REPO, DATA, WORKSPACE, file_sha, load_json, save_json

CONFIG='b5eeb844d0adcdfe97648351a0654a2575a98942e1627c98896ba302555f7b04'
SHORT=DATA/'q9_video4_jitter_20261113_03'
LONG=DATA/'sq3_long_fourway_20261113_01'
DEST=STAGE/'保留版本/SQ3_用户认可_20261113'
QUOTE='嗯，目前这个SQ3我看很不错啊，抖动减轻很多，虽然没那么锐利，但还是可以的，SQ3这个可以留一下，然后先继续优化，后面优化的不好就采用SQ3了'

def main():
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    assert not DEST.exists(), 'Never overwrite an accepted snapshot'
    DEST.mkdir(parents=True)
    feedback=dict(date='2026-11-13',quote=QUOTE,scope='Presented SQ3 full 82.6-second four-way and preceding short clips',
        interpretation='User accepts reduced jitter and some loss of sharpness. Preserve SQ3 as preferred fallback while exploring further optimization.',
        formal_deployment_change=False,preferred_fallback='SQ3',future_candidate_acceptance='requires comparative viewing')
    save_json(DEST/'用户反馈.json',feedback)
    for label,run in [('short',SHORT),('long',LONG)]:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.suffix in ('.py','.json') and 'entries_' not in str(p.relative_to(run)):
                dst=DEST/'provenance'/label/p.relative_to(run)
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    sources=['short_history_filter.py','received_structure_fusion.py','run_video4_jitter.py',
        'run_sq3_long.py','run_sq3_long_receiver.py','finish_sq3_long.py','publish_sq3_long.py',
        'runtime_speed_common.py','q9_chunk_protocol.py','structure_guide_stream.py','preserve_sq3_accepted.py']
    for name in sources:
        (DEST/'source').mkdir(exist_ok=True);shutil.copy2(STAGE/name,DEST/'source'/name)
    outputs={f'{case}_SQ3.npy':SHORT/case/'receive/SQ3/rgb.npy' for case in ('video1','video2','video3','video4')}
    outputs['long_SQ3.npy']=LONG/'receive/SQ3.npy'
    movie=WORKSPACE/'05_成果与观看/GLC_SQ3长片四宫格_20261113/82秒_四宫格_原片_H265_原GLC_SQ3.mp4'
    outputs[movie.name]=movie
    saved={}
    for name,src in outputs.items():
        dst=DEST/'outputs'/name;dst.parent.mkdir(exist_ok=True)
        shutil.copy2(src,dst);h=file_sha(src);assert file_sha(dst)==h
        saved[name]=dict(original=str(src),snapshot=str(dst),sha256=h,bytes=dst.stat().st_size,independent_copy=True)
        print('PRESERVED',name,flush=True)
    from short_history_filter import SETTINGS
    save_json(DEST/'回退配置.json',dict(name='SQ3',status='user_accepted_preferred_fallback',feedback=feedback,
        settings=SETTINGS,upstream='Existing Q9_R25 -> GS25 -> nonrecursive guide-aligned SQ3',
        short_run=str(SHORT),long_run=str(LONG),outputs=saved,formal_config_sha256=CONFIG,
        no_new_transmission=True,source_at_receiver=False,added_history_frames=2,
        upstream_Q9_long_reference_remains=True,scene_cut_detection=False,
        restore='Use the independent output copies directly; reproduce with frozen source and provenance execution plans against their hashed inputs. Do not overwrite current deployment automatically.'))
    (DEST/'README.md').write_text('# SQ3 用户认可的回退版本\n\n2026-11-13：用户认可完整长片中抖动明显减轻，接受锐度下降；继续优化不佳时采用本版本。\n\n'
        '本目录保存独立的四短片/完整长片 SQ3 数组、长片四宫格、算法源码、冻结计划、输入和输出哈希以及用户原话。大型光流和上游码流通过原运行目录及哈希引用，未重复复制。正式部署配置未改。\n\n'
        '后续候选必须另建文件和目录，以 SQ3 为比较基准；本快照不作原地修改。复现入口见 `回退配置.json` 和 `provenance/long/execution_plan.json`。\n',encoding='utf-8')
    files={str(p.relative_to(DEST)).replace('\\','/'):dict(sha256=file_sha(p),bytes=p.stat().st_size)
        for p in sorted(DEST.rglob('*')) if p.is_file()}
    save_json(DEST/'manifest.json',dict(complete=True,date='2026-11-13',files=files,self_excluded=True))
    sp=REPO/'05_资料与索引/会话收束_20261110/接续状态.json'
    state=load_json(sp);state['previous_feedback_before_sq3_acceptance']=state.get('latest_user_perceptual_feedback')
    state.update(date='2026-11-13',status='SQ3_accepted_and_preserved',latest_user_perceptual_feedback=feedback,
        user_latest_decision=QUOTE,preferred_fallback=dict(profile='SQ3',archive=str(DEST),manifest=str(DEST/'manifest.json')),
        current_session_work='SQ3 preserved with independent outputs; next round tests restrained detail enhancement against SQ3.',
        next_experiment_plan=dict(status='authorized_preparing',baseline='SQ3',focus='Recover modest sharpness while preserving stability across video1/2/3/4; no automatic replacement'))
    save_json(sp,state)
    for p in (REPO/'交接文档.md',REPO/'当前项目状态与接续.md'):
        old=p.read_text(encoding='utf-8')
        msg='> 最新（2026-11-13）：用户已认可 SQ3 完整长片，接受一定锐度损失。已独立保留为首选回退版本：`21_优化阶段_GLC解码后处理/保留版本/SQ3_用户认可_20261113`。下一轮以 SQ3 为基线温和恢复锐度，正式部署配置暂不改变。下方旧“等待长片反馈”是历史记录。\n\n'
        p.write_text(msg+old,encoding='utf-8')
    assert file_sha(STAGE/'当前采用配置.json')==CONFIG
    for name,r in files.items():assert file_sha(DEST/name)==r['sha256']
    print('SQ3 BASELINE PRESERVED',DEST,'files',len(files),flush=True)

if __name__=='__main__':main()
