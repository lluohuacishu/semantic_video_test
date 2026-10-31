"""Build and verify an explicit cleanup proposal. This script has NO delete operation."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
WORKSPACE=REPO.parents[2]
AUDIT=Path('C:/Users/lenovo/.codex/tmp/semantic-handoff-cleanup-20261031')


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    if (Path(__file__).parent/'待确认清理清单.json').exists():
        raise ValueError('Immutable proposal already exists; do not overwrite approval/deletion provenance')
    root=WORKSPACE.resolve();groups=defaultdict(list);protected=set();notes=[]
    records=[json.loads(line) for line in (AUDIT/'files.jsonl').open(encoding='utf-8')]
    decoder=root/'02_实验数据/优化_解码器推理/decoder_20261031_05'
    encoder=root/'02_实验数据/优化_输入编码器推理/inference_20261030_01'
    batch=root/'02_实验数据/优化_解码批量/batch_pilot_20261031_01'
    for clip in load(encoder/'plan.json')['clips']:
        protected.add(Path(clip['source']).resolve())
    for clip in ('long108_q0_640','sf007_q12_512','sf025_q32_512','uv096_q63_512_255'):
        for variant in ('eager','trt_fp32_stream','trt_fp16_stream'):
            protected.add((encoder/'full'/clip/variant/'receiver/decoded_rgb.npy').resolve())
    for backend in ('eager','trt_fp32','trt_fp16_safe'):
        name='eager1' if backend=='eager' else backend+'_1'
        protected.add((decoder/'short'/name/'decoded_rgb.npy').resolve())
    required_roots=[root/'02_实验数据/优化_部署精简/role_assets_20261029_01',
                    root/'02_实验数据/优化_输入编码器推理/engine_packages_1050',
                    root/'02_实验数据/C_编解码加速与网页展示/C5_loopback',
                    root/'02_实验数据/D_语义目标增强/原始数据']
    for row in records:
        relative=Path(row['path']);parts=relative.parts;path=(root/relative).resolve()
        if not path.is_relative_to(root) or '.git' in parts:continue
        group=None;retained=None
        if len(parts)>=3 and parts[0]=='02_实验数据' and parts[1]=='优化_解码器推理' and parts[2] in (
                'decoder_20261031_01','decoder_20261031_02','decoder_20261031_03','decoder_20261031_04'):
            if path.suffix.lower() in ('.engine','.onnx'):group='A1'
            elif path.name=='decoded_rgb.npy':group='A2'
        elif path.suffix.lower()=='.onnx' and any(path.is_relative_to(folder/'engines') for folder in (encoder,decoder,batch)):
            group='A3'
        elif path.name=='decoded_rgb.npy' and path.is_relative_to(decoder/'short'):
            folder=path.parent.name
            if folder in ('eager2','eager3','trt_fp32_2','trt_fp32_3','trt_fp16_safe_2','trt_fp16_safe_3'):
                group='A4';name='eager1' if folder.startswith('eager') else folder.rsplit('_',1)[0]+'_1'
                retained=decoder/'short'/name/'decoded_rgb.npy'
        elif path.name=='decoded_rgb.npy' and path.is_relative_to(decoder/'full1239'):group='B1'
        elif path.name=='decoded_rgb.npy' and path.is_relative_to(decoder/'multi'):group='B2'
        if group is None:continue
        if path in protected or any(path.is_relative_to(folder) for folder in required_roots):
            raise ValueError('Proposed target intersects protected dependencies: '+str(path))
        current=path.stat()
        if current.st_size!=row['bytes'] or current.st_mtime_ns!=row['mtime_ns']:
            raise ValueError('File changed since census: '+str(path))
        if getattr(current,'st_file_attributes',0)&0x400:raise ValueError('Reparse file in proposal')
        digest=sha(path)
        item=dict(relative_path=str(relative),bytes=current.st_size,mtime_ns=current.st_mtime_ns,
                  sha256=digest,hardlinks=current.st_nlink)
        if retained is not None:
            if retained.resolve() not in protected or sha(retained)!=digest:
                raise ValueError('Repeated RGB is not identical to its retained original')
            item['retained_identical_file']=str(retained.relative_to(root))
        groups[group].append(item)
    descriptions={
        'A1':('旧解码调试引擎与导出图','01–04已被05替代，仅保留JSON/CSV、日志、代码快照。当前网页和CLI不用这些二进制。'),
        'A2':('旧调试版本的解码RGB数组','01–04已结束的临时输出；失败诊断、摘要和原始码流保留。'),
        'A3':('已缓存引擎对应的ONNX中间图','当前16/17/18有效.engine及build.json保留，正常运行命中缓存不读取ONNX。手动分析/重导图需重新导出；不是删除运行引擎。'),
        'A4':('重复计时生成的完全相同RGB副本','6份重复输出已与各自保留的首轮数组逐文件SHA核对。计时JSON和首轮数组保留。'),
        'B1':('完整1239帧实验的两份无压缩RGB','可选：视频、码流、逐帧指标和摘要保留；删除后重算画质/逐像素分析需重新解码，约数分钟。'),
        'B2':('多素材解码实验的无压缩RGB','可选：007/025/096的9份派生RGB。已有指标/视频保留，但以后重算这些画质指标需重新解码。'),
    }
    proposal=[]
    for identifier in sorted(groups):
        title,impact=descriptions[identifier];files=groups[identifier]
        proposal.append(dict(id=identifier,title=title,impact=impact,recommendation='recommended' if identifier.startswith('A') else 'optional',
                             file_count=len(files),logical_bytes=sum(f['bytes'] for f in files),files=files))
    result=dict(schema='semantic-cleanup-proposal-v1',workspace=str(root),created_date='2026-10-31',
        approved=False,deletion_performed=False,groups=proposal,
        recommended_bytes=sum(g['logical_bytes'] for g in proposal if g['id'].startswith('A')),
        optional_bytes=sum(g['logical_bytes'] for g in proposal if g['id'].startswith('B')),
        preserved_required_files=[str(p.relative_to(root)) for p in sorted(protected)],
        preserved_roots=[str(p.relative_to(root)) for p in required_roots],
        preserve_all=['all .git directories and branches','all source datasets and original uploads','all model/role/student checkpoints and environments',
                      'active TensorRT .engine files in stage16/17/18','all reports JSON CSV logs and source-code snapshots',
                      'all finished playback MP4s and transport streams','all junctions and external runtime directories'],
        note='NO DELETION. Logical sizes are estimates; hardlinks/NTFS allocation may reduce actual physical reclaim. User confirmation is still required, followed by another exact file check.')
    destination=Path(__file__).parent/'待确认清理清单.json'
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Proposal SHA256:',sha(destination),flush=True)
    for group in proposal:print(f"{group['id']}: {group['file_count']} files, {group['logical_bytes']/1024**3:.3f} GiB, {group['title']}",flush=True)
    print('No files deleted.',flush=True)


if __name__=='__main__':main()
