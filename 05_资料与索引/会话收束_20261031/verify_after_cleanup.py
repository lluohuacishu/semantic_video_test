"""Verify the approved deletion receipt and the preserved assets, without GPU work."""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parents[2]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    proposal_file = HERE / '待确认清理清单.json'
    proposal, approval, execution, before = [load(HERE / name) for name in (
        '待确认清理清单.json', '清理授权.json', '清理执行记录.json', '清理前只读校验.json')]
    if not approval['approved'] or not execution['passed']:
        raise ValueError('No successful approved execution')
    if not (sha(proposal_file) == approval['proposal_sha256'] == execution['proposal_sha256'] == before['proposal_sha256']):
        raise ValueError('Proposal identity mismatch')
    expected_deleted = {item['relative_path']: item for group in proposal['groups']
                        if group['id'] in approval['approved_groups'] for item in group['files']}
    deleted = {item['relative_path']: item for item in execution['deleted_files']}
    if len(deleted) != 48 or set(deleted) != set(expected_deleted):
        raise ValueError('Deletion scope does not equal the approved 48 files')
    for relative, item in deleted.items():
        if (WORKSPACE / relative).exists() or item['sha256'] != expected_deleted[relative]['sha256']:
            raise ValueError('Deleted target or receipt mismatch: ' + relative)
    retained_optional = []
    for group in proposal['groups']:
        if group['id'] not in approval['preserve_groups']:
            continue
        for item in group['files']:
            path = WORKSPACE / item['relative_path']
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('B file changed: ' + str(path))
            retained_optional.append(item['relative_path'])
    for item in before['retained_files'] + before['retained_engines']:
        if sha(WORKSPACE / item['relative_path']) != item['sha256']:
            raise ValueError('Retained dependency changed: ' + item['relative_path'])
    for item in before['frozen_code_checks']:
        if sha(REPO / item['file']) != item['sha256']:
            raise ValueError('Frozen producer or web source changed: ' + item['file'])
    role_root = WORKSPACE / '02_实验数据/优化_部署精简/role_assets_20261029_01'
    if sha(role_root / 'manifest.json') != before['role_asset_manifest_sha256']:
        raise ValueError('Role manifest changed')
    for role in ('sender_image', 'sender_video', 'receiver_image', 'receiver_video'):
        if not (role_root / (role + '.pt')).is_file():
            raise ValueError('Role checkpoint missing')
    for relative in proposal['preserved_roots']:
        if not (WORKSPACE / relative).is_dir():
            raise ValueError('Preserved directory missing')
    dataset = WORKSPACE / '02_实验数据/D_语义目标增强/原始数据/SFISHTRACK.zip'
    upload = WORKSPACE / '02_实验数据/C_编解码加速与网页展示/C5_loopback/sender/inference_lab/cfb1a1b4cba37b3d569a04b4ab73bcf2/upload.mp4'
    for path in (dataset, upload):
        if not path.is_file():
            raise ValueError('Original media missing')
    with urllib.request.urlopen('http://127.0.0.1:8773/inference/', timeout=10) as response:
        web = dict(url=response.url, status=response.status, bytes=len(response.read()))
    result = dict(
        checked_at=datetime.now().astimezone().isoformat(), passed=True,
        proposal_sha256=sha(proposal_file), approval_sha256=sha(HERE / '清理授权.json'),
        execution_receipt_sha256=sha(HERE / '清理执行记录.json'),
        approved_groups=approval['approved_groups'], deleted_files=len(deleted),
        deleted_logical_bytes=sum(item['bytes'] for item in deleted.values()),
        retained_B_files=retained_optional, retained_array_checks=len(before['retained_files']),
        retained_engine_checks=len(before['retained_engines']), frozen_code_checks=len(before['frozen_code_checks']),
        dataset_bytes=dataset.stat().st_size, original_upload_bytes=upload.stat().st_size,
        role_weights_present=4, web=web, new_GPU_jobs=0,
        note='B groups unchanged by SHA256; all 28 retained arrays and 20 active engines unchanged; 39 frozen code checks passed. HTTP check only, no inference or service restart.',
    )
    (HERE / '清理后核验.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
