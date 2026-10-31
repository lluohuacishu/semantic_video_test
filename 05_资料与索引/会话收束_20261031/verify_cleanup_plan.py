"""Recheck an explicit cleanup proposal and its retained dependencies. NO deletion."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parents[2]
EXPECTED_PROPOSAL_SHA256 = '508e71836e82d11ebabf37a36987cc71064d43882c0d86b309758b8a12fea4b6'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_path(relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
        raise ValueError('Invalid relative path: ' + str(relative))
    path = WORKSPACE / relative
    cursor = path
    while True:
        info = cursor.lstat()
        if getattr(info, 'st_file_attributes', 0) & 0x400 or cursor.is_symlink():
            raise ValueError('Reparse point in target ancestry: ' + str(cursor))
        if cursor == WORKSPACE:
            break
        cursor = cursor.parent
    if path.resolve() != path.absolute() or not path.resolve().is_relative_to(WORKSPACE):
        raise ValueError('Path escapes the named workspace: ' + str(path))
    if not path.is_file():
        raise ValueError('Expected a regular file: ' + str(path))
    return path


def main():
    proposal_path = HERE / '待确认清理清单.json'
    proposal_sha = digest(proposal_path)
    if proposal_sha != EXPECTED_PROPOSAL_SHA256:
        raise ValueError('Proposal changed; review and record a new approval scope first')
    data = load(proposal_path)
    if Path(data['workspace']).resolve() != WORKSPACE.resolve():
        raise ValueError('Unexpected workspace')
    candidates = []
    proposed_paths = set()
    protected_files = {WORKSPACE / name for name in data['preserved_required_files']}
    protected_roots = [WORKSPACE / name for name in data['preserved_roots']]
    sha_cache = {}

    def sha(path):
        if path not in sha_cache:
            sha_cache[path] = digest(path)
        return sha_cache[path]

    for group in data['groups']:
        for item in group['files']:
            path = checked_path(item['relative_path'])
            if path in proposed_paths:
                raise ValueError('Repeated candidate: ' + str(path))
            proposed_paths.add(path)
            if path in protected_files or any(path.is_relative_to(root) for root in protected_roots):
                raise ValueError('Candidate overlaps a preserved dependency')
            before = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_nlink) != (item['bytes'], item['mtime_ns'], item['hardlinks']):
                raise ValueError('Candidate changed since proposal: ' + str(path))
            if sha(path) != item['sha256']:
                raise ValueError('Candidate checksum changed: ' + str(path))
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Candidate changed while being checked')
            retained = item.get('retained_identical_file')
            if retained:
                original = checked_path(retained)
                if original not in protected_files or sha(original) != item['sha256']:
                    raise ValueError('Identical retained copy missing or changed')
            candidates.append(dict(group=group['id'], relative_path=item['relative_path'], bytes=item['bytes'], sha256=item['sha256']))
    print(f'Verified {len(candidates)} proposed files; no files deleted.', flush=True)

    encoder = WORKSPACE / '02_实验数据/优化_输入编码器推理/inference_20261030_01'
    decoder = WORKSPACE / '02_实验数据/优化_解码器推理/decoder_20261031_05'
    batch = WORKSPACE / '02_实验数据/优化_解码批量/batch_pilot_20261031_01'
    retained_receipts = []
    for root in protected_roots:
        if not root.is_dir():
            raise ValueError('Preserved root missing: ' + str(root))
    for clip in load(encoder / 'plan.json')['clips']:
        source = checked_path(Path(clip['source']).relative_to(WORKSPACE))
        if sha(source) != clip['source_sha256']:
            raise ValueError('Viewer source checksum changed')
    for path in sorted(protected_files):
        path = checked_path(path.relative_to(WORKSPACE))
        expected = None
        if path.name == 'decoded_rgb.npy':
            expected = load(path.with_name('result.json'))['decoded_rgb_sha256']
        if expected is not None and sha(path) != expected:
            raise ValueError('Retained RGB checksum mismatch: ' + str(path))
        retained_receipts.append(dict(relative_path=str(path.relative_to(WORKSPACE)), bytes=path.stat().st_size, sha256=sha(path)))
    for path in sorted(batch.rglob('*.npy')):
        checked_path(path.relative_to(WORKSPACE))
        if path in proposed_paths:
            raise ValueError('Batch pilot evidence must remain intact')
        retained_receipts.append(dict(relative_path=str(path.relative_to(WORKSPACE)), bytes=path.stat().st_size, sha256=sha(path)))

    engines = []
    for root in (encoder, decoder, batch):
        for path in sorted((root / 'engines').rglob('*.engine')):
            checked_path(path.relative_to(WORKSPACE))
            if path in proposed_paths:
                raise ValueError('Active engine proposed for deletion')
            receipt = path.with_name('build.json') if path.name == 'generator.engine' else path.with_name(path.stem + '_build.json')
            if sha(path) != load(receipt)['engine_sha256']:
                raise ValueError('Cached engine checksum mismatch: ' + str(path))
            engines.append(dict(relative_path=str(path.relative_to(WORKSPACE)), sha256=sha(path), receipt=str(receipt.relative_to(WORKSPACE))))
    if len(engines) != 20:
        raise ValueError('Expected 20 retained engines')

    frozen = []
    for relative, fields in (
        ('16_优化阶段_输入编码器推理加速/evidence_20261030/execution_manifest.json', ['final_code_sha256']),
        ('17_优化阶段_解码器推理加速/evidence_trt_20261031/execution_manifest.json', ['producer_code_sha256', 'web_code_sha256']),
        ('18_优化阶段_解码批量推理/evidence_20261031/execution_manifest.json', ['producer_code_sha256']),
    ):
        manifest = load(REPO / relative)
        for field in fields:
            for filename, expected in manifest[field].items():
                path = REPO / filename
                if sha(path) != expected:
                    raise ValueError('Frozen producer/web code changed: ' + filename)
                frozen.append(dict(manifest=relative, field=field, file=filename, sha256=expected))
    role = WORKSPACE / '02_实验数据/优化_部署精简/role_assets_20261029_01/manifest.json'
    if sha(role) != '7901386c96f9aecc230682676b9518f3b2d2d04831bd02ac5a5fa9b11626701a':
        raise ValueError('Role asset manifest changed')

    statuses = Counter()
    active_jobs = []
    lab = WORKSPACE / '02_实验数据/C_编解码加速与网页展示/C5_loopback/sender/inference_lab'
    for path in lab.glob('*/job.json'):
        job = load(path)
        statuses[job['status']] += 1
        if job['status'] in ('running', 'queued', 'cancelling', 'uploading'):
            active_jobs.append(dict(id=job['id'], status=job['status']))
    with urllib.request.urlopen('http://127.0.0.1:8773/inference/', timeout=10) as response:
        web = dict(url=response.url, status=response.status, bytes=len(response.read()))
    if active_jobs:
        raise ValueError('New live jobs exist; defer cleanup and inspect their dependencies')
    receipt = dict(
        checked_at=datetime.now().astimezone().isoformat(), passed=True,
        proposal_sha256=proposal_sha, deletion_performed=False, user_approval=False,
        candidate_count=len(candidates), candidate_bytes=sum(item['bytes'] for item in candidates),
        candidates=candidates, retained_files=retained_receipts, retained_engines=engines,
        frozen_code_checks=frozen, role_asset_manifest_sha256=sha(role),
        inference_job_statuses=dict(statuses), active_jobs=active_jobs, web=web,
        note='Read-only preflight only. No GPU inference, deletion, service restart or user approval. Re-run immediately before any approved deletion; do not infer authorization from this receipt.',
    )
    (HERE / '清理前只读校验.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(candidates)} candidates; {len(retained_receipts)} retained arrays; {len(engines)} engines; {len(frozen)} frozen-code checks; web HTTP {web["status"]}. No files deleted.', flush=True)


if __name__ == '__main__':
    main()
