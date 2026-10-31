"""Copy the small pre-handover entry documents, preserving their hashes. No moves/deletes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = []
    sources = [(REPO / name, '仓库/' + name) for name in
               ('README.md', '交接文档.md', '当前项目状态与接续.md', '新会话接续.txt', '项目总结与未来方向.md')]
    sources += [(WORKSPACE / name, '工作区入口/' + name) for name in ('开始这里.md', '新会话接续.txt')]
    for source, relative in sources:
        target = HERE / '整理前文档' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError('Snapshot already exists; do not overwrite historical copies')
        shutil.copy2(source, target)
        if sha(source) != sha(target):
            raise ValueError('Snapshot verification failed')
        records.append(dict(original=str(source), snapshot=str(target.relative_to(HERE)), bytes=target.stat().st_size, sha256=sha(target)))
    (HERE / '整理前文档/manifest.json').write_text(json.dumps(dict(created_date='2026-10-31', files=records,
        note='Small document copies only; historical statements are not current instructions. Relative links retain their original repository/workspace base.'),
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Archived and verified', len(records), 'small entry documents;', sum(item['bytes'] for item in records), 'bytes. No files moved or deleted.')


if __name__ == '__main__':
    main()
