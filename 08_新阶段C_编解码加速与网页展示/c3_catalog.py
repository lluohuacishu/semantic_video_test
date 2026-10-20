"""Read-only C3 projection of frozen local evidence. Importing never loads CUDA."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parents[2]
CATALOG = HERE / 'c3_existing_assets_20261020.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def transmission(file_bytes, digest):
    """Use transport.send_file's exact wire header, including its four-byte length."""
    header = json.dumps({'bytes': file_bytes, 'sha256': digest}, separators=(',', ':')).encode()
    return {'file_bytes': file_bytes, 'header_bytes': 4 + len(header),
            'application_bytes': file_bytes + 4 + len(header), 'measured_seconds': None,
            'theoretical_seconds': {str(rate): (file_bytes + 4 + len(header)) * 8 / (rate * 1000)
                                    for rate in (20, 30, 40)}}


class Catalog:
    def __init__(self, manifest=CATALOG, workspace=WORKSPACE):
        self.base = Path(workspace).resolve()
        self.manifest = read_json(manifest)
        self.assets = {}
        for key, info in self.manifest['assets'].items():
            path = (self.base / info['workspace_relative_path']).resolve()
            if not path.is_relative_to(self.base):
                raise ValueError('Catalog path outside workspace')
            exists = path.is_file()
            valid = exists and path.stat().st_size == info['bytes'] and sha256(path) == info['sha256']
            self.assets[key] = {**info, 'path': path, 'available': bool(valid),
                                'reason': None if valid else '本地文件缺失或与归档摘要不符'}
        summary = self.json_asset('evidence.c2_summary')
        self.stages = {stage['name']: stage for stage in summary['stages']}
        self.presets = read_json(REPO / '04_外部依赖快照/semantic-video-training/scripts/glc_presets.json')
        reports = {
            'report.a': REPO / '06_新阶段A_压缩路线选型/README.md',
            'report.b': REPO / '07_新阶段B_GLC语义保护与传输/B阶段收尾总结_20261019.md',
            'report.b2': REPO / '07_新阶段B_GLC语义保护与传输/B2首片实验报告_20261018.md',
            'report.c1': HERE / 'C1阶段报告_20261019.md',
            'report.c2': HERE / 'C2阶段报告_20261020.md',
        }
        for key, path in reports.items():
            self.assets[key] = {'path': path, 'kind': 'report', 'available': path.is_file(),
                                'sha256': sha256(path) if path.is_file() else None}
        self.records = [self.record(row) for row in self.manifest['main_clip_presets']]
        second = self.manifest['second_clip']
        self.records.append(self.record({'preset': 'compact', 'label': '第二片段 · 小体积',
            'source_asset': second['source_asset'], 'c2': second}, second=True))
        combined = self.stages[self.manifest['final_combined_performance']['stage_name']]
        cold = self.stages[self.manifest['cold_performance']['stage_name']]
        self.payload = {'records': self.records, 'presets': self.presets,
            'combined': {'experiment': combined['name'], **combined['performance']},
            'cold': {'experiment': cold['name'], **cold['cold']},
            'speed_goals': summary['speed_goals'],
            'history': {'videos': [self.asset_ref(k) for k in self.manifest['research_history']['assets']],
                        'reports': [self.asset_ref(k) for k in reports]},
            'evidence': self.asset_ref('evidence.c2_summary'),
            'asset_check': {'total': len(self.assets),
                            'available': sum(v['available'] for v in self.assets.values())},
            'limits': ['工程提速不证明语义 ROI 的等预算净收益。',
                       '尚未验证真实水声链路、跨设备重建或实时编解码。',
                       '第二片段的摄录来源独立性未证明；其他三档仅一对测量。']}
        report_text = reports['report.b2'].read_text(encoding='utf-8')
        base = re.search(r'\| GLC完整基础码流 \| ([\d,]+)', report_text)
        whole = re.search(r'\| 完整发送文件 \| ([\d,]+)', report_text)
        if base and whole:
            base_bytes, whole_bytes = [int(match[1].replace(',', '')) for match in (base, whole)]
            self.payload['history']['byte_note'] = (f'B2 同一片段：GLC 完整基础流 {base_bytes:,} B；'
                f'加入 ROI 后完整发送包 {whole_bytes:,} B，约 {whole_bytes/base_bytes:.3f} 倍。'
                '两者包含不同层与封装，非等预算对照；来源：B2 首片报告。')

    def asset_ref(self, key):
        info = self.assets[key]
        return {'id': key, 'url': '/assets/' + quote(key, safe=''),
                'available': info['available'], 'kind': info['kind'],
                'sha256': info.get('sha256'), 'reason': info.get('reason')}

    def path(self, key):
        info = self.assets[key]
        if not info['available'] or not info['path'].is_file():
            raise FileNotFoundError(key)
        return info['path']

    def json_asset(self, key):
        return read_json(self.path(key))

    def record(self, row, second=False):
        c2, preset = row['c2'], row['preset']
        assets = c2['assets']
        enc = self.json_asset(assets['encode_result'])
        dec = self.json_asset(assets['decode_result'])
        stage = self.stages[c2['experiment']]
        reference = row.get('b4_assets', {})
        b4_key = reference.get('stream', c2.get('reference_stream_asset'))
        stream_key = assets['stream']
        matched = bool(b4_key and self.assets[b4_key]['sha256'] == self.assets[stream_key]['sha256'])
        inputs = enc['input']
        return {'id': ('second.' if second else 'main.') + preset, 'preset': preset,
                'label': row['label'], 'clip': 'second' if second else 'main',
                'source_asset': row['source_asset'], 'kind': 'historical',
                'output': {k: inputs[k] for k in ('width', 'height', 'fps', 'processed_frames', 'duration_seconds')},
                'file_bytes': enc['file_bytes'], 'transport': transmission(enc['file_bytes'], enc['file_sha256']),
                'videos': {'source': self.asset_ref(assets['preprocessed_source_preview']),
                           'baseline': self.asset_ref(reference['decoded_preview']) if reference else None,
                           'optimized': self.asset_ref(assets['independent_decoded_preview']),
                           'comparison': self.asset_ref(reference['synchronized_comparison']) if reference.get('synchronized_comparison') else None},
                'checks': {'stream_matches_reference': matched,
                           'all_frame_checksums_match': dec['all_frame_checksums_match'],
                           'independent_decoder': dec['independent_decoder'],
                           'source_media_reads': dec['source_media_reads']},
                'performance': {'experiment': stage['name'], **stage['performance']},
                'timings': {role: self.json_asset(assets[role + '_timing']) for role in ('encode', 'decode')},
                'downloads': [self.asset_ref(value) for value in assets.values()],
                'reference_label': '同片 C1 Python' if second else '同档 B4',
                'preview_note': '代表性第 1 对用于回放；性能统计使用本组全部配对。'}
