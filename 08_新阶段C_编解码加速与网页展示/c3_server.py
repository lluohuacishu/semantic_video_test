"""C3 local replay and explicit single-job execution; no GPU work on page load."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import traceback
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlsplit
from urllib.request import ProxyHandler, build_opener

from c3_catalog import Catalog, HERE, REPO, WORKSPACE, read_json, sha256, transmission
from c2_watch import inspect, supervise, write
from c4_config import load_settings, UploadCatalog

LEGACY = REPO / '04_外部依赖快照/semantic-video-training'
sys.path.insert(0, str(LEGACY))
from demo_server import Handler as BaseHandler, JobStore, MAX_UPLOAD, TERMINAL, filename_info, now

APP_ID = 'glc-c3-local-v1'
STATIC = HERE / 'c3_console'
DATA = WORKSPACE / '02_实验数据/C_编解码加速与网页展示'
JOB_ASSETS = {'source.mp4': 'encode/encoder/source_preview.mp4',
              'decoded.mp4': 'decode/receiver/reconstruction_preview.mp4',
              'stream.glcv': 'received.glcv', 'encode.json': 'encode/encoder/encode.json',
              'decode.json': 'decode/receiver/decode.json', 'result.json': 'result.json',
              'encode-timing.json': 'encode/timing.json', 'decode-timing.json': 'decode/timing.json',
              'encode.log': 'encode_process/worker.log', 'decode.log': 'decode_process/worker.log',
              'manifest.json': 'manifest.json', 'failure.json': 'failure.json',
              'browser.json': 'browser.json', 'events.jsonl': 'events.jsonl'}


class DirectoryLease:
    """Hold an OS lock so a second server cannot relabel an active task store."""
    def __init__(self, directory):
        self.handle = (Path(directory) / '.server.lock').open('a+b')
        if self.handle.tell() == 0:
            self.handle.write(b'0')
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError('此任务目录已有服务在使用') from None

    def close(self):
        self.handle.close()


def parse_options(value, presets):
    if not isinstance(value, dict) or set(value) - {'preset', 'frames', 'backend'}:
        raise ValueError('新任务只接受档位、帧数和熵后端参数')
    preset = value.get('preset')
    if preset not in presets:
        raise ValueError('请手动选择四档之一')
    frames = value.get('frames')
    if frames is not None and (type(frames) is not int or not 1 <= frames <= presets[preset]['frames']):
        raise ValueError('帧数必须在 1 与该档默认帧数之间')
    backend = value.get('backend', 'native')
    if backend not in ('native', 'python'):
        raise ValueError('未知熵后端')
    return {'preset': preset, 'frames': frames, 'backend': backend}


def verify_result(folder):
    folder = Path(folder)
    enc = read_json(folder / JOB_ASSETS['encode.json'])
    dec = read_json(folder / JOB_ASSETS['decode.json'])
    digest = sha256(folder / 'received.glcv')
    if enc['file_sha256'] != digest or dec['input_sha256'] != digest:
        raise ValueError('完整码流摘要不一致')
    if (not dec['independent_decoder'] or dec['source_media_reads'] or
            not dec['all_frame_checksums_match'] or not dec['whole_file_checksum_verified']):
        raise ValueError('独立解码或容器校验失败')
    if len(enc['frames']) != len(dec['frames']) or not enc['frames']:
        raise ValueError('编解码帧数不一致')
    for a, b in zip(enc['frames'], dec['frames']):
        if any(a[key] != b[key] for key in ('frame', 'rgb_sha256', 'latent_sha256')):
            raise ValueError('逐帧 RGB/潜变量摘要不一致')
    return enc, dec


class C3Store(JobStore):
    def create(self, *args, **kwargs):
        with self.lock:
            job = super().create(*args, **kwargs)
            write(self.directory / job['id'] / 'initial.json', job)
            with (self.directory / job['id'] / 'events.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps({key: job.get(key) for key in ('updated_at','status','stage','phase')}, ensure_ascii=False) + '\n')
            return job

    def update(self, identifier, **changes):
        with self.lock:
            job = self.jobs[identifier]
            updated = {**job, **changes, 'updated_at': now()}
            write(self.directory / identifier / 'job.json', updated)
            self.jobs[identifier] = updated
            if any(job.get(key) != updated.get(key) for key in ('status', 'stage', 'phase')):
                with (self.directory / identifier / 'events.jsonl').open('a', encoding='utf-8') as log:
                    log.write(json.dumps({key: updated.get(key) for key in ('updated_at','status','stage','phase')}, ensure_ascii=False) + '\n')
            return copy.deepcopy(updated)


class C3App:
    def __init__(self, directory, catalog=None, runtime=None):
        self.catalog = catalog or Catalog()
        self.runtime = {key: Path(value) for key, value in (runtime or load_settings()['runtime']).items()}
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lease = DirectoryLease(self.directory)
        self.blockers = []
        # Inspect original process records before JobStore labels interrupted jobs.
        for record in self.directory.glob('*/*_process/process.json'):
            try:
                state = inspect(record.parent)['observed_state']
                if state in ('running', 'orphan_running', 'liveness_unknown'):
                    self.blockers.append(str(record.parent))
            except (OSError, ValueError, KeyError):
                self.blockers.append(str(record.parent))
        self.store = C3Store(self.directory)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='c3-serial')
        self.closed = False

    def config(self):
        missing = [key for key, path in self.runtime.items() if not path.exists()]
        return {'app': APP_ID, 'runtime_available': not set(missing) - {'native_library'} and not self.blockers and not self.closed,
                'native_available': 'native_library' not in missing,
                'missing': missing, 'blocked_processes': self.blockers,
                'max_upload_bytes': MAX_UPLOAD, 'presets': self.catalog.presets,
                'execution_mode': 'separate_fresh_processes', 'gpu_started_on_open': False,
                'history_available': bool(self.catalog.records), 'version': 'C4'}

    def reserve(self, value):
        options = parse_options(value.get('options'), self.catalog.presets['presets'])
        source_id = value.get('source_asset')
        if source_id:
            if source_id not in ('source.main', 'source.second'):
                raise ValueError('只能选用已登记的素材')
            source = self.catalog.path(source_id)
            job = self.store.create(source.name, 'video', options)
            return self.store.update(job['id'], source_asset=source_id, bytes=source.stat().st_size,
                                     status='staged', phase='素材已准备，请明确点击开始运行')
        filename, extension, kind = filename_info(value.get('filename', ''))
        size = value.get('bytes')
        if kind != 'video' or type(size) is not int or not 0 < size <= MAX_UPLOAD:
            raise ValueError('请选择 1 GiB 以内的非空视频')
        job = self.store.create(filename, 'video', options)
        return self.store.update(job['id'], expected_bytes=size, extension=extension, receiving=False)

    def source(self, job):
        if job.get('source_asset'):
            return self.catalog.path(job['source_asset'])
        return self.directory / job['id'] / ('source' + job['extension'])

    def start(self, identifier):
        with self.store.lock:
            job = self.store.get(identifier)
            if job['status'] != 'staged':
                raise ValueError('仅能启动已准备且尚未运行的任务')
            if not self.config()['runtime_available']:
                raise ValueError('运行环境缺失或遗留进程状态不确定；已有成果仍可回放')
            if job['options']['backend'] == 'native' and not self.config()['native_available']:
                raise ValueError('原生 DLL 不可用，请新建任务选择 Python 回退路径')
            if not self.source(job).is_file():
                raise ValueError('素材已不存在，请重新上传')
            queued = self.store.update(identifier, status='queued', phase='等待单 GPU 队列', queued_at=now())
            self.executor.submit(self.run, identifier)
            return queued

    def cancel(self, identifier):
        with self.store.lock:
            job = self.store.get(identifier)
            if job['status'] in TERMINAL:
                return job
            self.store.events[identifier].set()
            (self.directory / identifier / 'STOP_REQUESTED').touch()
            immediate = job['status'] in ('staged', 'queued') or (job['status'] == 'uploading' and not job.get('receiving'))
            return self.store.update(identifier, status='cancelled' if immediate else 'cancelling',
                                     cancel_requested_at=now(),
                                     phase='已停止' if immediate else '正在等待当前帧结束并释放资源')

    def execute(self, role, task, folder):
        if (folder / 'STOP_REQUESTED').exists():
            raise InterruptedError('任务已停止')
        process_folder = folder / (role + '_process')
        config = {key: str(value) for key, value in self.runtime.items() if key != 'python'}
        config.update(role=role, device='cuda', process_folder=str(process_folder),
                      stop_file=str(folder / 'STOP_REQUESTED'), tasks=[task])
        # supervise owns creation of its fresh process directory.
        config_path = folder / (role + '_config.json')
        write(config_path, config)
        try:
            result = supervise([str(self.runtime['python']), '-B', str(HERE / 'c2_worker.py'), '--config', str(config_path)],
                               process_folder, REPO, folder / 'STOP_REQUESTED', poll_seconds=0.2)
        finally:
            try:
                uncertain = inspect(process_folder)['observed_state'] in ('running', 'orphan_running', 'liveness_unknown')
            except (OSError, ValueError, KeyError):
                uncertain = True
            if uncertain:
                self.blockers.append(str(process_folder))
        if uncertain:
            raise RuntimeError('工作进程退出状态不确定，已阻止后续任务；请检查进程记录')
        if (folder / 'STOP_REQUESTED').exists():
            raise InterruptedError('任务已停止')
        if result['returncode'] != 0:
            raise RuntimeError(f'{role} 进程未完成，请查看日志')

    def run(self, identifier):
        folder = self.directory / identifier
        stop = self.store.events[identifier]
        try:
            if stop.is_set() or self.closed:
                return
            if self.blockers:
                raise RuntimeError('遗留工作进程状态不确定，队列已停止')
            started = time.perf_counter()
            job = self.store.get(identifier)
            from datetime import datetime
            queue_seconds = (datetime.fromisoformat(now()) - datetime.fromisoformat(job['queued_at'])).total_seconds()
            self.store.update(identifier, status='running', phase='编码：准备模型与素材', stage='encode', started_at=now())
            source = self.source(job)
            files = list(HERE.glob('c3_*.py')) + [HERE / n for n in ('c4_config.py','c1_runtime.py','c1_io.py','c2_worker.py','c2_native.py','c2_watch.py')]
            files += [STATIC / n for n in ('index.html','app.js','styles.css')]
            files += list((LEGACY / 'scripts').glob('glc_*.py')) + [LEGACY / 'scripts/glc_presets.json']
            write(folder / 'manifest.json', {'created_at': now(), 'source_sha256': sha256(source),
                  'source_bytes': source.stat().st_size, 'source': str(source), 'options': job['options'],
                  'code_sha256': {str(p.relative_to(REPO)): sha256(p) for p in files},
                  'native_sha256': sha256(self.runtime['native_library']) if job['options']['backend'] == 'native' else None,
                  'mode': 'separate_fresh_processes', 'network_transfer_performed': False})
            task = {'id': 'encode', 'input': str(source), 'output': str(folder / 'encode'),
                    'output_mode': 'display', 'profile': False, **job['options']}
            self.execute('encode', task, folder)
            if stop.is_set():
                raise InterruptedError()
            shutil.copyfile(folder / 'encode/stream.glcv', folder / 'received.glcv')
            self.store.update(identifier, phase='独立解码：仅接收完整码流', stage='decode')
            self.execute('decode', {**task, 'id': 'decode', 'input': str(folder / 'received.glcv'),
                                   'output': str(folder / 'decode')}, folder)
            self.store.update(identifier, phase='核对逐帧摘要与预览', stage='verify')
            enc, dec = verify_result(folder)
            timings = {role: read_json(folder / role / 'timing.json') for role in ('encode', 'decode')}
            for name in ('source.mp4', 'decoded.mp4'):
                if not (folder / JOB_ASSETS[name]).is_file() or not (folder / JOB_ASSETS[name]).stat().st_size:
                    raise ValueError('预览未就绪')
            result = {'kind': 'new_run', 'id': identifier, 'preset': job['options']['preset'],
                      'backend': job['options']['backend'], 'output': enc['input'],
                      'file_bytes': enc['file_bytes'], 'transport': transmission(enc['file_bytes'], enc['file_sha256']),
                      'timings': timings, 'queue_seconds': queue_seconds,
                      'service_seconds': time.perf_counter() - started,
                      'browser_load_seconds': None, 'network_transfer_seconds': None,
                      'execution_mode': 'separate_fresh_processes', 'verified': True,
                      'checks': {'frames': len(dec['frames']), 'all_rgb_and_latent_equal': True,
                                 'source_media_reads': dec['source_media_reads'], 'stream_sha256': enc['file_sha256']},
                      'videos': {key: '/jobs/' + identifier + '/' + name for key, name in [('source','source.mp4'),('optimized','decoded.mp4')]},
                      'downloads': [{'name': name, 'url': '/jobs/' + identifier + '/' + name}
                                    for name in JOB_ASSETS if (folder / JOB_ASSETS[name]).is_file() or name == 'result.json']}
            write(folder / 'result.json', result)
            with self.store.lock:
                if stop.is_set():
                    raise InterruptedError()
                self.store.update(identifier, status='done', stage='done', phase='已完成：逐帧校验通过', finished_at=now())
        except BaseException as exc:
            cancelled = isinstance(exc, InterruptedError) or stop.is_set()
            write(folder / 'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
            self.store.update(identifier, status='cancelled' if cancelled else 'error',
                              phase='已停止，未完成结果不计入成功' if cancelled else '任务失败，请查看日志',
                              error=str(exc), finished_at=now())

    def get_job(self, identifier):
        job = self.store.get(identifier)
        folder = self.directory / identifier
        if job['status'] in ('running', 'cancelling'):
            role = job.get('stage', 'encode')
            try:
                beat = read_json(folder / (role + '_process') / 'heartbeat.json')
                job['heartbeat'] = {k: beat.get(k) for k in ('state','heartbeat_utc','last_progress','task_state')}
            except (OSError, ValueError):
                job['heartbeat'] = None
        if job['status'] == 'done':
            job['result'] = read_json(folder / 'result.json')
        job['logs'] = [{'name': name, 'url': '/jobs/' + identifier + '/' + name}
                       for name in ('encode.log','decode.log','failure.json','manifest.json')
                       if (folder / JOB_ASSETS[name]).is_file()]
        return job

    def browser_metric(self, identifier, value):
        with self.store.lock:
            job = self.store.get(identifier)
            if job['status'] != 'done':
                raise ValueError('预览计时只接受已完成任务')
            for key in ('preview_load_seconds', 'dispatch_to_preview_seconds'):
                number = value.get(key)
                if number is None and key == 'dispatch_to_preview_seconds':
                    continue
                if type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= 86400:
                    raise ValueError('浏览器计时无效')
            path = self.directory / identifier / 'browser.json'
            rows = read_json(path) if path.exists() else []
            rows.append({'recorded_at': now(), 'preview_load_seconds': value['preview_load_seconds'],
                         'dispatch_to_preview_seconds': value.get('dispatch_to_preview_seconds'),
                         'after_refresh': value.get('after_refresh') is True,
                         'source': 'browser performance.now; self-reported; excludes upload and real network transfer'})
            write(path, rows[-100:])
            return {'saved': True}

    def close(self):
        self.closed = True
        for job in self.store.list():
            if job['status'] not in TERMINAL:
                self.cancel(job['id'])
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.lease.close()


class Handler(BaseHandler):
    server_version = 'GLC-C3/1'

    def send_header(self, keyword, value):
        if keyword.lower() == 'content-disposition':
            name = value.split('filename="', 1)[-1].rstrip('"')
            value = "attachment; filename*=UTF-8''" + quote(name, safe='')
        super().send_header(keyword, value)

    def do_GET(self):
        try:
            self.check_origin()
            url = urlsplit(self.path)
            route = unquote(url.path)
            if route == '/api/config':
                return self.json(self.app.config())
            if route == '/api/catalog':
                return self.json(self.app.catalog.payload)
            if route == '/api/jobs':
                return self.json(self.app.store.list())
            match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})', route)
            if match:
                return self.json(self.app.get_job(match[1]))
            if route.startswith('/assets/'):
                return self.file(self.app.catalog.path(route.removeprefix('/assets/')), download='download' in parse_qs(url.query))
            match = re.fullmatch(r'/jobs/([0-9a-f]{32})/([^/]+)', route)
            if match:
                job = self.app.store.get(match[1])
                name = match[2]
                if name not in JOB_ASSETS or (job['status'] != 'done' and name not in ('encode.log','decode.log','failure.json','manifest.json')):
                    raise KeyError(route)
                return self.file(self.app.directory / match[1] / JOB_ASSETS[name], download='download' in parse_qs(url.query))
            if route == '/archive' and not self.app.catalog.records:
                return self.json({'error': '本机未配置历史素材；首页上传功能可独立使用'}, 404)
            files = {'/': 'index.html', '/app.js': 'app.js', '/styles.css': 'styles.css', '/favicon.svg':'favicon.svg',
                     '/archive': 'archive.html', '/archive.js': 'archive.js', '/archive.css': 'archive.css'}
            return self.file(STATIC / files[route])
        except PermissionError as exc:
            self.json({'error': str(exc)}, 403)
        except (KeyError, FileNotFoundError):
            self.json({'error': '文件或结果不可用'}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def body(self):
        if self.headers.get('Transfer-Encoding'):
            raise ValueError('不支持分块请求')
        size = int(self.headers.get('Content-Length', '0'))
        if not 0 < size <= 4096:
            raise ValueError('参数长度无效')
        self.connection.settimeout(15)
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, dict):
            raise ValueError('参数应为对象')
        return value

    def upload(self, identifier):
        if self.headers.get('Transfer-Encoding'):
            raise ValueError('上传需要 Content-Length')
        size = int(self.headers.get('Content-Length', '0'))
        with self.app.store.lock:
            job = self.app.store.get(identifier)
            if job['status'] != 'uploading' or job.get('receiving') or size != job['expected_bytes']:
                raise ValueError('上传长度或任务状态不匹配')
            self.app.store.update(identifier, receiving=True)
        started = time.perf_counter()
        try:
            source = self.app.source(job)
            temporary = source.with_suffix(source.suffix + '.part')
            remaining = size
            self.connection.settimeout(30)
            with temporary.open('xb') as out:
                while remaining:
                    if self.app.store.events[identifier].is_set():
                        raise InterruptedError('上传已停止')
                    data = self.rfile.read(min(1024 * 1024, remaining))
                    if not data:
                        raise ValueError('上传中断')
                    out.write(data)
                    remaining -= len(data)
            with self.app.store.lock:
                if self.app.store.events[identifier].is_set():
                    raise InterruptedError('上传已停止')
                temporary.replace(source)
                return self.app.store.update(identifier, status='staged', receiving=False, bytes=size,
                    upload_seconds=time.perf_counter() - started, phase='上传完成，请明确点击开始运行')
        except BaseException as exc:
            self.app.store.update(identifier, status='cancelled' if self.app.store.events[identifier].is_set() else 'error',
                                  receiving=False, phase='上传未完成', error=str(exc))
            raise

    def do_POST(self):
        try:
            self.check_origin()
            route = urlsplit(self.path).path
            if route == '/api/jobs/reserve':
                return self.json(self.app.reserve(self.body()), 201)
            match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})/(source|start|cancel|browser)', route)
            if not match:
                raise KeyError(route)
            identifier, action = match.groups()
            if action == 'browser':
                return self.json(self.app.browser_metric(identifier, self.body()))
            if action == 'source':
                return self.json(self.upload(identifier))
            if action == 'start':
                return self.json(self.app.start(identifier), 202)
            return self.json(self.app.cancel(identifier))
        except PermissionError as exc:
            self.json({'error': str(exc)}, 403)
        except KeyError:
            self.json({'error': '任务不存在'}, 404)
        except (ValueError, OSError) as exc:
            self.close_connection = True
            try:
                self.json({'error': str(exc)}, 400)
            except (BrokenPipeError, ConnectionError):
                pass


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app):
        self.app = app
        super().__init__(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8773)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--check', action='store_true', help='Check paths without opening a server or starting GPU work')
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.check:
        missing = [key for key, path in settings['runtime'].items() if not path.exists()]
        print(json.dumps({'config': str(settings['config_path']), 'runtime': {k:str(v) for k,v in settings['runtime'].items()},
                          'missing': missing, 'data_dir': str(settings['data_dir']), 'gpu_started': False}, ensure_ascii=False, indent=2))
        return 1 if set(missing) - {'native_library'} else 0
    url = f'http://127.0.0.1:{args.port}/'
    try:
        with build_opener(ProxyHandler({})).open(url + 'api/config', timeout=1) as response:
            existing = json.load(response)
        if existing.get('app') == APP_ID:
            print('C3 服务已启动：' + url, flush=True)
            if args.open:
                import webbrowser
                webbrowser.open(url)
            return
    except (OSError, ValueError):
        pass
    # Bind before any task store changes; a duplicate launch cannot disrupt jobs.
    with Server(('127.0.0.1', args.port), None) as server:
        catalog = UploadCatalog()
        if settings['history_enabled']:
            try:
                catalog = Catalog(workspace=settings['workspace'])
            except (OSError, ValueError, KeyError) as exc:
                catalog = UploadCatalog('历史素材未完整配置：' + str(exc))
                print(catalog.payload['history_reason'], flush=True)
        app = C3App(args.data_dir or settings['data_dir'], catalog=catalog, runtime=settings['runtime'])
        server.app = app
        print(f'C3_READY http://127.0.0.1:{args.port}/', flush=True)
        if args.open:
            import webbrowser
            webbrowser.open(url)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
        finally:
            app.close()


if __name__ == '__main__':
    sys.exit(main())
