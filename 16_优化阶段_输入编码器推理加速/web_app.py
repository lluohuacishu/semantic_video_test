"""C5-mounted inference lab. Shares the existing sender GPU executor; no GPU imports."""
from __future__ import annotations
import copy
import json
import math
import re
import secrets
import threading
import time
from urllib.parse import urlsplit
from experiment_common import *
from c2_watch import supervise,write,inspect
from web_frames import Frames,discard_working_frames

MODES={'eager':'原GLC','amp_fp16':'局部FP16','trt_fp32_stream':'TensorRT FP32','trt_fp16_stream':'TensorRT FP16'}
PROFILES={'amp_fp16':'pt251-amp16','trt_fp32_stream':'trt105-f32','trt_fp16_stream':'trt105-f16'}
TERMINAL={'done','error','cancelled','interrupted'}
BASE='/inference/'


def options(value):
    if not isinstance(value,dict) or set(value)-{'sample','source_job','filename','bytes','modes','frames','size','q','range','fps_mode'}:raise ValueError('无效实验参数')
    modes=value.get('modes')
    if not isinstance(modes,list) or not modes or len(modes)>4 or len(set(modes))!=len(modes) or any(m not in MODES for m in modes):
        raise ValueError('请选择一种至四种编码模式')
    scope=value.get('range','first_frames' if value.get('frames') is not None else 'full')
    if scope not in ['full','first_frames']:raise ValueError('请选择完整视频或快速测试')
    if scope=='first_frames' and (type(value.get('frames')) is not int or not 1<=value['frames']<=10000):raise ValueError('快速测试帧数需为1–10000')
    for key,low,high in [('q',0,63)]:
        if type(value.get(key)) is not int or not low<=value[key]<=high:raise ValueError('无效参数：'+key)
    if type(value.get('size')) is not int or value['size'] not in (512,640):raise ValueError('请选择512或640像素最长边')
    fps_mode=value.get('fps_mode','original')
    if not isinstance(fps_mode,str) or fps_mode not in ('original','half'):raise ValueError('请选择原帧率或帧率减半')
    return dict(modes=list(modes),range=scope,frames=value['frames'] if scope=='first_frames' else None,q=value['q'],size=value['size'],fps_mode=fps_mode)


class InferenceLab:
    def __init__(self,owner):
        self.owner=owner;self.lock=threading.RLock();self.jobs={};self.closed=False;self.blockers=[]
        self.directory=owner.directory/'inference_lab';self.directory.mkdir(exist_ok=True)
        self.trial=WORKSPACE/'02_实验数据/优化_输入编码器推理/inference_20261030_01'
        self.replay=self.trial/'viewer'
        self.plan=load_json(self.trial/'plan.json') if (self.trial/'plan.json').exists() else None
        for f in self.directory.glob('*/job.json'):
            try:
                j=load_json(f)
                if j['status'] not in TERMINAL and j['status']!='staged':
                    for record in f.parent.glob('*_process/process.json'):
                        if inspect(record.parent)['observed_state'] in ['running','orphan_running','liveness_unknown']:self.blockers.append(str(record.parent))
                    j.update(status='interrupted',phase='上次服务中断，未记为成功');write(f,j)
                self.jobs[j['id']]=j
            except (OSError,ValueError,KeyError):continue

    def refresh_blockers(self):
        # Startup can catch a transient/old process identity. Recheck it rather
        # than keeping a retired task blocked for the entire server lifetime.
        # Active or uncertain processes remain blocked; no worker is started.
        for holder in (self.owner,self):
            for item in list(holder.blockers):
                folder=Path(item)
                if folder.name=='process.json':folder=folder.parent
                try:state=inspect(folder)['observed_state']
                except (OSError,ValueError,KeyError,TypeError):continue
                if state in ('running','orphan_running','liveness_unknown'):continue
                try:holder.blockers.remove(item)
                except ValueError:pass

    def config(self):
        self.refresh_blockers()
        uploads=[]
        with self.lock:jobs=copy.deepcopy(list(self.jobs.values()))
        for j in sorted(jobs,key=lambda x:x['created_at'],reverse=True):
            f=self.directory/j['id']/('upload'+j.get('extension',''))
            if j.get('filename') and not j.get('receiving') and f.is_file() and f.stat().st_size==j.get('expected_bytes'):
                uploads.append(dict(id=j['id'],filename=j['filename'],options=j['options']))
        return dict(app='glc-inference-lab-v1',modes=[dict(id=k,label=v) for k,v in MODES.items()],
            samples=[{k:c[k] for k in ['id','width','height','frames','fps','q']} for c in self.plan['clips'] if c['id']!='uv108_q17_512'] if self.plan else [],
            ready=bool(self.plan and not self.closed and not self.owner.closed and not self.blockers and not self.owner.blockers and self.owner.role=='sender'),
            blocked_processes=list(self.owner.blockers)+list(self.blockers),
            uploads=uploads[:20],full_video_supported=True,frame_rate_modes=['original','half'],max_upload_bytes=1024**3,gpu_started_on_open=False,shared_gpu_queue=True)

    def update(self,identifier,**changes):
        with self.lock:
            self.jobs[identifier].update(changes,updated_at=time.time())
            write(self.directory/identifier/'job.json',self.jobs[identifier])
            return copy.deepcopy(self.jobs[identifier])

    def get(self,identifier):
        with self.lock:j=copy.deepcopy(self.jobs[identifier])
        if j['status']=='running' and j.get('worker_output'):
            path=Path(j['worker_output'])/'progress.json'
            try:j['progress']=load_json(path)
            except (OSError,ValueError):pass
        j.pop('worker_output',None);return j

    def create(self,value):
        if not self.config()['ready'] or self.owner.blockers:raise ValueError('当前服务或运行环境不可用')
        opts=options(value);sample=value.get('sample');reuse=value.get('source_job');identifier=secrets.token_hex(16)
        if sample and reuse:raise ValueError('请选择一种输入来源')
        if sample and sample not in {c['id'] for c in self.plan['clips']}:raise ValueError('示例不存在')
        if reuse:
            if not isinstance(reuse,str) or not re.fullmatch('[0-9a-f]{32}',reuse):raise ValueError('无效上传记录')
            old=self.jobs[reuse];file=self.directory/reuse/('upload'+old.get('extension',''))
            if old.get('receiving') or not file.is_file() or file.stat().st_size!=old.get('expected_bytes'):raise ValueError('原上传文件不可用')
            name,ext,size=old['filename'],old['extension'],file.stat().st_size
        elif not sample:
            name=value.get('filename','').replace('\\','/').rsplit('/',1)[-1]
            ext=Path(name).suffix.lower();size=value.get('bytes')
            if ext not in ['.mp4','.mkv','.mov','.avi','.webm'] or type(size) is not int or not 0<size<=1024**3:raise ValueError('请选择1GiB以内的视频')
        folder=self.directory/identifier;folder.mkdir()
        job=dict(id=identifier,created_at=time.time(),updated_at=time.time(),status='staged' if sample or reuse else 'uploading',
                 phase='已准备，等待开始' if sample or reuse else '等待上传',options=opts,sample=sample,source_job=reuse,results=[],receiving=False)
        if not sample:job.update(filename=name,extension=ext,expected_bytes=size)
        with self.lock:self.jobs[identifier]=job;write(folder/'job.json',job)
        return self.get(identifier)

    def upload(self,handler,identifier):
        with self.lock:
            j=self.jobs[identifier]
            if j['status']!='uploading' or j['receiving']:raise ValueError('任务不接受上传')
            size=int(handler.headers.get('Content-Length','0'))
            if handler.headers.get('Transfer-Encoding') or size!=j['expected_bytes']:raise ValueError('上传长度不一致')
            self.update(identifier,receiving=True)
        path=self.directory/identifier/('upload'+j['extension']);handler.connection.settimeout(30)
        try:
            remaining=size
            with path.open('xb') as f:
                while remaining:
                    if (path.parent/'STOP').exists():raise InterruptedError('取消上传')
                    data=handler.rfile.read(min(2**20,remaining))
                    if not data:raise ValueError('上传未完成')
                    f.write(data);remaining-=len(data)
            return self.update(identifier,status='staged',phase='上传完成，等待开始',receiving=False)
        except BaseException:
            self.update(identifier,status='error',phase='上传未完成',receiving=False);raise

    def start(self,identifier):
        with self.lock:
            j=self.jobs[identifier]
            if j['status']!='staged' or self.closed or self.owner.closed or self.blockers or self.owner.blockers:raise ValueError('任务不能启动')
            self.update(identifier,status='queued',phase='等待同一GPU队列')
            self.owner.executor.submit(self.run,identifier)
        return self.get(identifier)

    def cancel(self,identifier):
        with self.lock:
            j=self.jobs[identifier]
            if j['status'] in TERMINAL:return self.get(identifier)
            (self.directory/identifier/'STOP').touch()
            return self.update(identifier,status='cancelling' if j['status']=='running' else 'cancelled',phase='正在停止' if j['status']=='running' else '已取消')

    def execute(self,identifier,name,cfg):
        folder=self.directory/identifier;out=folder/name
        cfg.update(output=str(out),stop_file=str(folder/'STOP'))
        path=folder/(name+'.config.json');write(path,cfg);self.update(identifier,worker_output=str(out),current_action=cfg['action'])
        r=supervise([str(self.owner.runtime['python']),'-X','utf8','-u',str(HERE/'web_video_worker.py'),'--config',str(path)],
                    folder/(name+'_process'),REPO,folder/'STOP',poll_seconds=.2)
        if r.get('child_may_still_be_running'):self.blockers.append(str(folder/(name+'_process')));raise RuntimeError('进程状态不确定，暂停新任务')
        if (folder/'STOP').exists():raise InterruptedError('用户取消')
        if r['state']!='completed':
            log=(folder/(name+'_process')/'worker.log').read_text(encoding='utf-8',errors='replace')
            raise RuntimeError(log[-1800:] or '运行失败，请查看日志')
        return load_json(out/'result.json')

    def run(self,identifier):
        folder=self.directory/identifier;began=time.perf_counter()
        try:
            if (folder/'STOP').exists() or self.closed:return
            self.update(identifier,status='running',phase='准备同一份输入',started_at=time.time())
            j=self.get(identifier);opts=j['options'];cfg=dict(action='prepare',frames=opts['frames'],size=opts['size'],fps_mode=opts.get('fps_mode','original'))
            if j['sample']:cfg['sample']=next(c for c in self.plan['clips'] if c['id']==j['sample'])
            elif j.get('source_job'):
                old=self.jobs[j['source_job']];cfg['input']=str(self.directory/old['id']/('upload'+old['extension']))
            else:cfg['input']=str(folder/('upload'+j['extension']))
            source=self.execute(identifier,'prepare',cfg);self.update(identifier,source={k:source[k] for k in ['frames','width','height','fps','source_fps','frame_stride','duration_seconds','read_until_eof']})
            for variant in opts['modes']:
                self.update(identifier,phase=MODES[variant]+'：编码',current_mode=variant)
                send=self.execute(identifier,variant+'_sender',dict(action='sender',assets=self.plan['assets'],source=source,
                    variant=variant,q=opts['q'],engine_cache=str(self.trial/'engines')))
                self.update(identifier,phase=MODES[variant]+'：独立解码')
                stream=folder/(variant+'_sender')/send['stream_file']
                receive=self.execute(identifier,variant+'_receiver',dict(action='receiver',assets=self.plan['assets'],input=str(stream)))
                if not receive['reference_verified'] or send['reference_checksums']!=receive['reference_checksums'] or send['stream_sha256']!=receive['stream_sha256']:
                    raise ValueError('独立解码校验失败')
                a,b=Frames(source),Frames(receive['decoded_storage'])
                if receive['frames']!=source['frames']:raise ValueError('解码帧数与输入不同')
                try:mse=sum(float(np.mean((a[i].astype('float64')-b[i])**2)) for i in range(source['frames']))/source['frames']
                finally:a.close();b.close()
                self.cleanup_frames(receive['decoded_storage'],folder)
                duration=source['frames']/source['fps']
                item=dict(mode=variant,label=MODES[variant],frames=source['frames'],encode_core_seconds=send['core_seconds'],decode_core_seconds=receive['core_seconds'],
                    encode_fps=send['fps'],decode_fps=receive['fps'],model_load_seconds=send['model_load_seconds']+receive['model_load_seconds'],
                    engine_setup_seconds=send['engine_setup_seconds'],warmup_seconds=send['warmup_seconds']+receive['warmup_seconds'],
                    elapsed_seconds=send['worker_seconds']+receive['worker_seconds'],stream_bytes=send['stream_bytes'],stream_sha256=send['stream_sha256'],
                    duration_seconds=duration,bitrate_kbps=send['stream_bytes']*8/duration/1000,
                    bpp=send['stream_bytes']*8/(source['frames']*source['width']*source['height']),stream_file=send['stream_file'],segments=send['segments'],
                    psnr_db=10*math.log10(255**2/max(mse,1e-12)),reference_verified=True,
                    video=BASE+f'jobs/{identifier}/{variant}.mp4',stream=BASE+f'jobs/{identifier}/{variant}'+stream.suffix)
                with self.lock:self.jobs[identifier]['results'].append(item)
                self.update(identifier,phase=MODES[variant]+'已完成')
            self.cleanup_frames(source,folder)
            self.update(identifier,status='done',phase='全部完成，独立解码校验通过',total_seconds=time.perf_counter()-began,completed_at=time.time())
        except BaseException as exc:
            cancelled=isinstance(exc,InterruptedError) or (folder/'STOP').exists()
            self.update(identifier,status='cancelled' if cancelled else 'error',phase='已停止' if cancelled else '任务失败',
                        error=None if cancelled else str(exc),total_seconds=time.perf_counter()-began)

    def cleanup_frames(self,storage,folder):
        try:discard_working_frames(storage,folder)
        except OSError:pass  # A locked working cache must not invalidate a verified video.

    def asset(self,identifier,name):
        j=self.get(identifier);folder=self.directory/identifier
        if name=='source.mp4' and j.get('source'):return folder/'prepare/source.mp4'
        for result in j['results']:
            m=result['mode']
            if name==m+'.mp4':return folder/(m+'_receiver')/'video.mp4'
            if name==m+Path(result.get('stream_file','stream.glcsnd')).suffix:return folder/(m+'_sender')/result.get('stream_file','stream.glcsnd')
        raise KeyError(name)

    def close(self):
        self.closed=True
        for j in list(self.jobs.values()):
            if j['status'] not in TERMINAL:self.cancel(j['id'])

    def get_route(self,handler,route):
        suffix=route[len(BASE):]
        if suffix=='':return handler.file(HERE/'viewer.html')
        if suffix=='live.js':return handler.file(HERE/'live.js')
        if suffix=='summary.json':return handler.file(self.replay/'summary.json')
        if suffix=='api/config':return handler.json(self.config())
        if suffix=='api/jobs':return handler.json([self.get(k) for k in sorted(self.jobs,key=lambda x:self.jobs[x]['created_at'],reverse=True)])
        match=re.fullmatch(r'api/jobs/([0-9a-f]{32})',suffix)
        if match:return handler.json(self.get(match[1]))
        match=re.fullmatch(r'jobs/([0-9a-f]{32})/([^/]+)',suffix)
        if match:return handler.file(self.asset(*match.groups()),download='download' in urlsplit(handler.path).query)
        if re.fullmatch(r'[a-zA-Z0-9_]+\.mp4',suffix):return handler.file(self.replay/suffix)
        raise KeyError(route)

    def post_route(self,handler,route,value):
        suffix=route[len(BASE):]
        if suffix=='api/jobs':return handler.json(self.create(value),201)
        match=re.fullmatch(r'api/jobs/([0-9a-f]{32})/(start|cancel)',suffix)
        if match:return handler.json(getattr(self,match[2])(match[1]))
        raise KeyError(route)


def get_lab(owner):
    if owner.role!='sender':raise ValueError('请在本机发送端打开编码加速对比')
    with owner.store.lock:
        if not hasattr(owner,'inference_lab'):owner.inference_lab=InferenceLab(owner)
        return owner.inference_lab
