const $ = id => document.getElementById(id);
let config, selected = sessionStorage.getItem('c5.selected'), busy = false;
const terminal = new Set(['done','error','cancelled','interrupted']);
async function api(path, body, method) {
  const response = await fetch(path, {method:method || (body === undefined ? 'GET' : 'POST'),
    headers:body === undefined ? {} : {'Content-Type':'application/json'}, body:body === undefined ? undefined : JSON.stringify(body)});
  const value = await response.json();
  if (!response.ok) throw Error(value.error || '请求失败');
  return value;
}
function notice(text='') {$('notice').textContent=text;}
function action(id, fn) {
  $(id).onclick = async () => {
    $(id).disabled=true;
    try {notice(); await fn();} catch(e) {notice(e.message);}
    finally {$(id).disabled=false; if(config) await refresh().catch(e=>notice(e.message));}
  };
}
function select(id) {selected=id;sessionStorage.setItem('c5.selected',id);return refresh();}
function setVideo(id, box, url) {
  $(box).hidden=!url;
  const node=$(id);
  if(url && node.getAttribute('src')!==url) node.src=url;
  if(!url && node.hasAttribute('src')) {node.removeAttribute('src');node.load();}
}
async function refresh() {
  const rows=await api('/api/jobs');
  $('history').replaceChildren();
  for(const row of rows) {
    const item=document.createElement('div');item.className='history-row';
    const button=document.createElement('button');button.textContent=`${row.filename} · ${row.id.slice(0,8)}`;
    button.onclick=()=>select(row.id).catch(e=>notice(e.message));
    const state=document.createElement('span');state.textContent=row.status;
    item.append(button,state);$('history').append(item);
  }
  if(!selected && rows.length) {selected=rows[0].id;sessionStorage.setItem('c5.selected',selected);}
  if(!selected) return;
  const id=selected, job=await api('/api/jobs/'+id);
  if(id!==selected)return;
  $('jobName').textContent=job.filename+' · '+id.slice(0,8);
  $('phase').textContent=job.phase;
  $('progress').value=job.progress || 0;
  const beat=job.heartbeat?.last_progress || job.remote_heartbeat?.last_progress;
  const frame=beat ? `已处理 ${beat.frame_completed} / ${beat.frames} 帧` : '';
  const transfer=job.total_bytes ? `传输 ${job.transferred_bytes || 0} / ${job.total_bytes} B` : '';
  const result=job.result;
  const rate=job.rate_mode ? (job.rate_mode==='max' ? '最大速度' : job.rate_mode+' kbps') : '';
  $('metrics').textContent=result ? `${result.frames} 帧 · ${result.file_bytes} B · 服务 ${result.service_seconds.toFixed(2)} 秒 · 校验通过${rate?' · '+rate:''}` : [frame,transfer,rate].filter(Boolean).join(' · ');
  $('cancel').disabled=terminal.has(job.status);
  $('startJob').disabled=job.status!=='staged' || busy;
  $('resend').disabled=config.role!=='sender' || !job.downloads?.some(d=>d.name==='stream.glcv') || !terminal.has(job.status);
  $('remote').hidden=!job.receiver_job_id;
  setVideo('sourceVideo','sourceBox',job.status==='done' && config.role==='sender' && result.encoded_this_task ? `/jobs/${id}/source.mp4` : null);
  setVideo('decodedVideo','decodedBox',job.status==='done' && config.role==='receiver' ? `/jobs/${id}/decoded.mp4` : null);
  $('downloads').replaceChildren();
  for(const item of job.downloads || []) {
    const link=document.createElement('a');link.href=item.url;link.textContent=item.name;link.download='';$('downloads').append(link);
  }
  $('details').textContent=JSON.stringify(job,null,2);
}
function encodeOptions() {
  const mode=$('clip').value, clip={mode};
  if(mode==='segment'){clip.start=$('start').value;clip.end=$('end').value;}
  return {preset:$('preset').value,backend:$('backend').value,sampling_mode:'time',clip};
}
action('prepare',async()=>{
  const file=$('file').files[0];if(!file)throw Error('请先选择文件');
  busy=true;
  try {
    const job=await api('/api/jobs',{filename:file.name,bytes:file.size,options:encodeOptions()});
    await select(job.id);
    const response=await fetch(`/api/jobs/${job.id}/upload`,{method:'PUT',body:file});
    const value=await response.json();if(!response.ok)throw Error(value.error);
    await refresh();
    if(job.kind==='video'){
      notice('文件已准备，正在读取实际时间轴…');
      const plan=await api(`/api/jobs/${job.id}/probe`,{});
      notice(`已准备：${plan.frames} 帧，${plan.target_fps} fps，${plan.duration_seconds.toFixed(2)} 秒。点击开始发送。`);
    } else notice('已有码流已准备，点击开始发送；本次不会编码。');
  } finally {busy=false;await refresh();}
});
action('startJob',async()=>{await api(`/api/jobs/${selected}/start`,{});await refresh();});
action('cancel',async()=>{await api(`/api/jobs/${selected}/cancel`,{});await refresh();});
action('resend',async()=>{const job=await api(`/api/jobs/${selected}/resend`,{});await select(job.id);});
action('remote',async()=>{const remote=await api(`/api/jobs/${selected}/remote`,{});notice('原接收任务：'+remote.status+' · '+remote.phase);await refresh();});
action('connect',async()=>{
  if($('token').value) await api('/api/peer',{url:$('peer').value,token:$('token').value});
  else if($('peer').value!==config.peer_url)throw Error('修改地址时请同时填写连接口令');
  const value=await api('/api/peer/check',{});
  config=await api('/api/config');$('token').value='';
  $('connectionState').textContent=`已连接 ${value.environment.arithmetic.device} · ${value.rate_mode==='max'?'最大速度':value.rate_mode+' kbps'}`;
});
async function preferences(){config=await api('/api/preferences',{rate_mode:$('rate').value,compatibility_policy:$('policy').value,backend:$('rxBackend').value});notice('已保存，作用于新任务。');}
action('saveRate',preferences);action('saveAdvanced',preferences);
action('preflight',async()=>{notice('正在检查本机环境…');const value=await api('/api/preflight',{});notice('环境已通过：'+value.environment.arithmetic.device);});
action('pairing',async()=>{const value=await api('/api/pairing');$('pairingToken').textContent=value.token;});
action('release',async()=>{await api('/api/release',{});notice('空闲模型已释放；下一任务会重新加载。');});
$('clip').onchange=()=>{$('bounds').hidden=$('clip').value!=='segment';};
$('file').onchange=()=>{$('encodingOptions').hidden=$('file').files[0]?.name.toLowerCase().endsWith('.glcv');};
async function init(){
  config=await api('/api/config');
  const sender=config.role==='sender';
  $('sender').hidden=!sender;$('receiver').hidden=sender;
  $('title').textContent=sender?'发送端 · 语义视频':'接收端 · 独立重建';
  $('subtitle').textContent=sender?'选择视频，编码并发送到另一台电脑。':'接收完整码流，校验后在本机解码播放。';
  $('peer').value=config.peer_url || '';
  $('connection').open=sender && !config.token_configured;
  $('rate').value=config.rate_mode;$('policy').value=config.compatibility_policy;
  $('rxBackend').value=config.backend;$('httpPort').textContent=location.port;$('dataPort').textContent=config.data_port;
  $('workerMode').textContent=config.worker_mode==='resident'?'单角色常驻：首次任务加载模型，任务结束后保留；可在空闲时释放。':'逐任务冷启动：每项任务结束后释放模型。';
  await refresh();
  setInterval(()=>refresh().catch(e=>notice(e.message)),1500);
}
init().catch(e=>notice(e.message));
