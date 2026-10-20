const $ = id => document.getElementById(id);
const linkedJob = new URLSearchParams(location.search).get('job');
let config, selected = /^[0-9a-f]{32}$/.test(linkedJob || '') ? linkedJob : sessionStorage.getItem('c5.selected'), busy = false;
let currentJob;
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
function select(id) {
  selected=id;sessionStorage.setItem('c5.selected',id);
  history.replaceState(null,'','?job='+id);
  return refresh();
}
const number = (value, digits=2) => value.toLocaleString('zh-CN',{maximumFractionDigits:digits});
function rate(value) {const parts=String(value).split('/').map(Number);return number(parts[0]/(parts[1] || 1),4);}
function renderPlan(plan) {
  $('videoPlan').hidden=!plan;
  if(!plan)return;
  const original=`原视频：${plan.source_width} × ${plan.source_height} · ${rate(plan.source_fps)} fps${plan.source_variable_fps?'（平均值，可变帧率）':''} · ${number(plan.source_duration_seconds)} 秒`;
  const output=`本次处理：${plan.width} × ${plan.height} · ${rate(plan.target_fps)} fps · ${plan.frames} 帧 · ${number(plan.duration_seconds)} 秒`;
  $('videoPlan').textContent=original+'\n→ '+output+(plan.source_variable_fps?'\n可变帧率输入按真实时间戳采样，输出为恒定帧率。':'');
}
function compression(result) {
  const data=result?.compression;
  $('compression').hidden=!data;
  $('compressionCards').replaceChildren();
  if(!data)return;
  function metric(label,value,note) {
    const card=document.createElement('div');card.className='metric';
    const title=document.createElement('span');title.textContent=label;
    const strong=document.createElement('strong');strong.textContent=value;
    const sub=document.createElement('small');sub.textContent=note;
    card.append(title,strong,sub);$('compressionCards').append(card);
  }
  metric('完整码流',number(data.file_bytes,0)+' B',number(data.file_bytes/1024)+' KiB · 实际传输的 .glcv');
  metric('平均视频码率',number(data.video_kbps)+' kbps',number(data.duration_seconds)+' 秒 · 含码流容器');
  metric('同档 RGB24 压缩比',number(data.raw_rgb24_ratio)+':1',`${data.width} × ${data.height} · ${number(data.fps)} fps · ${data.frames} 帧`);
  if(data.source_file_bytes!==null) {
    if(data.source_file_ratio!==null) {
      const change=data.source_file_reduction_percent;
      metric('原文件 / 码流',number(data.source_file_ratio)+':1',`原文件 ${number(data.source_file_bytes,0)} B · ${change>=0?'减少':'增加'} ${number(Math.abs(change))}%`);
    } else {
      metric('原文件 / 码流','不作整片比较',`上传文件 ${number(data.source_file_bytes,0)} B · ${data.whole_source_processed===false?'本次只处理选定片段':'未确认覆盖整片'}`);
    }
  }
  if(data.mean_psnr_db!==null)metric('重建 PSNR',number(data.mean_psnr_db)+' dB','与同档采样画面比较 · 越高越接近');
  let note='压缩比 = 基准字节数 ÷ 完整 .glcv 字节数；同档 RGB24 为处理后画面的未压缩大小。模型预共享，预览 MP4 不计入码流大小。';
  if(data.source_file_ratio!==null)note+=' 原文件大小比包含缩放、抽帧及去音频的收益。';
  if(data.whole_source_processed===false)note+=` 本次范围 ${number(data.selected_start_seconds)}–${number(data.selected_end_seconds)} 秒，整片 ${number(data.source_duration_seconds)} 秒；处理完整视频后才显示整片文件比。`;
  $('compressionNote').textContent=note;
}
function setVideo(id, box, url) {
  $(box).hidden=!url;
  const node=$(id);
  if(url && node.getAttribute('src')!==url) node.src=url;
  if(!url && node.hasAttribute('src')) {node.removeAttribute('src');node.load();}
}
async function refresh() {
  const rows=await api('/api/jobs');
  if(selected && !rows.some(row=>row.id===selected)) {selected=null;sessionStorage.removeItem('c5.selected');}
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
  currentJob=job;
  renderPlan(job.video_plan);
  $('jobName').textContent=job.filename+' · '+id.slice(0,8);
  $('phase').textContent=job.phase;
  $('progress').value=job.progress || 0;
  const beat=job.heartbeat?.last_progress || job.remote_heartbeat?.last_progress;
  const frame=beat ? `已处理 ${beat.frame_completed} / ${beat.frames} 帧` : '';
  const transfer=job.total_bytes ? `传输 ${job.transferred_bytes || 0} / ${job.total_bytes} B` : '';
  const result=job.result;
  const rate=job.rate_mode ? (job.rate_mode==='max' ? '最大速度' : job.rate_mode+' kbps') : '';
  $('metrics').textContent=result ? `${result.frames} 帧 · ${result.file_bytes} B · 服务 ${result.service_seconds.toFixed(2)} 秒 · 校验通过${rate?' · '+rate:''}` : [frame,transfer,rate].filter(Boolean).join(' · ');
  compression(result);
  const receiverUrl=config.loopback_urls?.receiver;
  $('receiverResult').hidden=!(receiverUrl && config.role==='sender' && job.receiver_job_id);
  if(!$('receiverResult').hidden)$('receiverResult').href=receiverUrl+'?job='+job.receiver_job_id;
  $('cancel').disabled=terminal.has(job.status);
  $('startJob').disabled=job.status!=='staged' || busy;
  $('applyOptions').disabled=job.status!=='staged' || job.kind!=='video' || busy;
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
  const choice=$('preset').value;
  let video={fps_mode:'source',size_mode:'source'};
  if(choice==='custom') {
    video={fps_mode:$('fpsMode').value,size_mode:$('sizeMode').value};
    if(video.fps_mode==='limit')video.fps_limit=$('fpsLimit').value;
    if(video.size_mode==='limit')video.long_edge=Number($('longEdge').value);
  } else if(choice!=='source') {
    video={fps_mode:['compact','detail'].includes(choice)?'half':'source',size_mode:'limit',
           long_edge:['compact','motion'].includes(choice)?640:960};
  }
  return {preset:['source','custom'].includes(choice)?'compact':choice,backend:$('backend').value,sampling_mode:'time',clip,video};
}
async function prepareCurrentOptions() {
  const id=selected, options=encodeOptions();
  notice('正在读取源视频参数并确认处理范围…');
  const plan=await api(`/api/jobs/${id}/probe`,{options});
  if(id!==selected)throw Error('已切换任务，请在当前任务重新确认参数');
  renderPlan(plan);
  notice(`参数已确认：${plan.width} × ${plan.height}，${rate(plan.target_fps)} fps，${plan.frames} 帧。点击开始发送。`);
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
      await prepareCurrentOptions();
    } else notice('已有码流已准备，点击开始发送；本次不会编码。');
  } finally {busy=false;await refresh();}
});
action('applyOptions',async()=>{busy=true;try{await prepareCurrentOptions();}finally{busy=false;}});
action('startJob',async()=>{
  busy=true;
  try {
    if(currentJob?.kind==='video')await prepareCurrentOptions();
    await api(`/api/jobs/${selected}/start`,{});notice('已开始，按任务中确认的参数处理。');
  } finally {busy=false;await refresh();}
});
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
$('preset').onchange=()=>{$('customVideo').hidden=$('preset').value!=='custom';};
$('fpsMode').onchange=()=>{$('fpsLimitBox').hidden=$('fpsMode').value!=='limit';};
$('sizeMode').onchange=()=>{$('longEdgeBox').hidden=$('sizeMode').value!=='limit';};
$('file').onchange=()=>{$('encodingOptions').hidden=$('file').files[0]?.name.toLowerCase().endsWith('.glcv');};
async function init(){
  config=await api('/api/config');
  const sender=config.role==='sender';
  $('sender').hidden=!sender;$('receiver').hidden=sender;
  $('title').textContent=sender?'发送端 · 语义视频':'接收端 · 独立重建';
  $('subtitle').textContent=sender?'选择视频，编码并发送到另一台电脑。':'接收完整码流，校验后在本机解码播放。';
  if(config.loopback_urls){
    $('title').textContent=sender?'本机回环 · 视频压缩试用':'本机回环 · 重建结果';
    $('subtitle').textContent=sender?'上传自己的视频，压缩后通过本机回环传输，在接收端重建播放。':'本机接收完整码流，校验后独立解码；可从发送端跳转到对应任务。';
    $('loopbackNav').hidden=false;
    $('senderPage').href=config.loopback_urls.sender;$('receiverPage').href=config.loopback_urls.receiver;
    $('connection').hidden=true;
    $('footer').textContent='本机回环：收发两端均在这台电脑上，已自动配对。启动时使用最大速度；关闭启动终端会停止两端服务。';
  }
  $('peer').value=config.peer_url || '';
  $('connection').open=sender && !config.token_configured;
  $('rate').value=config.rate_mode;$('policy').value=config.compatibility_policy;
  $('rxBackend').value=config.backend;$('httpPort').textContent=location.port;$('dataPort').textContent=config.data_port;
  $('workerMode').textContent=config.worker_mode==='resident'?'单角色常驻：首次任务加载模型，任务结束后保留；可在空闲时释放。':'逐任务冷启动：每项任务结束后释放模型。';
  await refresh();
  setInterval(()=>refresh().catch(e=>notice(e.message)),1500);
}
init().catch(e=>notice(e.message));
