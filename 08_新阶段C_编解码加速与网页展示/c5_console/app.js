const $ = id => document.getElementById(id);
const linkedJob = new URLSearchParams(location.search).get('job');
let config, selected = /^[0-9a-f]{32}$/.test(linkedJob || '') ? linkedJob : sessionStorage.getItem('c5.selected'), busy = false;
let currentJob;
let optionsJob;
let restoredClip;
const modeLabels={ordinary:'普通 GLC',lowpass:'全图低通 1.2',light_lowpass:'全图轻度低通 0.6',semantic_original:'语义保护 · 目标原像素',semantic_light:'语义保护 · 目标轻度低通'};
function jobName(job){
  const frames=job.result?.frames ?? job.video_plan?.frames;
  return job.demo_source && Number.isInteger(frames)?`示例 ${job.demo_source} · ${frames} 帧`:job.filename;
}
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
function fileSize(bytes) {
  if(bytes>=1000000)return number(bytes/1000000)+' MB';
  if(bytes>=1000)return number(bytes/1000)+' KB';
  return number(bytes,0)+' B';
}
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
  if(data.source_file_bytes!=null)metric('上传的原文件',fileSize(data.source_file_bytes),number(data.source_file_bytes,0)+' B');
  metric('压缩后码流',fileSize(data.file_bytes),number(data.file_bytes,0)+' B · 实际传输的 .glcv');
  if(data.source_file_bytes!=null) {
    if(data.source_file_ratio!=null) {
      const change=data.source_file_reduction_percent;
      metric('文件压缩比',number(data.source_file_ratio)+':1',`原文件 ÷ 完整码流 · 体积${change>=0?'减少':'增加'} ${number(Math.abs(change))}%`);
    } else {
      metric('文件压缩比','暂不计算',data.whole_source_processed===false?'只处理了片段，不能与整片文件相比':'未确认覆盖完整原视频');
    }
  } else {
    metric('文件压缩比','无原文件数据','仅有码流时不推算原文件大小');
  }
  metric('平均视频码率',number(data.video_kbps)+' kbps',number(data.duration_seconds)+' 秒 · 含码流容器');
  if(data.mean_psnr_db!=null)metric('全图重建 PSNR',number(data.mean_psnr_db)+' dB','与低通 / 保护前的采样原画面比较');
  let note=`输出 ${data.width} × ${data.height} · ${number(data.fps)} fps · ${data.frames} 帧。文件压缩比 = 上传的原文件字节数 ÷ 完整 .glcv 字节数。KB/MB 按十进制计量，模型预共享；观看用 MP4 不计入码流大小。`;
  if(data.source_file_ratio!=null)note+=' 若降低尺寸、帧率或去除音频，文件大小比也包含这些影响。';
  if(data.whole_source_processed===false)note+=` 本次范围 ${number(data.selected_start_seconds)}–${number(data.selected_end_seconds)} 秒，整片 ${number(data.source_duration_seconds)} 秒；处理完整视频后才显示文件压缩比。`;
  $('compressionNote').textContent=note;
}
function methodHint(){
  const mode=$('enhancement').value;
  const notes={ordinary:'原画面直接进行 GLC 编码。',lowpass:'全图高斯低通，σ 1.2。',light_lowpass:'全图高斯低通，σ 0.6。',semantic_original:'鱼与潜水员区域保留原像素，背景低通 σ 1.2。',semantic_light:'鱼与潜水员区域低通 σ 0.6，背景低通 σ 1.2。'};
  const unavailable=mode!=='ordinary' && config?.enhancement && (!config.enhancement.preprocess_available || (mode.startsWith('semantic') && !config.enhancement.semantic_available));
  $('enhancementHint').textContent=(notes[mode] || '')+(unavailable?' 本机尚未配置所需环境。':'');
}
function restoreOptions(job){
  const opts=job.options;if(!opts || job.kind!=='video')return;
  const video=opts.video || {fps_mode:['compact','detail'].includes(opts.preset)?'half':'source',size_mode:'limit',long_edge:['compact','motion'].includes(opts.preset)?640:960};
  $('enhancement').value=opts.enhancement || 'ordinary';
  $('preset').value=video.fps_mode==='source' && video.size_mode==='limit' && video.long_edge===512?'demo':'custom';
  $('fpsMode').value=video.fps_mode;$('sizeMode').value=video.size_mode;
  if(video.fps_limit)$('fpsLimit').value=video.fps_limit;
  if(video.long_edge)$('longEdge').value=video.long_edge;
  $('clip').value=opts.clip?.mode || 'first10';
  restoredClip=null;
  if(opts.clip?.mode==='segment'){
    const seconds=value=>{const [n,d=1]=String(value).split('/').map(Number);return String(n/d);};
    $('start').value=seconds(opts.clip.start);$('end').value=seconds(opts.clip.end);
    restoredClip={start:opts.clip.start,end:opts.clip.end,displayStart:$('start').value,displayEnd:$('end').value};
  }
  $('backend').value=opts.backend || 'native';
  $('bounds').hidden=$('clip').value!=='segment';$('customVideo').hidden=$('preset').value!=='custom';
  $('fpsLimitBox').hidden=video.fps_mode!=='limit';$('longEdgeBox').hidden=video.size_mode!=='limit';
  $('encodingOptions').hidden=false;methodHint();
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
    const button=document.createElement('button');button.textContent=`${jobName(row)} · ${row.id.slice(0,8)}`;
    if(row.options?.enhancement)button.textContent+=' · '+modeLabels[row.options.enhancement];
    button.onclick=()=>select(row.id).catch(e=>notice(e.message));
    const state=document.createElement('span');state.textContent=row.status==='done' && row.phase?.includes('预览已完成')?'预览（RGB 非逐位一致）':row.status;
    item.append(button,state);$('history').append(item);
  }
  if(!selected && rows.length) {selected=rows[0].id;sessionStorage.setItem('c5.selected',selected);}
  if(!selected) return;
  const id=selected, job=await api('/api/jobs/'+id);
  if(id!==selected)return;
  currentJob=job;
  if(optionsJob!==id){restoreOptions(job);optionsJob=id;}
  renderPlan(job.video_plan);
  $('jobName').textContent=jobName(job)+' · '+id.slice(0,8);
  $('phase').textContent=job.phase;
  $('progress').value=job.progress || 0;
  const beat=job.heartbeat?.last_progress || job.remote_heartbeat?.last_progress;
  const frame=beat ? `已处理 ${beat.frame_completed} / ${beat.frames} 帧` : '';
  const transfer=job.total_bytes ? `传输 ${job.transferred_bytes || 0} / ${job.total_bytes} B` : '';
  const result=job.result;
  const previewOnly=result?.verification?.status==='reference_verified_preview';
  $('verificationWarning').hidden=!previewOnly;
  $('verificationWarning').textContent=previewOnly?result.verification.warning:'';
  $('decodedCaption').textContent=previewOnly?'接收端预览 · RGB 非逐位一致':'接收端重建';
  const rate=job.rate_mode ? (job.rate_mode==='max' ? '最大速度' : job.rate_mode+' kbps') : '';
  const verificationLabel=previewOnly?'参考状态通过 · RGB 非逐位一致':(result?.verified?'码流、RGB、latent 校验通过':'校验未确认');
  $('metrics').textContent=result ? `${result.frames} 帧 · ${result.file_bytes} B · 服务 ${result.service_seconds.toFixed(2)} 秒 · ${verificationLabel}${rate?' · '+rate:''}` : [frame,transfer,rate].filter(Boolean).join(' · ');
  compression(result);
  const prep=result?.preprocessing;
  $('methodResult').hidden=config.role!=='sender';
  $('methodResult').textContent=job.kind==='stream'?'已有码流直接传输，本次没有运行预处理或编码。':
    (prep?`${prep.label} · ${prep.frames} 帧 · 预处理 ${number(prep.seconds)} 秒`+(prep.semantic?` · 保护启用 ${prep.active_frames} 帧 / 回退原画面 ${prep.fallback_frames} 帧。掩码仅在发送端使用。`:'。'):
    `${modeLabels[job.options?.enhancement || 'ordinary']} · ${job.status==='done'?'已完成':'等待本任务完成后显示实际处理统计'}`);
  const receiverUrl=config.loopback_urls?.receiver;
  $('receiverResult').hidden=!(receiverUrl && config.role==='sender' && job.receiver_job_id);
  if(!$('receiverResult').hidden)$('receiverResult').href=receiverUrl+'?job='+job.receiver_job_id;
  $('cancel').disabled=terminal.has(job.status);
  $('startJob').disabled=job.status!=='staged' || busy;
  $('applyOptions').disabled=job.status!=='staged' || job.kind!=='video' || busy;
  $('resend').disabled=config.role!=='sender' || !job.downloads?.some(d=>d.name==='stream.glcv') || !terminal.has(job.status);
  $('remote').hidden=!job.receiver_job_id;
  setVideo('sourceVideo','sourceBox',job.status==='done' && config.role==='sender' && result.encoded_this_task ? `/jobs/${id}/source.mp4` : null);
  setVideo('codingVideo','codingBox',job.status==='done' && config.role==='sender' && prep ? `/jobs/${id}/coding-input.mp4` : null);
  setVideo('decodedVideo','decodedBox',job.status==='done' ? (config.role==='receiver' ? `/jobs/${id}/decoded.mp4` : (receiverUrl && job.receiver_job_id ? `/jobs/${id}/receiver-preview.mp4`:null)) : null);
  $('playbackTools').hidden=job.status!=='done';
  $('downloads').replaceChildren();
  for(const item of job.downloads || []) {
    const link=document.createElement('a');link.href=item.url;link.textContent=item.name;link.download='';$('downloads').append(link);
  }
  $('details').textContent=JSON.stringify(job,(key,value)=>
    ['raw_rgb24_bytes','raw_rgb24_ratio'].includes(key)?undefined:value,2);
}
function encodeOptions() {
  const mode=$('clip').value, clip={mode};
  if(mode==='segment'){
    clip.start=restoredClip && $('start').value===restoredClip.displayStart?restoredClip.start:$('start').value;
    clip.end=restoredClip && $('end').value===restoredClip.displayEnd?restoredClip.end:$('end').value;
  }
  const choice=$('preset').value;
  let video={fps_mode:'source',size_mode:'source'};
  if(choice==='custom') {
    video={fps_mode:$('fpsMode').value,size_mode:$('sizeMode').value};
    if(video.fps_mode==='limit')video.fps_limit=$('fpsLimit').value;
    if(video.size_mode==='limit')video.long_edge=Number($('longEdge').value);
  } else if(choice==='demo') {
    video={fps_mode:'source',size_mode:'limit',long_edge:512};
  } else if(choice!=='source') {
    video={fps_mode:['compact','detail'].includes(choice)?'half':'source',size_mode:'limit',
           long_edge:['compact','motion'].includes(choice)?640:960};
  }
  return {preset:['source','custom','demo'].includes(choice)?'compact':choice,backend:$('backend').value,sampling_mode:'time',clip,video,enhancement:$('enhancement').value};
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
for(const sample of ['011','021'])action('sample'+sample,async()=>{
  busy=true;
  try{
    const job=await api('/api/demo-jobs',{sample,options:encodeOptions()});
    await select(job.id);await prepareCurrentOptions();
  }finally{busy=false;}
});
action('syncPlay',async()=>{
  const players=['sourceVideo','codingVideo','decodedVideo'].map($).filter(v=>v.getAttribute('src'));
  for(const player of players){player.pause();player.currentTime=0;}
  await Promise.all(players.map(v=>v.play()));
});
action('syncPause',async()=>{for(const id of ['sourceVideo','codingVideo','decodedVideo'])$(id).pause();});
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
$('enhancement').onchange=methodHint;
async function init(){
  config=await api('/api/config');
  const sender=config.role==='sender';
  $('sender').hidden=!sender;$('receiver').hidden=sender;
  $('title').textContent=sender?'发送端 · 语义视频':'接收端 · 独立重建';
  $('subtitle').textContent=sender?'选择视频，编码并发送到另一台电脑。':'接收完整码流，校验后在本机解码播放。';
  if(config.loopback_urls){
    $('title').textContent=sender?'语义视频传输':'本机回环 · 重建结果';
    $('subtitle').textContent=sender?'选择视频和处理方法，开始发送。':'本机接收完整码流，校验后独立解码；可从发送端跳转到对应任务。';
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
  methodHint();
  await refresh();
  setInterval(()=>refresh().catch(e=>notice(e.message)),1500);
}
init().catch(e=>notice(e.message));
