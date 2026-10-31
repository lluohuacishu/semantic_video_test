/* Replay uses external JS to honor the existing C5 script-src 'self' policy. */
const $=id=>document.getElementById(id),v=$('video');let data,clip,ready=false,frame=0;
const fmt=(n,d=3)=>Number(n).toFixed(d),signed=n=>(n>=0?'+':'')+fmt(n,4);
function setReady(value){ready=value;for(const id of ['play','prev','next','restart'])$(id).disabled=!value}
function position(){if(!clip)return;frame=Math.max(0,Math.min(clip.frames-1,Math.floor(v.currentTime*clip.fps+1e-4)));$('seek').value=frame;$('position').textContent=`第 ${frame+1} / ${clip.frames} 帧 · ${fmt(v.currentTime,2)}秒`}
function seek(index){if(!ready)return;v.pause();v.currentTime=(Math.max(0,Math.min(clip.frames-1,index))+.01)/clip.fps;$('play').textContent='播放';$('status').textContent='已暂停，可逐帧比较或切换放大画面。';position()}
function loadClip(id){
 v.pause();setReady(false);clip=data.clips.find(c=>c.id===id);frame=0;
 document.querySelectorAll('#clips button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.id===id)));
 $('clip-title').textContent=clip.title;$('clip-info').textContent=`${clip.width} × ${clip.height} · ${clip.frames}帧 · q${clip.q}`;
 $('seek').max=clip.frames-1;$('seek').value=0;$('position').textContent=`第 1 / ${clip.frames} 帧`;$('status').textContent='正在加载视频…';
 $('metrics').innerHTML=clip.methods.map(m=>`<tr><td>${m.label}</td><td><strong>${fmt(m.core_seconds)}秒</strong></td><td>${fmt(m.speedup,2)}×</td><td>${fmt(m.p_psnr)} dB<small>相对原版 ${signed(m.p_delta)} dB</small></td><td>${m.fish_psnr===null?'—':fmt(m.fish_psnr)+' dB'}</td><td>${fmt(m.stream_bytes*8/(clip.frames/clip.fps)/1000)} kbps<small>${fmt(m.stream_bytes*8/(clip.frames*clip.width*clip.height),6)} bpp</small></td><td>${m.stream_bytes.toLocaleString()} B<small>${m.file_delta===null?'原版基线':`相对原版 ${signed(m.file_delta)}%`}</small></td></tr>`).join('');
 v.src=clip.video;v.load();
}
v.addEventListener('loadeddata',()=>{setReady(true);$('status').textContent='已就绪：左上原始画面，右上原GLC，左下FP32，右下FP16。';position()});
v.addEventListener('error',()=>{$('status').textContent='视频加载失败，请刷新页面或重新启动本地入口。';setReady(false)});
v.addEventListener('play',()=>{$('play').textContent='暂停';$('status').textContent='同步回放中'});
v.addEventListener('pause',()=>{$('play').textContent='播放';if(ready)$('status').textContent='已暂停，可逐帧比较或切换放大画面。'});
v.addEventListener('timeupdate',position);v.addEventListener('seeked',position);v.addEventListener('ended',()=>{if(v.ended){$('play').textContent='重新播放';$('status').textContent='已播放完毕'}});
$('play').onclick=()=>{if(v.paused){if(v.ended)v.currentTime=0;v.play().catch(()=>{$('status').textContent='请再次点击播放'})}else v.pause()};
$('prev').onclick=()=>{position();seek(frame-1)};$('next').onclick=()=>{position();seek(frame+1)};$('restart').onclick=()=>{seek(0);v.play()};$('seek').oninput=e=>seek(Number(e.target.value));
$('view').onchange=e=>{$('screen').dataset.view=e.target.value};$('rate').onchange=e=>{v.playbackRate=Number(e.target.value)};
$('fullscreen').onclick=()=>{if(!document.fullscreenElement)$('screen').requestFullscreen?.();else document.exitFullscreen?.()};
document.addEventListener('visibilitychange',()=>{if(document.hidden)v.pause()});
fetch('summary.json').then(r=>{if(!r.ok)throw Error('data');return r.json()}).then(d=>{data=d;for(const c of d.clips){const b=document.createElement('button');b.textContent=c.tab;b.dataset.id=c.id;b.setAttribute('aria-pressed','false');b.onclick=()=>loadClip(c.id);$('clips').append(b)}loadClip(d.clips[0].id)}).catch(()=>{$('status').textContent='结果数据加载失败，请重新启动本地入口。'});

/* Actual on-demand encoding, isolated original decoding and explicit live timings. */
(()=>{
const el=id=>document.getElementById(id),prefix='./api/';let config,current=null,poll=null,rendered='',submitting=false;
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const n=(x,d=3)=>Number(x).toFixed(d);
const bytes=x=>x>=1048576?n(x/1048576,2)+' MiB':x>=1024?n(x/1024,2)+' KiB':x+' B';
const methodLabel=r=>r.decoder_mode?r.label:r.label+'编码 / 原GLC FP32解码';
async function api(path,body){const r=await fetch(prefix+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw Error(j.error||'请求失败');return j}
function tab(live){el('live-panel').hidden=!live;el('replay-panel').hidden=live;el('tab-run').setAttribute('aria-pressed',String(live));el('tab-replay').setAttribute('aria-pressed',String(!live));document.querySelectorAll('video').forEach(v=>v.pause())}
el('tab-run').onclick=()=>tab(true);el('tab-replay').onclick=()=>tab(false);
function inputChange(){const id=el('live-source').value;el('live-file').hidden=id!=='upload';if(id.startsWith('reuse:')){const old=config.uploads.find(x=>x.id===id.slice(6));el('live-size').value=String(old.options.size);el('live-q').value=old.options.q;el('live-modes').querySelectorAll('input').forEach(x=>x.checked=old.options.modes.includes(x.value))}else if(id!=='upload'){const c=config.samples.find(x=>x.id===id);el('live-size').value=c.width>512?'640':'512';el('live-q').value=c.q;el('live-frames').value=Math.min(c.frames,66)}}
function restoreInput(j){const value=j.sample||'reuse:'+(j.source_job||j.id);if([...el('live-source').options].some(o=>o.value===value)){el('live-source').value=value;inputChange()}el('live-size').value=String(j.options.size);el('live-q').value=j.options.q;el('live-fps-mode').value=j.options.fps_mode||'original';el('live-modes').querySelectorAll('input').forEach(x=>x.checked=j.options.modes.includes(x.value));el('live-decoders').querySelectorAll('input').forEach(x=>x.checked=(j.options.decoder_modes||['eager']).includes(x.value))}
function buttons(active){el('live-start').disabled=active||submitting||!config?.ready;el('live-cancel').disabled=!active}
function render(j){
current=j;const active=['queued','running','cancelling','uploading'].includes(j.status);buttons(active);
const p=j.progress,part=p?.total?` · ${p.index}/${p.total}帧`:'';
el('live-status').textContent=j.phase+(p?' · '+p.phase+part:'')+(j.error?'：'+j.error:'');
const done=j.results.length,progress=p?.total?p.index/p.total:0,phasePart=j.current_action==='receiver' ? .55+progress*.4 : j.current_action==='sender' ? .1+progress*.4 : progress*.05;el('live-progress').value=j.status==='done'?1:Math.min(.99,(done+phasePart)/(j.options.modes.length*(j.options.decoder_modes||['eager']).length));
const key=j.id+':'+j.results.length+':'+j.status;
if(key!==rendered){
 rendered=key;
 if(!j.results.length){el('live-result').innerHTML='';return}
 const baseline=j.results.find(r=>r.mode==='eager');
 const rows=j.results.map(r=>{const duration=j.source.frames/j.source.fps,rate=r.bitrate_kbps??r.stream_bytes*8/duration/1000,bpp=r.bpp??r.stream_bytes*8/(j.source.frames*j.source.width*j.source.height),decodeBase=j.results.find(x=>x.mode===r.mode&&(x.decoder_mode||'eager')==='eager');return `<tr><td>${escape(methodLabel(r))}<small>独立校验通过</small></td><td><strong>${n(r.encode_core_seconds)}秒</strong><small>${n(r.encode_fps,2)}帧/秒 · ${baseline?n(baseline.encode_core_seconds/r.encode_core_seconds,2)+'×':'—'}</small></td><td><strong>${n(r.decode_core_seconds)}秒</strong><small>${n(r.decode_fps,2)}帧/秒${decodeBase?' · '+n(decodeBase.decode_core_seconds/r.decode_core_seconds,2)+'×':''}</small></td><td><strong>${n(rate)} kbps</strong><small><a href="${r.stream}?download=1">${bytes(r.stream_bytes)} · 下载码流</a></small><small>${r.stream_bytes.toLocaleString()} B · ${n(bpp,6)} bpp</small></td><td>${n(r.psnr_db)} dB</td><td>${n(r.elapsed_seconds)}秒<small>引擎 ${n(r.engine_setup_seconds)}秒 · 加载 ${n(r.model_load_seconds)}秒</small></td></tr>`}).join('');
 el('live-result').innerHTML=`<h2 style="font-size:18px">任务结果 · ${j.options.range==='full'?'完整视频 · ':''}${j.options.fps_mode==='half'?'帧率减半':'原帧率'} · ${j.source.frames}帧 · ${n(j.source.fps,2)}fps · ${n(j.source.frames/j.source.fps,2)}秒视频 · ${j.source.width}×${j.source.height} · q${j.options.q}</h2><div class="table-wrap"><table><thead><tr><th>模式</th><th>编码核心 / 加速</th><th>解码核心</th><th>码率 / 码流大小</th><th>全图PSNR</th><th>编解码任务</th></tr></thead><tbody>${rows}</tbody></table></div><p class="note">kbps=1000bit/s，按实际处理帧数/标称帧率折算；码率与大小包括完整传输文件的封装。核心耗时完成输入上卡后计时，任务耗时含加载、引擎准备、暖机与输出；数据准备另计。${j.total_seconds?'全部用时 '+n(j.total_seconds)+'秒。':''}</p>`;
 {
  const previews=[{label:j.options.fps_mode==='half'?'同一份输入（隔帧采样）':'同一份输入',video:`./jobs/${j.id}/source.mp4`},...j.results.map(r=>({...r,label:methodLabel(r)}))];
  const box=document.createElement('div');box.innerHTML=`<div class="controls"><button id="live-playall">同步播放结果</button><button id="live-pauseall">暂停全部</button></div><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px">${previews.map(r=>`<div><h3 style="font-size:15px">${escape(r.label)}</h3><video class="live-output" controls muted playsinline preload="metadata" style="width:100%;background:black;border:1px solid #30434f;border-radius:6px" src="${r.video}"></video></div>`).join('')}</div>`;el('live-result').append(box);
  const videos=[...box.querySelectorAll('video')];
  el('live-playall').onclick=async()=>{videos.forEach(v=>{v.currentTime=0;v.playbackRate=1});await Promise.all(videos.map(v=>v.play().catch(()=>{})))};
  el('live-pauseall').onclick=()=>videos.forEach(v=>v.pause());
  let seeking=false;videos[0].addEventListener('seeking',()=>{if(seeking)return;seeking=true;videos.slice(1).forEach(v=>{if(Math.abs(v.currentTime-videos[0].currentTime)>.08)v.currentTime=videos[0].currentTime});seeking=false});
 }
}
}
async function history(){const jobs=await api('jobs');el('live-history').innerHTML=jobs.slice(0,12).map(j=>`<p><button data-job="${j.id}">${new Date(j.created_at*1000).toLocaleString()} · ${j.options.fps_mode==='half'?'帧率减半':'原帧率'} · ${escape(j.phase)}</button></p>`).join('');el('live-history').querySelectorAll('button').forEach(b=>b.onclick=()=>watch(b.dataset.job));return jobs}
async function watch(id){clearTimeout(poll);try{const j=await api('jobs/'+id);sessionStorage.setItem('glc-inference.selected',id);render(j);if(['queued','running','cancelling','uploading'].includes(j.status))poll=setTimeout(()=>watch(id),800);else history().catch(()=>{})}catch(e){el('live-status').textContent=e.message}}
el('live-source').onchange=inputChange;el('live-range').onchange=()=>{el('live-frames-wrap').hidden=el('live-range').value==='full'};
el('live-start').onclick=async()=>{
 if(submitting)return;submitting=true;buttons(true);document.querySelectorAll('video').forEach(v=>v.pause());
 try{
  const modes=[...el('live-modes').querySelectorAll('input:checked')].map(x=>x.value);
 const decoder_modes=[...el('live-decoders').querySelectorAll('input:checked')].map(x=>x.value);
 const opts={modes,decoder_modes,range:el('live-range').value,frames:el('live-range').value==='full'?null:Number(el('live-frames').value),size:Number(el('live-size').value),q:Number(el('live-q').value),fps_mode:el('live-fps-mode').value};
  if(el('live-source').value==='upload'){
   const file=el('live-file').files[0];if(!file)throw Error('请先选择视频文件');if(file.size>config.max_upload_bytes)throw Error('视频不能超过1GiB');
   opts.filename=file.name;opts.bytes=file.size;const j=await api('jobs',opts);current=j;
   el('live-status').textContent='正在上传视频…';const r=await fetch(prefix+'jobs/'+j.id+'/upload',{method:'PUT',body:file});const response=await r.json();if(!r.ok)throw Error(response.error||'上传失败');
  }else{if(el('live-source').value.startsWith('reuse:'))opts.source_job=el('live-source').value.slice(6);else opts.sample=el('live-source').value;current=await api('jobs',opts)}
  await api('jobs/'+current.id+'/start',{});submitting=false;await watch(current.id);
 }catch(e){el('live-status').textContent=e.message;submitting=false;buttons(false)}
};
el('live-cancel').onclick=async()=>{if(current){try{render(await api('jobs/'+current.id+'/cancel',{}));watch(current.id)}catch(e){el('live-status').textContent=e.message}}};
api('config').then(async c=>{config=c;c.uploads=c.uploads||[];el('live-source').innerHTML=c.uploads.map(u=>`<option value="reuse:${u.id}">已上传：${escape(u.filename)}（直接复用）</option>`).join('')+c.samples.map(s=>`<option value="${s.id}">示例片段：${escape(s.id)} · ${s.frames}帧</option>`).join('')+'<option value="upload">上传自己的视频</option>';el('live-modes').innerHTML=c.modes.map(m=>`<label style="padding:7px 10px;border:1px solid #415763;border-radius:6px"><input type="checkbox" value="${m.id}" checked> ${escape(m.label)}</label>`).join('');el('live-decoders').innerHTML=(c.decoder_modes||[{id:'eager',label:'原GLC FP32'}]).map(m=>`<label style="padding:7px 10px;border:1px solid #415763;border-radius:6px"><input type="checkbox" value="${m.id}" ${m.id==='eager'?'checked':''}> ${escape(m.label)}</label>`).join('');inputChange();el('live-start').disabled=!c.ready;el('live-status').textContent=c.ready?'已就绪。可分别选择编码与解码模式，默认处理完整视频。':'当前环境未就绪；仍可查看已完成对比。';try{const jobs=await history(),restore=jobs.find(j=>j.status==='running'||j.status==='cancelling')||jobs.find(j=>j.status==='queued')||jobs.find(j=>j.id===sessionStorage.getItem('glc-inference.selected'))||jobs.find(j=>j.status==='done');if(restore){restoreInput(restore);await watch(restore.id)}}catch(e){el('live-status').textContent=e.message}}).catch(()=>{el('live-status').innerHTML='当前为独立回放入口。<a href="http://127.0.0.1:8773/inference/">进入主网页运行编解码对照 →</a>';tab(false)});
})();
