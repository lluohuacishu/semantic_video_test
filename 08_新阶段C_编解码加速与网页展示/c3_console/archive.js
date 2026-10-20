"use strict";
const $ = id => document.getElementById(id);
const terminal = new Set(["done", "error", "cancelled", "interrupted"]);
const labels = {staged:"已准备",uploading:"上传中",queued:"排队中",running:"处理中",cancelling:"停止中",cancelled:"已停止",error:"失败",done:"已完成",interrupted:"已中断"};
const names = {"report.a":"A 路线归档","report.b":"B 收尾与研究边界","report.b2":"B2 ROI 实验报告","report.c1":"C1 流程优化报告","report.c2":"C2 原生加速报告"};
let catalog, config, selected = "compact", qualityMode = false, players = [], playing = false;
let currentJob = null, busy = false, polling = false, upload = null, historyKey = "", resultId = null, renderVersion = 0;
let driftMax = 0, dispatchStart = null;
const fmt = (n, digits=2) => typeof n === "number" && Number.isFinite(n) ? n.toFixed(digits) : "待测";
const bytes = n => typeof n === "number" ? n.toLocaleString("zh-CN") + " B" : "待测";
function el(tag, text, className) {const node=document.createElement(tag); if(text!==undefined)node.textContent=text; if(className)node.className=className;return node;}
function error(message="") {$("error").textContent=message; $("error").hidden=!message;}
async function request(url, options={}) {const response=await fetch(url,options); const value=await response.json();if(!response.ok)throw new Error(value.error||"本机请求失败");return value;}
const post = (url, body) => request(url,{method:"POST",headers:body?{"Content-Type":"application/json"}:{},body:body?JSON.stringify(body):undefined});
function link(url,label,download=false) {const a=el("a",label);a.href=url+(download?"?download=1":"");if(download)a.download="";else a.target="_blank";a.rel="noopener";return a;}
function assetLink(asset, label) {return asset?.available?link(asset.url,label||asset.id,true):el("span",(label||asset?.id||"文件")+" · 不可用","hint");}
function group(perf,role) {return perf.groups.find(row=>row.role===role);}
function record() {return catalog.records.find(r=>r.clip===$("clip").value&&r.preset===selected);}
function pausePlayers() {playing=false;players.forEach(v=>v.pause());$("play").textContent="同步播放";}
function makePlayer(title,asset,note) {
  const card=el("article",undefined,"player"),head=el("div",undefined,"player-head");head.append(el("span",title),el("small","MP4 观看副本"));card.append(head);
  if(asset?.available){const video=el("video");video.src=asset.url;video.preload="metadata";video.muted=true;video.playsInline=true;video.setAttribute("aria-label",title);video.addEventListener("error",()=>{const message=el("div","预览不可用，可查看证据或下载文件。","hint");card.append(message);});players.push(video);card.append(video);const cap=el("div",undefined,"player-caption");cap.append(el("span",note));const zoom=el("button","放大 ↗","zoom");zoom.addEventListener("click",()=>video.requestFullscreen().catch(()=>error("当前浏览器不支持全屏，可打开视频文件查看。")));cap.append(zoom);card.append(cap);}else card.append(el("div","该片段没有此项对照材料","unavailable"));
  return card;
}
function renderPresets() {
  $("presets").replaceChildren(...Object.entries(catalog.presets.presets).map(([key,p])=>{const b=el("button",undefined,"preset");b.classList.toggle("active",key===selected);b.setAttribute("aria-pressed",String(key===selected));b.disabled=$("clip").value==="second"&&key!=="compact";b.append(el("b",p.label),el("small",`${p.width} 宽度上限 · ${p.frame_stride===1?"保留帧率":"隔帧采样"}`));b.onclick=()=>{selected=key;renderReplay();};return b;}));
}
function table(perf) {const t=el("table",undefined,"data-table"),head=el("tr");["阶段","优化前","优化后","降幅"].forEach(x=>head.append(el("th",x)));t.append(head);for(const role of ["encode","decode","sum"]){const row=group(perf,role),tr=el("tr");[role==="encode"?"编码":role==="decode"?"独立解码":"两端合计",fmt(row.before_seconds.median,3)+" s",fmt(row.after_seconds.median,3)+" s",fmt(row.reduction_percent.median)+"%"].forEach(x=>tr.append(el("td",x)));t.append(tr);}return t;}
function metric(label,value,note) {const node=el("div",undefined,"metric");node.append(el("small",label),el("strong",value),el("em",note));return node;}
function renderMetrics(r) {
  const rate=$("rate").value,perf=r.performance;
  $("metrics").replaceChildren(metric("C2 编码 · 到 MP4 就绪",fmt(group(perf,"encode").after_seconds.median)+" s",`${perf.complete_pair_count} 对 · 历史常驻`),metric("C2 独立解码 · 到 MP4 就绪",fmt(group(perf,"decode").after_seconds.median)+" s","不读取原始媒体"),metric("完整通信包",bytes(r.file_bytes),".glcv · 不含预共享模型"),metric(`${rate} kbps 理论发送`,fmt(r.transport.theoretical_seconds[rate])+" s",`包含 ${r.transport.header_bytes} B 应用层头；编解码另计`));
  $("presetPerformance").replaceChildren(table(perf),el("p",`${perf.experiment} · ${perf.complete_pair_count} 对${perf.complete_pair_count===1?"，单对不代表稳定性能":"，报告中位数"}`,"hint"));
  $("checks").replaceChildren(...[[r.checks.stream_matches_reference,`完整码流与${r.reference_label}逐字节一致`],[r.checks.all_frame_checksums_match,"RGB / 潜变量摘要通过"],[r.checks.independent_decoder&&!r.checks.source_media_reads.length,"独立接收端未读取原媒体"]].map(([ok,label])=>{const p=el("div",undefined,"check-row");p.append(el("span",ok?"✓":"待核对",ok?"ok":""),el("span",label));return p;}));
  $("downloads").replaceChildren(...r.downloads.map(a=>assetLink(a,a.id.split(".").slice(2).join("."))));
  $("timingDetails").replaceChildren(...Object.entries(r.timings).map(([role,t])=>{const d=el("div");d.append(el("p",`${role} · ${r.performance.experiment} · 第 1 对`),el("pre",JSON.stringify(t.task_seconds,null,2)));return d;}));
}
function renderReplay() {
  renderVersion++; pausePlayers();players=[];driftMax=0;$("seek").value=0;$("drift").textContent="按播放时间对齐；多播放器不是严格逐帧锁定。";
  const second=$("clip").value==="second";if(second){selected="compact";qualityMode=false;}
  $("qualityView").disabled=second;$("qualityView").classList.toggle("active",qualityMode);$("sameView").classList.toggle("active",!qualityMode);renderPresets();
  const r=record(),out=r.output,size=`${out.width} × ${out.height} · ${out.fps} fps · ${out.processed_frames} 帧`;
  const cards=qualityMode?catalog.records.filter(v=>v.clip==="main").map(v=>makePlayer(v.label,v.videos.optimized,`${v.output.width} × ${v.output.height} · ${v.output.fps} fps · ${bytes(v.file_bytes)}`)):[makePlayer("原始素材 · 同档预处理",r.videos.source,size),makePlayer("B4 · 同档基线",r.videos.baseline,second?"第二片没有 B4 主片对照":size),makePlayer("C2 · 优化后独立重建",r.videos.optimized,size)];
  $("players").classList.toggle("four",qualityMode);$("players").replaceChildren(...cards);$("seek").max=out.duration_seconds;$("clock").textContent=`0.00 / ${fmt(out.duration_seconds)} s`;
  $("viewNote").textContent=qualityMode?"四档独立解码结果按同一时间播放；降采样造成的画质和体积变化，不计作 C 轮加速。":`${size} · ${fmt(out.duration_seconds)} s。${second?"第二片使用同片 C1 参考；摄录来源独立性未证明。":"比较同一档位的优化前后；源画面也经过该档预处理。"}`;
  $("composite").hidden=!r.videos.comparison?.available;if(r.videos.comparison?.available){$("composite").href=r.videos.comparison.url;$("composite").target="_blank";}
  renderMetrics(r);
  players.forEach(v=>{v.playbackRate=Number($("speed").value);v.addEventListener("ended",pausePlayers);});
}
function renderCombined() {
  const sum=group(catalog.combined,"sum"),body=el("div",undefined,"performance-main"),bars=el("div",undefined,"bars");
  for(const [label,n,after] of [["原 legacy",sum.before_seconds.median,false],["C1 + 原生 rANS",sum.after_seconds.median,true]]){const row=el("div"),cap=el("div",undefined,"bar-label"),track=el("div",undefined,"bar-track"),fill=el("div",undefined,"bar-fill"+(after?" after":""));cap.append(el("span",label),el("span",fmt(n,3)+" s","mono"));fill.style.width=n/sum.before_seconds.median*100+"%";track.append(fill);row.append(cap,track);bars.append(row);}
  const outcome=el("div",undefined,"outcome");for(const [number,label] of [[fmt(sum.speedup.median,3)+"×","同期配对加速比中位数"],[fmt(sum.reduction_percent.median)+"%","配对耗时降幅中位数"]]){const cell=el("div");cell.append(el("strong",number),el("small",label));outcome.append(cell);}body.append(bars,outcome);
  $("combined").replaceChildren(body,el("span","20% 降时目标通过 · 严格 2× 未达到","goal-note"));
  const cold=catalog.cold.data.comparison;$("cold").textContent=`冷启动单对成本示例：${fmt(cold.before_seconds,3)} → ${fmt(cold.after_seconds,3)} s（${fmt(cold.speedup,3)}×）。不与常驻数据混算；不是稳定冷态收益。`;
  $("canonical").href=catalog.evidence.url;$("canonical").target="_blank";
  $("reports").replaceChildren(...catalog.history.reports.map(a=>assetLink(a,names[a.id])));
  $("historyVideos").replaceChildren(...catalog.history.videos.map((a,i)=>{const d=el("div");d.append(el("p",i?"固定鱼纹区域 · 三路半速":"原图 / GLC / 自动 ROI · 同步合成"));if(a.available){const v=el("video");v.controls=true;v.preload="none";v.src=a.url;d.append(v);}else d.append(el("p","本地材料不可用"));return d;}));
  $("historyBytes").textContent=catalog.history.byte_note||"B2 旧实验只作历史研究展示，不能作为 C 轮提速或等预算语义优势的证明。包大小与评价边界见 B2 原始报告。";
  $("assetStatus").textContent=`本次启动核对 ${catalog.asset_check.available}/${catalog.asset_check.total} 份本地资产 · 2026-10-20`;
}
function setView(view) {pausePlayers();document.querySelectorAll(".nav").forEach(n=>n.classList.toggle("active",n.dataset.view===view));$("replayView").hidden=view!=="replay";$("runView").hidden=view!=="run";}
function updateControls() {
  const active=currentJob&&!terminal.has(currentJob.status),editable=!busy&&!active;
  ["source","file","taskPreset","frames","backend","prepare"].forEach(id=>$(id).disabled=!editable);
  $("start").disabled=busy||currentJob?.status!=="staged"||!config?.runtime_available;
  $("cancel").disabled=!active||currentJob.status==="cancelling";
  $("newJob").disabled=busy||Boolean(active);
  document.querySelectorAll("#jobs button").forEach(b=>b.disabled=busy||Boolean(upload));
}
function renderJob(job) {
  currentJob=job;localStorage.setItem("c3-job",job.id);$("phase").textContent=job.phase;
  $("taskMeta").textContent=`${job.filename} · ${catalog.presets.presets[job.options.preset].label} · ${job.options.backend} · ${job.id.slice(0,8)}`;
  const progress=job.heartbeat?.last_progress;
  const elapsed=job.started_at ? `已用 ${fmt((Date.now()-Date.parse(job.started_at))/1000,1)} s · ` : "";
  if(progress){$("progress").value=progress.frame_completed/progress.frames*100;$("taskTime").textContent=elapsed+`${job.stage==="decode"?"独立解码":"编码"} ${progress.frame_completed}/${progress.frames} 帧 · 心跳 ${new Date(job.heartbeat.heartbeat_utc).toLocaleTimeString("zh-CN")}`;}
  else if(job.status==="running"||job.status==="cancelling"){$("progress").removeAttribute("value");$("taskTime").textContent=elapsed+"正在准备模型或完成预览，等待工作进程回报";}
  else{$("progress").value=job.status==="done"?100:0;$("taskTime").textContent=job.error||"";}
  $("taskLogs").replaceChildren(...(job.logs||[]).map(a=>link(a.url,a.name,true)));
  if(job.status!=="done"){$("newResult").hidden=true;resultId=null;}
  if(job.status==="done"&&resultId!==job.id){renderNewResult(job.result);resultId=job.id;}
  updateControls();
}
function renderNewResult(r) {
  const panel=$("newResult");panel.hidden=false;const videos=el("div",undefined,"new-videos");
  const mediaStarted=performance.now();let loaded=0;
  for(const [label,url] of [["本次输入 · 同档预处理",r.videos.source],["本次独立重建",r.videos.optimized]]){const card=el("div"),v=el("video");v.controls=true;v.preload="auto";v.src=url;v.muted=true;v.playsInline=true;card.append(el("h3",label),v);videos.append(card);v.addEventListener("loadeddata",()=>{if(++loaded===2&&currentJob?.id===r.id&&currentJob.status==="done"){const detail={preview_load_seconds:(performance.now()-mediaStarted)/1000,dispatch_to_preview_seconds:dispatchStart===null?null:(performance.now()-dispatchStart)/1000,after_refresh:dispatchStart===null};panel.append(el("p",`当前浏览器预览加载 ${fmt(detail.preview_load_seconds,3)} s${detail.dispatch_to_preview_seconds===null?"；本页没有完整点击起点":"；本次点击至两个预览可用 "+fmt(detail.dispatch_to_preview_seconds,3)+" s"}。不含传输网络。`,"hint"));post(`/api/jobs/${r.id}/browser`,detail).catch(e=>error(e.message));}},{once:true});}
  const taskMetrics=el("div",undefined,"metrics");taskMetrics.append(metric("本次编码 · 任务内到 MP4",fmt(r.timings.encode.task_seconds.offline_view_ready_seconds)+" s","新进程，含任务内模型加载"),metric("本次独立解码 · 任务内到 MP4",fmt(r.timings.decode.task_seconds.offline_view_ready_seconds)+" s","进程导入等额外成本见服务总耗时"),metric("完整通信包",bytes(r.file_bytes),"与接收端摘要一致"),metric("20 kbps 理论发送",fmt(r.transport.theoretical_seconds["20"])+" s","不含物理链路或编解码"));
  const info=r.output;const downloads=el("div",undefined,"downloads");downloads.append(...r.downloads.map(a=>link(a.url,a.name,true)));
  panel.replaceChildren(el("h2","本次结果 · 校验通过"),el("p",`${info.width} × ${info.height} · ${info.fps} fps · ${info.processed_frames} 帧 · ${fmt(info.duration_seconds)} s · ${bytes(r.file_bytes)}`,"hint"),videos,taskMetrics,el("p",`排队 ${fmt(r.queue_seconds,3)} s；服务处理至完整结果 ${fmt(r.service_seconds,3)} s。发送/接收各自新进程，含冷启动。20 kbps 理论发送 ${fmt(r.transport.theoretical_seconds["20"])} s。`,"hint"),downloads);
}
async function selectJob(id){dispatchStart=null;renderJob(await request(`/api/jobs/${id}`));}
async function poll() {
  if(polling)return;polling=true;
  try{config=await request("/api/config");$("connection").textContent=config.runtime_available?"本机已连接 · 运行环境路径就绪":"本机已连接 · 仅回放 / 运行环境需检查";const jobs=await request("/api/jobs");const key=JSON.stringify([currentJob?.id,jobs.map(j=>[j.id,j.status,j.updated_at])]);if(key!==historyKey){historyKey=key;$("jobs").replaceChildren(...jobs.map(j=>{const b=el("button",undefined,"job");b.classList.toggle("active",j.id===currentJob?.id);b.append(el("span",j.filename),el("small",`${labels[j.status]||j.status} · ${j.options.preset}`));b.onclick=()=>selectJob(j.id).catch(e=>error(e.message));return b;}));if(!jobs.length)$("jobs").append(el("p","还没有新任务","hint"));}if(currentJob&&!upload){const id=currentJob.id;const fresh=await request(`/api/jobs/${id}`);if(currentJob?.id===id)renderJob(fresh);}updateControls();}
  catch(e){config=null;$("connection").textContent="本机服务未连接";updateControls();}finally{polling=false;}
}
$("taskForm").onsubmit=async event=>{event.preventDefault();if(busy||!$("taskForm").reportValidity())return;busy=true;error();updateControls();try{const frameText=$("frames").value,options={preset:$("taskPreset").value,frames:frameText?Number(frameText):null,backend:$("backend").value},file=$("file").files[0];let payload={options};if($("source").value==="upload"){if(!file)throw new Error("请先选择视频");payload={...payload,filename:file.name,bytes:file.size};}else payload.source_asset=$("source").value;renderJob(await post("/api/jobs/reserve",payload));if(file&&$("source").value==="upload"){await new Promise((resolve,reject)=>{const xhr=new XMLHttpRequest();upload=xhr;xhr.open("POST",`/api/jobs/${currentJob.id}/source`);xhr.setRequestHeader("Content-Type","application/octet-stream");xhr.upload.onprogress=e=>{if(e.lengthComputable){$("progress").value=e.loaded/e.total*100;$("taskTime").textContent=`上传 ${bytes(e.loaded)} / ${bytes(e.total)}`;}};xhr.onload=()=>{try{const value=JSON.parse(xhr.responseText);if(xhr.status!==200)throw new Error(value.error);renderJob(value);resolve();}catch(e){reject(e);}};xhr.onerror=()=>reject(new Error("上传失败，请检查本机服务"));xhr.onabort=()=>reject(new Error("上传已停止"));xhr.send(file);});}}catch(e){error(e.message);}finally{busy=false;upload=null;updateControls();}};
$("start").onclick=async()=>{if(!currentJob||busy)return;busy=true;updateControls();try{dispatchStart=performance.now();renderJob(await post(`/api/jobs/${currentJob.id}/start`));}catch(e){error(e.message);dispatchStart=null;}finally{busy=false;updateControls();}};
$("cancel").onclick=async()=>{if(!currentJob)return;try{const job=await post(`/api/jobs/${currentJob.id}/cancel`);if(upload)upload.abort();renderJob(job);}catch(e){error(e.message);}};
$("newJob").onclick=()=>{currentJob=null;resultId=null;dispatchStart=null;localStorage.removeItem("c3-job");$("newResult").hidden=true;$("phase").textContent="等待选择素材和档位";$("taskMeta").textContent="准备素材不会启动 GPU。";$("taskTime").textContent="";$("progress").value=0;$("taskLogs").replaceChildren();updateControls();error();};
$("source").onchange=()=>$("fileLabel").hidden=$("source").value!=="upload";
$("taskPreset").onchange=()=>{const p=catalog.presets.presets[$("taskPreset").value];$("frames").max=p?.frames||250;$("frames").placeholder=p?`默认 ${p.frames} 帧`:"沿用档位默认";};
$("clip").onchange=renderReplay;$("rate").onchange=()=>renderMetrics(record());$("sameView").onclick=()=>{qualityMode=false;renderReplay();};$("qualityView").onclick=()=>{qualityMode=true;renderReplay();};
document.querySelectorAll(".nav").forEach(b=>b.onclick=()=>setView(b.dataset.view));
$("play").onclick=async()=>{if(playing){pausePlayers();return;}if(!players.length)return;const version=renderVersion;try{const t=players[0].ended?0:players[0].currentTime;players.forEach(v=>{v.currentTime=t;v.playbackRate=Number($("speed").value);});await Promise.all(players.map(v=>v.play()));if(version!==renderVersion)return;playing=true;$("play").textContent="同步暂停";}catch(e){pausePlayers();error("浏览器暂时无法播放，可下载对应预览查看。");}};
$("rewind").onclick=()=>{pausePlayers();players.forEach(v=>v.currentTime=0);$("seek").value=0;};
$("seek").oninput=()=>{players.forEach(v=>{if(v.readyState)v.currentTime=Number($("seek").value);});};
$("speed").onchange=()=>players.forEach(v=>v.playbackRate=Number($("speed").value));
setInterval(()=>{if(!players.length)return;const master=players[0];if(playing){for(const v of players.slice(1)){if(v.readyState&&!v.seeking&&!master.seeking){const drift=Math.abs(master.currentTime-v.currentTime);driftMax=Math.max(driftMax,drift);if(drift>0.12)v.currentTime=master.currentTime;}}$("drift").textContent=`本次同步期间最大观察时间差 ${fmt(driftMax,3)} s；超过 0.12 s 时校正，不保证逐帧锁定。`;}$("seek").value=master.currentTime;$("clock").textContent=`${fmt(master.currentTime)} / ${fmt(Number($("seek").max))} s`;},100);
(async()=>{try{catalog=await request("/api/catalog");renderReplay();renderCombined();for(const [key,p]of Object.entries(catalog.presets.presets)){const o=el("option",p.label);o.value=key;$("taskPreset").append(o);}await poll();const last=localStorage.getItem("c3-job");if(last){try{await selectJob(last);}catch{localStorage.removeItem("c3-job");}}setInterval(poll,1000);}catch(e){error(e.message);$("connection").textContent="成果索引加载失败";}})();
