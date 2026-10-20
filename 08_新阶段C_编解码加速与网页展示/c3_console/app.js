"use strict";
const $=id=>document.getElementById(id), terminal=new Set(["done","error","cancelled","interrupted"]);
const labels={uploading:"上传中",staged:"已准备",queued:"排队中",running:"处理中",cancelling:"取消中",cancelled:"已取消",error:"失败",interrupted:"已中断",done:"已完成"};
let catalog,config,currentJob=null,busy=false,polling=false,upload=null,resultKey=null,historyKey="",renderToken=0,playing=false,players=[];
const dispatch=new Map(), fmt=(v,n=2)=>typeof v==="number"&&Number.isFinite(v)?v.toFixed(n):"—";
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function error(t=""){$("error").textContent=t;$("error").hidden=!t;}
async function request(url,o={}){const r=await fetch(url,o),v=await r.json();if(!r.ok)throw new Error(v.error||"本机请求失败");return v;}
const post=(url,body)=>request(url,{method:"POST",headers:body?{"Content-Type":"application/json"}:{},body:body?JSON.stringify(body):undefined});
function link(url,t){const a=el("a",t);a.href=url+"?download=1";a.download="";return a;}
function controls(){$("prepare").disabled=busy||Boolean(currentJob&&!terminal.has(currentJob.status));$("start").disabled=busy||currentJob?.status!=="staged"||!config?.runtime_available;$("cancel").disabled=!currentJob||terminal.has(currentJob.status);$("newJob").disabled=busy;}
function pause(){playing=false;players.forEach(v=>v.pause());$("play").textContent="同步播放";}
function clearResult(){pause();renderToken++;resultKey=null;$("result").hidden=true;$("empty").hidden=false;$("resultTag").textContent="预览";$("resultMeta").textContent="";$("resultTitle").textContent="输入与重建";for(const v of players){v.removeAttribute("src");v.load();}players=[];}
function metric(t,v,n){const d=el("div",undefined,"metric");d.append(el("small",t),el("strong",v),el("em",n));return d;}
function ready(v){return new Promise((resolve,reject)=>{if(v.readyState>=2)return resolve();const cleanup=()=>{clearTimeout(timer);v.removeEventListener("loadeddata",ok);v.removeEventListener("error",bad);};const ok=()=>{cleanup();resolve();},bad=()=>{cleanup();reject(new Error("预览无法加载，请下载文件查看。"));};const timer=setTimeout(bad,45000);v.addEventListener("loadeddata",ok,{once:true});v.addEventListener("error",bad,{once:true});});}
async function showResult(r,j=null){
 const key=j?j.id:"sample."+r.id;if(resultKey===key)return;clearResult();resultKey=key;const token=renderToken,t=performance.now(),o=r.output;
 $("empty").hidden=true;$("result").hidden=false;$("resultTag").textContent=j?"本次结果":"已有示例";$("resultTitle").textContent=j?j.filename:"压缩前后对比";$("resultMeta").textContent=o.width+" × "+o.height+" · "+o.processed_frames+" 帧\n"+fmt(o.fps)+" fps · "+fmt(o.duration_seconds)+" 秒";
 players=[$("sourceVideo"),$("decodedVideo")];players[0].src=j?r.videos.source:r.videos.source.url;players[1].src=j?r.videos.optimized:r.videos.optimized.url;
 $("metrics").replaceChildren(metric("完整码流",r.file_bytes.toLocaleString("zh-CN")+" B",".glcv · 预共享模型另计"),metric(j?"本次服务处理":"视频时长",fmt(j?r.service_seconds:o.duration_seconds)+" s",j?"含两端冷启动，不含上传与排队":"示例回放，不会启动编码"),metric("20 kbps 理论发送",fmt(r.transport.theoretical_seconds["20"])+" s","包含应用层头，编解码另计"));
 $("verified").textContent="✓ 独立解码 · 逐帧摘要校验通过";
 const stream=j?"/jobs/"+j.id+"/stream.glcv":r.downloads.find(a=>a.id.endsWith(".stream"))?.url;
 $("downloads").replaceChildren(...[stream?link(stream,"下载码流"):null,link(j?r.videos.optimized:r.videos.optimized.url,"下载重建视频")].filter(Boolean));
 $("evidence").replaceChildren(...(j?r.downloads.filter(x=>["result.json","encode.json","decode.json","manifest.json","events.jsonl","encode.log","decode.log"].includes(x.name)).map(x=>link(x.url,x.name)):[]));
 $("timingNote").textContent=j?"编码到预览 "+fmt(r.timings.encode.task_seconds.offline_view_ready_seconds,3)+" s；独立解码到预览 "+fmt(r.timings.decode.task_seconds.offline_view_ready_seconds,3)+" s；排队 "+fmt(r.queue_seconds,3)+" s。服务时间包含两个新进程的启动与检查。":"这是已归档示例；播放不产生新的编解码计时。";
 $("checkNote").textContent="接收端只读取完整码流，未读取源视频。逐帧 RGB 与潜变量摘要和发送端一致；不代表重建与原视频无损。";
 try{await Promise.all(players.map(ready));if(token!==renderToken||!j)return;const end=performance.now(),click=dispatch.get(j.id),m={preview_load_seconds:(end-t)/1000,dispatch_to_preview_seconds:click===undefined?null:(end-click)/1000,after_refresh:click===undefined};
 await post("/api/jobs/"+j.id+"/browser",m);if(token!==renderToken)return;$("timingNote").textContent+=click===undefined?" 刷新后加载 "+fmt(m.preview_load_seconds,3)+" s；原始点击计时不可用。":" 点击开始至双预览可用 "+fmt(m.dispatch_to_preview_seconds,3)+" s；其中预览加载 "+fmt(m.preview_load_seconds,3)+" s。";
 }catch(e){if(token===renderToken)error(e.message);}
}
function renderJob(j){const changed=currentJob?.id!==j.id;currentJob=j;sessionStorage.setItem("c4-job",j.id);if(changed||(j.status!=="done"&&resultKey!==null))clearResult();
 $("phase").textContent=j.phase||labels[j.status];const p=j.heartbeat?.last_progress;$("progress").value=j.status==="done"?100:p?Math.min(98,(j.stage==="decode"?50:0)+p.frame_completed/p.frames*48):0;
 $("taskTime").textContent=j.status==="running"?(j.stage==="decode"?"独立解码":"编码")+(p?" · "+p.frame_completed+" / "+p.frames+" 帧":" · 正在准备")+(dispatch.has(j.id)?" · 已等待 "+fmt((performance.now()-dispatch.get(j.id))/1000,1)+" 秒":""):j.status==="staged"?"视频已准备。点击开始压缩与重建。":j.status==="done"?"完整码流与逐帧校验均通过。":j.error||labels[j.status]||"";
 if(j.status==="done")void showResult(j.result,j);controls();
}
async function selectJob(id){renderJob(await request("/api/jobs/"+id));}
async function poll(){if(polling)return;polling=true;try{config=await request("/api/config");$("connection").textContent=config.runtime_available?"本机服务已连接":"本机服务已连接 · 环境待配置";$("archive").hidden=!config.history_available;
 const jobs=await request("/api/jobs"),key=JSON.stringify([currentJob?.id,jobs.map(j=>[j.id,j.status,j.updated_at])]);if(key!==historyKey){historyKey=key;$("jobs").replaceChildren(...jobs.slice(0,12).map(j=>{const b=el("button",undefined,"job");b.classList.toggle("active",j.id===currentJob?.id);b.append(el("span",j.filename),el("small",(labels[j.status]||j.status)+" · "+j.options.preset));b.onclick=()=>selectJob(j.id).catch(e=>error(e.message));return b;}));}
 if(currentJob&&!upload){const id=currentJob.id,j=await request("/api/jobs/"+id);if(currentJob?.id===id)renderJob(j);}controls();
 }catch(e){config=null;$("connection").textContent="本机服务未连接";controls();}finally{polling=false;}}
$("taskForm").onsubmit=async e=>{e.preventDefault();if(busy||!$("taskForm").reportValidity())return;busy=true;error();controls();try{
 const options={preset:$("taskPreset").value,frames:$("frames").value?Number($("frames").value):null,backend:$("backend").value};let payload={options};const f=$("file").files[0];
 if($("source").value==="upload"){if(!f)throw new Error("请先选择视频");payload={...payload,filename:f.name,bytes:f.size};}else payload.source_asset=$("source").value;
 renderJob(await post("/api/jobs/reserve",payload));if($("source").value==="upload")await new Promise((resolve,reject)=>{const x=new XMLHttpRequest();upload=x;x.open("POST","/api/jobs/"+currentJob.id+"/source");x.setRequestHeader("Content-Type","application/octet-stream");
 x.upload.onprogress=p=>{if(p.lengthComputable){$("progress").value=p.loaded/p.total*100;$("taskTime").textContent="上传 "+fmt(p.loaded/p.total*100,0)+"%";}};x.onload=()=>{try{const j=JSON.parse(x.responseText);if(x.status!==200)throw new Error(j.error);renderJob(j);resolve();}catch(e){reject(e);}};x.onerror=()=>reject(new Error("上传失败"));x.onabort=()=>reject(new Error("上传已取消"));x.send(f);});
 }catch(e){error(e.message);}finally{busy=false;upload=null;controls();}};
$("start").onclick=async()=>{if(!currentJob||busy)return;busy=true;error();controls();const id=currentJob.id;dispatch.set(id,performance.now());try{renderJob(await post("/api/jobs/"+id+"/start"));}catch(e){dispatch.delete(id);error(e.message);}finally{busy=false;controls();}};
$("cancel").onclick=async()=>{if(!currentJob)return;try{const j=await post("/api/jobs/"+currentJob.id+"/cancel");if(upload)upload.abort();renderJob(j);}catch(e){error(e.message);}};
$("newJob").onclick=()=>{currentJob=null;sessionStorage.removeItem("c4-job");clearResult();$("phase").textContent="等待选择视频";$("taskTime").textContent="准备视频后，点击开始运行。";$("progress").value=0;error();controls();};
$("source").onchange=()=>{$("fileLabel").hidden=$("source").value!=="upload";};
$("taskPreset").onchange=()=>{const p=catalog.presets.presets[$("taskPreset").value];$("frames").value="";$("frames").max=p?.frames||250;$("frames").placeholder=p?"默认 "+p.frames+" 帧":"沿用档位默认";$("presetNote").textContent=p?"宽度上限 "+p.width+" · "+(p.frame_stride===1?"保留源帧率":"隔帧采样")+" · 默认 "+p.frames+" 帧":"请选择画质";};
$("showExample").onclick=()=>{currentJob=null;sessionStorage.removeItem("c4-job");void showResult(catalog.records.find(r=>r.clip==="main"&&r.preset===($("taskPreset").value||"compact"))||catalog.records[0]);$("phase").textContent="正在查看已有示例";$("taskTime").textContent="选择视频后可开始新任务。";$("progress").value=0;controls();};
$("play").onclick=async()=>{if(playing){pause();return;}try{if(players[0]?.ended)players.forEach(v=>v.currentTime=0);else if(players.length===2)players[1].currentTime=players[0].currentTime;await Promise.all(players.map(v=>v.play()));playing=true;$("play").textContent="同步暂停";}catch(e){pause();error("无法播放，可下载视频查看。");}};
$("rewind").onclick=()=>{pause();players.forEach(v=>v.currentTime=0);void $("play").onclick();};
for(const id of ["sourceVideo","decodedVideo"])$(id).addEventListener("ended",()=>{if(players.every(v=>v.ended))pause();});
setInterval(()=>{if(playing&&players.length===2){const[a,b]=players;if(!a.paused&&!a.seeking&&!b.seeking&&Math.abs(a.currentTime-b.currentTime)>.12)b.currentTime=a.currentTime;}},100);
(async()=>{try{catalog=await request("/api/catalog");for(const[k,p]of Object.entries(catalog.presets.presets)){const o=el("option",p.label);o.value=k;$("taskPreset").append(o);}
 for(const[id,n]of [["source.main","使用演示素材"],["source.second","使用第二片段"]])if(catalog.records.some(r=>r.source_asset===id)){const o=el("option",n);o.value=id;$("source").append(o);}
 $("showExample").hidden=!catalog.records.length;await poll();const last=sessionStorage.getItem("c4-job");if(last){try{await selectJob(last);}catch{sessionStorage.removeItem("c4-job");}}else if(catalog.records.length)void showResult(catalog.records.find(r=>r.id==="main.compact")||catalog.records[0]);setInterval(poll,500);
 }catch(e){error(e.message);}})();
