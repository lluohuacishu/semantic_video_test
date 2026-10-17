"use strict";

const $ = (id) => document.getElementById(id);
const state = {config: null, jobs: [], active: null, view: "overlay", uploading: false,
  xhr: null, rendered: null, polling: false, connected: false, inputVideo: null, outputVideo: null, historySignature: ""};
const terminal = new Set(["done", "error", "cancelled", "interrupted"]);
const statusNames = {done: "已完成", error: "失败", cancelled: "已停止", interrupted: "已中断", running: "处理中", queued: "等待中", uploading: "上传中"};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function message(text) { $("errorBox").textContent = text || ""; $("errorBox").hidden = !text; }
function formatBytes(bytes) { return bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(0)} KiB` : `${(bytes / 1024 ** 2).toFixed(1)} MiB`; }
function mediaUrl(job, filename, download = false) { return `/media/${job.id}/${encodeURIComponent(filename)}${download ? "?download=1" : ""}`; }
function codecMode(mode) { return ["image_codec", "semantic_codec"].includes(mode); }
function semanticJob(job) { return (job?.summary?.mode || job?.options?.mode) === "semantic_codec"; }
function codecJob(job) { return codecMode(job?.summary?.mode || job?.options?.mode); }
function updateMode() {
  const semantic = $("operationMode").value === "semantic_codec";
  const codec = codecMode($("operationMode").value);
  $("codecQuality").closest("label").hidden = semantic;
  $("inputSize").disabled = semantic;
  if (semantic) { $("inputSize").value = "256"; $("codecQuality").value = "3"; }
  $("codecSettings").hidden = !codec;
  $("pageTitle").textContent = semantic ? "从同一份码流还原图片和语义" : codec ? "压缩、传文件，再还原图片" : "看看模型识别到了什么";
  $("pageIntro").textContent = semantic ? "发送端语义参与编码，接收端直接解出图片与类别图。选择限速后拖入图片。" : codec ? "选择质量档位与传输限速，拖入图片，对照原图查看还原质量。" : "拖入一张图片或一段视频，对照原画面查看分割结果。";
  $("temporalWindow").closest("label").hidden = codec;
  $("temporalHint").hidden = codec;
  document.querySelector(".advanced").hidden = codec;
  $("fileInput").accept = codec ? ".jpg,.jpeg,.png,.webp,.bmp" : ".jpg,.jpeg,.png,.webp,.bmp,.mp4,.avi,.mov,.mkv,.webm,.m4v";
  $("dropZone").querySelector("strong").textContent = codec ? "把图片拖到这里" : "把图片或视频拖到这里";
  $("dropZone").querySelector("small").textContent = codec ? "JPG / PNG / WebP / BMP" : "JPG / PNG / WebP / BMP · MP4 / AVI / MOV / MKV / WebM / M4V";
  $("dropZone").setAttribute("aria-label", codec ? "选择或拖入图片文件" : "选择或拖入图片、视频文件");
  $("inputSizeLabel").textContent = semantic ? "发送端语义输入 · 与训练一致" : codec ? "语义检查输入边长" : "模型输入边长";
  $("uploadHint").textContent = codec ? "拖入即按下方参数开始。图片按原尺寸编码，建议先试 720p 或 1080p。" : "拖入即按下方参数开始处理。单个文件 ≤ 1 GiB，演示建议用 720p 或 1080p 短视频。";
  $("emptyHint").textContent = codec ? "拖入一张图片，完成后可查看还原画面、传输记录和语义变化。" : "先试一张图片，再播放一段视频。完成后可在这里切换显示方式、下载结果。";
  if (!state.rendered) {
    state.view = codec ? "reconstructed" : "overlay";
    $("reconstructedTab").hidden = !codec;
    $("legend").hidden = codec;
    $("resultSubtitle").textContent = codec ? "原图与接收端还原图片并排显示" : "原画面与预测结果并排显示";
    document.querySelectorAll("[data-view]").forEach(tab => tab.setAttribute("aria-selected", String(tab.dataset.view === state.view)));
  }
  $("experimentNote").textContent = semantic ? "联合语义源编码原型；类别图直接从接收码流解码。" : codec ? "神经图片编码与 TCP 文件收发；语义结果用于评价。" : "图片分割与视频逐帧推理。";
}
function busy() { const current = state.jobs.find((job) => job.id === state.active); return state.uploading || (current && !terminal.has(current.status)); }
function refreshButtons() {
  const unavailable = !state.connected || state.config?.model.state !== "ready";
  $("dropZone").disabled = unavailable || state.uploading;
  $("rerunButton").disabled = unavailable || busy() || state.jobs.find(job => job.id === state.active)?.status !== "done";
  $("settingsFields").disabled = state.uploading;
  $("resetSettings").disabled = state.uploading;
}
function progress(title, detail, percent) {
  $("progressPanel").hidden = false;
  $("progressTitle").textContent = title;
  $("progressDetail").textContent = detail;
  const value = Math.max(0, Math.min(100, Number(percent) || 0));
  $("progressBar").style.width = `${value}%`;
  $("progressPercent").textContent = `${Math.round(value)}%`;
}
function options() {
  if (!$("maxFrames").checkValidity()) throw new Error("最多处理帧数应为 0～1000000 的整数");
  return {size: $("inputSize").value, alpha: Number($("alpha").value) / 100,
    temporal_window: $("temporalWindow").value, max_frames: $("maxFrames").value || "0", save_masks: $("saveMasks").checked,
    mode: $("operationMode").value, codec_quality: $("codecQuality").value, transfer_kbps: $("transferRate").value};
}
function stopPlayback() {
  state.inputVideo?.pause(); state.outputVideo?.pause();
  state.inputVideo = null; state.outputVideo = null;
}
function clearResults() {
  stopPlayback(); state.rendered = null;
  $("comparison").hidden = true; $("metricsSection").hidden = true; $("detailsSection").hidden = true;
  $("inputStage").replaceChildren(); $("outputStage").replaceChildren();
}

async function upload(file) {
  if (!file || state.uploading) return;
  message("");
  if (!state.connected || state.config?.model.state !== "ready") return message("模型还没有就绪，请稍等右上角状态更新。");
  const extension = `.${file.name.split(".").pop().toLowerCase()}`;
  if (!state.config.extensions.includes(extension)) return message("暂不支持这个文件类型，请选择页面中列出的图片或视频格式。");
  if (!file.size || file.size > state.config.max_upload_bytes) return message("文件为空或超过 1 GiB，请换一份较小的素材。");
  const isVideo = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"].includes(extension);
  if (isVideo && codecMode($("operationMode").value)) return message("当前编码模式请拖入图片。视频分割可切换到语义分割模式。");
  if (isVideo && !state.config.video_available) return message("未找到本机 FFmpeg。图片可以使用，视频需先按演示说明配置 FFmpeg。");
  let settings;
  try { settings = options(); } catch (error) { return message(error.message); }
  state.active = null; state.uploading = true;
  state.view = codecMode(settings.mode) ? "reconstructed" : "overlay";
  $("selectedFile").hidden = false; $("fileName").textContent = file.name; $("fileSize").textContent = formatBytes(file.size);
  clearResults(); $("emptyState").hidden = true; $("resultSubtitle").textContent = file.name;
  progress("正在上传到本机", "文件只会保存到这台电脑", 0); refreshButtons();
  try {
    const job = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest(); state.xhr = xhr;
      xhr.open("POST", `/api/jobs?${new URLSearchParams({filename: file.name, ...settings})}`);
      xhr.setRequestHeader("Content-Type", "application/octet-stream");
      xhr.upload.onprogress = (event) => { if (event.lengthComputable) progress("正在上传到本机", `${formatBytes(event.loaded)} / ${formatBytes(event.total)}`, 100 * event.loaded / event.total); };
      xhr.onload = () => {
        let data; try { data = JSON.parse(xhr.responseText); } catch { return reject(new Error("服务返回异常，请检查启动窗口。")); }
        if (xhr.status >= 200 && xhr.status < 300) resolve(data); else reject(new Error(data.error || "上传失败"));
      };
      xhr.onerror = () => reject(new Error("连接中断，请确认 run_demo.cmd 的窗口仍然开着。"));
      xhr.onabort = () => reject(new Error("上传已停止。若文件已传完，可在最近处理中停止对应任务。"));
      xhr.send(file);
    });
    state.active = job.id; state.jobs.unshift(job); state.uploading = false;
    progress("等待模型处理", "同一时间处理一个任务", 0);
    await poll();
  } catch (error) {
    message(error.message); $("progressPanel").hidden = true; $("emptyState").hidden = false;
  } finally {
    state.uploading = false; state.xhr = null; $("fileInput").value = ""; refreshButtons();
  }
}

function renderLegend(classes) {
  $("legend").replaceChildren(...classes.map((row) => {
    const item = element("span", "legend-item");
    const dot = element("span", "swatch"); dot.style.backgroundColor = `rgb(${row.rgb.join(",")})`;
    item.append(dot, document.createTextNode(row.name_zh)); return item;
  }));
}
function renderHistory() {
  const jobs = state.jobs.slice(0, 15);
  const signature = JSON.stringify([state.active, state.uploading, jobs.map(job => [job.id, job.status])]);
  if (signature === state.historySignature) return;
  state.historySignature = signature;
  $("historyCount").textContent = jobs.length ? `${jobs.length} 条` : "";
  if (!jobs.length) { $("historyList").replaceChildren(element("p", "hint", "处理过的结果会保存在这里。")); return; }
  $("historyList").replaceChildren(...jobs.map((job) => {
    const button = element("button", `history-item${job.id === state.active ? " active" : ""}${["error", "cancelled", "interrupted"].includes(job.status) ? " failed" : ""}`);
    button.type = "button"; button.title = job.filename;
    const copy = element("div", "history-copy");
    const date = new Date(job.created_at).toLocaleString("zh-CN", {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false});
    copy.append(element("div", "history-name", job.filename), element("div", "history-meta", `${date} · ${statusNames[job.status] || job.status}`));
    button.append(element("span", "history-icon", job.kind === "image" ? "图片" : "视频"), copy, element("span", "history-dot"));
    button.disabled = state.uploading;
    button.addEventListener("click", () => { state.active = job.id; state.view = codecJob(job) ? "reconstructed" : "overlay"; clearResults(); message(""); renderJob(job); renderHistory(); refreshButtons(); });
    return button;
  }));
}
function renderMetrics(job) {
  const s = job.summary;
  const metrics = codecJob(job) ? [
    ["实际压缩文件", (s.codec.file_bytes / 1024).toFixed(1), "KiB", `含文件头 · ${s.codec.bpp.toFixed(4)} bpp`],
    ["还原 PSNR", s.quality.psnr_db?.toFixed(2) ?? "相同", "dB", "原尺寸 RGB 像素误差"],
    ["还原 SSIM", s.quality.ssim?.toFixed(4) ?? "—", "", "结构相似度，越接近 1 越好"],
    ["实际平均发送", s.transfer.average_application_kbps.toFixed(2), "kbps", `限速 ${s.transfer.limit_kbps} kbps · 含应用层传输头`],
    ["文件传输耗时", s.transfer.elapsed_seconds.toFixed(2), "秒", "TCP 收发与接收确认"],
    ["编码计算", (s.codec.encode_compute_seconds * 1000).toFixed(1), "ms", "CPU 神经编码与熵编码，不含加载"],
    ["解码计算", (s.codec.decode_compute_seconds * 1000).toFixed(1), "ms", "CPU 从接收文件解码，不含加载"],
    [semanticJob(job) ? "两端类别一致" : "前后类别一致", (s.semantic_consistency.pixel_agreement * 100).toFixed(1), "%", semanticJob(job) ? "发送端条件与直接解码类别；不是真值准确率" : "分割预测一致比例，不是真值准确率"],
  ] : job.kind === "video" ? [
    ["实际处理速度", s.processing_fps.toFixed(1), "FPS", "含读帧、分割及三路视频写入"],
    ["平均模型前向", s.timing_ms.inference_ms.toFixed(1), "ms / 帧", "仅神经网络前向计算"],
    ["已处理帧数", s.processed_frames.toLocaleString(), "帧", `预览 ${s.duration_seconds.toFixed(1)} 秒 · ${s.input_fps.toFixed(2)} FPS`],
    ["处理与保存", s.elapsed_seconds.toFixed(2), "秒", "不含上传、启动预热和 ZIP 打包"],
  ] : [
    ["模型前向", s.timing_ms.inference_ms.toFixed(1), "ms", "仅神经网络前向计算"],
    ["预处理与还原", (s.timing_ms.preprocess_ms + s.timing_ms.postprocess_ms).toFixed(1), "ms", "缩放、补边、还原原图尺寸"],
    ["模型输入", s.model_input_size, "px", "等比缩放后的长边尺寸"],
    ["处理与保存", s.elapsed_seconds.toFixed(2), "秒", "不含上传、启动预热和 ZIP 打包"],
  ];
  $("metricsSection").replaceChildren(...metrics.map(([label, value, unit, note]) => {
    const card = element("div", "metric"); const number = element("div", "metric-value", value);
    number.append(element("small", "", unit));
    card.append(element("div", "metric-label", label), number, element("div", "metric-note", note)); return card;
  }));
  $("metricsSection").hidden = false;
}
function renderDetails(job) {
  const s = job.summary;
  $("fractionNote").textContent = job.kind === "video" ? "全部已处理帧的像素占比" : "像素占比";
  $("classBars").replaceChildren(...[...s.classes].sort((a,b) => b.predicted_fraction - a.predicted_fraction).map((row) => {
    const wrapper = element("div", "class-row"); const name = element("div", "class-name");
    const dot = element("span", "swatch"); dot.style.backgroundColor = `rgb(${row.rgb.join(",")})`;
    name.append(dot, document.createTextNode(row.name_zh));
    const track = element("div", "class-track"); const fill = element("span"); fill.style.width = `${100 * row.predicted_fraction}%`; track.append(fill);
    wrapper.append(name, track, element("span", "class-percent", `${(100 * row.predicted_fraction).toFixed(1)}%`)); return wrapper;
  }));
  $("savedNote").textContent = codecJob(job) ? "压缩文件、接收副本、还原图片和传输报告已保存。" : "预览、分割结果和参数报告已保存。刷新页面后，可从最近处理重新打开。";
  $("downloadAll").href = mediaUrl(job, "results.zip", true);
  const links = [["叠加结果", s.outputs.overlay], ["彩色类别", s.outputs.color], ["数据报告 JSON", "summary.json"]];
  if (job.kind === "image") links.push(["类别编号 PNG", s.outputs.index]); else links.push(["逐帧耗时 CSV", "frames.csv"]);
  if (codecJob(job)) links.unshift(["还原图片", "reconstructed.png"], ["发送端压缩文件", "encoded.svic"],
    ["接收端压缩文件", "received.svic"], ["传输记录 CSV", "transfer.csv"], [semanticJob(job) ? "发送端语义条件" : "原图分割结果", "semantic_before.png"]);
  $("fileLinks").replaceChildren(...links.map(([label, filename]) => { const link = element("a", "", label); link.href = mediaUrl(job, filename, true); link.download = filename; return link; }));
  $("runDetails").textContent = codecJob(job) ? `压缩质量 ${s.codec.quality} · ${s.transfer.limit_kbps} kbps · ${s.width} × ${s.height} 原尺寸编码 · 文件校验${s.transfer.verified ? "通过" : "失败"}。语义检查输入 ${s.model_input_size} px。处理与保存 ${s.elapsed_seconds.toFixed(2)} 秒，不含上传和 ZIP。记录编号：${job.id.slice(0, 8)}` : `本次参数：${s.model_input_size} px · 叠加 ${Math.round(s.options.alpha * 100)}% · ${s.device.toUpperCase()}${job.kind === "video" ? ` · ${s.options.temporal_window === 3 ? "三帧投票" : "独立预测"}` : ""}。记录编号：${job.id.slice(0, 8)}`;
  if (semanticJob(job)) $("runDetails").textContent = `联合训练第 ${s.codec.training_epoch} 轮权重 · ${s.transfer.limit_kbps} kbps · ${s.width} × ${s.height} 原尺寸编码。发送端语义输入固定为 ${s.model_input_size} px；接收端只使用联合解码器。处理与保存 ${s.elapsed_seconds.toFixed(2)} 秒。记录编号：${job.id.slice(0, 8)}`;
  const notes = [...(s.warnings || [])];
  if (job.kind === "video") notes.push("处理速度不等于网络传输帧率；相邻预测变化也包含实际运动。");
  $("resultWarnings").replaceChildren(...notes.map((text) => element("div", "", text)));
  $("detailsSection").hidden = false;
}

function mediaError() { $("mediaWarning").hidden = false; $("mediaWarning").textContent = "浏览器暂时无法播放这个预览。可以下载结果，用本机播放器打开，或换用 Edge / Chrome。"; }
function createMedia(job, filename, output) {
  const node = document.createElement(job.kind === "video" ? "video" : "img");
  if (job.kind === "video") {
    node.controls = output; node.muted = true; node.playsInline = true; node.preload = "auto";
    node.setAttribute("aria-label", output ? "分割结果视频" : "输入视频");
  } else { node.alt = filename === "reconstructed.png" ? "接收端还原图片" : filename === "semantic_before.png" ? "原图分割结果" : output ? "模型分割结果" : "输入图片"; }
  node.addEventListener("error", mediaError);
  node.src = mediaUrl(job, filename);
  return node;
}
function setVideoSync() {
  const input = state.inputVideo, output = state.outputVideo;
  if (!input || !output) return;
  const sync = () => { if (Number.isFinite(input.duration) && Math.abs(input.currentTime - output.currentTime) > 0.12) input.currentTime = output.currentTime; };
  output.addEventListener("play", () => { sync(); input.play().catch(mediaError); $("playBoth").textContent = "暂停播放"; });
  output.addEventListener("pause", () => { input.pause(); $("playBoth").textContent = "同步播放"; });
  output.addEventListener("seeking", sync); output.addEventListener("timeupdate", sync);
  output.addEventListener("ratechange", () => { input.playbackRate = output.playbackRate; });
  output.addEventListener("ended", () => { input.pause(); $("playBoth").textContent = "重新播放"; });
}
function renderResult(job) {
  const s = job.summary;
  const codec = codecJob(job);
  if (!codec && state.view === "reconstructed") state.view = "overlay";
  $("reconstructedTab").hidden = !codec;
  document.querySelectorAll("[data-view]").forEach(tab => tab.setAttribute("aria-selected", String(tab.dataset.view === state.view)));
  const sameVideo = job.kind === "video" && state.rendered?.startsWith(`${job.id}:`);
  const position = sameVideo ? state.outputVideo?.currentTime || 0 : 0;
  const wasPlaying = sameVideo && state.outputVideo && !state.outputVideo.paused;
  stopPlayback();
  $("emptyState").hidden = true; $("progressPanel").hidden = true; $("comparison").hidden = false;
  $("inputDimensions").textContent = `${s.width} × ${s.height}`;
  $("inputLabel").textContent = codec && state.view === "color" ? semanticJob(job) ? "发送端语义条件" : "原图分割结果" : "输入画面";
  $("outputKind").textContent = state.view === "reconstructed" ? "文件解码" : "模型输出";
  $("colorTab").textContent = semanticJob(job) ? "两端类别" : codec ? "前后类别" : "纯色类别";
  $("outputLabel").textContent = state.view === "reconstructed" ? "接收端还原图片" : state.view === "overlay" ? semanticJob(job) ? "直接解码类别叠加" : "预测叠加" : semanticJob(job) ? "接收端解码类别" : codec ? "还原图分割结果" : "纯色类别";
  $("mediaWarning").hidden = true;
  const input = createMedia(job, codec && state.view === "color" ? s.outputs.semantic_before : s.outputs.input, false);
  const output = createMedia(job, s.outputs[state.view], true);
  $("inputStage").replaceChildren(input); $("outputStage").replaceChildren(output);
  $("playbackBar").hidden = job.kind !== "video";
  if (job.kind === "video") {
    state.inputVideo = input; state.outputVideo = output; $("playBoth").textContent = "同步播放"; setVideoSync();
    output.addEventListener("loadedmetadata", () => {
      if (position) output.currentTime = Math.min(position, output.duration);
      if (wasPlaying) output.play().catch(mediaError);
    }, {once:true});
    input.addEventListener("loadedmetadata", () => { if (position) input.currentTime = Math.min(position, input.duration); }, {once:true});
  }
  renderMetrics(job); renderDetails(job); renderLegend(s.classes); state.rendered = `${job.id}:${state.view}`;
  $("legend").hidden = state.view === "reconstructed";
  $("modelName").textContent = semanticJob(job) ? "SUIM 联合语义编码 · 图片与类别双解码" : codec ? "CompressAI Hyperprior · SegFormer 语义检查" : s.model_name;
  $("experimentNote").textContent = s.note;
}
function renderJob(job) {
  $("resultSubtitle").textContent = job.filename;
  $("selectedFile").hidden = false;
  $("fileName").textContent = job.filename;
  $("fileSize").textContent = job.bytes ? formatBytes(job.bytes) : "";
  if (job.status === "done") {
    if (state.rendered !== `${job.id}:${state.view}`) renderResult(job);
  } else if (terminal.has(job.status)) {
    clearResults(); $("progressPanel").hidden = true; $("emptyState").hidden = false;
    message(job.error || job.phase);
  } else {
    $("emptyState").hidden = true;
    let detail = job.phase;
    if (job.processed_frames) detail = `已处理 ${job.processed_frames}${job.total_frames ? ` / ${job.total_frames}` : ""} 帧${job.live_fps ? ` · 当前 ${job.live_fps.toFixed(1)} FPS` : ""}`;
    progress(job.phase, detail, job.progress);
  }
}

async function poll() {
  if (state.polling) return;
  state.polling = true;
  try {
    const [configResponse, jobsResponse] = await Promise.all([fetch("/api/config", {cache: "no-store", signal: AbortSignal.timeout(5000)}), fetch("/api/jobs", {cache: "no-store", signal: AbortSignal.timeout(5000)})]);
    if (!configResponse.ok || !jobsResponse.ok) throw new Error("服务状态异常");
    const [config, jobs] = await Promise.all([configResponse.json(), jobsResponse.json()]);
    state.config = config; state.jobs = jobs; state.connected = true; $("offlineBox").hidden = true;
    const codecOption = $("operationMode").querySelector('option[value="image_codec"]');
    codecOption.disabled = !config.image_codec_available;
    $("operationMode").querySelector('option[value="semantic_codec"]').disabled = !config.semantic_codec_available;
    if (!state.settingsInitialized) {
      $("operationMode").value = config.default_mode === "semantic_codec" && config.semantic_codec_available ? "semantic_codec" : config.default_mode === "image_codec" && config.image_codec_available ? "image_codec" : "segmentation";
      state.settingsInitialized = true; updateMode();
    }
    const model = config.model;
    $("modelStatus").textContent = model.state === "ready" ? `模型已就绪 · ${model.device_name}` : model.message;
    $("statusDot").className = `status-dot ${model.state}`;
    $("outputPath").textContent = `本机保存目录：${config.output_directory}`;
    if (model.classes && !$("legend").children.length) renderLegend(model.classes);
    if (!config.video_available && !$("errorBox").textContent) message("图片推理可用。视频演示需要本机 FFmpeg，请按项目演示说明配置路径。");
    if (!state.active && !state.uploading && jobs.length) state.active = (jobs.find(job => !terminal.has(job.status)) || jobs.find(job => job.status === "done") || jobs[0]).id;
    const active = jobs.find((job) => job.id === state.active);
    if (active && !state.uploading) renderJob(active);
    renderHistory(); refreshButtons();
  } catch {
    state.connected = false; $("offlineBox").hidden = false;
    $("modelStatus").textContent = "服务未连接"; $("statusDot").className = "status-dot error"; refreshButtons();
  } finally { state.polling = false; }
}

$("dropZone").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", () => upload($("fileInput").files[0]));
for (const name of ["dragenter", "dragover"]) $("dropZone").addEventListener(name, (event) => { event.preventDefault(); $("dropZone").classList.add("dragging"); });
for (const name of ["dragleave", "drop"]) $("dropZone").addEventListener(name, (event) => { event.preventDefault(); $("dropZone").classList.remove("dragging"); });
$("dropZone").addEventListener("drop", (event) => { if (event.dataTransfer.files.length > 1) return message("一次拖入一个文件即可。多个实验结果可以在最近处理中切换。"); upload(event.dataTransfer.files[0]); });
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
$("alpha").addEventListener("input", () => { $("alphaValue").value = `${$("alpha").value}%`; });
$("operationMode").addEventListener("change", updateMode);
$("resetSettings").addEventListener("click", () => { $("inputSize").value = "256"; $("alpha").value = "45"; $("alphaValue").value = "45%"; $("temporalWindow").value = "3"; $("maxFrames").value = "0"; $("saveMasks").checked = false; $("codecQuality").value = "3"; $("transferRate").value = "20"; });
$("rerunButton").addEventListener("click", async () => {
  if (!state.active || busy()) return;
  message("");
  let settings;
  try { settings = options(); } catch (error) { return message(error.message); }
  state.uploading = true; refreshButtons();
  try {
    const response = await fetch(`/api/jobs/${state.active}/rerun`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(settings)});
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "重新处理失败");
    state.active = job.id; state.jobs.unshift(job); state.uploading = false;
    state.view = codecJob(job) ? "reconstructed" : "overlay";
    clearResults(); $("emptyState").hidden = true; renderJob(job); await poll();
  } catch (error) { message(error.message); }
  finally { state.uploading = false; refreshButtons(); }
});
$("cancelButton").addEventListener("click", async () => {
  if (state.xhr) { state.xhr.abort(); return; }
  if (!state.active) return;
  try { const response = await fetch(`/api/jobs/${state.active}/cancel`, {method: "POST"}); if (!response.ok) throw new Error("停止任务失败，请稍后再试"); await poll(); } catch (error) { message(error.message); }
});
document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
  state.view = button.dataset.view;
  document.querySelectorAll("[data-view]").forEach((tab) => tab.setAttribute("aria-selected", String(tab === button)));
  const job = state.jobs.find((row) => row.id === state.active);
  if (job?.status === "done") renderResult(job);
}));
$("playBoth").addEventListener("click", () => { const video = state.outputVideo; if (!video) return; if (video.paused) video.play().catch(mediaError); else video.pause(); });
$("helpButton").addEventListener("click", () => $("helpDialog").showModal());
poll(); setInterval(poll, 1200);
