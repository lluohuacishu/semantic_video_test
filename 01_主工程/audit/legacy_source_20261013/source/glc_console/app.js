"use strict";

const $ = id => document.getElementById(id);
const terminal = new Set(["done", "error", "cancelled", "interrupted"]);
let config = null, selectedFile = null, current = null, upload = null;
let mediaKey = null, frameOffset = 0, frameTotal = 0, historyKey = "", pollBusy = false, submitting = false, selectionVersion = 0;
const videos = [$("inputVideo"), $("outputVideo")];

function number(value, digits = 2) {
  return value === null || value === undefined || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);
}
function bytes(value) {
  if (!Number.isFinite(Number(value))) return "—";
  return value >= 1024 ** 2 ? `${number(value / 1024 ** 2, 2)} MiB` : value >= 1024 ? `${number(value / 1024, 2)} KiB` : `${value} B`;
}
function error(message = "") { $("alert").textContent = message; $("alert").hidden = !message; }
async function request(path, options = {}) {
  const response = await fetch(path, {cache: "no-store", ...options});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}
function settings() {
  return {q: Number($("quality").value), width: Number($("width").value), frames: Number($("frames").value),
    kbps: Number($("kbps").value), start_frame: Number($("startFrame").value),
    intra_period: Number($("intraPeriod").value), reset_interval: Number($("resetInterval").value)};
}
function showSettings(options) {
  const fields = {q: "quality", width: "width", frames: "frames", kbps: "kbps", start_frame: "startFrame",
    intra_period: "intraPeriod", reset_interval: "resetInterval"};
  for (const [key, id] of Object.entries(fields)) if (options[key] !== undefined) $(id).value = options[key];
  refreshParameterLabels();
}
function refreshParameterLabels() {
  $("qualityValue").textContent = `q = ${$("quality").value}`;
  document.querySelectorAll("[data-rate]").forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.rate) === Number($("kbps").value))));
}
function controls() {
  const active = Boolean(current && !terminal.has(current.status));
  const rerun = current && !current.is_example && terminal.has(current.status) && current.bytes > 0;
  $("settings").disabled = Boolean(upload) || submitting || active;
  $("dropZone").disabled = Boolean(upload) || submitting;
  document.querySelectorAll(".history-item,.example-button").forEach(button => button.disabled = Boolean(upload) || submitting);
  $("startButton").disabled = !config || config.model.state !== "ready" || submitting || Boolean(upload) || active || !(selectedFile || rerun);
  $("startButton").textContent = upload ? "正在上传…" : active ? "实验进行中…" : selectedFile ? "开始实验 →" : rerun ? "按当前参数重新运行 →" : "开始实验 →";
  $("cancelButton").disabled = !upload && (!active || current.status === "cancelling");
}
function mediaUrl(identifier, name, download = false) {
  return `/media/${encodeURIComponent(identifier)}/${name}${download ? "?download=1" : ""}`;
}
function clearMedia() {
  videos.forEach(video => {video.pause(); video.removeAttribute("src"); video.removeAttribute("poster"); video.load();});
  mediaKey = null;
  $("comparison").hidden = true;
  $("results").hidden = true;
  $("verifiedBadge").hidden = true;
  $("emptyState").hidden = false;
}
function pick(file) {
  if (!file || upload || submitting) return;
  if (!/\.(mp4|avi|mov|mkv|webm|m4v)$/i.test(file.name)) return error("请选择 MP4、AVI、MOV、MKV、WebM 或 M4V 视频。");
  if (!file.size || file.size > (config?.max_upload_bytes || 1024 ** 3)) return error("文件为空或超过 1 GiB。");
  selectedFile = file;
  selectionVersion += 1;
  current = null;
  historyKey = "";
  error();
  clearMedia();
  $("fileName").textContent = file.name;
  $("fileMeta").textContent = `${bytes(file.size)} · 等待开始`;
  $("resultTitle").textContent = file.name;
  $("emptyTitle").textContent = "视频已选择";
  $("emptyText").textContent = "确认左侧参数后，点击“开始实验”。";
  $("progressPanel").hidden = true;
  controls();
}
async function selectRun(identifier) {
  if (upload) return;
  const version = ++selectionVersion;
  error();
  const run = await request(`/api/runs/${encodeURIComponent(identifier)}`);
  if (version !== selectionVersion) return;
  selectedFile = null;
  $("fileInput").value = "";
  current = run;
  localStorage.setItem("glc-last-run", run.id);
  frameOffset = 0;
  if (run.options) showSettings(run.options);
  else if (config) showSettings({...config.defaults, q: run.summary.q});
  $("fileName").textContent = run.is_example ? "选择自己的视频，开始新实验" : run.filename;
  $("fileMeta").textContent = run.is_example ? "正在查看已有的 720p 实验" : `${bytes(run.bytes)} · 原文件已保存在本机`;
  renderRun(run);
  historyKey = "";
}
function renderRun(run) {
  $("resultTitle").textContent = run.filename;
  $("progressPanel").hidden = run.status === "done";
  $("phase").textContent = run.phase;
  $("progress").value = Math.min(100, Math.max(0, run.progress || 0));
  $("percent").textContent = `${number(run.progress || 0, 0)}%`;
  let detail = "";
  if (run.stage === "transfer") detail = `${bytes(run.transferred_bytes || 0)} / ${bytes(run.transfer_total_bytes)} · 限速 ${run.options.kbps} kbps`;
  else if (run.total_frames) detail = `${run.processed_frames || 0} / ${run.total_frames} 帧`;
  $("progressDetail").textContent = detail;
  const stages = ["encoding", "transfer", "decoding", "verifying"];
  document.querySelectorAll("[data-stage]").forEach(item => {
    item.classList.toggle("active", item.dataset.stage === run.stage && !terminal.has(run.status));
    item.classList.toggle("complete", stages.indexOf(item.dataset.stage) < stages.indexOf(run.stage));
  });
  $("elapsed").textContent = terminal.has(run.status) ? run.phase : run.started_at ? `已运行 ${number((Date.now() - Date.parse(run.started_at)) / 1000, 0)} 秒` : "等待空闲的处理进程";
  if (run.status === "error") error(run.error || "处理失败，请调整参数后重试。");
  if (run.status !== "done") {
    if (mediaKey !== null) clearMedia();
    $("emptyTitle").textContent = run.status === "cancelled" ? "任务已停止" : run.status === "error" ? "本次处理未完成" : run.status === "interrupted" ? "上次任务已中断" : "正在处理视频";
    $("emptyText").textContent = terminal.has(run.status)
      ? run.bytes > 0 ? "原视频仍在，可以调整左侧参数后重新运行。" : "文件尚未完整上传，请重新选择视频。"
      : "完成后会显示输入片段、接收端还原视频和实验数据。";
  } else if (mediaKey !== run.id) {
    renderResult(run);
  }
  controls();
}
function renderResult(run) {
  const data = run.summary;
  videos.forEach(video => video.pause());
  $("inputVideo").src = mediaUrl(run.id, "input.mp4");
  $("inputVideo").poster = mediaUrl(run.id, "input.png");
  $("outputVideo").src = mediaUrl(run.id, "reconstructed.mp4");
  $("outputVideo").poster = mediaUrl(run.id, "reconstructed.png");
  mediaKey = run.id;
  $("emptyState").hidden = true;
  $("comparison").hidden = false;
  $("results").hidden = false;
  $("verifiedBadge").hidden = !data.verified;
  $("inputSize").textContent = `${data.width} × ${data.height}`;
  $("clipInfo").textContent = `${data.frames} 帧 · ${number(data.fps, 1)} FPS · ${number(data.duration_seconds, 2)} 秒`;
  $("fileBytes").textContent = bytes(data.file_bytes);
  $("fileBytes").title = `${data.file_bytes.toLocaleString()} 字节`;
  $("bpp").textContent = `${number(data.actual_bpp, 6)} bpp · 含文件头`;
  $("videoRate").textContent = `${number(data.actual_video_kbps)} kbps`;
  $("transferTime").textContent = `${number(data.transfer_seconds)} s`;
  $("transferRate").textContent = `文件传输限速 ${data.limit_kbps} kbps`;
  $("psnr").textContent = `${number(data.psnr_db)} dB`;
  $("encodeTime").textContent = `${number(data.encode_seconds, 1)} s`;
  $("decodeTime").textContent = `${number(data.decode_seconds, 1)} s`;
  $("gpuMemory").textContent = data.peak_allocated_mib === null ? "—" : `${number(data.peak_allocated_mib / 1024)} GiB`;
  $("actualRate").textContent = `${number(data.average_application_kbps)} kbps`;
  $("qualityTag").textContent = `q = ${data.q}${run.is_example ? " · 已有实验" : ""}`;
  $("resultDirectory").textContent = run.result_directory;
  const items = [["reconstructed.mp4", "还原视频"], ["comparison.mp4", "并排对比视频"], ["encoded.glcv", "压缩码流 .glcv"], ["encode.json", "编码数据 JSON"], ["decode.json", "解码数据 JSON"], ["transfer.csv", "传输记录 CSV"]];
  if (!run.is_example) items.push(["frames.csv", "逐帧数据 CSV"], ["log.txt", "运行日志"]);
  $("downloads").replaceChildren(...items.map(([name, label]) => {
    const link = document.createElement("a"); link.href = mediaUrl(run.id, name, true); link.textContent = label; return link;
  }));
  loadFrames().catch(exc => error(exc.message));
}
async function loadFrames() {
  if (!current || current.status !== "done") return;
  const id = current.id, offset = frameOffset;
  const data = await request(`/api/runs/${encodeURIComponent(id)}/frames?offset=${offset}`);
  if (current?.id !== id || frameOffset !== offset) return;
  frameTotal = data.total;
  const fragment = document.createDocumentFragment();
  data.rows.forEach(row => {
    const tr = document.createElement("tr");
    [row.frame, row.type, `${row.bytes.toLocaleString()} B`, `${number(row.psnr_db)} dB`, `${number(row.encode_seconds)} s`, `${number(row.decode_seconds)} s`].forEach((value, index) => {
      const cell = document.createElement("td"); cell.textContent = value;
      if (index === 1 && row.type === "I") cell.classList.add("i-frame"); tr.append(cell);
    }); fragment.append(tr);
  });
  $("frameRows").replaceChildren(fragment);
  $("pageInfo").textContent = `第 ${data.offset + 1}–${data.offset + data.rows.length} 帧，共 ${data.total} 帧`;
  $("previousPage").disabled = frameOffset === 0;
  $("nextPage").disabled = frameOffset + 50 >= frameTotal;
}
function renderHistory(jobs) {
  const signature = JSON.stringify([current?.id, jobs.map(job => [job.id, job.status, job.updated_at])]);
  if (signature === historyKey) return;
  historyKey = signature;
  const labels = {done: "已完成", queued: "排队中", running: "处理中", error: "失败", cancelled: "已停止", cancelling: "停止中", interrupted: "已中断", uploading: "上传中"};
  $("history").replaceChildren(...jobs.map(job => {
    const button = document.createElement("button"); button.className = "history-item"; button.classList.toggle("active", job.id === current?.id);
    const name = document.createElement("strong"); name.textContent = job.filename;
    const meta = document.createElement("small"); meta.textContent = `${labels[job.status] || job.status} · q=${job.options.q} · ${job.options.kbps} kbps`;
    button.append(name, meta); button.addEventListener("click", () => selectRun(job.id).catch(exc => error(exc.message))); return button;
  }));
  if (!jobs.length) {const text = document.createElement("p"); text.className = "hint"; text.textContent = "还没有新任务"; $("history").append(text);}
  document.querySelectorAll("[data-example]").forEach(button => button.classList.toggle("active", button.dataset.example === current?.id));
}
async function start() {
  if (submitting || upload) return;
  if (!$("settingsForm").reportValidity()) return;
  submitting = true;
  controls();
  try {
  error();
  const options = settings();
  if (!selectedFile) {
    if (!current || current.is_example || !terminal.has(current.status)) return;
    const job = await request(`/api/jobs/${current.id}/rerun`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(options)});
    await selectRun(job.id); return;
  }
  const file = selectedFile;
  const reservation = await request("/api/jobs/reserve", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({filename: file.name, bytes: file.size, options})});
  current = reservation;
  const xhr = new XMLHttpRequest(); upload = xhr;
  $("progressPanel").hidden = false; $("phase").textContent = "正在上传到本机工作目录";
  $("progressDetail").textContent = file.name; $("elapsed").textContent = "上传完成后开始编码";
  $("progress").value = 0; $("percent").textContent = "0%";
  document.querySelectorAll("[data-stage]").forEach(item => item.classList.remove("active", "complete"));
  controls();
  xhr.open("POST", `/api/jobs/${reservation.id}/source`);
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.upload.onprogress = event => {
    if (!event.lengthComputable) return;
    const percent = event.loaded / event.total * 100;
    $("progress").value = percent; $("percent").textContent = `${number(percent, 0)}%`;
    $("progressDetail").textContent = `${bytes(event.loaded)} / ${bytes(event.total)}`;
  };
  xhr.onload = async () => {
    upload = null;
    try {
      const data = JSON.parse(xhr.responseText);
      if (xhr.status !== 202) throw new Error(data.error || "上传失败");
      await selectRun(data.id);
    } catch (exc) {error(exc.message); $("phase").textContent = "上传失败";}
    controls();
  };
  xhr.onerror = () => {upload = null; error("无法连接本机服务，请确认启动窗口仍在运行。"); controls();};
  xhr.onabort = () => {upload = null; $("phase").textContent = "已停止上传"; $("elapsed").textContent = "可以重新开始"; controls();};
  xhr.send(file);
  } finally {submitting = false; controls();}
}
async function poll() {
  if (pollBusy) return;
  pollBusy = true;
  try {
    config = await request("/api/config");
    $("environmentText").textContent = config.model.state === "ready" ? `${config.model.gpu.replace("NVIDIA GeForce ", "")} · 环境就绪` : config.model.message;
    $("environmentDot").className = `dot ${config.model.state}`;
    if (config.model.state === "error") error(config.model.message);
    const jobs = await request("/api/jobs");
    renderHistory(jobs);
    if (current && !terminal.has(current.status) && !upload) {
      const id = current.id;
      const fresh = await request(`/api/runs/${id}`);
      if (current?.id === id) {current = fresh; renderRun(fresh);}
    }
    controls();
  } catch (exc) {
    $("environmentDot").className = "dot error"; $("environmentText").textContent = "本机服务未连接";
    config = null; controls();
  } finally {pollBusy = false;}
}

$("dropZone").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", event => pick(event.target.files[0]));
for (const name of ["dragenter", "dragover"]) $("dropZone").addEventListener(name, event => {event.preventDefault(); $("dropZone").classList.add("dragging");});
$("dropZone").addEventListener("dragleave", () => $("dropZone").classList.remove("dragging"));
$("dropZone").addEventListener("drop", event => {event.preventDefault(); $("dropZone").classList.remove("dragging"); if (event.dataTransfer.files.length !== 1) return error("请一次选择一个视频。"); pick(event.dataTransfer.files[0]);});
document.addEventListener("dragover", event => event.preventDefault());
document.addEventListener("drop", event => event.preventDefault());
$("settingsForm").addEventListener("submit", event => {event.preventDefault(); start().catch(exc => error(exc.message));});
$("quality").addEventListener("input", refreshParameterLabels);
$("kbps").addEventListener("input", refreshParameterLabels);
document.querySelectorAll("[data-rate]").forEach(button => button.addEventListener("click", () => {$("kbps").value = button.dataset.rate; refreshParameterLabels();}));
$("cancelButton").addEventListener("click", async () => {
  if (upload) upload.abort();
  if (!current || terminal.has(current.status)) return;
  try {const job = await request(`/api/jobs/${current.id}/cancel`, {method: "POST"}); current = job; renderRun(job);} catch (exc) {error(exc.message);}
});
$("refreshButton").addEventListener("click", () => {historyKey = ""; poll();});
$("previousPage").addEventListener("click", () => {frameOffset = Math.max(0, frameOffset - 50); loadFrames().catch(exc => error(exc.message));});
$("nextPage").addEventListener("click", () => {if (frameOffset + 50 < frameTotal) frameOffset += 50; loadFrames().catch(exc => error(exc.message));});
for (const id of ["helpButton", "aboutButton"]) $(id).addEventListener("click", () => $("helpDialog").showModal());
$("closeHelp").addEventListener("click", () => $("helpDialog").close());
$("playButton").addEventListener("click", async () => {
  try {
    if (videos.some(video => !video.paused)) videos.forEach(video => video.pause());
    else {if (videos[0].ended) videos.forEach(video => video.currentTime = 0); videos[1].currentTime = videos[0].currentTime; await Promise.all(videos.map(video => video.play()));}
  } catch (exc) {error("预览暂时无法播放，可以使用“还原视频”下载后查看。");}
});
$("rewindButton").addEventListener("click", () => {videos.forEach(video => {video.pause(); video.currentTime = 0;});});
videos.forEach((video, index) => {
  const other = videos[1 - index];
  video.addEventListener("play", () => {if (other.paused) other.play().catch(() => {}); $("playButton").textContent = "同步暂停";});
  video.addEventListener("pause", () => {if (!other.paused) other.pause(); $("playButton").textContent = "同步播放";});
  video.addEventListener("seeking", () => {if (other.readyState && Math.abs(video.currentTime - other.currentTime) > 0.08) other.currentTime = video.currentTime;});
  video.addEventListener("ratechange", () => {if (other.playbackRate !== video.playbackRate) other.playbackRate = video.playbackRate;});
  video.addEventListener("timeupdate", () => {if (index === 0 && !video.paused && !video.seeking && !other.seeking && Math.abs(video.currentTime - other.currentTime) > 0.3) other.currentTime = video.currentTime;});
});

(async () => {
  await poll();
  if (config) {
    $("examples").replaceChildren(...config.examples.map(example => {
      const button = document.createElement("button"); button.className = "example-button"; button.dataset.example = example.id;
      button.textContent = `查看 ${example.label}`; button.addEventListener("click", () => selectRun(example.id).catch(exc => error(exc.message))); return button;
    }));
    try {
      const jobs = await request("/api/jobs");
      const last = localStorage.getItem("glc-last-run");
      const allIds = [...jobs.map(job => job.id), ...config.examples.map(example => example.id)];
      const id = allIds.includes(last) ? last : jobs[0]?.id || config.examples[0]?.id;
      if (id) await selectRun(id);
    } catch (exc) {error(exc.message);}
  }
  setInterval(poll, 1000);
})();
