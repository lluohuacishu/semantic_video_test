const $ = id => document.getElementById(id);
const source = $('sourceVideo'), reconstruction = $('reconstructionVideo');
let selected = null, selectedFile = null, current = null, busy = false, uploading = false;
let selectionVersion = 0, playing = false, sampleAvailable = false;
const sizes = n => n >= 1e6 ? `${(n / 1e6).toFixed(2)} MB` : `${(n / 1000).toFixed(2)} KB`;
const integer = n => Number(n).toLocaleString('en-US');
const rate = n => `${Number(n).toFixed(2)} kbps`;
const seconds = n => Number.isFinite(n) ? `${n.toFixed(1)} s` : '—';
const clock = n => `${Math.floor(n / 60).toString().padStart(2, '0')}:${Math.floor(n % 60).toString().padStart(2, '0')}`;
const codecName = codec => codec === 'semantic-h265' ? '语义 H.265' : '普通 H.265';
const optionIds = ['target', 'background_crf', 'roi_crf', 'background_scale', 'roi_layout', 'resolution', 'duration'];
function notice(message = '') { $('notice').hidden = !message; $('notice').textContent = message; }
async function api(url, options) {
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || '请求失败');
  return value;
}
function buttons() {
  $('run').disabled = busy || uploading || !selectedFile;
  $('sampleRun').disabled = busy || uploading || !sampleAvailable;
  for (const id of optionIds) $(id).disabled = uploading;
}
function pickFile(file) {
  if (!file) return;
  if (!/\.(mp4|mov|mkv|avi|webm|m4v)$/i.test(file.name) || file.size > 1024 ** 3 || !file.size) {
    notice('请选择有效视频（最大 1 GB）'); return;
  }
  selectedFile = file;
  $('fileName').textContent = file.name;
  $('fileInfo').textContent = sizes(file.size);
  notice(); buttons();
}
$('file').addEventListener('change', event => pickFile(event.target.files[0]));
$('dropzone').addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('file').click(); } });
for (const name of ['dragover', 'dragenter']) $('dropzone').addEventListener(name, event => { event.preventDefault(); $('dropzone').classList.add('dragover'); });
for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, event => { event.preventDefault(); $('dropzone').classList.remove('dragover'); });
$('dropzone').addEventListener('drop', event => pickFile(event.dataTransfer.files[0]));
function pause() { source.pause(); reconstruction.pause(); playing = false; $('play').textContent = '播放'; }
function seek(value) {
  pause();
  if (source.readyState) source.currentTime = value;
  if (reconstruction.readyState) reconstruction.currentTime = value;
  $('seek').value = value;
}
async function togglePlayback() {
  if (playing) { pause(); return; }
  if (source.ended || reconstruction.ended) seek(0);
  reconstruction.currentTime = source.currentTime;
  try {
    await Promise.all([source.play(), reconstruction.play()]);
    playing = true; $('play').textContent = '暂停';
  } catch { pause(); notice('视频尚未就绪，请稍后再试'); }
}
$('play').addEventListener('click', togglePlayback);
for (const video of [source, reconstruction]) {
  video.addEventListener('click', () => { if (!$('play').disabled) togglePlayback(); });
  video.addEventListener('ended', pause);
  video.addEventListener('canplay', () => { $('play').disabled = source.readyState < 2 || reconstruction.readyState < 2; });
  video.addEventListener('error', () => { $('play').disabled = true; notice('网页预览加载失败，请刷新或下载视频查看'); });
}
$('restart').addEventListener('click', () => seek(0));
$('seek').addEventListener('input', event => seek(Number(event.target.value)));
$('speed').addEventListener('change', event => { source.playbackRate = reconstruction.playbackRate = Number(event.target.value); });
$('showOverlay').addEventListener('change', () => {
  if (!current?.media.overlay_preview) return;
  const time = source.currentTime;
  pause(); $('play').disabled = true;
  reconstruction.addEventListener('loadedmetadata', () => { reconstruction.currentTime = time; }, {once: true});
  reconstruction.src = $('showOverlay').checked ? current.media.overlay_preview : current.media.reconstruction_preview;
  reconstruction.playbackRate = Number($('speed').value);
  $('rightLabel').textContent = $('showOverlay').checked ? '右 · 重建 + 预测目标框' : '右 · 语义 H.265 重建';
});
function frame() {
  if (current?.metrics) {
    if (playing && Math.abs(source.currentTime - reconstruction.currentTime) > .16 && !reconstruction.seeking) reconstruction.currentTime = source.currentTime;
    if (playing) $('seek').value = source.currentTime;
    $('clock').textContent = `${clock(source.currentTime || 0)} / ${clock(current.metrics.duration_seconds)}`;
  }
  if (current?.status === 'running') $('elapsed').textContent = seconds(Math.max(0, Date.now() / 1000 - current.started_at));
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
function render(record) {
  current = record;
  $('resultName').textContent = `${record.source_name} · ${codecName(record.codec)}`;
  $('resultStatus').textContent = record.stage;
  $('resultStatus').classList.toggle('failed', record.status === 'failed');
  const running = !['done', 'failed'].includes(record.status);
  $('taskProgress').hidden = !running;
  $('stage').textContent = record.stage;
  $('progress').value = record.progress;
  if (!running) $('elapsed').textContent = '';
  $('logBox').hidden = !record.logs?.length;
  $('logs').textContent = (record.logs || []).join('\n');
  $('resultContent').hidden = !record.metrics;
  if (record.error) notice(record.error);
  if (!record.metrics) { pause(); return; }
  const m = record.metrics;
  const semantic = record.codec === 'semantic-h265';
  $('packageLabel').textContent = semantic ? '完整传输包 · .scv' : '普通 H.265 封装视频 · MP4';
  $('streamLabel').textContent = semantic ? '各层裸 H.265 码流合计' : '普通 H.265 裸码流';
  $('semanticInfo').hidden = !semantic;
  $('overlayControl').hidden = !record.media.overlay_preview;
  if (semantic) {
    const target = record.parameters.target === 'fish' ? '鱼类' : '潜水员';
    $('semanticInfo').textContent = m.roi_segments
      ? `${target}预测命中 ${m.target_present_frames}/${m.frames} 帧 · 目标区域码流 ${m.roi_segments} 段 · 平均保护矩形覆盖 ${(m.average_roi_rectangle_fraction * 100).toFixed(1)}%`
      : `未检测到达到阈值的${target}区域：本次只传背景层和语义信息，未产生目标增强层。`;
  }
  $('streamSize').textContent = sizes(m.stream_bytes); $('streamRate').textContent = rate(m.stream_kbps); $('streamBytes').textContent = `${integer(m.stream_bytes)} B`;
  $('packageSize').textContent = sizes(m.package_bytes); $('packageRate').textContent = rate(m.package_kbps); $('packageBytes').textContent = `${integer(m.package_bytes)} B`;
  $('sourceSize').textContent = sizes(m.source_bytes); $('reconstructionSize').textContent = sizes(m.reconstruction_bytes);
  $('sourceGeometry').textContent = `${m.source_width} × ${m.source_height}`; $('outputGeometry').textContent = `${m.width} × ${m.height}`;
  $('videoInfo').textContent = `${m.duration_seconds.toFixed(2)} s · ${m.fps.toFixed(2)} fps · ${integer(m.frames)} 帧`;
  $('sourceCaption').textContent = semantic ? `${m.width} × ${m.height}` : `${m.source_width} × ${m.source_height}`; $('reconstructionCaption').textContent = `${m.width} × ${m.height}`;
  $('leftLabel').textContent = semantic ? '左 · 编码输入' : '左 · 原始视频';
  if (source.dataset.url !== record.media.source_preview) {
    pause(); $('play').disabled = true;
    $('showOverlay').checked = false;
    $('rightLabel').textContent = semantic ? '右 · 语义 H.265 重建' : '右 · 普通 H.265 重建';
    source.src = source.dataset.url = record.media.source_preview;
    reconstruction.src = record.media.reconstruction_preview;
    $('seek').max = m.duration_seconds; $('seek').value = 0;
    source.playbackRate = reconstruction.playbackRate = Number($('speed').value);
  }
  $('streamsTable').replaceChildren();
  const composition = semantic ? [
    ['背景层 H.265 · MP4', m.context_bytes], ['目标区域 H.265 · MP4', m.roi_stream_bytes],
    ['语义信息 · GZIP', m.semantic_bytes], ['清单与封装开销', m.manifest_and_zip_bytes], ['完整传输包 · SCV', m.package_bytes]
  ] : [['H.265 裸码流', m.stream_bytes], ['H.265 视频 · MP4', m.package_bytes], ['重建预览 · MP4', m.reconstruction_bytes]];
  for (const [label, bytes] of composition) {
    const row = [label, integer(bytes), rate(bytes * 8 / m.duration_seconds / 1000)];
    const tr = document.createElement('tr');
    for (const value of row) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
    $('streamsTable').append(tr);
  }
  $('timings').replaceChildren();
  for (const [label, key] of [[semantic ? '语义提取与编码' : '编码', 'compression_seconds'], ['解码预览', 'preview_seconds'], ['流程合计', 'pipeline_seconds']]) {
    const div = document.createElement('div'), span = document.createElement('span'), strong = document.createElement('strong');
    span.textContent = label; strong.textContent = seconds(record.timings?.[key]); div.append(span, strong); $('timings').append(div);
  }
  const p = record.parameters;
  $('encodingInfo').textContent = semantic
    ? `旧项目模型 · 阈值 ${record.teacher.threshold} · 背景 ${p.background_scale * 100}% / CRF ${p.background_crf} · ROI CRF ${p.roi_crf} · ${p.roi_layout === 'halves' ? '左右分区' : '合并区域'} · P 帧 / GOP 32`
    : `libx265 · ${p.preset} · GOP ${p.gop} · CRF ${p.crf} · ${p.bframes ? 'B4' : 'P 帧'}`;
  $('packetNote').textContent = semantic
    ? '完整 .scv 包含背景、目标区域、语义信息及封装。裸码流合计仅统计导出的 Annex B 视频数据；H.264 预览不参与传输。目标框为模型预测。'
    : '历史普通 H.265 记录。新提交任务使用语义分层编码。H.264 重建预览仅用于播放。';
  const labels = {packet: '完整传输包 .scv', raw_streams: '各层裸 H.265 码流', context: '背景层 H.265',
    stream: 'H.265 码流', package: 'H.265 视频', source: '原始视频', reconstruction: '重建预览', overlay: '预测目标框视频',
    semantics: '语义信息', manifest: '码流清单', decode_audit: '接收端读取记录', metrics: '数据 JSON', log: '日志'};
  $('downloads').replaceChildren();
  for (const key of Object.keys(labels)) {
    const file = record.downloads.find(item => item.key === key); if (!file) continue;
    const a = document.createElement('a'); a.href = file.url; a.textContent = labels[key]; a.download = file.name; a.title = `${file.name} · ${integer(file.bytes)} B`; $('downloads').append(a);
  }
}
async function selectRecord(id) {
  selected = id; notice(); const version = ++selectionVersion;
  try {
    const record = await api(`/api/jobs/${id}`);
    if (version === selectionVersion) { render(record); document.querySelectorAll('#history button').forEach(button => button.classList.toggle('active', button.dataset.id === id)); }
  } catch (error) { notice(error.message); }
}
async function refresh() {
  try {
    const listing = await api('/api/jobs');
    busy = listing.busy; sampleAvailable = listing.sample_available; buttons();
    $('serverState').textContent = busy ? '处理中' : '本机服务就绪';
    $('history').replaceChildren();
    for (const record of listing.jobs) {
      const button = document.createElement('button'); button.dataset.id = record.id; button.classList.toggle('active', selected === record.id);
      const badge = document.createElement('span'); badge.className = 'codec-badge'; badge.textContent = codecName(record.codec);
      const info = document.createElement('span'); info.className = 'record-info';
      const name = document.createElement('strong'); name.textContent = record.source_name;
      const date = document.createElement('small');
      date.textContent = `${record.created_at.replace('T', ' ')} · ${record.codec === 'semantic-h265' ? `${record.parameters.target === 'fish' ? '鱼类' : '潜水员'} · CRF ${record.parameters.background_crf}/${record.parameters.roi_crf}` : `CRF ${record.parameters.crf}`}`;
      info.append(name, date);
      const status = document.createElement('span'); status.className = 'record-status'; status.textContent = record.stage;
      button.append(badge, info, status); button.addEventListener('click', () => selectRecord(record.id)); $('history').append(button);
    }
    if (!selected && listing.jobs.length) await selectRecord(listing.jobs[0].id);
    else if (selected && current && !['done', 'failed'].includes(current.status)) {
      const id = selected, version = selectionVersion;
      const record = await api(`/api/jobs/${id}`);
      if (version === selectionVersion && id === selected) render(record);
    }
    if (!listing.jobs.length) $('resultName').textContent = '上传视频开始处理';
  } catch (error) { $('serverState').textContent = '连接已断开'; notice(error.message); }
}
async function run(sample = false) {
  if (busy || uploading || (!sample && !selectedFile)) return;
  uploading = true; buttons(); notice();
  if (!['background_crf', 'roi_crf'].every(id => $(id).reportValidity())) { uploading = false; buttons(); return; }
  const query = new URLSearchParams({...Object.fromEntries(optionIds.map(id => [id, $(id).value])), ...(sample ? {sample: '1'} : {name: selectedFile.name})});
  const xhr = new XMLHttpRequest();
  xhr.open('POST', `/api/jobs?${query}`); xhr.setRequestHeader('Content-Type', 'application/octet-stream');
  $('run').textContent = sample ? '准备示例…' : '上传中…';
  xhr.upload.onprogress = event => { if (event.lengthComputable && !sample) $('run').textContent = `上传 ${Math.round(event.loaded / event.total * 100)}%`; };
  xhr.onload = async () => {
    uploading = false; $('run').textContent = '上传并处理 →';
    try { const record = JSON.parse(xhr.responseText); if (xhr.status !== 202) throw new Error(record.error || '提交失败'); busy = true; await selectRecord(record.id); await refresh(); }
    catch (error) { notice(error.message); } buttons();
  };
  xhr.onerror = () => { uploading = false; $('run').textContent = '上传并处理 →'; buttons(); notice('上传失败，请检查本机服务'); };
  xhr.send(sample ? null : selectedFile);
}
$('run').addEventListener('click', () => run()); $('sampleRun').addEventListener('click', () => run(true));
async function poll() { await refresh(); setTimeout(poll, 1500); }
poll();
