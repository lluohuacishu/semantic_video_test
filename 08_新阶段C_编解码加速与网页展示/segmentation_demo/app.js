const $ = id => document.getElementById(id);
const videos = [$('sourceVideo'), $('overlayVideo'), $('colorVideo')];
const state = { file: null, busy: false, uploading: false, selected: null, rendered: null, record: null, version: 0, view: 'overlay', sample: false };
const statusNames = { queued: '等待处理', running: '分割中', done: '已完成', failed: '失败' };
const downloadNames = { source: '原始文件', source_image: '原始画面 PNG', mask_image: '类别标签 PNG', color_image: '分类色块 PNG', overlay_image: '叠加效果 PNG', source_video: '原始画面预览', color_video: '分类色块视频', overlay_video: '叠加效果视频', labels: '逐帧标签 ZIP', report: '分割统计 JSON', log: '运行日志' };
const number = new Intl.NumberFormat('zh-CN');
function size(bytes) { if (bytes < 1024) return `${bytes} B`; const i = Math.min(3, Math.floor(Math.log(bytes) / Math.log(1024))); return `${(bytes / 1024 ** i).toFixed(i > 1 ? 2 : 1)} ${['B', 'KiB', 'MiB', 'GiB'][i]}`; }
function clock(value) { value = Math.max(0, Math.floor(Number.isFinite(value) ? value : 0)); return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`; }
function notice(message = '') { $('notice').textContent = message; $('notice').hidden = !message; }
function controls() {
  const locked = state.busy || state.uploading;
  $('run').disabled = locked || !state.file;
  $('sampleRun').disabled = locked || !state.sample;
  for (const id of ['file', 'mode', 'fps', 'duration', 'alpha']) $(id).disabled = locked;
  if (!locked && state.file && /\.(png|jpe?g|webp|bmp)$/i.test(state.file.name)) { $('fps').disabled = true; $('duration').disabled = true; }
  $('run').textContent = state.uploading ? '上传中…' : '上传并分割 →';
}
function choose(file) {
  if (!file || state.uploading || state.busy) return;
  if (!/\.(png|jpe?g|webp|bmp|mp4|mov|mkv|avi|webm|m4v)$/i.test(file.name)) return notice('请选择支持的图片或视频文件');
  if (file.size === 0 || file.size > 1024 ** 3) return notice('文件必须大于 0，且不超过 1 GB');
  state.file = file; $('fileName').textContent = file.name; $('fileInfo').textContent = size(file.size); notice(); controls();
}
$('file').addEventListener('change', event => choose(event.target.files[0]));
$('dropzone').addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('file').click(); } });
for (const name of ['dragenter', 'dragover']) $('dropzone').addEventListener(name, event => { event.preventDefault(); $('dropzone').classList.add('dragover'); });
for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, event => { event.preventDefault(); $('dropzone').classList.remove('dragover'); if (name === 'drop') choose(event.dataTransfer.files[0]); });
async function api(url, options) {
  const response = await fetch(url, options);
  const record = await response.json();
  if (!response.ok) throw new Error(record.error || `请求失败（${response.status}）`);
  return record;
}
function pause() { videos.forEach(video => video.pause()); $('play').textContent = '播放'; }
function resetPlayers() {
  pause();
  for (const video of videos) { video.removeAttribute('src'); video.load(); }
  for (const id of ['sourceImage', 'resultImage']) $(id).removeAttribute('src');
  $('seek').value = 0; $('clock').textContent = '00:00 / 00:00'; $('play').disabled = true;
}
function view(mode) {
  state.view = mode;
  $('overlayView').classList.toggle('active', mode === 'overlay'); $('colorView').classList.toggle('active', mode === 'color');
  $('overlayVideo').hidden = mode !== 'overlay'; $('colorVideo').hidden = mode !== 'color';
  $('videoCaption').textContent = mode === 'overlay' ? '右 · 分割叠加' : '右 · 分类色块';
  $('imageCaption').textContent = mode === 'overlay' ? '右 · 分割叠加（模型预测）' : '右 · 分类色块（模型预测）';
  if (state.record?.kind === 'image' && state.record.status === 'done') $('resultImage').src = state.record.media[mode + '_image'];
}
$('overlayView').addEventListener('click', () => view('overlay'));
$('colorView').addEventListener('click', () => view('color'));
function renderDownloads(record) {
  $('downloads').replaceChildren();
  for (const item of record.downloads || []) {
    const a = document.createElement('a'); a.href = item.url; a.download = item.name;
    const prefix = record.kind === 'video' && item.key.endsWith('_image') ? '首帧 · ' : '';
    a.textContent = `${prefix}${downloadNames[item.key] || item.name} · ${size(item.bytes)}`;
    a.title = `${item.name} · ${number.format(item.bytes)} 字节`;
    $('downloads').append(a);
  }
}
function renderResult(record) {
  state.rendered = record.id;
  const m = record.metrics, p = record.parameters, video = record.kind === 'video';
  $('geometry').textContent = `${m.width} × ${m.height}`;
  $('frameCount').textContent = video ? `${number.format(m.frames)} 帧` : '1 张图片';
  $('timeInfo').textContent = video ? `${m.duration_seconds.toFixed(2)} 秒 · ${m.fps.toFixed(2).replace(/\.00$/, '')} fps` : '单张完整图片';
  $('inferenceTime').textContent = `${m.inference_seconds.toFixed(2)} s`;
  $('speedInfo').textContent = `${m.inference_fps.toFixed(2)} 帧/秒 · 总耗时 ${m.pipeline_seconds.toFixed(1)} s`;
  $('maskSize').textContent = size(m.mask_bytes);
  $('maskInfo').textContent = video ? 'ZIP · 无损逐帧 PNG' : '无损单通道 PNG';
  $('videoPlayers').hidden = !video; $('imagePlayers').hidden = video; $('playback').hidden = !video;
  if (video) {
    for (const key of ['source', 'overlay', 'color']) { const element = $(key + 'Video'); element.src = record.media[key + '_video']; element.playbackRate = Number($('speed').value); }
  } else $('sourceImage').src = record.media.source_image;
  $('distributionInfo').textContent = video ? '全部采样帧平均' : '整张图片';
  $('legend').replaceChildren();
  for (const item of record.classes) {
    const row = document.createElement('div'); row.className = 'class-row';
    const swatch = document.createElement('i'); swatch.style.background = `rgb(${item.color.join(',')})`;
    const label = document.createElement('span'); label.textContent = `${item.id} · ${item.name}`;
    const track = document.createElement('div'); track.className = 'class-track';
    const bar = document.createElement('i'); bar.style.width = `${item.fraction * 100}%`; bar.style.background = swatch.style.background; track.append(bar);
    const value = document.createElement('b'); value.textContent = `${(item.fraction * 100).toFixed(2)}%`;
    row.title = `${item.symbol} · ${number.format(item.pixels)} 像素`; row.append(swatch, label, track, value); $('legend').append(row);
  }
  $('modelInfo').textContent = `${record.model} · ${p.mode === 'tiles512' ? '原尺寸分块 512 / 步长 384' : '整图长边 256'} · ${m.device}${video ? ` · 预览 ${m.preview_width} × ${m.preview_height}` : ''}`;
  renderDownloads(record); view(state.view);
}
function render(record) {
  state.record = record;
  $('resultName').textContent = record.source_name;
  $('resultStatus').textContent = statusNames[record.status] || record.status;
  $('resultStatus').classList.toggle('failed', record.status === 'failed');
  $('taskProgress').hidden = record.status === 'done' || record.status === 'failed';
  $('stage').textContent = record.stage; $('progress').value = record.progress;
  $('elapsed').textContent = record.started_at ? `${Math.max(0, Math.floor((record.finished_at || Date.now() / 1000) - record.started_at))} 秒` : '';
  $('resultContent').hidden = record.status !== 'done';
  $('logBox').hidden = record.status !== 'failed';
  $('logs').textContent = record.error || '';
  if (record.status === 'done' && state.rendered !== record.id) renderResult(record);
}
async function select(id) {
  state.selected = id; state.rendered = null; state.record = null; const version = ++state.version;
  resetPlayers(); $('resultContent').hidden = true; $('logBox').hidden = true;
  for (const button of $('history').querySelectorAll('button')) button.classList.toggle('active', button.dataset.id === id);
  try { const record = await api(`/api/jobs/${id}`); if (version === state.version) render(record); } catch (error) { notice(error.message); }
}
function history(data) {
  $('history').replaceChildren();
  if (!data.jobs.length) { const empty = document.createElement('p'); empty.className = 'empty-history'; empty.textContent = '处理完成后，可在这里再次查看'; $('history').append(empty); }
  for (const record of data.jobs) {
    const button = document.createElement('button'); button.dataset.id = record.id; button.classList.toggle('active', record.id === state.selected);
    const badge = document.createElement('span'); badge.className = 'codec-badge'; badge.textContent = record.kind === 'video' ? '视频' : '图片';
    const info = document.createElement('span'); info.className = 'record-info';
    const title = document.createElement('strong'); title.textContent = record.source_name;
    const date = document.createElement('small'); date.textContent = `${record.created_at.replace('T', ' ')} · ${record.parameters.mode === 'tiles512' ? '分块' : '快速整图'}`;
    info.append(title, date); const status = document.createElement('span'); status.className = 'record-status'; status.textContent = statusNames[record.status];
    button.append(badge, info, status); button.addEventListener('click', () => select(record.id)); $('history').append(button);
  }
}
async function refresh() {
  try {
    const data = await api('/api/jobs'); state.busy = data.busy; state.sample = data.sample_available;
    $('serverState').textContent = data.busy ? '正在处理' : '服务就绪'; controls(); history(data);
    if (!state.selected && data.jobs.length) await select(data.jobs[0].id);
    else if (state.selected) { const id = state.selected, version = state.version; const record = await api(`/api/jobs/${id}`); if (version === state.version && id === state.selected) render(record); }
  } catch (error) { $('serverState').textContent = '连接断开'; state.busy = true; controls(); }
}
async function submit(sample) {
  if (state.busy || state.uploading || (!sample && !state.file)) return;
  notice(); state.uploading = true; controls();
  const query = new URLSearchParams({ mode: $('mode').value, fps: $('fps').value, duration: $('duration').value, alpha: $('alpha').value });
  if (sample) query.set('sample', '1'); else query.set('name', state.file.name);
  try {
    const record = await api('/api/jobs?' + query, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: sample ? null : state.file });
    state.busy = true; await select(record.id); await refresh();
  } catch (error) { notice(error.message); }
  finally { state.uploading = false; controls(); }
}
$('run').addEventListener('click', () => submit(false)); $('sampleRun').addEventListener('click', () => submit(true));
function seek(value) { const maximum = videos[0].duration; if (!Number.isFinite(maximum)) return; videos.forEach(video => { video.currentTime = Math.max(0, Math.min(value, maximum)); }); }
videos.forEach(video => {
  video.addEventListener('loadedmetadata', () => { $('play').disabled = !videos.every(v => Number.isFinite(v.duration)); $('seek').max = videos[0].duration || 1; $('clock').textContent = `${clock(videos[0].currentTime)} / ${clock(videos[0].duration)}`; });
  video.addEventListener('error', () => { if (video.getAttribute('src')) notice('预览加载失败，请刷新页面或下载结果查看'); });
});
videos[0].addEventListener('timeupdate', () => {
  $('seek').value = videos[0].currentTime; $('clock').textContent = `${clock(videos[0].currentTime)} / ${clock(videos[0].duration)}`;
  if (!videos[0].paused) for (const video of videos.slice(1)) if (Math.abs(video.currentTime - videos[0].currentTime) > .15) video.currentTime = videos[0].currentTime;
});
videos[0].addEventListener('ended', pause);
$('play').addEventListener('click', async () => {
  if (!videos[0].paused) return pause();
  if (videos[0].ended || videos[0].currentTime >= videos[0].duration - .05) seek(0);
  try { await Promise.all(videos.map(video => video.play())); $('play').textContent = '暂停'; } catch (error) { pause(); notice('播放未开始，请再次点击播放'); }
});
$('seek').addEventListener('input', event => seek(Number(event.target.value)));
$('restart').addEventListener('click', () => { pause(); seek(0); });
$('speed').addEventListener('change', event => videos.forEach(video => { video.playbackRate = Number(event.target.value); }));
async function poll() { await refresh(); setTimeout(poll, 1500); }
poll();
