// Exercise the actual UI render functions with terminal job states, without CUDA.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, 'c5_console/index.html'), 'utf8');
const source = fs.readFileSync(path.join(__dirname, 'c5_console/app.js'), 'utf8');
const ids = [...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]);
function element() {
  return {hidden:false, textContent:'', children:[], attributes:{},
    append(...nodes){this.children.push(...nodes);}, replaceChildren(){this.children=[];},
    getAttribute(key){return this.attributes[key] ?? null;},
    hasAttribute(key){return key in this.attributes;}, removeAttribute(key){delete this.attributes[key];}, load(){}};
}
async function check(role, preview) {
  const nodes = Object.fromEntries(ids.map(id => [id, element()]));
  const id = 'a'.repeat(32);
  const config = {role, rate_mode:'max', compatibility_policy:'reference_preview', backend:'native', worker_mode:'cold'};
  const result = {role, durations:{encode_seconds:2,decode_seconds:3,codec_total_seconds:5,task_total_seconds:9,preprocess_seconds:1,transfer_seconds:.5,encoded_this_task:true},frames:8, file_bytes:1234, service_seconds:1, verified:!preview, encoded_this_task:false,
    verification:{status:preview?'reference_verified_preview':'fully_verified', warning:preview?'参考状态校验通过；RGB 非逐位一致，仅供预览，画面误差未验收。':null}};
  let job = {id, filename:'test.glcv', kind:'stream', status:'done', phase:preview?'预览已完成：参考状态通过':'已完成：全部通过', result, downloads:[]};
  const context = vm.createContext({URLSearchParams, location:{search:'',port:'8774'},
    document:{getElementById(id){assert.ok(nodes[id], 'Missing element: '+id);return nodes[id];},createElement:element},
    sessionStorage:{getItem(){return id;},setItem(){},removeItem(){}}, history:{replaceState(){}},setInterval(){},
    fetch:async route=>({ok:true,json:async()=>route==='/api/config'?config:route==='/api/jobs'?[job]:job})});
  vm.runInContext(source.replace('init().catch(e=>notice(e.message));', 'globalThis.testUI={init,refresh,restoreOptions,encodeOptions,compression};'), context);
  await context.testUI.init();
  assert.equal(nodes.timings.hidden, false);
  const durationText=nodes.timingCards.children.map(card=>card.children.map(n=>n.textContent).join(" ")).join(" | ");
  assert.match(durationText, /解码耗时 3 秒/);
  if(role==='sender')assert.match(durationText, /编解码总耗时 5 秒/);
  else assert.doesNotMatch(durationText, /编码耗时/);
  assert.equal(nodes.verificationWarning.hidden, !preview);
  assert.equal(nodes.decodedBox.hidden, role!=='receiver');
  if(preview){
    assert.match(nodes.verificationWarning.textContent, /未验收/);
    assert.match(nodes.metrics.textContent, /RGB 非逐位一致/);
    assert.doesNotMatch(nodes.metrics.textContent, /码流、RGB、latent 校验通过/);
    assert.match(nodes.history.children[0].children[1].textContent, /预览/);
  } else assert.match(nodes.metrics.textContent, /码流、RGB、latent 校验通过/);
  job = {...job, status:'error', phase:'failed', result:undefined};
  await context.testUI.refresh();
  assert.equal(nodes.timings.hidden, true);
  assert.equal(nodes.timingCards.children.length, 0);
  assert.equal(nodes.verificationWarning.hidden, true);
  assert.equal(nodes.decodedBox.hidden, true);
  context.testUI.restoreOptions({kind:'video',options:{preset:'compact',enhancement:'semantic_original',
    clip:{mode:'segment',start:'353353/30000',end:'417417/30000'},
    video:{fps_mode:'source',size_mode:'limit',long_edge:512}}});
  const exact=context.testUI.encodeOptions();
  assert.equal(exact.clip.start,'353353/30000');assert.equal(exact.clip.end,'417417/30000');
  assert.equal(exact.enhancement,'semantic_original');assert.equal(exact.video.long_edge,512);
  nodes.end.value='13';assert.equal(context.testUI.encodeOptions().clip.end,'13');
  if(role==='sender' && !preview){
    const metrics={width:640,height:360,fps:12.5,frames:125,duration_seconds:10,
      file_bytes:25000,video_kbps:20,raw_rgb24_bytes:86400000,raw_rgb24_ratio:3456,
      source_file_bytes:1000000,source_file_ratio:40,source_file_reduction_percent:97.5,
      whole_source_processed:true,selected_start_seconds:0,selected_end_seconds:10,source_duration_seconds:10};
    function cards(overrides={}){
      context.testUI.compression({compression:{...metrics,...overrides}});
      return Object.fromEntries(nodes.compressionCards.children.map(card=>[
        card.children[0].textContent,card.children[1].textContent]));
    }
    const full=cards();
    assert.equal(full['文件压缩比'],'40:1');
    assert.equal(full['帧数据压缩比（RGB24）'],'3,456:1');
    const segment=cards({whole_source_processed:false,source_file_ratio:null,
      source_file_reduction_percent:null,source_duration_seconds:30});
    assert.equal(segment['文件压缩比'],'不适用（片段）');
    assert.equal(segment['帧数据压缩比（RGB24）'],'3,456:1');
    assert.match(nodes.compressionNote.textContent,/本次范围 0–10 秒，整片 30 秒/);
    const stream=cards({source_file_bytes:null,source_file_ratio:null,whole_source_processed:null});
    assert.equal(stream['文件压缩比'],'无原文件数据');
    assert.equal(stream['帧数据压缩比（RGB24）'],'3,456:1');
    const legacy=cards({source_file_ratio:null,whole_source_processed:null,raw_rgb24_ratio:undefined,raw_rgb24_bytes:undefined});
    assert.equal(legacy['文件压缩比'],'缺少范围记录');
    assert.ok(!('帧数据压缩比（RGB24）' in legacy));
    context.testUI.compression(undefined);
    assert.equal(nodes.compression.hidden,true);
    assert.equal(nodes.compressionCards.children.length,0);
  }
}
(async()=>{for(const role of ['sender','receiver'])for(const preview of [false,true])await check(role,preview);
  console.log('4 UI states and full/segment/stream/legacy/empty compression displays passed.');})().catch(e=>{console.error(e);process.exitCode=1;});
