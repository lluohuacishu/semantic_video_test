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
  const result = {frames:8, file_bytes:1234, service_seconds:1, verified:!preview, encoded_this_task:false,
    verification:{status:preview?'reference_verified_preview':'fully_verified', warning:preview?'参考状态校验通过；RGB 非逐位一致，仅供预览，画面误差未验收。':null}};
  let job = {id, filename:'test.glcv', kind:'stream', status:'done', phase:preview?'预览已完成：参考状态通过':'已完成：全部通过', result, downloads:[]};
  const context = vm.createContext({URLSearchParams, location:{search:'',port:'8774'},
    document:{getElementById(id){assert.ok(nodes[id], 'Missing element: '+id);return nodes[id];},createElement:element},
    sessionStorage:{getItem(){return id;},setItem(){},removeItem(){}}, history:{replaceState(){}},setInterval(){},
    fetch:async route=>({ok:true,json:async()=>route==='/api/config'?config:route==='/api/jobs'?[job]:job})});
  vm.runInContext(source.replace('init().catch(e=>notice(e.message));', 'globalThis.testUI={init,refresh};'), context);
  await context.testUI.init();
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
  assert.equal(nodes.verificationWarning.hidden, true);
  assert.equal(nodes.decodedBox.hidden, true);
}
(async()=>{for(const role of ['sender','receiver'])for(const preview of [false,true])await check(role,preview);
  console.log('4 UI states passed; failed jobs clear warning and hide video.');})().catch(e=>{console.error(e);process.exitCode=1;});
