'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),codec=require('../web/inference.js');
const source=fs.readFileSync('web/inference.js','utf8'),asset=fs.readFileSync('web/inference-import.js','utf8');
class Element{constructor(){this.value='';this.textContent='';this.disabled=false;this.hidden=false;this.checked=false;this.listeners={};this.children=[];}replaceChildren(...a){this.children=a;}append(...a){this.children.push(...a);}addEventListener(k,f){this.listeners[k]=f;}remove(){}getContext(){return {clearRect(){},fillRect(){}};}}
const tick=async()=>{for(let i=0;i<16;i++)await Promise.resolve();};
const request={layer:7,activation_site:'mlp',max_new_tokens:1,prompt:'synthetic',edits:[],source_model:{}},record=codec.experimentRecord(request,{status:'complete',worker_alive:false,steps:[]}),text=JSON.stringify(record);
function fixture(){
 const elements=new Map(),scripts=[],requests=[],timers=new Map();let timerId=0;
 const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
 for(const [k,v] of Object.entries({rate:'2',limit:'1',layer:'7',site:'mlp',mode:'step',prompt:'synthetic'}))get('infer-'+k).value=v;
 const context=vm.createContext({console,TextEncoder,document:{getElementById:get,createElement:()=>new Element(),head:{append:s=>scripts.push(s)},body:{append(){}}},window:{addEventListener(){}},setTimeout:(fn,ms)=>{timers.set(++timerId,{fn,ms});return timerId;},clearTimeout:id=>timers.delete(id),fetch:(url,options)=>new Promise(resolve=>requests.push({url,options,resolve:body=>resolve({ok:true,json:async()=>body})}))});
 vm.runInContext(source,context);assert.equal(scripts.length,0);requests.shift().resolve({model:'fixture',engine:'CPU'});
 const click=id=>get('infer-'+id).listeners.click(),take=suffix=>{const i=requests.findIndex(r=>r.url.endsWith(suffix));assert(i>=0,suffix);return requests.splice(i,1)[0];};
 const load=s=>{assert.equal(s.src,'/inference-import.js');vm.runInContext(asset,context);s.onload();};
 get('infer-import-file').files=[{size:Buffer.byteLength(text),text:async()=>text}];
 return {get,click,scripts,requests,timers,context,load,take};
}
(async()=>{
 assert(asset.length<65536,'separate trusted codec has a finite 64 KiB source cap');
 const server=fs.readFileSync('src/server.rs','utf8');assert(server.includes('"/inference-import.js" => Some(('));assert(server.includes('include_bytes!("../web/inference-import.js")'));assert(server.includes("script-src 'self'"));
 const files=['vendor/openseadragon.min.js','atlas-tools.js','app.js','workspace-tools.js','inference.js'],bytes=files.reduce((n,f)=>n+fs.statSync('web/'+f).size,12);assert(bytes<600*1024);assert(!server.slice(server.indexOf('fn viewer_scripts()'),server.indexOf('fn reply_viewer_bundle')).includes('inference-import'));
 const f=fixture();await tick();const started=f.click('import-log');await tick();assert.equal(f.scripts.length,1);assert(f.get('infer-start').disabled);assert(!f.get('infer-cancel-import').disabled);assert(f.get('infer-log-status').textContent.includes('0 / 8'));assert.equal(f.requests.length,0);
 f.scripts[0].onerror();await started;assert(f.get('infer-import-status').textContent.includes('unavailable'));assert(f.get('infer-log-status').textContent.includes('0 / 8'));assert(!f.get('infer-start').disabled);assert.equal(f.requests.length,0);
 const badInit=f.click('import-log');await tick();f.scripts[1].onload();await badInit;assert(f.get('infer-import-status').textContent.includes('initialize'));assert(f.get('infer-log-status').textContent.includes('0 / 8'));
 const retry=f.click('import-log');await tick();f.load(f.scripts[2]);await retry;assert(f.get('infer-log-status').textContent.includes('1 / 8'));assert.equal(f.requests.length,0);assert.equal(f.get('infer-run-inputs').textContent,'No accepted run.');
 const c=fixture();await tick();const cancelled=c.click('import-log');await tick();c.click('cancel-import');assert(!c.get('infer-start').disabled);c.load(c.scripts[0]);await cancelled;assert(c.get('infer-log-status').textContent.includes('0 / 8'));assert(c.get('infer-import-status').textContent.includes('cancelled'));await c.click('import-log');assert(c.get('infer-log-status').textContent.includes('1 / 8'));assert.equal(c.scripts.length,1);assert.equal(c.requests.length,0);
 const slow=fixture();await tick();let resolveText;slow.get('infer-import-file').files=[{size:text.length,text:()=>new Promise(r=>resolveText=r)}];const reading=slow.click('import-log');await tick();slow.click('cancel-import');resolveText(text);await reading;assert.equal(slow.scripts.length,0);assert(slow.get('infer-log-status').textContent.includes('0 / 8'));
 const timeout=fixture();await tick();const timed=timeout.click('import-log');await tick();const timer=[...timeout.timers.values()].find(t=>t.ms===15000);assert(timer);timer.fn();await timed;assert(timeout.get('infer-import-status').textContent.includes('unavailable'));timeout.load(timeout.scripts[0]);assert(timeout.get('infer-log-status').textContent.includes('0 / 8'));
 // A reset of the existing terminal worker view invalidates a pending import.
 const reset=fixture();await tick();const submit=reset.get('infer-form').listeners.submit({preventDefault(){}});reset.take('/start').resolve({session:'synthetic-owner',status:'loading',worker_alive:true,steps:[],details:{}});await submit;reset.take('/poll').resolve({session:'synthetic-owner',status:'complete',worker_alive:false,steps:[],details:{}});await tick();const delayed=reset.click('import-log');await tick();const clear=reset.click('reset');reset.take('/reset').resolve({session:null,status:'idle',worker_alive:false,steps:[],details:{}});await clear;reset.load(reset.scripts[0]);await delayed;assert(reset.get('infer-log-status').textContent.includes('0 / 8'));assert(reset.get('infer-import-status').textContent.includes('cancelled'));assert.equal(reset.requests.length,0);
 console.log(JSON.stringify({status:'PASS',startup_bytes:bytes,startup_headroom:600*1024-bytes,import_asset_bytes:Buffer.byteLength(asset),scope:'Fixed same-origin/CSP-compatible route; no startup load; delayed/error/missing initializer/explicit retry/timeout/cancel/reset stale import refusal; no imported-data execution, worker adoption or run input changes. UI and transport doubles only.'},null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
