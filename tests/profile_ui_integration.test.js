'use strict';
const assert=require('node:assert/strict');
const profiles=require('../web/profile-client.js'),hosts=require('../web/host-client.js');
const cases=[];const test=(name,fn)=>cases.push({name,fn});
const context={model_id:'m_'+'a'.repeat(64),context_id:'c'.repeat(32),tab_capability:'b'.repeat(64)};
const binding={version:2,source_identity:'e'.repeat(64),model_identity:'f'.repeat(64),tensor:1,name:'matrix',dtype:'BF16',shape:[2,3],rows:2,cols:3,slice:{leading_indices:[],display_axes:[0,1]}};
const admitted={version:1,model_id:context.model_id,context_id:context.context_id,job_id:'1'.repeat(32),job_capability:'2'.repeat(64),state:'running',accepted:null,cleanup_pending:false,resume_available:false};
const accepted={revision:'3'.repeat(64),visited_values:2,total_values:6,complete:false};
const flush=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
class Element{
 constructor(tag,doc){this.tag=tag;this.ownerDocument=doc;this.children=[];this.events={};this.textContent='';this.value='';this.disabled=false;this.hidden=false;this.attributes={};}
 append(...items){this.children.push(...items);}replaceChildren(...items){this.children=items;this.textContent='';}
 setAttribute(key,value){this.attributes[key]=value;}addEventListener(event,fn){this.events[event]=fn;}
}
function fixture(post){
 const doc={createElement(tag){return new Element(tag,doc);}},root=new Element('section',doc),timers=[];
 const ui=profiles.mount(root,{post,setTimer:fn=>{timers.push(fn);return timers.length;},clearTimer:()=>{}});
 ui.configure({profiles_enabled:true},context,binding);
 const all=()=>{const list=[];const visit=n=>{list.push(n);n.children.forEach(visit);};visit(root);return list;};
 return {ui,root,timers,find:text=>all().find(n=>n.textContent===text),all};
}
const textOf=node=>[node.textContent,...node.children.map(textOf)].join(' ');
test('selection and timers never start work; full-slice click is explicit and running stays indeterminate',async()=>{
 const calls=[];const f=fixture(async(p,b)=>{calls.push({p,b});return admitted;});
 assert.equal(calls.length,0);await f.timers.shift()();assert.equal(calls.length,0);
 f.find('Start full slice').events.click();await flush();
 assert.equal(calls.length,1);assert.equal(calls[0].b.values,6);assert.equal(calls[0].b.restart,false);
 assert.equal(f.find('Start full slice').disabled,true);assert.match(textOf(f.root),/counts are unavailable/);
 await assert.rejects(f.ui.client.start(binding,0,6,{restart:true}));assert.equal(calls.length,1);
 assert(!textOf(f.root).includes('Resume available'));
});
test('partial paired pages label observed coverage and unvisited means without silent continuation',async()=>{
 const calls=[];const partial={...admitted,state:'partial',accepted};
 const records=[{index:0,sum_abs:3,mean_abs:1.5,visited_count:2,expected_count:3,complete:false},
 {index:1,sum_abs:0,mean_abs:null,visited_count:0,expected_count:3,complete:false}];
 const f=fixture(async(p,b)=>{calls.push({p,b});return p.endsWith('/page')?{revision:accepted.revision,binding,axis:'rows',start:0,end:2,axis_length:2,visited_values:2,total_values:6,original:records,control:records}:partial;});
 f.find('Start full slice').events.click();await flush();assert.match(textOf(f.root),/2 \/ 6 values validated · partial/);
 const seed=f.all().find(n=>n.tag==='input'&&n.max==='4294967295');seed.value='17';
 await f.find('Read paired page').events.click();assert.match(textOf(f.root),/control seed 0/);assert.match(textOf(f.root),/Not visited/);assert.match(textOf(f.root),/0 \/ 3 · partial/);
 assert.match(textOf(f.root),/Original sum/);assert.match(textOf(f.root),/Control sum/);assert.match(textOf(f.root),/Means divide by visited counts/);
 await f.timers.shift()();assert.equal(calls.filter(c=>c.p.endsWith('/start')).length,1);
 assert.equal(calls.find(c=>c.p.endsWith('/page')).b.count,128);
});
test('reset cancels exact owner and acknowledges storage cleanup before another explicit start',async()=>{
 const calls=[];const f=fixture(async(p,b)=>{calls.push({p,b});return p.endsWith('/start')?{...admitted,state:'complete',accepted:{...accepted,visited_values:6,complete:true}}:{...admitted,state:'cancelled',accepted:null};});
 f.find('Start full slice').events.click();await flush();await f.ui.reset();
 assert.equal(f.ui.client.snapshot().cleanup_pending,false);assert.equal(f.ui.client.snapshot().accepted,null);
 assert.equal(calls.find(c=>c.p.endsWith('/cancel')).b.job_capability,admitted.job_capability);
 f.find('Start full slice').events.click();await flush();assert.equal(calls.filter(c=>c.p.endsWith('/start')).length,2);
});
test('old paired page never paints after tensor or slice selection changes',async()=>{
 const pending=deferred();const f=fixture(async p=>p.endsWith('/start')?{...admitted,state:'partial',accepted}:p.endsWith('/page')?pending.promise:admitted);
 f.find('Start full slice').events.click();await flush();const page=f.find('Read paired page').events.click();
 f.ui.configure({profiles_enabled:true},context,{...binding,tensor:2});
 pending.resolve({revision:accepted.revision,binding,axis:'rows',start:0,end:1});await page;
 assert(!textOf(f.root).includes('Original sum'));assert.equal(f.ui.client.snapshot().cleanup_pending,true);
});
test('native binding and owner tuple reach only the private profile bridge; mismatch disables start',async()=>{
 const calls=[];let configured;const panel={reset:async()=>{},client:{snapshot:()=>({cleanup_pending:false})},configure:(...args)=>{configured=args;},unavailable:message=>{configured=['unavailable',message];}};
 const model=context.model_id;let native=binding;
 const host=hosts.create({fetchImpl:async(url,options={})=>{
  calls.push({url,options});let body;
  if(url==='/api/models')body={api_version:1,profiles_enabled:true,resume_available:false,models:[{model_id:model,name:'fixture',fixture_eligible:true}]};
  else if(url==='/api/view-contexts')body={api_version:1,model_id:model,context_id:context.context_id,capability:context.tab_capability};
  else body={api_version:1,source_binding:native};
  return {ok:true,json:async()=>body};},schedule:()=>1,cancel:()=>{}});
 host.mountProfiles({}, {mount:()=>panel});await host.initialize(async()=>{},()=>{});await host.setProfileSelection(binding);
 assert.deepEqual(configured[1],context);assert.deepEqual(configured[2],binding);
 assert(calls.some(c=>c.url.includes('/binding?')));assert(calls.every(c=>!c.url.includes(context.tab_capability)));
 assert(!JSON.stringify(host.snapshot()).includes(context.tab_capability));assert(!calls.some(c=>c.url.includes('/profiles/start')));
 native={...binding,source_identity:'9'.repeat(64)};await host.setProfileSelection(binding);assert.equal(configured[0],'unavailable');
});
test('unconfirmed profile cleanup prevents model replacement before any acquire request',async()=>{
 let calls=0,initializing=true;const host=hosts.create({fetchImpl:async url=>{
  if(initializing&&url==='/api/models')return {ok:true,json:async()=>({api_version:1,models:[{model_id:context.model_id,name:'fixture',fixture_eligible:true}]})};
  if(initializing&&url==='/api/view-contexts')return {ok:true,json:async()=>({api_version:1,model_id:context.model_id,context_id:context.context_id,capability:context.tab_capability})};
  calls++;throw new Error('must not request');},schedule:()=>1,cancel:()=>{}});
 await host.initialize(async()=>{},()=>{});initializing=false;
 host.mountProfiles({}, {mount:()=>({reset:async()=>{},client:{snapshot:()=>({cleanup_pending:true})}})});
 await assert.rejects(host.select(context.model_id),/cleanup pending/);assert.equal(calls,0);
 assert.equal(host.snapshot().context_id,context.context_id);
});
(async()=>{for(const {name,fn} of cases){await fn();console.log('PASS '+name);}console.log(cases.length+' profile UI integration checks passed');})().catch(e=>{console.error(e);process.exitCode=1;});
