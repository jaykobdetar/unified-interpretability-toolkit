'use strict';
const architecture=require('./fixtures/inference-architecture.json');
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const {observationSelection,experimentRecord,redactExperiment}=require('../web/inference.js');
assert.deepEqual(observationSelection('attention','8','attention',architecture),{kind:'attention',head:8});
assert.deepEqual(observationSelection('logit_lens','0','block',architecture),{kind:'logit_lens'});
assert.equal(observationSelection('none','0','mlp',architecture),null);
for(const head of ['',-1,9,.5,Infinity,true,false,null])assert.throws(()=>observationSelection('attention',head,'attention',architecture));
assert.throws(()=>observationSelection('attention',0,'block',architecture));assert.throws(()=>observationSelection('logit_lens',0,'mlp',architecture));
const activation=Array(576).fill(.25),base={index:0,activation,layer:7,position:2,input_token_id:3,token_id:4,token_piece:'x',generated_text:'x',phase:'prefill',compute_ms:1,compute_total_ms:1};
const attention={layer:7,query_head:8,kv_head:2,head_dim:64,query_position:2,key_positions:[0,1,2],key_token_ids:[1,2,3],probabilities:[.25,.25,.5],semantics:'fixture'};
const lens={layer:29,position:2,score_kind:'raw FP32 logits',lens_argmax_id:3,final_argmax_id:4,semantics:'fixture',candidates:[{id:3,piece:'a',lens_logit:2,final_logit:1,delta_lens_minus_final:1},{id:4,piece:'b',lens_logit:1,final_logit:2,delta_lens_minus_final:-1}]};
const request={prompt:'private source',layer:7,activation_site:'attention',observation:{kind:'attention',head:8}},snapshot={status:'complete',worker_alive:false,details:{prompt_ids:[1,2,3]},steps:[{...base,attention,activation_site:'attention'}]};
const omitted=experimentRecord(request,snapshot),included=experimentRecord(request,snapshot,{includePrompt:true});
assert(!Object.hasOwn(omitted.steps[0].attention,'key_token_ids'));assert.deepEqual(included.steps[0].attention.key_token_ids,[1,2,3]);assert.deepEqual(omitted.request.observation,request.observation);
assert(!Object.hasOwn(redactExperiment(included).steps[0].attention,'key_token_ids'));
assert.deepEqual(experimentRecord({...request,observation:{kind:'logit_lens'}},{...snapshot,steps:[{...base,logit_lens:lens}]}).steps[0].logit_lens.candidates,lens.candidates);
class Element{constructor(){this.value='';this.textContent='';this.hidden=false;this.disabled=false;this.checked=false;this.children=[];this.listeners={};this.width=512;this.height=288;}replaceChildren(...v){this.children=v;}append(...v){this.children.push(...v);}addEventListener(k,f){this.listeners[k]=f;}getContext(){return {clearRect(){},fillRect(){}};}}
function setup(){
 const elements=new Map(),requests=[],get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
 for(const [key,value] of Object.entries({rate:'2',limit:'2',layer:'7',site:'block',mode:'step',prompt:'public fixture',observation:'none','head-index':'8'}))get('infer-'+key).value=value;
 const context=vm.createContext({console,document:{getElementById:get,createElement:()=>new Element()},window:{addEventListener(){}},fetch:(url,options)=>new Promise(resolve=>requests.push({url,options,resolve:body=>resolve({ok:true,status:200,json:async()=>body})})),setTimeout:()=>1,clearTimeout(){}});
 vm.runInContext(fs.readFileSync('web/inference.js','utf8'),context);
 const take=part=>{const i=requests.findIndex(r=>r.url.endsWith(part));assert(i>=0,part);return requests.splice(i,1)[0];};
 return {get,take,requests,click:id=>get('infer-'+id).listeners.click(),change:(id,value)=>{get('infer-'+id).value=value;get('infer-'+id).listeners.change();},submit:()=>get('infer-form').listeners.submit({preventDefault(){}})};
}
const tick=async()=>{for(let i=0;i<15;i++)await Promise.resolve();};
(async()=>{
 const old=setup();old.take('/api/inference').resolve({architecture,model:'fixture',engine:'fixture'});await tick();assert(old.get('infer-observation').disabled);
 const ui=setup();ui.take('/api/inference').resolve({architecture,model:'fixture',engine:'fixture',observations:{kinds:['attention','logit_lens']}});await tick();assert(!ui.get('infer-observation').disabled);
 ui.change('observation','attention');assert.equal(ui.get('infer-site').value,'attention');assert(ui.get('infer-site').disabled);assert(!ui.get('infer-head-index').disabled);
 const start=ui.submit(),req=ui.take('/start');assert.deepEqual(JSON.parse(req.options.body).observation,{kind:'attention',head:8});req.resolve({session:'owner',status:'running',steps:[],details:{}});await start;assert(ui.get('infer-observation').disabled);assert(ui.get('infer-head-index').disabled);
 ui.take('/poll').resolve({session:'owner',status:'complete',steps:[{...base,attention,activation_site:'attention'}],details:{}});await tick();assert(ui.get('infer-observation-result').hidden);ui.click('step');assert(!ui.get('infer-observation-result').hidden);assert.equal(ui.get('infer-observation-values').children.length,3);assert(ui.get('infer-observation-context').textContent.includes('KV head 2'));assert(ui.get('infer-observation-context').textContent.includes('not causal attribution'));assert.equal(ui.get('infer-observation-values').children[2].children[2].children[0].value,.5);
 ui.click('replay');assert(ui.get('infer-observation-result').hidden);
 ui.change('observation','logit_lens');assert.equal(ui.get('infer-site').value,'block');assert(ui.get('infer-head-index').disabled);ui.get('infer-layer').value='29';
 const second=ui.submit();const secondReq=ui.take('/start');assert.deepEqual(JSON.parse(secondReq.options.body).observation,{kind:'logit_lens'});secondReq.resolve({session:'owner2',status:'running',steps:[],details:{}});await second;
 const stale=ui.take('/poll');const reset=ui.click('reset');ui.take('/reset').resolve({session:null,status:'idle',steps:[],details:{}});await reset;stale.resolve({session:'owner2',status:'complete',steps:[{...base,layer:29,logit_lens:lens,activation_site:'block'}],details:{}});await tick();assert(ui.get('infer-observation-result').hidden);
 const third=ui.submit();ui.take('/start').resolve({session:'owner3',status:'running',steps:[],details:{}});await third;ui.take('/poll').resolve({session:'owner3',status:'complete',steps:[{...base,layer:29,logit_lens:lens,activation_site:'block'}],details:{}});await tick();ui.click('step');assert.equal(ui.get('infer-observation-values').children.length,2);assert(ui.get('infer-observation-context').textContent.includes('not an early-exit prediction'));assert.equal(ui.get('infer-observation-values').children[0].children[3].textContent,'1.000000');
 console.log(JSON.stringify({status:'PASS',scope:'Strict mode/head selection, matching capture sites, immutable accepted request, backend capability gating, attention distribution/meter and lens union display, playback/rewind/reset/stale response, prompt-key ID export omission and later redaction'},null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
