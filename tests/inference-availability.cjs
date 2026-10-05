'use strict';
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const source=fs.readFileSync('web/inference.js','utf8');
const {AtlasPlayback}=require('../web/inference.js');
const step=i=>({index:i,activation:Array(576).fill(i+.5),position:4+i,input_token_id:10+i,token_id:11+i,token_piece:' token',generated_text:' token'.repeat(i+1),compute_ms:20,compute_total_ms:20*(i+1),layer:0,phase:i?'decode':'prefill'});
const p=new AtlasPlayback();p.accept({status:'running',steps:[step(0)]});assert.equal(p.advance(),true);assert.equal(p.current.index,0);assert.equal(p.advance(),false);p.accept({status:'complete',steps:[step(0),step(1)]});assert.equal(p.advance(),true);p.rewind();assert.equal(p.current,null);assert.equal(p.replaying,true);assert.equal(p.paused,true);p.reset();assert.equal(p.steps.length,0);assert.throws(()=>p.accept({steps:Array.from({length:33},(_,i)=>step(i))}));assert.throws(()=>p.accept({steps:[{...step(0),activation:[NaN]}]}));
const {createFixture}=require('./support/inference-fixture.cjs');
const {Element,elems,get,timers,requests,context,tick,take,click,submit}=createFixture();
(async()=>{
 take('/api/inference').resolve({error:'renderer unavailable',code:'backend_unavailable'},503);await tick();
 assert.equal(get('inference-panel').hidden,true);assert(get('mode-status').textContent.includes('unavailable'));
 assert(get('infer-status').textContent.includes('unavailable'));get('infer-rate').listeners.change();get('infer-index').listeners.input();click('replay');assert(get('infer-status').textContent.includes('unavailable'));assert.equal(get('infer-start').disabled,true);assert.equal(requests.length,0);
 // Recreate a browser page after explicit manual recovery, then reject start.
 vm.runInNewContext(source,{console,document:context.document,window:context.window,fetch:context.fetch,setTimeout:context.setTimeout,clearTimeout:context.clearTimeout});take('/api/inference').resolve({model:'fixture',engine:'fixture'});await tick();assert(get('infer-status').textContent.startsWith('Ready'));
 assert.equal(get('inference-panel').hidden,false);
 const busyCheck=click('check');take('/api/inference').resolve({model:'fixture',engine:'fixture',busy:true,busy_owner:'analytics job',queue_capacity:0});await busyCheck;await tick();assert.equal(get('infer-start').disabled,true);assert(get('infer-status').textContent.includes('analytics job'));assert(get('infer-status').textContent.includes('No queue'));await submit();assert.equal(requests.length,0);
 const idleCheck=click('check');take('/api/inference').resolve({model:'fixture',engine:'fixture',busy:false});await idleCheck;await tick();assert.equal(get('infer-start').disabled,false);
 const start=submit();await submit();assert.equal(requests.length,1);take('/start').resolve({error:'backend unavailable',code:'backend_unavailable'},503);await start;
 assert(get('infer-status').textContent.includes('unavailable'));get('infer-rate').listeners.change();get('infer-index').listeners.input();assert(get('infer-status').textContent.includes('unavailable'));assert.equal(requests.length,0);assert.equal(timers.size,0);assert.equal(get('infer-start').disabled,false);
 const recovery=submit();take('/start').resolve({session:'manual-only',status:'running',steps:[],details:{}});await recovery;assert(get('infer-status').textContent.includes('Live generation'));assert.equal(requests.length,1);assert(requests[0].url.endsWith('/poll'));
 console.log(JSON.stringify({status:'PASS',checks:5,scope:'Unavailable inference is hidden; busy analytics ownership disables start without queuing; manual readiness refresh recovers; 503 start is never automatically retried'},null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
