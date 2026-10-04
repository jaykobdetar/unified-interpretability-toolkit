'use strict';
// Reuse only the deterministic DOM/transport setup. Run production readiness
// handlers with reversed response order; no server, worker or network traffic.
const fs=require('node:fs'),path=require('node:path'),{createRequire}=require('node:module');
const fixture=path.resolve('tests/inference-availability.cjs');
const setup=fs.readFileSync(fixture,'utf8').split('(async()=>{')[0];
const scenario=process.argv[2]||'busy';
if(!['busy','failure','finally','session'].includes(scenario))throw Error('Unknown scenario');
new Function('require','scenario',setup+`
(async()=>{
 const idle={model:'fixture',engine:'fixture',busy:false};
 const busy={...idle,busy:true,busy_owner:'analytics job',queue_capacity:0};
 const initial=take('/api/inference');
 if(scenario==='busy'){
  const newest=context.window.atlasRefreshAvailability();take('/api/inference').resolve(busy);await newest;
  initial.resolve(idle);await tick();
  assert.equal(get('infer-start').disabled,true,'Older idle response must not clear newer busy state');
  assert.match(get('infer-status').textContent,/analytics job.*No queue/);
  assert.equal(get('mode-status').textContent,'Local worker busy');
  await submit();assert.equal(requests.length,0,'Busy readiness cannot submit a job');
 }else if(scenario==='failure'){
  const newest=context.window.atlasRefreshAvailability();take('/api/inference').resolve(idle);await newest;
  initial.resolve({error:'old unavailable result'},503);await tick();
  assert.equal(get('inference-panel').hidden,false,'Older failure must not hide newer available experiments');
  assert.match(get('infer-status').textContent,/^Ready/);assert.equal(get('infer-start').disabled,false);
 }else if(scenario==='finally'){
  const newest=context.window.atlasRefreshAvailability(),latest=take('/api/inference');
  initial.resolve({error:'old unavailable result'},503);await tick();
  assert.equal(get('infer-check').disabled,true,'Older finally must not unlock the newer pending check');
  latest.resolve(idle);await newest;assert.equal(get('infer-check').disabled,false);
 }else{
  initial.resolve(idle);await tick();
  const oldCheck=context.window.atlasRefreshAvailability(),old=take('/api/inference');
  const starting=submit(),start=take('/start');
  old.resolve({error:'stale failure after a new session epoch'},503);await oldCheck;
  assert.equal(get('inference-panel').hidden,false);assert.equal(get('infer-check').disabled,true);
  start.resolve({session:'mock-owned',status:'running',steps:[],details:{}});await starting;
  assert.equal(get('infer-check').disabled,true);assert.match(get('infer-status').textContent,/compute running/);
  assert.equal(requests.length,1);assert(requests[0].url.endsWith('/poll'));
 }
 console.log(JSON.stringify({status:'PASS',scenario,scope:'Pure DOM/transport only; readiness and session generations'}));
})().catch(e=>{console.error(e);process.exitCode=1;});
`)(createRequire(fixture),scenario);
