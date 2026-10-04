'use strict';
// Prepared opt-in qualification: run only in an assigned shared heavy slot.
const {chromium,expect}=require('playwright/test'),fs=require('fs'),assert=require('assert/strict');
const base=process.env.ATLAS_TEST_URL||'http://127.0.0.1:8816',out=process.env.ATLAS_EVIDENCE_DIR||'results/capture-logs-browser';
assert(/^http:\/\/127\.0\.0\.1:\d+$/.test(base));fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,chromiumSandbox:true,executablePath:process.env.ATLAS_CHROMIUM,args:['--renderer-process-limit=1','--disable-gpu']});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1100}});await context.route('**/*',route=>route.request().url().startsWith(base+'/')?route.continue():route.abort());
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));const el=id=>page.locator('#infer-'+id);
  await page.goto(base);await page.locator('#inference-panel > summary').click();await expect(el('start')).toBeEnabled();await expect(el('logging')).not.toBeChecked();await expect(el('include-prompt')).not.toBeChecked();await el('mode').selectOption('step');await el('limit').fill('2');await el('logging').check();
  async function run(layer,site){
   await el('layer').selectOption(String(layer));await el('site').selectOption(site);const response=page.waitForResponse(r=>r.url()===base+'/api/inference/start'&&r.request().method()==='POST');await el('start').click();const owner=await (await response).json();await expect(el('status')).toContainText('compute complete',{timeout:30000});await el('step').click();await expect(el('alignment')).toContainText(site==='attention'?'after o_proj':'after down_proj');
   return owner.session;
  }
  async function download(name){const event=page.waitForEvent('download');await el('export-log').click();const file=await event;const path=out+'/'+name;await file.saveAs(path);return JSON.parse(fs.readFileSync(path,'utf8'));}
  const firstOwner=await run(7,'attention');await expect(el('log-status')).toContainText('1 / 8');let log=await download('synthetic-omitted-prompt.json');assert.equal(log.records.length,1);assert.equal(log.records[0].request.activation_site,'attention');assert.equal(log.records[0].request.layer,7);assert(!Object.hasOwn(log.records[0].request,'prompt'));assert(!Object.hasOwn(log.records[0].steps[0],'input_token_id'));assert(log.records[0].runtime.torch);assert.equal(log.records[0].worker_cleanup_confirmed,true);assert(!JSON.stringify(log).includes(firstOwner));
  await el('include-prompt').check();const secondOwner=await run(29,'mlp');await expect(el('log-status')).toContainText('2 / 8');log=await download('synthetic-private-prompt.json');assert.equal(log.records[1].request.prompt,'The capital of France is');assert(Array.isArray(log.records[1].request.prompt_ids));assert.equal(log.records[1].request.activation_site,'mlp');assert(log.records[1].runtime.transformers);assert(!JSON.stringify(log).includes(secondOwner));
  await el('include-prompt').uncheck();log=await download('synthetic-redacted-history.json');assert(log.records.every(record=>!Object.hasOwn(record.request,'prompt')&&!Object.hasOwn(record.request,'prompt_ids')&&!record.privacy.request_replayable));
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.locator('.experiment-log').scrollIntoViewIfNeeded();await page.screenshot({path:out+'/private-log-mobile.jpg',type:'jpeg',quality:85});
  await page.reload();await expect(el('log-status')).toContainText('0 / 8');await expect(el('logging')).not.toBeChecked();assert.deepEqual(errors,[]);
  const result={status:'PASS',fixture:'capital-france-public-synthetic',checks:['Layer7 attention and layer29 MLP observations','Version/source/edit/token/score provenance in actual downloads','Separate opt-in prompt inclusion and history redaction','No capability in files','Reload loses unsaved memory and disables logging','390px layout'],page_errors:errors};fs.writeFileSync(out+'/capture-logs-browser.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
