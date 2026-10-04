'use strict';
// Prepared benign actual-model/UI qualification. Run only in a granted heavy slot.
const {chromium,expect}=require('playwright/test'),fs=require('fs'),assert=require('assert/strict');
const base=process.env.ATLAS_TEST_URL||'http://127.0.0.1:8816',out=process.env.ATLAS_EVIDENCE_DIR||'results/observations-browser';
assert(/^http:\/\/127\.0\.0\.1:\d+$/.test(base));assert(process.env.ATLAS_CHROMIUM);fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,chromiumSandbox:true,executablePath:process.env.ATLAS_CHROMIUM,args:['--renderer-process-limit=1','--disable-gpu']});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});await context.route('**/*',r=>r.request().url().startsWith(base+'/')?r.continue():r.abort());
  const page=await context.newPage(),errors=[],results=[];page.on('pageerror',e=>errors.push(e.message));const el=id=>page.locator('#infer-'+id);
  await page.goto(base);await page.locator('#inference-panel > summary').click();await expect(el('start')).toBeEnabled();await expect(el('observation')).toBeEnabled();await el('mode').selectOption('step');await el('limit').fill('2');await el('prompt').fill('The capital of France is');
  for(const [kind,layer] of [['attention','7'],['logit_lens','29']]){
   await el('layer').selectOption(layer);await el('observation').selectOption(kind);await expect(el('site')).toBeDisabled();if(kind==='attention')await el('head-index').fill('8');
   await el('start').click();await expect(el('status')).toContainText('compute complete',{timeout:30000});await expect(el('observation-result')).toBeHidden();await el('step').click();await expect(el('observation-result')).toBeVisible();
   await expect(el('observation-context')).toContainText(kind==='attention'?'query head 8 (KV head 2':'not an early-exit prediction');
   assert(await el('observation-values').locator('tr').count()>0);if(kind==='attention')assert(await el('observation-values').locator('meter').count()>0);
   const download=page.waitForEvent('download');await el('export-run').click();const file=await download,path=out+'/'+kind+'-synthetic.json';await file.saveAs(path);const record=JSON.parse(fs.readFileSync(path,'utf8'));
   assert.equal(record.request.observation.kind,kind);assert.equal(record.worker_cleanup_confirmed,true);assert.equal(record.runtime.attention_backend,'eager');assert.equal(record.privacy.prompt_included,false);
   for(const step of record.steps){assert.equal(step.alignment,'matched_prefix');assert(step.candidates.every(c=>c.delta===0));}
   if(kind==='attention'){assert(!Object.hasOwn(record.steps[0].attention,'key_token_ids'));assert(Math.abs(record.steps[0].attention.probabilities.reduce((a,b)=>a+b,0)-1)<1e-5);}
   else assert(record.steps.every(step=>Math.max(...step.logit_lens.candidates.map(c=>Math.abs(c.delta_lens_minus_final)))<2e-4));
   results.push({kind,layer:Number(layer),records:record.steps.length,observed_branch:record.steps[0].activation_branch});
   await el('observation-result').scrollIntoViewIfNeeded();await page.screenshot({path:out+'/'+kind+'-desktop.jpg',type:'jpeg',quality:82,fullPage:false});await el('replay').click();await expect(el('observation-result')).toBeHidden();await el('step').click();
  }
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await el('observation-result').scrollIntoViewIfNeeded();await page.screenshot({path:out+'/lens-mobile.jpg',type:'jpeg',quality:82,fullPage:false});assert.deepEqual(errors,[]);
  const evidence={status:'PASS',fixture:'capital-france-public-synthetic',results,page_errors:errors,scope:'Selected layer7/queryhead8 genuine probabilities and final-layer lens, empty-edit exact displayed logit parity, actual private redacted downloads, step/rewind, desktop/mobile emulation'};fs.writeFileSync(out+'/observations-browser.json',JSON.stringify(evidence,null,2)+'\n');console.log(JSON.stringify(evidence,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
