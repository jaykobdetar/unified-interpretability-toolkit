'use strict';
const {chromium,expect}=require('playwright/test');
const fs=require('fs'),assert=require('node:assert/strict');
const base='http://127.0.0.1:8816',out='results/analytics-qualification';
const progress={checks:[]};
(async()=>{
 const browser=await chromium.launch({headless:true,chromiumSandbox:true,executablePath:process.env.ATLAS_CHROMIUM,args:['--renderer-process-limit=1','--disable-gpu']});
 const context=await browser.newContext({viewport:{width:1440,height:1000}});
 await context.route('**/*',r=>r.request().url().startsWith(base+'/')?r.continue():r.abort());
 const page=await context.newPage(),errors=[],requests=[],checks=progress.checks,reports=[];
 page.setDefaultTimeout(10000);
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 page.on('response',async r=>{if(r.url()===base+'/api/analytics/poll'&&r.status()===200){try{const x=await r.json();if(x.status==='complete'&&x.result)reports.push(x.result);}catch{}}});
 const host=page.locator('section[aria-label="Bounded weight analytics"]'),widget=host.locator('section.atlas-analytics');
 const view=()=>widget.getByLabel('View',{exact:true});
 const run=async()=>{const n=reports.length;await widget.getByRole('button',{name:'Compare original / shuffle',exact:true}).click();await expect.poll(()=>reports.length,{timeout:10000}).toBeGreaterThan(n);return reports.at(-1);};
 const select=async name=>{await page.locator('#tensor-search').fill(name);await page.getByRole('button',{name:'Select '+name,exact:true}).click();};
 const api=async(action,data)=>context.request.post(base+'/api/analytics/'+action,{data,headers:{'X-Atlas-Local':'1'}});
 let owned=null;
 try{
  await page.goto(base);await expect(host).toBeVisible();await expect(page.locator('#tensor-name')).toContainText('q_proj.weight',{timeout:15000});await expect(page.locator('#left-resolution')).toContainText('cells',{timeout:15000});
  let result=await run();assert.equal(result.coverage.visited_values,4096);assert.equal(result.coverage.full_tensor,false);assert.equal(result.heads.head_count,9);
  await widget.getByLabel('Sort strips').check();const expected=result.original.row_order[0];const canvas=widget.locator('.analytics-pair canvas').first();await canvas.scrollIntoViewIfNeeded();const rect=await canvas.boundingBox();await page.mouse.click(rect.x+1,rect.y+30);await expect(page.locator('#row')).toHaveValue(String(expected));
  checks.push('Actual native 64x64 paired strength window; sorted strip click preserves native row map and exact inspector');
  await view().selectOption('heads');await expect(widget.getByRole('heading',{name:'Q heads',exact:true})).toBeVisible();await widget.getByLabel('Jump to head group').selectOption('128');await widget.getByRole('button',{name:'Jump',exact:true}).click();await expect(page.locator('#row')).toHaveValue('128');
  await host.getByRole('button',{name:'Tall band',exact:true}).click();result=await run();assert.equal(result.region.rows,576);assert.equal(result.region.cols,16);assert.equal(result.original.folded.full_head_axis,true);await expect(widget).toContainText('full head axis');
  checks.push('Source-grounded 9 Q heads; jump to head 2 row128; full head-axis fold over 576x16, other columns explicitly excluded');
  await view().selectOption('outliers');await expect(widget.getByRole('heading',{name:'Original source coordinates',exact:true})).toBeVisible();assert(result.shuffled.top_values.every(x=>x.source_native_indices));checks.push('Original outlier coordinates and shuffled-position provenance remain distinct');
  await select('model.norm.weight');await view().selectOption('vector');result=await run();assert.equal(result.shape.length,1);assert.equal(result.original.vector.length,64);await expect(widget.getByRole('heading',{name:'Signed vector values',exact:true})).toBeVisible();checks.push('Actual one-dimensional model tensor: signed vector bars and paired control, partial64/576 coverage');
  await select('model.layers.0.self_attn.q_proj.weight');await view().selectOption('svd');let n=reports.length;await widget.getByRole('button',{name:'Compute bounded SVD comparison',exact:true}).click();await expect.poll(()=>reports.length,{timeout:10000}).toBeGreaterThan(n);result=reports.at(-1);assert.equal(result.svd.available,true);assert.equal(result.svd.scope,'selected native window only');await expect(widget).toContainText('separately fitted shuffled control');checks.push('Actual bounded64x64 NumPy SVD and leading rank-one residual, explicit selected-window scope');
  n=reports.length;await host.getByRole('button',{name:'Rank bounded model prefix',exact:true}).click();await expect.poll(()=>reports.length,{timeout:10000}).toBeGreaterThan(n);result=reports.at(-1);assert.equal(result.coverage.visited_values,65536);assert.equal(result.coverage.full_model,false);await expect(host.getByRole('heading',{name:'Partial-model outlier ranking',exact:true})).toBeVisible();checks.push('Bounded model prefix exactly65536/134515008 values; original/control rankings and tensor exclusions');
  // Actual ownership/cancellation, without any inference generation request.
  const metadata=await (await context.request.get(base+'/api/model')).json();const tensor=metadata.catalog.find(t=>t.name==='model.layers.0.self_attn.q_proj.weight');
  const start=await api('start',{tensor:tensor.id,region:{row:0,col:0,rows:128,cols:512},seed:99,svd:false});assert.equal(start.status(),202);owned=(await start.json()).job;
  const duplicate=await api('start',{scope:'model',seed:1});assert.equal(duplicate.status(),409);
  const stale=await api('cancel',{job:'not-the-owner'});assert.equal(stale.status(),409);
  const cancelled=await (await api('cancel',{job:owned})).json();assert.equal(cancelled.worker_alive,false);assert.equal(cancelled.status,'cancelled');owned=null;
  checks.push('Actual worker: duplicate admission409, stale cancel409, owned cancellation kills/reaps before slot release');
  const info=await (await context.request.get(base+'/api/analytics')).json();assert(!('job'in info)&&!('result'in info));const inference=await context.request.get(base+'/api/inference');assert.equal(inference.status(),503);checks.push('Public analytics metadata carries no capability/result; inference disabled in this no-inference qualification launcher');
  // Re-render real fold for representative desktop/mobile screenshots.
  await view().selectOption('heads');await host.getByRole('button',{name:'Tall band',exact:true}).click();await run();
  await host.evaluate(el=>el.scrollIntoView({block:'start'}));await page.screenshot({path:out+'/analytics-desktop.png'});
  await page.setViewportSize({width:390,height:844});await host.evaluate(el=>el.scrollIntoView({block:'start'}));assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:out+'/analytics-mobile-controls.png'});
  await widget.getByRole('heading',{name:'Signed mean across head groups (other axis preserved)',exact:true}).scrollIntoViewIfNeeded();await page.screenshot({path:out+'/analytics-mobile-fold.png'});
  checks.push('Desktop1440x1000 and mobile390x844 controls/fold: no horizontal overflow; paired charts stack');
  assert.deepEqual(errors,[]);assert(requests.every(u=>u.startsWith(base+'/')));assert(!requests.some(u=>u.includes('/api/inference/start')));
  const evidence={status:'PASS',browser:browser.version(),checks,page_errors:errors,no_inference_requests:true,external_page_requests:0,report_count:reports.length,scope:'Actual Smol source/worker; mobile emulation, not physical phone; Qwen numerical qualification separate',control:'Exact same-window multiset and separately fitted statistics; no causal/semantic significance claim'};
  fs.writeFileSync(out+'/browser-report.json',JSON.stringify(evidence,null,2)+'\n');console.log(JSON.stringify(evidence,null,2));
 }catch(error){await page.screenshot({path:out+'/analytics-browser-failure.png'}).catch(()=>{});throw error;}finally{if(owned)try{await api('cancel',{job:owned});}catch{}await browser.close();}
})().catch(e=>{fs.writeFileSync(out+'/browser-failure.json',JSON.stringify({status:'FAIL',...progress,error:e.stack},null,2));console.error(e);process.exitCode=1;});
