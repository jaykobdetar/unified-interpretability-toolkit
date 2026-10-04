'use strict';
// Reuse deterministic DOM/OSD transport doubles; no browser/network/model reads.
const fs=require('node:fs'),path=require('node:path'),{createRequire}=require('node:module');
const fixture=path.resolve('tests/ui-contract-races.cjs');
const setup=fs.readFileSync(fixture,'utf8').split('(async()=>{')[0];
new Function('require','__dirname',setup+`
vm.runInContext(fs.readFileSync(path.join(__dirname,'../web/atlas-tools.js'),'utf8'),context);
const high={id:21,name:'rank3',dtype:'BF16',element_bytes:2,shape:[2,3,4],rows:3,cols:4,count:24,max_level:2,min_level:0,available:true,calibration_complete:true,max_abs:12};
context.fixture={...model,source_identity:'a'.repeat(64),model_identity:'b'.repeat(64),catalog:[high],parameter_count:24};
get('left-rule').value='tensor_linear';get('right-rule').value='tensor_asinh';
function sliceView(){const t=copy(run('state.tensor'));return {api_version:1,tensor:t,source_binding:copy(run('AtlasTools.sourceBinding(state.model,state.tensor)')),legends:{left:{id:'tensor_linear',min:-12,max:12,zero:0,s:null,formula:'x/12',scope:'complete original tensor',units:'raw weight'},right:{id:'tensor_asinh',min:-12,max:12,zero:0,s:1,formula:'asinh',scope:'complete original tensor',units:'raw weight'}},tile_size:256,overlap:0,source_values_unchanged:true};}
async function openSlice(p){const r=take('/api/view'),before=viewers.length;r.resolve(sliceView());await tick();assert.equal(viewers.length-before,2);for(const v of viewers.slice(before))v.emit('open');await p;}
(async()=>{
 run('state.model=fixture;state.tensor=fixture.catalog[0]');await run('prepareTensor()');
 assert.equal(pending.length,0);assert.match(get('status').textContent,/Choose every leading index/);assert(get('inspect-submit').disabled);
 assert.equal(get('slice-picker').hidden,false);
 run('state.tensor=AtlasTools.withSlice(state.tensor,[1])');await openSlice(run('prepareTensor()'));
 assert.match(viewers.at(-1).source.getTileUrl(2,0,0),/slice=1/);
 run("$('overview-image').hidden=false;$('overview-viewport').style={};updateOverview()");assert.match(get('tensor-overview').attrs['aria-label'],/Selected 2D slice/);assert(get('tensor-overview').attrs['aria-label'].includes('[1]'));
 const pin=run('inspectAt(2,3)'),r=take('/api/inspect');assert.match(r.url,/slice=1/);
 const raw={api_version:1,source_binding:copy(run('AtlasTools.sourceBinding(state.model,state.tensor)')),tensor:21,row:2,col:3,dtype:'BF16',element_bytes:2,raw_hex_le:'803f',raw_exact:'1',native_indices:[1,2,3],shard:'fixture.safetensors',byte_offset:54,transformed:{left:1/12,right:.2}};
 r.resolve(raw);await pin;assert.equal(context.window.atlasInferenceSelection,null);assert.match(get('inspection').children[2].textContent,/1, 2, 3/);
 const stale=run('inspectAt(0,0)'),old=take('/api/inspect');
 run('state.tensor=AtlasTools.withSlice(fixture.catalog[0],[0])');await openSlice(run('prepareTensor()'));
 assert(old.options.signal.aborted);old.resolve({...raw,row:0,col:0,native_indices:[1,0,0]});await stale;
 assert.equal(run('state.inspectionData'),null);assert.equal(context.window.atlasInferenceSelection,null);
 assert.match(viewers.at(-1).source.getTileUrl(2,0,0),/slice=0/);
 run('state.tensor={...fixture.catalog[0],available:false,unavailable_reason:"Quantization scale semantics unavailable"}');await run('prepareTensor()');
 assert.equal(pending.length,0);assert.match(get('status').textContent,/Quantization/);assert(get('inspect-submit').disabled);
 console.log('PASS: explicit slice picker gate; slice-bound view/tile/inspect; full native indices; no higher-rank inference handoff; stale slice reply rejection; unavailable tensor leaves no fabricated view. Pure DOM doubles only.');
})().catch(e=>{console.error(e);process.exitCode=1;});
`)(createRequire(fixture),path.dirname(fixture));
