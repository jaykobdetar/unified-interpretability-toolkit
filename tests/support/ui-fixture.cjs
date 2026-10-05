'use strict';
// Fresh deterministic DOM/OSD/transport state for each test invocation.
function createFixture({appSource=require('node:path').join(__dirname,'../../web/app.js')}={}){
// Real frontend state machine + deterministic DOM/OSD/transport doubles.
// Aborted responses are deliberately delivered; this is not browser rendering QA.
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict'),crypto=require('crypto'),path=require('path');
const source=fs.readFileSync(appSource,'utf8');
class Element{
 constructor(tag='',id=''){this.tag=tag;this.id=id;this.value='';this.textContent='';this.hidden=false;this.disabled=false;this.children=[];this.dataset={};this.attrs={};this.listeners={};this.classes=new Set();this.classList={add:x=>this.classes.add(x),remove:x=>this.classes.delete(x),toggle:(x,v)=>v?this.classes.add(x):this.classes.delete(x),contains:x=>this.classes.has(x)};}
 get options(){return this.children;}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this.children=children;this.textContent='';if(this.tag==='select')this.value=children[0]?.value||'';}
 setAttribute(k,v){this.attrs[k]=v;}
 addEventListener(k,v){this.listeners[k]=v;}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element(['left-rule','right-rule','layer-filter'].includes(id)?'select':'div',id));return elements.get(id);};
const pending=[],viewers=[],frames=[],timers=[];
function OSD(){
 const v={handlers:{},destroyed:false,overlays:[],nav:false,bounds:{x:0,y:0,width:1,height:1},zoom:1,fitCalls:0,
  addHandler(k,f){(this.handlers[k]??=[]).push(f);},emit(k,e={}){for(const f of this.handlers[k]||[])f(e);},open(s){this.source=s;},destroy(){this.destroyed=true;},setMouseNavEnabled(v){this.nav=v;},clearOverlays(){this.overlays=[];},addOverlay(v){this.overlays.push(v);}};
 v.item={lastDrawn:[],getFullyLoaded:()=>true,viewportToImageCoordinates:(x,y)=>typeof x==='number'?{x,y}:x,imageToViewportRectangle:(x,y,width,height)=>({x,y,width,height}),imageToViewportCoordinates:(x,y)=>({x,y})};
 v.world={getItemCount:()=>v.destroyed?0:1,getItemAt:()=>v.item};
 v.viewport={getBounds:()=>({...v.bounds}),fitBounds:b=>{v.fitCalls++;assert(v.fitCalls<100,'viewport sync feedback loop');v.bounds={...b};v.emit('viewport-change');},goHome:()=>{v.bounds={x:0,y:0,width:1,height:1};v.emit('viewport-change');},pointFromPixel:p=>p,getZoom:()=>v.zoom,viewportToImageZoom:z=>z,imageToViewportZoom:z=>z,zoomBy:f=>{v.zoom*=f;v.bounds.width/=f;v.bounds.height/=f;v.emit('viewport-change');},zoomTo:z=>{v.zoom=z;},panTo:p=>{v.bounds.x=p.x;v.bounds.y=p.y;v.emit('viewport-change');},applyConstraints:()=>{}};
 viewers.push(v);return v;
}
const context=vm.createContext({console,AbortController,DOMException,URLSearchParams,performance,setTimeout:f=>timers.push(f),clearTimeout:i=>{timers[i-1]=null;},OpenSeadragon:OSD,requestAnimationFrame:f=>frames.push(f),window:{innerWidth:1200},document:{body:{dataset:{}},title:'',getElementById:get,createElement:t=>new Element(t),createDocumentFragment:()=>new Element('fragment')},fetch:(url,options)=>new Promise((resolve,reject)=>pending.push({url,options,resolve:(body,status=200)=>resolve({ok:status>=200&&status<300,status,json:async()=>body}),reject}))});
vm.runInContext(source.replace(/initialize\s*\(\s*\)\s*;\s*$/,''),context);
const run=s=>vm.runInContext(s,context),copy=x=>JSON.parse(JSON.stringify(x)),tick=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
const take=part=>{const i=pending.findIndex(p=>p.url.includes(part));assert(i>=0,'missing '+part);return pending.splice(i,1)[0];};
const tensor=(id,name,shape,max_abs)=>({id,name,shape,calibration_complete:true,rows:shape.length===1?1:shape[0],cols:shape.at(-1),count:shape.reduce((a,b)=>a*b,1),max_abs,max_level:Math.ceil(Math.log2(Math.max(...shape))),min_level:0});
const catalog=[tensor(11,'model.layers.0.self_attn.q_proj.weight',[4,4],.5),tensor(12,'model.layers.1.self_attn.q_proj.weight',[2,2],1),tensor(13,'model.layers.0.self_attn.k_norm.weight',[128],34)];
const model={api_version:1,source_identity:'a'.repeat(64),model_identity:'b'.repeat(64),name:'Qwen3-8B test double',revision:'fixture',representation:'BF16',source_directory:'/fixture',source_bytes:296,parameter_count:148,global_max:34,calibration_complete:true,catalog,rules:['global_linear','global_asinh','tensor_linear','tensor_asinh','tensor_magnitude','tensor_robust99','tensor_signed_percentile'].map(id=>({id,title:id,formula:'fixture formula'})),identity_validation:'Fixture identity',coverage:{active_tensor:null,all_requested:false,source_complete:true,sha_verified_shards:3,sha_hashed_shards:5,sha_expected_matched_shards:3,sha_missing_expected_shards:2,statistics_complete:true,values_streamed:148,materialized_tiles:2,materialized_bytes:100,all_pixels_materialized:false,rendering_policy:'Fixture only'},render_semantics:'Fixture only'};
function statusUpdate(t){return {api_version:1,model_status_version:1,source_identity:model.source_identity,model_identity:model.model_identity,global_max:34,calibration_complete:true,coverage:{calibrated_tensors:3,values_streamed:148,all_requested:false,active_tensor:null},tensor_status:{id:t.id,calibration_complete:true,max_abs:t.max_abs}};}
const currentSettings=()=>copy(run('settings()'));
function view(s){const t=catalog.find(t=>t.id===s.tensor),legends={};for(const side of ['left','right']){const max=s[side].startsWith('global')?34:t.max_abs;legends[side]={id:s[side],title:s[side],min:s[side]==='tensor_magnitude'?0:-max,max,zero:0,palette:s[side]==='tensor_magnitude'?'sequential-purple-v1':undefined,s:s[side].endsWith('asinh')?.01*max:null,formula:'fixture formula',scope:s[side].split('_')[0],units:'raw weight'};if(s[side]==='tensor_signed_percentile')Object.assign(legends[side],{min:-1,max:1,units:'dimensionless signed magnitude percentile'});if(s[side]==='tensor_robust99')Object.assign(legends[side],{q99:max,effective_divisor:max,zero_quantile_fallback:false,clipped_count:0,clipped_fraction:0});}return {api_version:1,tensor:t,legends,tile_size:256,overlap:0,source_values_unchanged:true};}
function inspect(t,row,col,raw){return {api_version:1,tensor:t.id,row,col,raw_exact:raw,bf16_hex_le:'0080',shard:'fixture.safetensors',byte_offset:8+2*(row*t.cols+col),native_indices:t.shape.length===1?[col]:[row,col],transformed:{left:0,right:0}};}
async function complete(req,s=currentSettings()){const before=viewers.length;req.resolve(view(s));await tick();const pair=viewers.slice(before);assert.equal(pair.length,2);for(const v of pair)v.emit('open');await tick();const restore=pending.findIndex(p=>p.url.includes('/api/inspect')&&p.options.signal===run('state.inspectController?.signal'));if(restore>=0){const req=pending.splice(restore,1)[0],data=copy(run('state.inspectionData'));req.resolve({...data,transformed:{left:0,right:0}});await tick();}return pair;}
async function activate(){const p=run('loadView()');await complete(take('/api/view'));await p;}
const passed=[];

return {fs,vm,assert,crypto,path,source,Element,elements,get,pending,viewers,frames,timers,OSD,context,run,copy,tick,take,tensor,catalog,model,statusUpdate,currentSettings,view,inspect,complete,activate,passed};
}
module.exports={createFixture};
