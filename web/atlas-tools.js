'use strict';
// Pure, bounded workspace contracts. No inference APIs, storage globals, or DOM.
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.AtlasTools=api;})(globalThis,()=>{
  const RULES=['global_linear','global_asinh','tensor_linear','tensor_asinh','tensor_magnitude','tensor_magnitude_asinh','tensor_robust99','tensor_signed_percentile'];
  const MAX_CELLS=256,MAX_BYTES=262144,MAX_NOTES=100,MAX_NOTE_TEXT=500;
  const check=(ok,message)=>{if(!ok)throw new Error(message);};
  const abort=()=>new DOMException('Cancelled','AbortError');
  const live=signal=>{if(signal?.aborted)throw abort();};
  const identity=s=>typeof s==='string'&&/^[0-9a-f]{64}$/.test(s);
  const integer=n=>Number.isSafeInteger(n)&&n>=0;
  const boundedText=(v,n)=>typeof v==='string'&&v.length<=n;
  const clamp=(n,a,b)=>Math.min(b,Math.max(a,n));
  function dimensions(t){check(t&&integer(t.rows)&&t.rows>0&&integer(t.cols)&&t.cols>0&&Number.isSafeInteger(t.rows*t.cols),'Invalid native dimensions');return t;}
  function region(value,t){dimensions(t);check(Array.isArray(value)&&value.length===4&&value.every(integer),'Use four nonnegative whole-number native bounds');const [r0,c0,r1,c1]=value;check(r0<=r1&&c0<=c1&&r1<t.rows&&c1<t.cols,'Region is empty or outside this tensor');return [...value];}
  function viewport(value,t){dimensions(t);check(Array.isArray(value)&&value.length===4&&value.every(Number.isFinite)&&value[2]>0&&value[3]>0,'Invalid viewport');const span=Math.min(Number.MAX_SAFE_INTEGER/8,Math.max(t.rows,t.cols))*4;return [clamp(value[0],-span,Math.min(Number.MAX_SAFE_INTEGER,t.cols+span)),clamp(value[1],-span,Math.min(Number.MAX_SAFE_INTEGER,t.rows+span)),clamp(value[2],1/24,span),clamp(value[3],1/24,span)];}
  function sliceIndices(t,leading=t?.slice){
    check(t&&t.available!==false&&Array.isArray(t.shape)&&t.shape.length>0,'Tensor is unavailable');
    const n=Math.max(0,t.shape.length-2),v=leading===undefined&&n===0?[]:leading;
    check(Array.isArray(v)&&v.length===n&&v.every((x,i)=>integer(x)&&x<t.shape[i]),'Select every leading axis explicitly');
    return [...v];
  }
  function withSlice(t,leading){return {...t,slice:sliceIndices(t,leading)};}
  function sliceQuery(t){return sliceIndices(t).join(',');}
  function nativeIndices(t,row,col){return [...sliceIndices(t),...(t.shape.length===1?[col]:[row,col])];}
  const pVersion=hash=>new URLSearchParams(hash.slice(1)).get('wa');
  function parseBookmark(hash){
    if(!hash)return null;
    try{
      check(typeof hash==='string'&&hash.length<=1024&&/^#wa=/.test(hash),'Invalid bookmark');
      const p=new URLSearchParams(hash.slice(1)),keys=['wa','s','m','t','l','r','b','z',...(pVersion(hash)==='3'?['i']:[])];
      check([...p.keys()].length===keys.length&&keys.every(k=>p.getAll(k).length===1),'Invalid bookmark fields');
      check(['2','3'].includes(p.get('wa'))&&identity(p.get('s'))&&identity(p.get('m'))&&/^\d{1,16}$/.test(p.get('t')),'Invalid bookmark identity');
      check(RULES.includes(p.get('l'))&&RULES.includes(p.get('r')),'Invalid rules');
      const numbers=(key,count,re)=>{const parts=p.get(key).split(',');check(parts.length===count&&parts.every(v=>re.test(v)),'Invalid coordinates');return parts.map(Number);};
      const b=numbers('b',4,/^\d{1,16}$/),z=numbers('z',4,/^-?\d{1,16}(?:\.\d{1,8})?$/),t=Number(p.get('t'));
      check(integer(t)&&b.every(integer)&&b[0]<=b[2]&&b[1]<=b[3]&&z.every(Number.isFinite)&&z[2]>0&&z[3]>0,'Invalid coordinates');
      const leading=p.get('wa')==='3'?p.get('i').split(',').map(v=>{check(/^\d{1,16}$/.test(v),'Invalid slice');return Number(v);}):[];
      check(leading.length<=30&&leading.every(integer),'Invalid slice indices or rank');
      return {v:Number(p.get('wa')),slice:leading,source:p.get('s'),model:p.get('m'),tensor:t,left:p.get('l'),right:p.get('r'),region:b,viewport:z};
    }catch{return null;}
  }
  function resolveBookmark(b,model){
    check(b&&[2,3].includes(b.v)&&b.source===model.source_identity&&b.model===model.model_identity,'Bookmark belongs to a different source identity or model revision');
    const t=model.catalog.find(t=>t.id===b.tensor);check(t,'Bookmarked tensor is absent');dimensions(t);sliceIndices(t,b.slice||[]);
    return {...b,region:[clamp(b.region[0],0,t.rows-1),clamp(b.region[1],0,t.cols-1),clamp(b.region[2],0,t.rows-1),clamp(b.region[3],0,t.cols-1)],viewport:viewport(b.viewport,t)};
  }
  function bookmark(value,model){
    check(identity(model.source_identity)&&identity(model.model_identity),'Exact source/revision identity unavailable');
    const t=model.catalog.find(t=>t.id===value.tensor);region(value.region,t);const leading=sliceIndices(t,value.slice===undefined?undefined:Array.isArray(value.slice)?value.slice:value.slice.split(',').filter(Boolean).map(Number));
    check(RULES.includes(value.left)&&RULES.includes(value.right),'Invalid rules');
    const z=viewport(value.viewport,t).map(n=>n.toFixed(8).replace(/\.?0+$/,''));
    return '#'+new URLSearchParams({wa:leading.length?'3':'2',...(leading.length?{i:leading.join(',')}:{}),s:model.source_identity,m:model.model_identity,t:String(value.tensor),l:value.left,r:value.right,b:value.region.join(','),z:z.join(',')}).toString();
  }
  function delay(ms,signal){return new Promise((resolve,reject)=>{live(signal);const onAbort=()=>{clearTimeout(timer);reject(abort());};const timer=setTimeout(()=>{signal?.removeEventListener('abort',onAbort);resolve();},ms);signal?.addEventListener('abort',onAbort,{once:true});});}
  async function readJSON(url,{signal,onState=()=>{},fetchImpl=globalThis.fetch,wait=delay}={}){
    // A closed allowlist: caller cannot smuggle generation/calibration/start into retry.
    check(typeof url==='string'&&/^\/api\/(model|view|inspect|progress|tensor-status)(?:\?[^#]*)?$/.test(url),'Read-only atlas route required');
    const bound=globalThis.AtlasHost?.bindRead(url),requestURL=bound?.url||url;
    const request={},emit=event=>onState({...event,request});
    live(signal);emit({phase:'started'});
    const waits=[250,750];
    for(let attempt=0;;attempt++){
      live(signal);let response,data;
      try{response=await fetchImpl(requestURL,{method:'GET',signal,cache:'no-store'});live(signal);bound?.assertCurrent();data=await response.json();live(signal);bound?.assertCurrent();}
      catch(e){live(signal);emit({phase:'failed',code:'transport_error'});throw e;}
      if(response.ok){if(data?.api_version!==1)emit({phase:'failed',code:'contract_error'});check(data?.api_version===1,'API contract: expected api_version 1.');bound?.check(data);emit({phase:'success'});return data;}
      const e=Object.assign(new Error(data?.error||`HTTP ${response.status}`),{status:response.status,code:data?.code});
      const retry=response.status===503&&['backend_unavailable','admission_full','dispatch_full'].includes(data?.code);
      if(!retry||attempt===waits.length){emit({phase:'failed',code:data?.code||(response.status===503?'calibration_readiness':'request_error'),status:response.status});throw e;}
      emit({phase:'retrying',code:data?.code,attempt:attempt+1,limit:waits.length,delay:waits[attempt]});
      await wait(waits[attempt],signal);live(signal);
    }
  }
  function scope(model,t){
    dimensions(t);check(identity(model.source_identity)&&integer(t.id)&&boundedText(t.name,512)&&boundedText(model.revision,512)&&boundedText(t.dtype,16)&&Array.isArray(t.shape),'Exact model/tensor identity unavailable');
    check(t.shape.length>=1&&t.shape.length<=32&&t.shape.every(n=>Number.isSafeInteger(n)&&n>0)&&t.cols===t.shape.at(-1)&&t.rows===(t.shape.length===1?1:t.shape.at(-2)),'Invalid tensor shape');
    const leading=sliceIndices(t);return {source_identity:model.source_identity,revision:model.revision,tensor:t.id,name:t.name,dtype:t.dtype,shape:[...t.shape],rows:t.rows,cols:t.cols,...(leading.length?{slice:leading}:{})};
  }
  function sourceBinding(model,t){
    const s=scope(model,t);check(identity(model.model_identity),'Exact model revision binding unavailable');
    return bindingFromScope(s,model.model_identity);
  }
  function bindingFromScope(s,model){return {version:2,model_identity:model,source_identity:s.source_identity,tensor:s.tensor,name:s.name,dtype:s.dtype,shape:[...s.shape],rows:s.rows,cols:s.cols,slice:{leading_indices:s.slice||[],display_axes:s.shape.length===1?[0]:[s.shape.length-2,s.shape.length-1]}};}
  function requireBinding(actual,expected){
    check(actual&&expected&&identity(expected.model_identity)&&Object.keys(expected).every(k=>k==='shape'?JSON.stringify(actual[k])===JSON.stringify(expected[k]):k==='slice'?actual.slice&&['leading_indices','display_axes'].every(field=>JSON.stringify(actual.slice[field])===JSON.stringify(expected.slice[field])):actual[k]===expected[k]),'Scalar source/revision/tensor binding mismatch; no file was produced');
  }
  function sameModelContext(a,b,id){
    try{return !!a.catalog.find(t=>t.id===id)&&!!b.catalog.find(t=>t.id===id)&&a.model_identity===b.model_identity&&identity(a.model_identity)&&JSON.stringify(a.catalog.find(t=>t.id===id)&&[a.revision,...['name','dtype','shape','rows','cols','available'].map(k=>a.catalog.find(t=>t.id===id)[k])])===JSON.stringify(b.catalog.find(t=>t.id===id)&&[b.revision,...['name','dtype','shape','rows','cols','available'].map(k=>b.catalog.find(t=>t.id===id)[k])]);}catch{return false;}
  }
  const noteKey=s=>'weight-atlas.notes.v1:'+JSON.stringify(s);
  function validNote(n,s){try{return n&&/^[a-zA-Z0-9-]{1,80}$/.test(n.id)&&['row','column','region'].includes(n.kind)&&boundedText(n.text,MAX_NOTE_TEXT)&&!!n.text.trim()&&region(n.region,s)&&
    (n.kind!=='row'||(n.region[0]===n.region[2]&&n.region[1]===0&&n.region[3]===s.cols-1))&&
    (n.kind!=='column'||(n.region[1]===n.region[3]&&n.region[0]===0&&n.region[2]===s.rows-1));}catch{return false;}}
  function loadNotes(storage,s){const raw=storage.getItem(noteKey(s));if(!raw)return [];check(raw.length<=65536,'Local note data exceeds limit');const notes=JSON.parse(raw);check(Array.isArray(notes)&&notes.length<=MAX_NOTES&&notes.every(n=>validNote(n,s))&&new Set(notes.map(n=>n.id)).size===notes.length,'Invalid local notes; storage left unchanged');return notes;}
  function saveNotes(storage,s,notes){check(Array.isArray(notes)&&notes.length<=MAX_NOTES&&notes.every(n=>validNote(n,s))&&new Set(notes.map(n=>n.id)).size===notes.length,'Invalid notes or 100-note limit reached');const raw=JSON.stringify(notes);check(raw.length<=65536,'Local notes exceed 64 KiB character limit');storage.setItem(noteKey(s),raw);}
  function updateNote(notes,s,n){check(validNote(n,s),'Invalid note');const next=notes.filter(x=>x.id!==n.id);next.push({...n,region:[...n.region]});check(next.length<=MAX_NOTES,'100-note limit reached');return next;}
  function decodeBF16(hex){check(typeof hex==='string'&&/^[0-9a-fA-F]{4}$/.test(hex),'Invalid original BF16 bytes');const bits=parseInt(hex.slice(2)+hex.slice(0,2),16),b=new ArrayBuffer(4),v=new DataView(b);v.setUint32(0,bits*65536,true);return v.getFloat32(0,true);}
  function csvCell(v,text=false){let s=String(v);if(text&&/^[\s\u0000-\u001f]*[=+\-@]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"';}
  function decodeSource(dtype,hex){
    check(['BF16','F16','F32'].includes(dtype),'Only original BF16/F16/F32 sources are supported');
    const bytes=dtype==='F32'?4:2;check(typeof hex==='string'&&new RegExp('^[0-9a-fA-F]{'+bytes*2+'}$').test(hex),'Invalid original source bytes');
    if(dtype==='BF16')return decodeBF16(hex);
    const data=Uint8Array.from(hex.match(/../g),h=>parseInt(h,16)),view=new DataView(data.buffer);
    if(dtype==='F32')return view.getFloat32(0,true);
    const bits=view.getUint16(0,true),sign=bits&32768?-1:1,exponent=(bits>>>10)&31,fraction=bits&1023;
    return sign*(exponent===0?fraction*2**-24:exponent===31?(fraction?NaN:Infinity):(1+fraction/1024)*2**(exponent-15));
  }
  function checkedRegion(s,b,values,binding){
    check(binding&&identity(binding.model_identity),'Exact scalar binding required');requireBinding(binding,bindingFromScope(s,binding.model_identity));
    region(b,s);const rows=b[2]-b[0]+1,cols=b[3]-b[1]+1,count=rows*cols;
    check(count<=MAX_CELLS&&values.length===count,'Export requires 1–256 native cells');
    check(['BF16','F16','F32'].includes(s.dtype),'Only original BF16/F16/F32 sources are supported');
    const hex=values.map((v,i)=>{
      const row=b[0]+Math.floor(i/cols),col=b[1]+i%cols;requireBinding(v.source_binding,binding);
      check(v.tensor===s.tensor&&v.row===row&&v.col===col&&JSON.stringify(v.native_indices)===JSON.stringify(nativeIndices(s,row,col)),'Export address mismatch');
      const h=v.raw_hex_le??(s.dtype==='BF16'?v.bf16_hex_le:undefined),decoded=decodeSource(s.dtype,h);
      check((v.dtype===undefined||v.dtype===s.dtype)&&(v.element_bytes===undefined||v.element_bytes===(s.dtype==='F32'?4:2)),'Export storage dtype mismatch');
      check(Number.isFinite(decoded)&&boundedText(v.raw_exact,256)&&/^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(v.raw_exact)&&Object.is(Number(v.raw_exact),decoded),'Exact decimal and source bytes disagree');return h.toLowerCase();
    });
    const metadata={format:'weight-atlas-region-v1',...s,model_identity:binding.model_identity,region:b,dimensions:[rows,cols],native_indices:s.shape.length===1?'[column]':'[...leading_indices,row,column]',source_dtype:s.dtype,decoded_dtype:s.dtype==='F16'?'IEEE754 binary16':'IEEE754 binary32'+(s.dtype==='BF16'?' (exact BF16 widening)':''),values:'original, untransformed',scale:'none; no normalization or color transform',decimal:'exact source decimal; '+s.dtype+' little-endian bytes retained'};
    return {rows,cols,hex,metadata};
  }
  function boundedCSV(s,b,values,binding){
    const {hex,metadata}=checkedRegion(s,b,values,binding),field=s.dtype==='BF16'?'bf16_hex_le':'raw_hex_le';
    const lines=[['metadata',JSON.stringify(metadata)].map(v=>csvCell(v,true)).join(','),'row,column,native_indices,raw_exact,'+field];
    for(let i=0;i<values.length;i++){const v=values[i];lines.push([v.row,v.col,JSON.stringify(v.native_indices),v.raw_exact,hex[i]].map(v=>csvCell(v)).join(','));}
    const csv=lines.join('\r\n')+'\r\n';check(new TextEncoder().encode(csv).byteLength<=MAX_BYTES,'Export exceeds 256 KiB');return csv;
  }
  function boundedNPY(s,b,values,binding){
    const {rows,cols,hex,metadata}=checkedRegion(s,b,values,binding),descr=s.dtype==='F16'?'<f2':'<f4';
    // Numeric NPY v1: fixed ASCII header, C order; never object arrays/pickle.
    const prefix="{'descr': '"+descr+"', 'fortran_order': False, 'shape': ("+rows+', '+cols+'), }';
    const header=prefix+' '.repeat((64-(10+prefix.length+1)%64)%64)+'\n',bytes=descr==='<f2'?2:4;
    const data=new Uint8Array(10+header.length+hex.length*bytes);data.set([147,78,85,77,80,89,1,0,header.length&255,header.length>>>8]);data.set(new TextEncoder().encode(header),10);
    hex.forEach((h,i)=>{const original=Uint8Array.from(h.match(/../g),x=>parseInt(x,16));data.set(s.dtype==='BF16'?[0,0,...original]:original,10+header.length+i*bytes);});
    check(data.byteLength<=MAX_BYTES,'Export exceeds 256 KiB');return {data,metadata:{...metadata,numpy:{format:'NPY v1.0',descr,fortran_order:false,shape:[rows,cols],allow_pickle:false},source_hex_le:hex}};
  }
  async function collectRegion({model,tensor,bounds,read,signal,onProgress=()=>{},format='csv'}){
    check(['csv','npy'].includes(format),'Unsupported export format');
    const s=scope(model,tensor),binding=sourceBinding(model,tensor),b=region(bounds,tensor),count=(b[2]-b[0]+1)*(b[3]-b[1]+1);check(count<=MAX_CELLS,'Export is capped at 256 native cells');check(['BF16','F16','F32'].includes(s.dtype),'Only original BF16/F16/F32 export is available');
    async function sameSource(){live(signal);const m=await read('/api/model',signal);live(signal);const t=m.catalog?.find(t=>t.id===s.tensor);check(t&&m.model_identity===binding.model_identity&&JSON.stringify(scope(m,withSlice(t,s.slice||[])))===JSON.stringify(s),'Source changed during export; no file was produced');}
    await sameSource();const values=[];
    for(let row=b[0];row<=b[2];row++)for(let col=b[1];col<=b[3];col++){live(signal);const value=await read('/api/inspect?'+new URLSearchParams({tensor:s.tensor,slice:(s.slice||[]).join(','),row,col,left:'tensor_linear',right:'tensor_asinh'}),signal);live(signal);requireBinding(value.source_binding,binding);values.push(value);onProgress(values.length,count);}
    await sameSource();live(signal);return format==='csv'?boundedCSV(s,b,values,binding):boundedNPY(s,b,values,binding);
  }
  // Observed native regions, bound to the exact local source snapshot. Evidence in
  // tests/fixtures/ui-polish-observations.json. No model-name/shape-only layout inference.
  const STARTER_SOURCE='2554a200ae640fd3b5bc7f91ffac1be0483efaf571b36dfac91f42eb5a40bc8d';
  const STARTER_REVISION='93efa2f097d58c2a74874c7e644dbc9b0cee75a2';
  const STARTERS=[
    {name:'model.layers.0.self_attn.q_proj.weight',shape:[576,576],region:[64,0,71,7],title:'Positive and negative share a head',description:'This 8 × 8 window has 36 positive and 28 negative weights (−0.65234375 to 0.671875). Compare signed color with magnitude; all eight rows belong to query head 1.',right:'tensor_magnitude'},
    {name:'model.layers.0.self_attn.k_proj.weight',shape:[192,576],region:[0,0,7,7],title:'Keys have fewer head groups',description:'This stored key projection has 192 rows: 3 KV heads of 64, versus 9 query heads. The selected 64 values span −0.74609375 to 0.4375. This is a layout observation, not a claim about head function.',right:'tensor_asinh'},
    {name:'model.layers.0.input_layernorm.weight',shape:[576],region:[0,0,0,31],title:'A vector stays a vector',description:'The first 32 normalization weights include 13 positive and 19 negative values (−0.1591796875 to 0.1953125). The 576-value tensor stays one native row; it is not a square image.',right:'tensor_magnitude'}
  ];
  const QWEN_SOURCE='cbf36b69cb8aab0ff2ec797fbd76cf756a0e72e151cdfac9c675d8bc9430e8f6';
  const QWEN_REVISION='b968826d9c46dd6066d109eabc6255188de91218';
  const QWEN_STARTERS=[{"name": "model.layers.0.self_attn.q_proj.weight", "shape": [4096, 4096], "region": [128, 0, 135, 7], "title": "Positive and negative share a head", "description": "This 8 × 8 window has 28 positive and 36 negative weights (−0.0439453125 to 0.0556640625). Compare signed color with magnitude; all eight rows belong to query head 1.", "right": "tensor_magnitude"}, {"name": "model.layers.0.self_attn.k_proj.weight", "shape": [1024, 4096], "region": [0, 0, 7, 7], "title": "Keys have fewer head groups", "description": "This stored key projection has 1,024 rows: 8 KV heads of 128, versus 32 query heads. The selected 64 values span −0.05712890625 to 0.06103515625. This is a layout observation, not a claim about head function.", "right": "tensor_asinh"}, {"name": "model.layers.0.input_layernorm.weight", "shape": [4096], "region": [0, 0, 0, 31], "title": "A positive normalization window", "description": "The first 32 normalization weights are positive (0.00860595703125 to 0.0125732421875). This local window does not establish the signs of the whole tensor. The 4,096-value vector stays one native row.", "right": "tensor_magnitude"}];
  function qwenSource(model){return model?.source_identity===QWEN_SOURCE&&model.revision===QWEN_REVISION;}
  function starterSource(model){return model?.source_identity===STARTER_SOURCE&&model.revision===STARTER_REVISION;}
  function guidedExamples(model){
    const entries=starterSource(model)?STARTERS:qwenSource(model)?QWEN_STARTERS:[];
    return entries.flatMap(e=>{
      const t=model.catalog.find(t=>t.name===e.name&&t.dtype==='BF16'&&JSON.stringify(t.shape)===JSON.stringify(e.shape));if(!t)return [];
      const [r0,c0,r1,c1]=e.region;
      return [{...e,location:`${e.name} · rows ${r0}–${r1}, columns ${c0}–${c1}`,href:bookmark({tensor:t.id,left:'tensor_linear',right:e.right,region:e.region,viewport:[c0,r0,c1-c0+1,r1-r0+1]},model)}];
    });
  }
  function hoverHead(model,t,row,col){
    const d=model?.head_layout,b=model?.head_layout_binding;
    if(!d||d.schema!=='weight-atlas-head-layout-v1'||d.adapter_version!==1||d.adapter_id!=='builtin-llama-eager'||d.model_type!=='llama'||!['pinned_configuration','loaded_builtin_layout'].includes(d.evidence)||!b||b.source_identity!==model.source_identity||b.model_identity!==model.model_identity||!identity(b.weights_sha256)||!identity(b.config_sha256)||b.weights_sha256!==d.source_model?.weights_sha256||b.config_sha256!==d.source_model?.config_sha256||d.source_model?.revision!==model.revision)return 'Head unavailable · no bound configuration layout descriptor';
    const m=/^model\.layers\.(\d+)\.self_attn\.([qkvo])_proj\.weight$/.exec(t.name);
    if(!m||t.shape.length!==2)return 'Head not applicable to this tensor';
    const key=m[2]+'_proj',p=d.projection_mappings?.[key],axis=key==='o_proj'?'columns':'rows';
    if(!p||JSON.stringify(d.native_weight_layout)!==JSON.stringify(['output_feature','input_feature'])||sliceIndices(t).length||![d.layers,d.width,d.query_heads,d.kv_heads,d.head_dim,d.queries_per_kv,p.heads,p.head_dim].every(n=>integer(n)&&n>0)||Number(m[1])>=d.layers||d.query_heads!==d.kv_heads*d.queries_per_kv||p.head_dim!==d.head_dim||p.heads!==(['k_proj','v_proj'].includes(key)?d.kv_heads:d.query_heads)||p.axis!==axis||JSON.stringify(t.shape)!==JSON.stringify(p.shape)||t.shape[axis==='rows'?0:1]!==p.heads*p.head_dim||t.shape[axis==='rows'?1:0]!==d.width||!integer(row)||!integer(col)||row>=t.rows||col>=t.cols)return 'Head unavailable · layout mismatch';
    const coordinate=axis==='rows'?row:col,index=Math.floor(coordinate/p.head_dim);
    const evidence=d.runtime_verified===true&&d.evidence==='loaded_builtin_layout'?'loaded built-in layout':'configuration-derived layout; runtime unverified';
    return `${key==='o_proj'?'Output input-column Q-head group':key==='q_proj'?'Query head':'KV '+m[2].toUpperCase()+' head'} ${index} · ${axis==='rows'?'row':'column'} offset ${coordinate%p.head_dim} (${evidence})`;
  }
  return {sliceIndices,sliceQuery,withSlice,nativeIndices,guidedExamples,hoverHead,RULES,MAX_CELLS,MAX_BYTES,MAX_NOTES,MAX_NOTE_TEXT,region,viewport,parseBookmark,resolveBookmark,bookmark,delay,readJSON,scope,sourceBinding,requireBinding,sameModelContext,noteKey,loadNotes,saveNotes,updateNote,decodeBF16,decodeSource,csvCell,boundedCSV,boundedNPY,collectRegion};
});
