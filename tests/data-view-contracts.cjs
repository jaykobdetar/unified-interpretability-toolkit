'use strict';
const assert=require('node:assert/strict'),A=require('../web/atlas-tools.js');
const fixture=require('../docs/SOURCE-BINDING-V2.json');
const t={id:7,name:'example.weight',dtype:'F32',shape:[2,3,4],rows:3,cols:4,count:24,available:true};
const model={model_identity:fixture.model_identity,source_identity:fixture.source_identity,revision:'example',catalog:[t]};
const selected=A.withSlice(t,[1]);
assert.deepEqual(A.sourceBinding(model,selected),fixture);
assert.throws(()=>A.sourceBinding(model,t),/explicitly/);
for(const indices of [[],[2],[-1],[.5],[0,0]])assert.throws(()=>A.withSlice(t,indices));
assert.deepEqual(A.nativeIndices(selected,2,3),[1,2,3]);
const binding=A.sourceBinding(model,selected);
A.requireBinding({...binding,slice:{display_axes:[1,2],leading_indices:[1]}},binding);
for(const slice of [{leading_indices:[0],display_axes:[1,2]},{leading_indices:[1],display_axes:[0,1]},undefined])assert.throws(()=>A.requireBinding({...binding,slice},binding));
assert.notEqual(A.noteKey(A.scope(model,selected)),A.noteKey(A.scope(model,A.withSlice(t,[0]))));
const b={tensor:7,slice:[1],left:'tensor_magnitude_asinh',right:'tensor_magnitude',region:[0,0,2,3],viewport:[0,0,4,3]};
const hash=A.bookmark(b,model);assert.match(hash,/wa=3/);
const parsed=A.resolveBookmark(A.parseBookmark(hash),model);assert.deepEqual(parsed.slice,[1]);
assert.equal(parsed.left,'tensor_magnitude_asinh');
assert.throws(()=>A.resolveBookmark({...parsed,slice:[]},model));
assert.throws(()=>A.resolveBookmark({...parsed,slice:[2]},model));
const matrix={...t,shape:[3,4],count:12};const legacyModel={...model,catalog:[matrix]};
const legacy=A.bookmark({...b,slice:[]},legacyModel);assert.match(legacy,/wa=2/);
assert.deepEqual(A.resolveBookmark(A.parseBookmark(legacy),legacyModel).slice,[]);
assert.throws(()=>A.withSlice({...t,available:false},[0]));
// Leading indices are checked against native bounds, not the display-axis cap.
for(const index of [200000,1000000]){
 const large={...t,shape:[index+1,1,1],rows:1,cols:1,count:index+1},m={...model,catalog:[large]},selected=A.withSlice(large,[index]);
 const link=A.bookmark({...b,slice:[index],region:[0,0,0,0],viewport:[0,0,1,1]},m);
 assert.deepEqual(A.resolveBookmark(A.parseBookmark(link),m).slice,[index]);
 assert.deepEqual(A.nativeIndices(selected,0,0),[index,0,0]);
 assert.throws(()=>A.withSlice(large,[index+1]));
}


// Higher-rank raw BF16 export validates every native index and source binding.
const bt={...selected,dtype:'BF16'},bm={...model,catalog:[{...t,dtype:'BF16'}]};
const bs=A.scope(bm,bt),bb=A.sourceBinding(bm,bt);
const value={source_binding:bb,tensor:7,row:2,col:3,native_indices:[1,2,3],bf16_hex_le:'0080',raw_exact:'-0'};
assert.match(A.boundedCSV(bs,[2,3,2,3],[value],bb),/leading_indices/);
assert.throws(()=>A.boundedCSV(bs,[2,3,2,3],[{...value,native_indices:[0,2,3]}],bb));

// Layout is entirely descriptor-derived, including explicit head_dim != width/heads.
const d=require('./fixtures/smollm2-head-layout-v1.json');
const hm={...model,revision:d.source_model.revision,head_layout:d,head_layout_binding:{source_identity:model.source_identity,model_identity:model.model_identity,weights_sha256:d.source_model.weights_sha256,config_sha256:d.source_model.config_sha256}};
const ht={id:0,name:'model.layers.5.self_attn.o_proj.weight',shape:d.projection_mappings.o_proj.shape,rows:d.width,cols:d.query_heads*d.head_dim};
assert.match(A.hoverHead(hm,ht,0,d.head_dim),/group 1/);
for(const changed of [{...hm,head_layout:undefined},{...hm,head_layout_binding:{...hm.head_layout_binding,config_sha256:'0'.repeat(64)}},{...hm,head_layout:{...d,queries_per_kv:1.5}},{...hm,head_layout:{...d,adapter_id:'unknown'}}])assert.match(A.hoverHead(changed,ht,0,0),/unavailable/);
assert.match(A.hoverHead(hm,{...ht,shape:[1,...ht.shape]},0,0),/not applicable/);
console.log('PASS: canonical binding v2; explicit slice/refusal; native indices; cache/note separation; v3 and legacy bookmarks; raw export; descriptor-only head labels. Pure contracts only.');
