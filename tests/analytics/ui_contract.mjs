// Lightweight DOM contract fixture, not a browser or visual validation.
import assert from 'node:assert/strict';
import fs from 'node:fs';
const source = fs.readFileSync(new URL('../../web/analytics-panel.js', import.meta.url), 'utf8');
const {mountAnalytics, renderModelOutliers} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const fixtures = JSON.parse(fs.readFileSync(new URL('../../web/analytics-examples.json', import.meta.url)));
class Node {
  constructor(tag, doc) { this.tag = tag; this.ownerDocument = doc; this.children = []; this.value = ''; this.textContent = ''; }
  append(...nodes) { this.children.push(...nodes); }
  prepend(...nodes) { this.children.unshift(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(key,value) { (this.attributes??={})[key]=value; }
  remove() { this.removed = true; }
  getContext() { return {beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, fillRect(){}}; }
  getBoundingClientRect() { return {left: 0, width: 800}; }
}
const doc = {createElement(tag) { return new Node(tag, doc); }};
const all = node => [node, ...node.children.flatMap(x => typeof x === 'object' ? all(x) : [])];
const root = new Node('root', doc), jumps = [];
let resolveA, resolveB, calls = 0;
const widget = mountAnalytics(root, {load: () => new Promise(resolve => { if (++calls === 1) resolveA = resolve; else resolveB = resolve; }), jump: x => jumps.push(x)});
const view = all(root).find(x => x.tag === 'select'); view.value = 'strength';
widget.setReport(fixtures.matrix);
assert.equal(view.attributes['aria-label'],'View');
const status = all(root).find(x => x.attributes?.role === 'status');
assert.match(status.textContent, /full tensor.*seed 42/);
widget.setReport(null);
assert.equal(status.textContent, '', 'Clearing a completed report must clear its tensor/coverage/seed status');
assert.equal(all(root).filter(x => x.tag === 'canvas').length, 0);
widget.setReport(fixtures.matrix);
const sorted = all(root).find(x => x.type === 'checkbox'); sorted.checked = true; sorted.onchange();
const canvas = all(root).find(x => x.tag === 'canvas'); canvas.onclick({clientX: 1});
assert.deepEqual(jumps.pop(), {axis: 'row', index: 3});
assert.deepEqual(fixtures.matrix.original.rows.map(x => x.index), [0,1,2,3]);
view.value = 'heads'; view.onchange();
assert.ok(all(root).some(x => x.textContent === 'Q heads'));
assert.ok(all(root).some(x => x.attributes?.['aria-label'] === 'Jump to head group'));
assert.ok(all(root).some(x => x.textContent.includes('full head axis')));
view.value = 'svd'; view.onchange();
assert.ok(all(root).some(x => x.textContent.includes('has not run')));
view.value = 'vector'; widget.setReport(fixtures.vector);
assert.ok(all(root).some(x => x.textContent === 'Signed vector values'));
const summary = {
  schema:'weight-atlas.svd-window-summary.v2', algorithm:'uncentered-independent-svd-xorshift32-v1',
  tensor:'synthetic128', region:{row:128,col:0,rows:128,cols:128},
  coverage:{visited_values:16384,total_tensor_values:16777216,tensor_fraction:1/1024,full_tensor:false,full_model:false},
  preview:{shape:[16,16],displayed_values:256,omitted_values:16128,positions:Array.from({length:256},(_,i)=>Math.floor(i/16)*128+i%16)},
  control:{seed:77,permutation_sha256:'a'.repeat(64),digest_encoding:'uint32 little-endian',preview_position_to_source:Array.from({length:256},(_,i)=>i),omitted_mapping_entries:16128},
  source_identity:'b'.repeat(64),model_identity:'c'.repeat(64),cache_key:'d'.repeat(64),
  results:Object.fromEntries(['original','shuffled'].map(side=>[side,{singular_values:Array(128).fill(0),energy_fractions:Array(128).fill(null),rank_one_residual_energy_fraction:null,rank_one_residual_preview:Array(256).fill(0)}]))
};
view.value = 'svd_summary'; widget.setReport(summary);
assert.match(status.textContent,/16,384.*16,777,216.*partial native window; no full-model claim/);
assert.ok(all(root).some(x=>x.textContent.includes('256 / 16384; 1.563%')));
assert.ok(all(root).some(x=>x.textContent.includes('16128 residual positions omitted per side')));
assert.ok(all(root).some(x=>x.textContent.includes('Complete permutation SHA-256')));
assert.equal(all(root).filter(x=>x.tag==='li' && x.textContent.startsWith('window [')).length,256);
view.value='strength'; view.onchange();
assert.ok(all(root).some(x=>x.textContent.includes('recompute regional analysis')));
view.value='vector'; widget.setReport(fixtures.vector);
const a = widget.reload(), b = widget.reload();
resolveB(fixtures.vector); await b; resolveA(fixtures.matrix); await a;
assert.ok(all(root).some(x => x.textContent.startsWith('synthetic.vector •')));
const pending = widget.reload();
assert.equal(status.textContent, 'Reading bounded native region…');
widget.setReport(null);
assert.equal(status.textContent, '', 'Clearing while loading must also clear status');
resolveB(fixtures.matrix); await pending;
assert.equal(status.textContent, '', 'A late response must not restore cleared coverage');
assert.equal(all(root).filter(x => x.tag === 'canvas').length, 0);
widget.destroy(); assert.equal(root.children[0].removed, true);
const model = new Node('root', doc);
renderModelOutliers(model, {coverage: {full_model:false, visited_values:2, total_values:8, visited_tensors:1, total_tensors:2, selection:'bounded windows', tensors:[]}, control:{kind:'same region',seed:1,statistics:'refit'}, rankings:{original:{values:[]},shuffled:{values:[]}},warning:'partial'});
assert.ok(all(model).some(x => x.textContent === 'Partial-model outlier ranking'));
const separate = new Node('root',doc), requests = [];
const summaryWidget = mountAnalytics(separate,{load:async data=>{requests.push(data);return summary;},jump(){}});
await all(separate).find(x=>x.textContent==='Compute 128-window SVD summary').onclick();
assert.deepEqual(requests,[{seed:1,scope:'svd_summary'}]);
assert.ok(all(separate).some(x=>x.textContent.includes('Residual preview only')));
await summaryWidget.reload();
assert.ok(all(separate).some(x=>x.textContent.includes('Unsupported analytics schema')));
console.log('PASS: paired views, sorting index maps, native jumps, unavailable SVD, vector, clear-status and pending-response invalidation, stale responses, cleanup, partial-model label');
