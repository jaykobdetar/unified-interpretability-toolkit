#!/usr/bin/env python3
"""Opt-in pinned CPU model checks. Declared prompts are public synthetic fixtures."""
import json
import os
from pathlib import Path
import resource
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from inference_worker import compare, load_engine
from inference_edits import SOURCE_MODEL, apply_edits, verified_parameters
from live_inference import verify_model
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
directory = Path(sys.argv[1])
verify_model(directory)
torch, tokenizer, model = load_engine(directory)
parameters = verified_parameters(model)
prompt = 'The capital of France is'
request = {'prompt': prompt, 'max_new_tokens': 4, 'layer': 0, 'source_model': SOURCE_MODEL, 'edits': []}
empty = []
compare(torch, tokenizer, model, request, empty.append)
empty_steps = [event for event in empty if event['type'] == 'step']
assert empty[-1]['baseline'] == empty[-1]['edited']
assert all(step['baseline']['generated_ids'] == step['edited']['generated_ids'] and step['alignment'] == 'matched_prefix' for step in empty_steps)
assert all(candidate['delta'] == 0.0 and candidate['baseline_logit'] == candidate['edited_logit'] for step in empty_steps for candidate in step['candidates'])
# Independent scalar/column/overlap arithmetic, directly against loaded FP32 values.
name = 'model.layers.0.self_attn.q_proj.weight'
value = parameters[name]
with torch.no_grad():
    saved_rows = value[:64, :].clone()
    expected_01 = float(value[0, 1]) * -2 * .5
    expected_12 = float(value[1, 2]) * -2
ops = [dict(tensor=name, shape=[576,576], kind='rows', operation='scale', start=0, end=2, scale=-2),
       dict(tensor=name, shape=[576,576], kind='columns', operation='zero', start=0, end=1),
       dict(tensor=name, shape=[576,576], kind='element', operation='scale', row=0, col=1, scale=.5)]
# Save only the changed column in addition to the small query-head slice.
with torch.no_grad(): saved_column = value[:, :1].clone()
apply_edits(torch, model, ops, SOURCE_MODEL)
assert float(value[0, 1]) == expected_01 and float(value[1, 2]) == expected_12
assert bool((value[:, 0] == 0).all())
with torch.no_grad():
    value[:64, :].copy_(saved_rows)
    value[:, :1].copy_(saved_column)
embed = parameters['model.embed_tokens.weight']
with torch.no_grad(): saved_embedding = embed[1, 2].clone()
apply_edits(torch, model, [dict(tensor='model.embed_tokens.weight', shape=[49152,576], kind='element', operation='scale', row=1, col=2, scale=.5)], SOURCE_MODEL)
assert float(parameters['lm_head.weight'][1,2]) == float(saved_embedding) * .5
with torch.no_grad(): embed[1,2].copy_(saved_embedding)
# Independent full-prefix prefill scores, no generate helper and no KV reuse.
ids = tokenizer.encode(prompt, add_special_tokens=True)
with torch.inference_mode(): baseline_logits = model(torch.tensor([ids]), use_cache=False).logits[0,-1].clone()
head_edit = dict(tensor=name, shape=[576,576], kind='rows', operation='zero', start=0, end=64)
changed = []
compare(torch, tokenizer, model, {**request, 'edits':[head_edit]}, changed.append)
changed_steps = [event for event in changed if event['type']=='step']
assert bool((value[:64] == 0).all())
assert torch.equal(value[64:, :1], saved_column[64:])
with torch.inference_mode(): edited_logits = model(torch.tensor([ids]), use_cache=False).logits[0,-1].clone()
delta = edited_logits - baseline_logits
first = changed_steps[0]
for candidate in first['candidates']:
    token = candidate['id']
    assert abs(candidate['baseline_logit'] - float(baseline_logits[token])) < 2e-4
    assert abs(candidate['edited_logit'] - float(edited_logits[token])) < 2e-4
    assert abs(candidate['delta'] - (float(edited_logits[token]) - float(baseline_logits[token]))) < 4e-4
# Restore only in this test so a second call demonstrates pristine generation.
# The production worker cannot be reused: one request, then it is reaped.
with torch.no_grad(): value[:64, :].copy_(saved_rows)
after = []
compare(torch, tokenizer, model, request, after.append)
assert after[-1]['baseline'] == empty[-1]['baseline'] == after[-1]['edited']
verify_model(directory)
result = {'status':'PASS', 'fixture':'capital-france-public-synthetic',
          'empty_edit_exact_parity':True, 'independent_arithmetic':'rows, columns, element, overlapping order, tied embedding/output head',
          'head':{'tensor':name,'rows_half_open':[0,64],'query_heads':9,'head_width':64,'kv_heads':3},
          'first_step_all_vocab_delta_max_abs':float(delta.abs().max()),
          'first_step_all_vocab_delta_l2':float(torch.linalg.vector_norm(delta)),
          'first_step_top_union_max_abs_delta':max(abs(c['delta']) for c in first['candidates']),
          'first_step_top_union':first['candidates'],
          'baseline_ids':changed[-1]['baseline']['generated_ids'], 'edited_ids':changed[-1]['edited']['generated_ids'],
          'alignments':[s['alignment'] for s in changed_steps],
          'disk_hashes_unchanged':True,'single_model_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
          'effect_scope':'One declared fixture and first-step matched prompt; effect is measured, never required to exceed an invented threshold.'}
print(json.dumps(result,indent=2,allow_nan=False))
