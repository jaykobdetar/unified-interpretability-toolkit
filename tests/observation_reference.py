#!/usr/bin/env python3
"""Prepared one-model observational oracle; requires a serialized heavy slot."""
import json
import math
import os
from pathlib import Path
import resource
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from inference_worker import generate, load_engine
from live_inference import verify_model

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (3*1024**3, 3*1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
directory = Path(sys.argv[1]);verify_model(directory)
torch, tokenizer, model = load_engine(directory)
prompt = 'The capital of France is'  # Public synthetic fixture, not a personal prompt.
ids = tokenizer.encode(prompt, add_special_tokens=True)
baseline = [];generate(torch, tokenizer, model, prompt, 2, 0, baseline.append)
baseline = [e for e in baseline if e['type'] == 'step']
signature = lambda steps: [(e['token_id'], e['top_logits']) for e in steps]
cases = []

for layer in (0, 7, 29):
    for kind, head in [('attention', 0), ('attention', 8), ('logit_lens', None)]:
        selection = {'kind': kind, **({'head': head} if head is not None else {})}
        site = 'attention' if kind == 'attention' else 'block'
        events = [];generate(torch, tokenizer, model, prompt, 2, layer, events.append, activation_site=site, observation=selection)
        steps = [e for e in events if e['type'] == 'step']
        assert signature(steps) == signature(baseline), 'Observation changed generated tokens or final logits'
        for index, step in enumerate(steps):
            consumed = ids + [s['token_id'] for s in steps[:index]]
            captured = {}
            def before(_module, args, kwargs):
                captured['hidden'] = kwargs['hidden_states'].detach().clone()
                captured['cos'], captured['sin'] = [t.detach().clone() for t in kwargs['position_embeddings']]
                mask = kwargs['attention_mask']
                captured['mask'] = None if mask is None else mask.detach().clone()
            def after(_module, _args, output):
                captured['residual'] = (output[0] if isinstance(output, tuple) else output)[0, -1].detach().clone()
            block = model.model.layers[layer]
            hook = (block.self_attn.register_forward_pre_hook(before, with_kwargs=True) if kind == 'attention'
                    else block.register_forward_hook(after))
            try:
                with torch.inference_mode():
                    oracle = model(torch.tensor([consumed]), use_cache=False, logits_to_keep=1)
            finally:
                hook.remove()
            with torch.inference_mode():
                if kind == 'attention':
                    # Independent selected-head arithmetic: no attention helper or output weights.
                    hidden = captured['hidden'][0]
                    query = (hidden @ block.self_attn.q_proj.weight.T).reshape(-1, 9, 64)[:, head]
                    key = (hidden @ block.self_attn.k_proj.weight.T).reshape(-1, 3, 64)[:, head // 3]
                    cos, sin = captured['cos'][0], captured['sin'][0]
                    rotate = lambda x: torch.cat((-x[:, 32:], x[:, :32]), dim=-1)
                    query = query * cos + rotate(query) * sin
                    key = key * cos + rotate(key) * sin
                    scores = (key @ query[-1]) / math.sqrt(64)
                    if captured['mask'] is not None:
                        scores += captured['mask'][0, 0, -1, :len(consumed)]
                    expected = torch.softmax(scores, dim=-1)
                    actual = torch.tensor(step['attention']['probabilities'])
                    error = float((actual - expected).abs().max())
                    assert error < 2e-5, (layer, head, index, error)
                    assert step['attention']['key_token_ids'] == consumed
                    assert step['attention']['key_positions'] == list(range(len(consumed)))
                    assert abs(sum(step['attention']['probabilities']) - 1) < 1e-5
                    cases.append({'kind': kind, 'layer': layer, 'head': head, 'step': index,
                                  'max_probability_error': error, 'key_count': len(consumed)})
                else:
                    # RMSNorm arithmetic and output projection independent of lens_record.
                    residual = captured['residual']
                    normalized = residual * torch.rsqrt(residual.pow(2).mean() + model.model.norm.variance_epsilon)
                    reference = (normalized * model.model.norm.weight) @ model.lm_head.weight.T
                    final = oracle.logits[0, -1]
                    candidate_error = delta_error = 0.
                    for candidate in step['logit_lens']['candidates']:
                        token = candidate['id']
                        candidate_error = max(candidate_error, abs(candidate['lens_logit'] - float(reference[token])))
                        delta_error = max(delta_error, abs(candidate['delta_lens_minus_final'] - (float(reference[token]) - float(final[token]))))
                    assert candidate_error < 2e-4 and delta_error < 2e-4
                    final_layer_error = float((reference-final).abs().max()) if layer == 29 else None
                    if layer == 29: assert final_layer_error < 2e-4, final_layer_error
                    cases.append({'kind': kind, 'layer': layer, 'step': index, 'max_candidate_error': candidate_error,
                                  'max_independent_delta_error': delta_error, 'final_layer_all_vocab_max_error': final_layer_error})
            del oracle, captured

# A test-only controlled record exception must release either observation hook.
for site, selection in [('attention', {'kind': 'attention', 'head': 0}), ('block', {'kind': 'logit_lens'})]:
    module = model.model.layers[7].self_attn if site == 'attention' else model.model.layers[7]
    prior = len(module._forward_hooks)
    def fail(event):
        if event['type'] == 'step': raise RuntimeError('controlled observation record failure')
    try: generate(torch, tokenizer, model, prompt, 1, 7, fail, activation_site=site, observation=selection)
    except RuntimeError: pass
    else: raise AssertionError('Expected controlled observer failure')
    assert len(module._forward_hooks) == prior
verify_model(directory)
print(json.dumps({'status': 'PASS', 'fixture': 'capital-france-public-synthetic', 'attention_backend': model.config._attn_implementation,
                  'cases': cases, 'on_off_exact_token_logit_parity': True, 'controlled_failure_hook_cleanup': True,
                  'disk_hashes_unchanged': True, 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}, indent=2))
