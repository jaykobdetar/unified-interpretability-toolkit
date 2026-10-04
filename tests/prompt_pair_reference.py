#!/usr/bin/env python3
"""Prepared bounded real-model two-prefill oracle; do not run without the slot."""
import json
import os
from pathlib import Path
import resource
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from inference_worker import load_engine, CAPTURE_SITES
from inference_edits import SOURCE_MODEL, verified_parameters
from inference_prompt_pair import token_preview, validate_request, validate_positions, run
from live_inference import verify_model

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (3*1024**3, 3*1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
directory = Path(sys.argv[1]);verify_model(directory)
from tokenizers import Tokenizer
raw_tokenizer = Tokenizer.from_file(str(directory/'tokenizer.json'))
torch, tokenizer, model = load_engine(directory);verified_parameters(model)
prompt = 'The capital of France is'  # Public synthetic fixture.
other = 'Paris is the capital of France.'
cases = []
for layer, site, prompts in [(0, 'block', [prompt, prompt]), (7, 'attention', [prompt, other]), (29, 'mlp', [prompt, other])]:
    preview = token_preview(raw_tokenizer, prompts)
    ids = [[t['id'] for t in ts] for ts in preview['tokens']]
    positions = [{'a': len(ids[0])-1, 'b': len(ids[1])-1}, {'a': 1, 'b': 1}]
    request = validate_request({'mode': 'prompt_pair', 'source_model': SOURCE_MODEL, 'prompts': prompts,
                                'layer': layer, 'activation_site': site, 'positions': positions, 'preview_digest': preview['digest']})
    validate_positions(request, preview)
    events = [];run(torch, tokenizer, model, request, preview, CAPTURE_SITES, events.append)
    steps = [e for e in events if e['type']=='step'];assert len(steps) == 2
    assert events[-1]['record_count'] == 2 and 'generated_tokens' not in events[-1]
    references = []
    for branch in range(2):
        residuals = {};block = model.model.layers[layer]
        def before(_module, args): residuals['before'] = args[0].detach().clone()
        def after_attn(_module, args): residuals['after_attention'] = args[0].detach().clone()
        def after(_module, _args, output): residuals['after'] = (output[0] if isinstance(output, tuple) else output).detach().clone()
        hooks = [block.register_forward_pre_hook(before), block.post_attention_layernorm.register_forward_pre_hook(after_attn), block.register_forward_hook(after)]
        try:
            with torch.inference_mode(): result = model(torch.tensor([ids[branch]]), use_cache=False, output_hidden_states=True, logits_to_keep=1)
        finally:
            for hook in hooks: hook.remove()
        reference = (result.hidden_states[layer+1] if site=='block' else residuals['after_attention']-residuals['before'] if site=='attention' else residuals['after']-residuals['after_attention'])
        references.append([reference[0,p['a' if branch==0 else 'b']].clone() for p in positions]);del result,residuals
    max_vector_error = max_metric_error = 0.
    for i, step in enumerate(steps):
        p = step['prompt_pair']
        a, b = [torch.tensor(p[k]['activation'], dtype=torch.float64) for k in ('a', 'b')]
        delta = b-a
        assert torch.equal(torch.tensor(step['activation'], dtype=torch.float64), delta)
        for side, values in enumerate((a,b)):
            error = float((values-references[side][i].double()).abs().max());max_vector_error=max(max_vector_error,error);assert error<2e-4
        na, nb, nd = [float(torch.linalg.vector_norm(v)) for v in (a,b,delta)]
        cos = float(torch.dot(a,b)/(na*nb)) if na and nb else None
        expected = {'a_l2':na,'b_l2':nb,'delta_l2':nd,'cosine':cos}
        for key, value in expected.items():
            if value is None: assert p['metrics'][key] is None
            else:
                error=abs(value-p['metrics'][key]);max_metric_error=max(max_metric_error,error);assert error<1e-8
        assert p['token_equal'] == (ids[0][positions[i]['a']]==ids[1][positions[i]['b']])
        assert p['prefix_equal'] == (ids[0][:positions[i]['a']+1]==ids[1][:positions[i]['b']+1])
        if prompts[0]==prompts[1]:assert p['a']['activation']==p['b']['activation'] and not any(step['activation'])
    cases.append({'layer':layer,'site':site,'prompt_lengths':[len(x) for x in ids], 'pairs':positions,
                  'same_prompt_exact_identity':prompts[0]==prompts[1], 'max_independent_vector_error':max_vector_error,
                  'max_independent_metric_error':max_metric_error})
verify_model(directory)
print(json.dumps({'status':'PASS','fixture':'public capital-country prompts','cases':cases,'disk_hashes_unchanged':True,
                  'scope':'Three selected layer/site cases, two pairs each; no all-layer coverage',
                  'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024},indent=2))
