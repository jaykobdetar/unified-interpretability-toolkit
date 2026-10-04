#!/usr/bin/env python3
"""Opt-in one-model selected-site oracle; run only in an assigned heavy slot."""
import json
import os
from pathlib import Path
import resource
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from inference_worker import generate,load_engine
from live_inference import verify_model
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(90,90))
directory=Path(sys.argv[1]);verify_model(directory)
torch,tokenizer,model=load_engine(directory)
prompt='The capital of France is'  # Explicit public synthetic fixture.
ids=tokenizer.encode(prompt,add_special_tokens=True)
results=[]
for layer in (0,7,29):
    block=model.model.layers[layer]
    residuals={}
    def before(_module,args):residuals['input']=args[0][0,-1].detach().clone()
    def before_mlp(_module,args):residuals['post_attention']=args[0][0,-1].detach().clone()
    def after(_module,_args,value):residuals['output']=(value[0] if isinstance(value,tuple) else value)[0,-1].detach().clone()
    hooks=[block.register_forward_pre_hook(before),block.post_attention_layernorm.register_forward_pre_hook(before_mlp),block.register_forward_hook(after)]
    try:
        with torch.inference_mode():oracle=model(torch.tensor([ids]),use_cache=False,output_hidden_states=True)
    finally:
        for hook in hooks:hook.remove()
    expected={'attention':residuals['post_attention']-residuals['input'], 'mlp':residuals['output']-residuals['post_attention']}
    baseline=None
    for site in ('block','attention','mlp'):
        events=[];generate(torch,tokenizer,model,prompt,2,layer,events.append,activation_site=site)
        steps=[r for r in events if r['type']=='step']
        signature=[(s['token_id'],s['top_logits']) for s in steps]
        if baseline is None:baseline=signature
        else:assert signature==baseline,'Observation hook changed generation'
        captured=torch.tensor(steps[0]['activation'])
        if site=='block':
            if layer==29:
                with torch.inference_mode():captured=model.model.norm(captured)
            reference=oracle.hidden_states[layer+1][0,-1]
        else:reference=expected[site]
        error=float((captured-reference).abs().max())
        assert error<2e-4,(layer,site,error)
        assert steps[0]['activation_site']==site and steps[0]['position']==len(ids)-1
        results.append({'layer':layer,'site':site,'max_abs_oracle_error':error,'generated_ids':[s['token_id'] for s in steps]})
    del oracle
# A bounded test-only record failure must remove the observation hook.
module=model.model.layers[7].self_attn
prior=len(module._forward_hooks)
def fail_on_step(event):
    if event['type']=='step':raise RuntimeError('controlled record failure')
try:generate(torch,tokenizer,model,prompt,1,7,fail_on_step,activation_site='attention')
except RuntimeError:pass
else:raise AssertionError('Expected controlled test-only exception')
assert len(module._forward_hooks)==prior
verify_model(directory)
print(json.dumps({'status':'PASS','fixture':'capital-france-public-synthetic','reference':'uncached hidden states and independent residual-addition differences',
                  'cases':results,'hooks_observational':True,'failure_hook_cleanup':True,'disk_hashes_unchanged':True,
                  'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024},indent=2))
