#!/usr/bin/env python3
"""Prepared guarded real-model full-layer sweep and independent concatenated-head oracle.
Run only after the exclusive heavy slot is granted. No user prompts or network.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from inference_architecture import ARCH
from inference_worker import load_engine,generate,configure_worker_limits
from inference_edits import SOURCE_MODEL, verified_parameters
from inference_sweep import build_plan,run,temporary_edits
from live_inference import verify_model

os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
configure_worker_limits()
started=time.monotonic();directory=Path(sys.argv[1]);verify_model(directory)
torch,tokenizer,model=load_engine(directory)
parameters=verified_parameters(model)

def fingerprint():
    digest=hashlib.sha256()
    # Stream existing CPU buffers: no second full model snapshot or pickle.
    for name,param in model.named_parameters():
        digest.update(name.encode());data=memoryview(param.detach().numpy()).cast('B')
        for offset in range(0,len(data),1024*1024):digest.update(data[offset:offset+1024*1024])
    return digest.hexdigest()

original=fingerprint()
request={'mode':'sweep','source_model':SOURCE_MODEL,'prompts':['The capital of France is'],
         'targets':[{'kind':'layer_heads','layer':0}],'operation':'zero','seed':7,'capture_layer':0,'activation_site':'attention'}
plan=build_plan(request,False);request['plan_digest']=plan['digest']
events=[]
run(torch,tokenizer,model,request,plan,started+120,generate,events.append,cpu_deadline=90)
steps=[e for e in events if e['type']=='step']
assert events[-1]['type']=='sweep_done'
assert fingerprint()==original
if events[-1]['status']!='complete':
    print(json.dumps({'status':'PARTIAL','coverage':events[-1]['coverage'],'restored_full_parameter_fingerprint':True,'remaining_checks':'independent head oracle not run'},indent=2))
    raise SystemExit(2)
assert len(steps)==19 and all(s['sweep']['restoration_verified'] for s in steps)
assert events[-1]['coverage']['unfinished_heads']==[]
ids=torch.tensor([tokenizer.encode(request['prompts'][0],add_special_tokens=True)])
oracles=[]
# GQA group edge, next group, and final query head share K/V in groups of 3.
for head in (2,3,8):
    if time.monotonic()-started>100 or time.process_time()>80:
        print(json.dumps({'status':'PARTIAL','sweep_complete':True,'oracles':oracles,'remaining_checks':'oracle total-budget margin'},indent=2));raise SystemExit(2)
    start,end=head*ARCH['head_dim'],(head+1)*ARCH['head_dim']
    projection=model.model.layers[0].self_attn.o_proj
    def remove_input(_module,args):
        hidden=args[0].clone();hidden[...,start:end]=0
        return (hidden,*args[1:])
    hook=projection.register_forward_pre_hook(remove_input)
    try:
        with torch.inference_mode():reference=model(ids,use_cache=False,logits_to_keep=1).logits[0,-1].clone()
    finally:hook.remove()
    case=plan['cases'][1+2*head]
    with temporary_edits(torch,model,case['edits'],SOURCE_MODEL):
        with torch.inference_mode():actual=model(ids,use_cache=False,logits_to_keep=1).logits[0,-1].clone()
    error=float((actual-reference).abs().max())
    assert error<=2e-5,(head,error)
    oracles.append({'query_head':head,'kv_head':head//ARCH['queries_per_kv'],'max_logit_error':error})
assert fingerprint()==original
verify_model(directory)
print(json.dumps({'status':'PASS','records':len(steps),'prefills_in_sweep':plan['prefills'],
    'coverage':events[-1]['coverage'],'oracle':oracles,'restored_full_parameter_fingerprint':True,
    'disk_hashes_unchanged':True,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time(),
    'limits':'one process, one CPU, one total 120s/90 CPU s budget; cross-session residency untested'},indent=2))
