"""Pure producer/consumer sweep archive compatibility; no model or NumPy imports."""

import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from inference_sweep import build_plan, coverage
from inference_edits import SOURCE_MODEL

ROOT = Path(__file__).resolve().parents[1]
script = r"""
const assert=require('node:assert/strict'),codec=require('./web/inference.js'),I=codec.AtlasExperimentImport;
let raw='';process.stdin.on('data',x=>raw+=x);process.stdin.on('end',()=>{
 const {request,plan,coverage}=JSON.parse(raw),first=plan.cases[0];
 const step={index:0,activation:Array(576).fill(0),layer:7,activation_site:'attention',input_token_id:42,sweep:{record_id:'empty/prompt-1',case_id:first.id,role:first.role,prompt_index:0,selected_cells:0,changed_cells:0,parameter_delta_l2:0,restoration_verified:true,metrics:{logit_delta_rms:0,logit_delta_max_abs:0,softmax_total_variation:0,baseline_argmax_id:1,baseline_argmax_logit_delta:0,edited_argmax_id:1,context:'matched fixed original prompt',semantics:'prompt-set sensitivity; no inferred causal purpose or general head importance'},candidates:[{id:1,piece:'fixture',baseline_logit:1,edited_logit:1,delta:0}]}};
 const r=codec.experimentRecord(request,{status:'time_limit',worker_alive:false,steps:[step],details:{sweep_plan:plan,coverage}},{includePrompt:true});
 for(const privatePrompt of [true,false]){const input=privatePrompt?r:codec.redactExperiment(r);assert.deepEqual(I.read(JSON.stringify(input),{...codec,includePrompt:privatePrompt}),[input]);}
 for(const mutation of [r=>r.request.seed=-1,r=>r.request.targets[0].layer=30,r=>r.sweep_plan.records++,r=>r.sweep_plan.limits.records=-1,r=>r.sweep_plan.cases[1].edits[0].end=999999,r=>r.sweep_plan.architecture.width=577,r=>r.summary.coverage.completed_ids=[],r=>r.summary.coverage.complete=true,r=>r.steps[0].sweep.prompt_index=1,r=>r.steps[0].sweep.restoration_verified=false,r=>r.steps[0].sweep.metrics.logit_delta_rms=-1,r=>r.steps[0].sweep.candidates[0].delta=2]){const bad=JSON.parse(JSON.stringify(r));mutation(bad);assert.throws(()=>I.read(JSON.stringify(bad),codec));}
 const altered=JSON.parse(JSON.stringify(r));altered.steps[0].sweep.metrics.hidden=1;assert.throws(()=>I.read(JSON.stringify(altered),codec));
 console.log('PASS: actual pure sweep producer schema imports, redacts, preserves partial/unrun coverage, and rejects unknown nested metrics.');
});"""
for targets in [
    [{"kind": "query_head", "layer": 0, "head": 1}],
    [{"kind": "layer_heads", "layer": 0}],
]:
    request = {
        "mode": "sweep",
        "source_model": SOURCE_MODEL,
        "prompts": ["synthetic fixture"],
        "targets": targets,
        "operation": "zero",
        "seed": 42,
        "capture_layer": 7,
        "activation_site": "attention",
    }
    plan = build_plan(request, False)
    request["plan_digest"] = plan["digest"]
    subprocess.run(
        ["node", "-e", script],
        input=json.dumps(
            {"request": request, "plan": plan, "coverage": coverage(plan, 1)}
        ),
        text=True,
        cwd=ROOT,
        check=True,
        timeout=10,
    )
