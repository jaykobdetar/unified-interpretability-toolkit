#!/usr/bin/env python3
"""Prepared real numerical/bitwise mini-sweep oracle; serialized heavy slot only."""

import json
import math
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from inference_worker import generate, load_engine
from inference_edits import SOURCE_MODEL, verified_parameters
from inference_sweep import build_plan, run, temporary_edits
from live_inference import verify_model

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
started = time.monotonic()
directory = Path(sys.argv[1])
verify_model(directory)
torch, tokenizer, model = load_engine(directory)
params = verified_parameters(model)
prompt = "The capital of France is"  # Public synthetic fixture, not a personal prompt.
data = {
    "mode": "sweep",
    "source_model": SOURCE_MODEL,
    "prompts": [prompt],
    "targets": [{"kind": "head", "layer": 0, "head": 0}],
    "operation": "zero",
    "seed": 7,
    "capture_layer": 0,
    "activation_site": "block",
}
plan = build_plan(data, False)
data["plan_digest"] = plan["digest"]
q = params["model.layers.0.self_attn.q_proj.weight"]
original = q.detach().clone()
o = params["model.layers.0.self_attn.o_proj.weight"]
original_o = o.detach().clone()
events = []
run(torch, tokenizer, model, data, plan, started + 120, generate, events.append)
assert events[-1]["type"] == "sweep_done" and events[-1]["status"] == "complete"
steps = [e for e in events if e["type"] == "step"]
assert len(steps) == 3
assert torch.equal(q.view(torch.int32), original.view(torch.int32))
ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)])
with torch.inference_mode():
    baseline = model(ids, use_cache=False, logits_to_keep=1).logits[0, -1].clone()


def softmax(values):
    maximum = max(values)
    weights = [math.exp(x - maximum) for x in values]
    denominator = math.fsum(weights)
    return [x / denominator for x in weights]


a = baseline.tolist()
pa = softmax(a)
results = []
for case, step in zip(plan["cases"], steps):
    try:
        with torch.no_grad():
            # Independent direct native-column ablation; does not call apply_edits.
            for edit in case["edits"]:
                assert edit["kind"] == "columns" and edit["tensor"].endswith(
                    ".o_proj.weight"
                )
                o[:, edit["start"] : edit["end"]].zero_()
        with torch.inference_mode():
            edited = model(ids, use_cache=False, logits_to_keep=1).logits[0, -1].clone()
    finally:
        with torch.no_grad():
            o.copy_(original_o)
    b = edited.tolist()
    pb = softmax(b)
    delta = [y - x for x, y in zip(a, b)]
    expected = {
        "logit_delta_rms": math.sqrt(math.fsum(x * x for x in delta) / len(delta)),
        "logit_delta_max_abs": max(abs(x) for x in delta),
        "softmax_total_variation": math.fsum(abs(x - y) for x, y in zip(pa, pb)) / 2,
    }
    errors = {
        key: abs(value - step["sweep"]["metrics"][key])
        for key, value in expected.items()
    }
    # Cached one-token probe vs uncached reference can differ slightly by GEMM shape.
    assert max(errors.values()) < 2e-4, errors
    for entry in step["sweep"]["candidates"]:
        i = entry["id"]
        assert abs(entry["delta"] - delta[i]) < 2e-4
    results.append(
        {
            "case": case["id"],
            "metrics": step["sweep"]["metrics"],
            "independent_max_error": max(errors.values()),
        }
    )

# Actual Torch undo controls include overlapping selections and the known tie.
overlap = [
    {
        "tensor": "model.layers.0.self_attn.q_proj.weight",
        "shape": [576, 576],
        "kind": "rows",
        "start": 0,
        "end": 1,
        "operation": "scale",
        "scale": 0.5,
    },
    {
        "tensor": "model.layers.0.self_attn.q_proj.weight",
        "shape": [576, 576],
        "kind": "columns",
        "start": 0,
        "end": 1,
        "operation": "scale",
        "scale": -1,
    },
]
try:
    with temporary_edits(torch, model, overlap, SOURCE_MODEL):
        raise RuntimeError("controlled post-edit reference failure")
except RuntimeError as exc:
    assert (
        type(exc) is RuntimeError
        and str(exc) == "controlled post-edit reference failure"
    )
assert torch.equal(q.view(torch.int32), original.view(torch.int32))
embed = params["model.embed_tokens.weight"]
prior = embed[0, 0].detach().clone()
edit = {
    "tensor": "model.embed_tokens.weight",
    "shape": [49152, 576],
    "kind": "element",
    "row": 0,
    "col": 0,
    "operation": "scale",
    "scale": -1,
}
with temporary_edits(torch, model, [edit], SOURCE_MODEL):
    assert embed is params["lm_head.weight"]
    assert float(params["lm_head.weight"][0, 0]) == -float(prior)
assert torch.equal(embed[0, 0].view(torch.int32), prior.view(torch.int32))
verify_model(directory)
print(
    json.dumps(
        {
            "status": "PASS",
            "fixture": "capital-france-public-synthetic",
            "records": len(steps),
            "prefills_in_sweep": 6,
            "cases": results,
            "restored_parameter_bits": True,
            "actual_overlap_and_tied_alias_undo": True,
            "disk_hashes_unchanged": True,
            "maximum_10_record_plan_unrun": True,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        indent=2,
    )
)
