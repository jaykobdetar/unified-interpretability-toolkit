#!/usr/bin/env python3
"""Opt-in real-model parity; never downloads a model or runtime."""

import json
import os
from pathlib import Path
import resource
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from inference_worker import generate, load_engine

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
torch, tokenizer, model = load_engine(Path(sys.argv[1]))
from transformers import AutoTokenizer

reference_tokenizer = AutoTokenizer.from_pretrained(
    sys.argv[1], local_files_only=True, trust_remote_code=False
)
results = []
with torch.inference_mode():
    for prompt, layer in [
        ("The capital of France is", 0),
        ("1, 2, 3,", 19),
        ("Hello world", 29),
    ]:
        ids = reference_tokenizer.encode(prompt, add_special_tokens=True)
        assert ids == tokenizer.encode(
            prompt, add_special_tokens=True
        ), "Tokenizer mismatch"
        records = []
        generate(torch, tokenizer, model, prompt, 4, layer, records.append)
        steps = [r for r in records if r["type"] == "step"]
        reference = model.generate(
            torch.tensor([ids]),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
            do_sample=False,
            max_new_tokens=4,
            pad_token_id=model.config.eos_token_id,
        ).tolist()[0][len(ids) :]
        assert reference == [r["token_id"] for r in steps], "HF generate mismatch"
        prefix = list(ids)
        errors = []
        for step in steps:
            out = model(
                torch.tensor([prefix]), use_cache=False, output_hidden_states=True
            )
            assert step["position"] == len(prefix) - 1
            assert step["input_token_id"] == prefix[-1]
            assert step["token_id"] == int(out.logits[0, -1].argmax())
            expected = out.hidden_states[layer + 1][0, -1]
            captured = torch.tensor(step["activation"])
            if layer == model.config.num_hidden_layers - 1:
                captured = model.model.norm(
                    captured
                )  # HF final hidden state includes final RMSNorm.
            err = float((captured - expected).abs().max())
            assert err < 2e-4, (layer, step["index"], err)
            for top in step["top_logits"]:
                assert abs(top["value"] - float(out.logits[0, -1, top["id"]])) < 2e-4
            errors.append(err)
            prefix.append(step["token_id"])
        results.append(
            {
                "prompt": prompt,
                "layer": layer,
                "prompt_ids": ids,
                "generated_ids": reference,
                "text": steps[-1]["generated_text"],
                "activation_max_errors": errors,
                "compute_ms": steps[-1]["compute_total_ms"],
            }
        )
    # Exercise an actual EOS return with a deterministic official model boundary.
    saved = model.config.eos_token_id
    model.config.eos_token_id = results[0]["generated_ids"][0]
    events = []
    generate(torch, tokenizer, model, results[0]["prompt"], 4, 0, events.append)
    model.config.eos_token_id = saved
    assert events[-1]["reason"] == "eos" and events[-1]["generated_tokens"] == 1
    for prompt in ["", "word " * 200]:
        try:
            generate(torch, tokenizer, model, prompt, 4, 0, lambda _: None)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid prompt accepted")
print(
    json.dumps(
        {
            "status": "PASS",
            "reference": "Transformers generate plus uncached full-prefix forward",
            "torch": torch.__version__,
            "prompts": results,
            "eos_test": "controlled EOS id override, not naturally occurring EOS",
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        indent=2,
    )
)
