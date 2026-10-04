#!/usr/bin/env python3
"""One bounded offline inference session; JSON lines on stdin/stdout, no HTTP."""
import json
import os
from pathlib import Path
import resource
import sys
import time

# Set before importing numerical libraries, including when imported by reference tests.
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
                  CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')

MAX_PROMPT = 128
MAX_NEW = 32
from inference_architecture import ARCH, WIDTH, LAYERS, HEADS, KV_HEADS, HEAD_DIM, VOCAB, CAPTURE_SITES, verify_attention_layout
from inference_observations import ATTENTION_SEMANTICS, lens_record, validate_observation, validate_record



def emit(record):
    print(json.dumps(record, allow_nan=False, separators=(',', ':')), flush=True)


def load_engine(directory):
    import torch
    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    # Explicit built-in implementation, no AutoModel, remote Python, or pickle.
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(directory / 'tokenizer.json'))
    model = LlamaForCausalLM.from_pretrained(
        str(directory), local_files_only=True, use_safetensors=True,
        dtype=torch.float32, attn_implementation='eager').eval()
    from inference_edits import verified_parameters
    verified_parameters(model)
    model._atlas_verified_layout = verify_attention_layout(model)
    return torch, tokenizer, model


def generate(torch, tokenizer, model, prompt, limit, layer, record=emit, capture_scores=None, activation_site='block', observation=None):
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    if not 1 <= len(ids) <= MAX_PROMPT:
        raise ValueError(f'Prompt must encode to 1–{MAX_PROMPT} tokens; got {len(ids)}')
    if type(limit) is not int or not 1 <= limit <= MAX_NEW or type(layer) is not int or not 0 <= layer < LAYERS:
        raise ValueError('Invalid token limit or activation layer')
    if type(activation_site) is not str or activation_site not in CAPTURE_SITES:
        raise ValueError('Choose activation site block, attention, or mlp')
    if observation is not None:
        observation = validate_observation(observation, activation_site)
        config = model.config
        if (config._attn_implementation != 'eager' or model.training or config.num_attention_heads != HEADS or
                config.num_key_value_heads != KV_HEADS or config.hidden_size != WIDTH or config.vocab_size != VOCAB or
                model.model.layers[layer].self_attn.head_dim != HEAD_DIM):
            raise ValueError('Observation requires the pinned eager SmolLM2 architecture in eval mode')
    record({'type': 'prefill', 'prompt_tokens': len(ids), 'prompt_ids': ids,
            'layer': layer, 'activation_site': activation_site, 'seed': 0, 'sampling': 'greedy', 'dtype': 'float32'})
    selected, observed = [], {}

    def capture(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        selected[:] = hidden[0, -1, :].detach().float().tolist()
        if observation and observation['kind'] == 'attention':
            weights = output[1] if isinstance(output, tuple) and len(output) == 2 else None
            if weights is None or weights.ndim != 4 or weights.shape[0] != 1 or weights.shape[1] != HEADS or not 1 <= weights.shape[3] <= 160:
                raise ValueError('Selected eager attention probabilities unavailable')
            observed['probabilities'] = weights[0, observation['head'], -1, :].detach().float().tolist()
        elif observation:
            observed['residual'] = hidden[0, -1, :].detach().clone()

    block = model.model.layers[layer]
    target = block if activation_site == 'block' else block.self_attn if activation_site == 'attention' else block.mlp
    hook = target.register_forward_hook(capture)
    past = None
    tokens = []
    inputs = torch.tensor([ids], dtype=torch.long)
    compute_total = 0.0
    try:
        with torch.inference_mode():
            for index in range(limit):
                start = time.perf_counter()
                out = model(input_ids=inputs, past_key_values=past, use_cache=True,
                            logits_to_keep=1)
                past = out.past_key_values
                logits = out.logits[0, -1, :]
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError('Nonfinite inference logits')
                if capture_scores is not None:
                    capture_scores(logits.detach().clone())
                position = len(ids) - 1 + index
                extra = {}
                if observation and observation['kind'] == 'attention':
                    extra['attention'] = {'layer': layer, 'query_head': observation['head'], 'kv_head': observation['head'] // (HEADS//KV_HEADS),
                        'head_dim': HEAD_DIM, 'query_position': position, 'key_positions': list(range(position + 1)),
                        'key_token_ids': ids + tokens, 'probabilities': observed['probabilities'], 'semantics': ATTENTION_SEMANTICS}
                elif observation:
                    extra['logit_lens'] = lens_record(torch, tokenizer, observed.pop('residual'), model, logits, layer, position)
                token = int(torch.argmax(logits))
                top = torch.topk(logits, 5)
                compute_ms = (time.perf_counter() - start) * 1000
                compute_total += compute_ms
                tokens.append(token)
                eos = token == model.config.eos_token_id
                step = {'type': 'step', 'index': index, 'phase': 'prefill' if index == 0 else 'decode',
                        'position': len(ids) - 1 + index, 'input_token_id': int(inputs[0, -1]),
                        'token_id': token, 'token_piece': tokenizer.decode([token], skip_special_tokens=False),
                        'generated_text': tokenizer.decode(tokens, skip_special_tokens=False),
                        'compute_ms': compute_ms, 'compute_total_ms': compute_total,
                        'layer': layer, 'activation': list(selected),
                        'activation_site': activation_site,
                        'activation_kind': CAPTURE_SITES[activation_site] + ', last consumed position',
                        'top_logits': [{'id': int(t), 'value': float(v)} for t, v in zip(top.indices, top.values)],
                        'eos': eos, **extra}
                validate_record(step, observation, layer)
                record(step)
                if eos:
                    break
                inputs = torch.tensor([[token]], dtype=torch.long)
        record({'type': 'done', 'reason': 'eos' if eos else 'token_limit',
                'generated_tokens': len(tokens), 'compute_total_ms': compute_total})
    finally:
        hook.remove()


def paired_step(index, baseline, edited, baseline_scores, edited_scores, tokenizer):
    """Union scores have both values; prefix identity is based on consumed IDs."""
    left = baseline[index] if index < len(baseline) else None
    right = edited[index] if index < len(edited) else None
    left_ids = [step['token_id'] for step in baseline[:index]]
    right_ids = [step['token_id'] for step in edited[:index]]
    alignment = ('branch_ended' if left is None or right is None else
                 'matched_prefix' if left_ids == right_ids else 'different_prefix')
    candidates = sorted({entry['id'] for step in (left, right) if step for entry in step['top_logits']})
    scores = []
    for token in candidates:
        a = float(baseline_scores[index][token]) if left else None
        b = float(edited_scores[index][token]) if right else None
        scores.append({'id': token, 'piece': tokenizer.decode([token], skip_special_tokens=False),
                       'baseline_logit': a, 'edited_logit': b,
                       'delta': b - a if a is not None and b is not None else None})
    def summary(step, prefix):
        return ({key: step[key] for key in ('token_id', 'token_piece', 'generated_text', 'eos', 'compute_ms', 'compute_total_ms')}
                | {'generated_ids': prefix + [step['token_id']]} if step else None)
    return {**(right or left), 'type': 'step', 'index': index,
            'baseline': summary(left, left_ids), 'edited': summary(right, right_ids),
            'alignment': alignment, 'score_kind': 'raw FP32 logits',
            'activation_branch': 'edited' if right else 'baseline', 'candidates': scores}


def compare(torch, tokenizer, model, request, record=emit):
    # This model belongs exclusively to one disposable worker. No later request
    # can reuse edited parameters. Cancellation/failure destroys this process.
    from inference_edits import apply_edits, validate_edits, verified_parameters
    edits = validate_edits(request['edits'], request['source_model'])
    verified_parameters(model)
    baseline, baseline_scores, edited, edited_scores = [], [], [], []
    def collect(target):
        def accept(event):
            if event['type'] == 'step':
                target.append(event)
            elif event['type'] == 'prefill':
                record(event)
        return accept
    record({'type': 'prefill', 'comparison_phase': 'baseline', 'edits': edits})
    generate(torch, tokenizer, model, request['prompt'], request['max_new_tokens'], request['layer'],
             collect(baseline), baseline_scores.append, activation_site=request.get('activation_site', 'block'), observation=request.get('observation'))
    apply_edits(torch, model, edits, request['source_model'])
    record({'type': 'prefill', 'comparison_phase': 'edited'})
    # Fresh generate invocation starts from prompt with no baseline KV cache.
    def accept_edited(event):
        collect(edited)(event)
        if event['type'] == 'step':
            record(paired_step(event['index'], baseline, edited, baseline_scores, edited_scores, tokenizer))
    generate(torch, tokenizer, model, request['prompt'], request['max_new_tokens'], request['layer'],
             accept_edited, edited_scores.append, activation_site=request.get('activation_site', 'block'), observation=request.get('observation'))
    for index in range(len(edited), len(baseline)):
        record(paired_step(index, baseline, edited, baseline_scores, edited_scores, tokenizer))
    def result(steps):
        return {'generated_ids': [step['token_id'] for step in steps],
                'generated_text': steps[-1]['generated_text'],
                'reason': 'eos' if steps[-1]['eos'] else 'token_limit'}
    record({'type': 'done', 'generated_tokens': max(len(baseline), len(edited)),
            'comparison_phase': 'complete', 'baseline': result(baseline), 'edited': result(edited),
            'reason': 'comparison_complete',
            'compute_total_ms': baseline[-1]['compute_total_ms'] + edited[-1]['compute_total_ms']})


def configure_worker_limits(cpu_seconds=90):
    if type(cpu_seconds) is not int or not 1 <= cpu_seconds <= 90:
        raise ValueError('Invalid remaining worker CPU allowance')
    for kind, limit in ((resource.RLIMIT_AS, 3*1024**3), (resource.RLIMIT_CPU, cpu_seconds)):
        inherited = resource.getrlimit(kind)
        limit = min([limit] + [v for v in inherited if v != resource.RLIM_INFINITY])
        resource.setrlimit(kind, (limit, limit))


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(10)
    if len(sys.argv) not in (2,4):
        raise ValueError('Invalid worker budget arguments')
    cpu_allowance = int(sys.argv[3]) if len(sys.argv)==4 else 90
    configure_worker_limits(cpu_allowance)
    # Entire request is bounded by coordinator and again here.
    request = json.loads(sys.stdin.buffer.readline(8193))
    from live_inference import verify_model, SweepAdmissionBudget
    directory = Path(sys.argv[1])
    import inference_prompt_pair as prompt_pair
    import inference_sweep as sweep
    sweep_plan, sweep_deadline = None, None
    if request.get('mode') == 'sweep':
        sweep_plan = sweep.build_plan(request)
        import math
        if len(sys.argv)!=4:
            raise ValueError('Sweep requires the coordinator total-job deadline')
        sweep_deadline = float(sys.argv[2])
        if not math.isfinite(sweep_deadline) or not 0 < sweep_deadline-time.monotonic() <= sweep.WALL_SECONDS:
            raise ValueError('Invalid or exhausted total sweep deadline')
        budget = SweepAdmissionBudget(sweep_deadline, cpu_allowance)
        with budget.verification():
            verify_model(directory, check=budget.check)
        from tokenizers import Tokenizer
        preview_tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        if any(not 1 <= len(preview_tokenizer.encode(p,add_special_tokens=True).ids) <= MAX_PROMPT for p in request['prompts']):
            raise ValueError('Each sweep prompt must encode to 1–128 tokens')
        del preview_tokenizer
    else:
        if len(sys.argv)!=2:
            raise ValueError('Budget arguments require a sweep request')
        verify_model(directory)
    pair_preview = None
    if request.get('mode') in prompt_pair.MODES:
        request = prompt_pair.validate_request(request)
        from tokenizers import Tokenizer
        preview_tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        pair_preview = prompt_pair.token_preview(preview_tokenizer, request['prompts'])
        del preview_tokenizer
        if request['mode'] == 'prompt_pair_preview':
            emit({'type': 'preview_done', 'preview': pair_preview, 'reason': 'token_preview'})
            return
        prompt_pair.validate_positions(request, pair_preview)
    started = time.perf_counter()
    torch, tokenizer, model = load_engine(directory)
    import platform
    import transformers, tokenizers, safetensors
    emit({'type': 'loaded', 'load_ms': (time.perf_counter() - started) * 1000,
          'head_layout': model._atlas_verified_layout,
          'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                      'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
                      'safetensors': safetensors.__version__, 'platform': sys.platform, 'machine': platform.machine(),
                      'device': 'cpu', 'dtype': 'float32', 'sampling': 'none' if request.get('mode') in ('prompt_pair','sweep') else 'greedy', 'seed': 0,
                      'deterministic_algorithms': True, 'numerical_threads': 1, 'attention_backend': model.config._attn_implementation}})
    if request.get('mode') == 'sweep':
        sweep.run(torch,tokenizer,model,request,sweep_plan,sweep_deadline,generate,emit,cpu_deadline=cpu_allowance)
    elif request.get('mode') == 'prompt_pair':
        from inference_edits import verified_parameters
        verified_parameters(model)
        prompt_pair.run(torch, tokenizer, model, request, pair_preview, CAPTURE_SITES, emit)
    elif 'edits' in request:
        compare(torch, tokenizer, model, request)
    else:
        generate(torch, tokenizer, model, request['prompt'], request['max_new_tokens'], request['layer'],
                 activation_site=request.get('activation_site', 'block'), observation=request.get('observation'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Never echo prompt text or model paths in the HTTP-visible error.
        emit({'type': 'error', 'error': str(exc) if isinstance(exc, ValueError) else type(exc).__name__})
        raise SystemExit(1)
