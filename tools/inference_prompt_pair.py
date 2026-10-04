"""Bounded two-prompt token preview and selected-position activation comparison."""
import hashlib
import json
import math
from inference_architecture import ARCH, WIDTH, LAYERS, HEADS, KV_HEADS, HEAD_DIM, VOCAB, CAPTURE_SITES
from inference_edits import SOURCE_MODEL, validate_edits

MODES = ('prompt_pair_preview', 'prompt_pair')
SHA256_HEX_LENGTH = 64


def validate_request(data):
    mode = data.get('mode')
    fields = {'mode', 'prompts', 'source_model'}
    if mode == 'prompt_pair':
        fields |= {'layer', 'activation_site', 'positions', 'preview_digest'}
    elif mode != 'prompt_pair_preview':
        raise ValueError('Unknown prompt-pair mode')
    if set(data) != fields:
        raise ValueError('Prompt-pair requests require exactly their declared fields')
    prompts = data['prompts']
    if type(prompts) is not list or len(prompts) != 2 or any(type(p) is not str or not p.strip() or len(p.encode('utf-8')) > 2048 for p in prompts):
        raise ValueError('Enter two nonempty prompts of at most 2048 UTF-8 bytes each')
    validate_edits([], data['source_model'])
    result = {'mode': mode, 'prompts': list(prompts), 'source_model': dict(SOURCE_MODEL)}
    if mode == 'prompt_pair':
        if type(data['layer']) is not int or not 0 <= data['layer'] < LAYERS or type(data['activation_site']) is not str or data['activation_site'] not in CAPTURE_SITES:
            raise ValueError(f'Select one native layer 0–{LAYERS-1} and capture output')
        positions = data['positions']
        if type(positions) is not list or not 1 <= len(positions) <= 8:
            raise ValueError('Select 1–8 explicit token position pairs')
        seen = set()
        for pair in positions:
            if type(pair) is not dict or set(pair) != {'a', 'b'} or any(type(pair[k]) is not int or not 0 <= pair[k] < 128 for k in ('a', 'b')):
                raise ValueError('Positions must be strict token integers 0–127')
            key = pair['a'], pair['b']
            if key in seen:
                raise ValueError('Duplicate position pair')
            seen.add(key)
        digest = data['preview_digest']
        if type(digest) is not str or len(digest) != SHA256_HEX_LENGTH or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Review the exact token preview before comparing prompts')
        result.update(layer=data['layer'], activation_site=data['activation_site'],
                      positions=[dict(p) for p in positions], preview_digest=digest)
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > 8192:
        raise ValueError('Complete prompt-pair request exceeds 8 KiB')
    return result


def digest_for(prompts, ids):
    value = {'source_model': SOURCE_MODEL, 'prompts': prompts, 'ids': ids}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def token_preview(tokenizer, prompts):
    # tokenizers.Tokenizer uses the pinned JSON's exact postprocessor, as the
    # existing PreTrainedTokenizerFast wrapper does with add_special_tokens=True.
    encodings = [tokenizer.encode(p, add_special_tokens=True).ids for p in prompts]
    if any(not 1 <= len(ids) <= 128 for ids in encodings):
        raise ValueError('Each prompt must encode to 1–128 tokens')
    tokens = [[{'position': i, 'id': token, 'piece': tokenizer.decode([token], skip_special_tokens=False)} for i, token in enumerate(ids)] for ids in encodings]
    value = {'digest': digest_for(prompts, encodings), 'tokens': tokens, 'model_loaded': False}
    validate_preview(value)
    return value


def validate_preview(value):
    if type(value) is not dict or set(value) != {'digest', 'tokens', 'model_loaded'} or value['model_loaded'] is not False:
        raise ValueError('Invalid tokenizer preview')
    digest = value['digest']
    if type(digest) is not str or len(digest) != SHA256_HEX_LENGTH or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid preview digest')
    if len(json.dumps(value, ensure_ascii=False).encode()) > 65536:
        raise ValueError('Token preview exceeds 64 KiB')
    arrays = value['tokens']
    if type(arrays) is not list or len(arrays) != 2:
        raise ValueError('Invalid preview branches')
    for tokens in arrays:
        if type(tokens) is not list or not 1 <= len(tokens) <= 128:
            raise ValueError('Invalid preview token count')
        for i, token in enumerate(tokens):
            if (type(token) is not dict or set(token) != {'position', 'id', 'piece'} or
                    type(token['position']) is not int or token['position'] != i or
                    type(token['id']) is not int or not 0 <= token['id'] < VOCAB or
                    type(token['piece']) is not str or len(token['piece'].encode('utf-8')) > 1024):
                raise ValueError('Invalid preview token')


def validate_positions(request, preview):
    if request['preview_digest'] != preview['digest']:
        raise ValueError('Prompts or tokenization changed; review a fresh preview')
    for pair in request['positions']:
        if any(pair[key] >= len(preview['tokens'][i]) for i, key in enumerate(('a', 'b'))):
            raise ValueError('Selected position exceeds its encoded prompt')


def difference(a, b):
    if any(type(v) is not list or len(v) != WIDTH or any(type(x) not in (int, float) or not math.isfinite(x) for x in v) for v in (a, b)):
        raise ValueError(f'Expected two finite {WIDTH}-value vectors')
    delta = [y-x for x, y in zip(a, b)]
    na, nb, nd = math.hypot(*a), math.hypot(*b), math.hypot(*delta)
    cosine = max(-1., min(1., math.fsum(x*y for x, y in zip(a, b)) / (na*nb))) if na and nb else None
    if not all(math.isfinite(x) for x in delta+[na, nb, nd]+([] if cosine is None else [cosine])):
        raise ValueError('Nonfinite activation difference/metric')
    return delta, {'a_l2': na, 'b_l2': nb, 'delta_l2': nd, 'cosine': cosine}


def validate_step(step, request):
    index = step.get('index')
    if type(index) is not int or not 0 <= index < len(request['positions']) or step.get('mode') != 'prompt_pair':
        raise ValueError('Invalid prompt-pair sequence')
    if type(step.get('layer')) is not int or step['layer'] != request['layer'] or step.get('activation_site') != request['activation_site']:
        raise ValueError('Prompt-pair capture differs from request')
    pair = step.get('prompt_pair')
    if type(pair) is not dict or set(pair) != {'a', 'b', 'token_equal', 'prefix_equal', 'metrics'}:
        raise ValueError('Invalid prompt-pair output')
    for key in ('a', 'b'):
        side = pair[key]
        if (type(side) is not dict or set(side) != {'position', 'token_id', 'token_piece', 'activation'} or
                type(side['position']) is not int or side['position'] != request['positions'][index][key] or
                type(side['token_id']) is not int or not 0 <= side['token_id'] < VOCAB or
                type(side['token_piece']) is not str or len(side['token_piece'].encode()) > 1024):
            raise ValueError('Invalid selected prompt token')
    if type(pair['token_equal']) is not bool or pair['token_equal'] != (pair['a']['token_id'] == pair['b']['token_id']) or type(pair['prefix_equal']) is not bool:
        raise ValueError('Invalid prompt alignment label')
    if pair['prefix_equal'] and (not pair['token_equal'] or pair['a']['position'] != pair['b']['position']):
        raise ValueError('Contradictory prefix alignment')
    delta, metrics = difference(pair['a']['activation'], pair['b']['activation'])
    actual = step.get('activation')
    reported = pair['metrics']
    if (type(actual) is not list or any(type(x) not in (int, float) or not math.isfinite(x) for x in actual) or
            type(reported) is not dict or set(reported) != set(metrics) or
            any((v is not None and (type(v) not in (int, float) or not math.isfinite(v))) for v in reported.values()) or
            actual != delta or reported != metrics):
        raise ValueError('Prompt difference/metrics disagree with captured values')


def run(torch, tokenizer, model, request, preview, capture_kinds, record):
    """Exactly two uncached prefills and one selected observational hook."""
    import time
    ids = [[t['id'] for t in tokens] for tokens in preview['tokens']]
    if ids != [tokenizer.encode(p, add_special_tokens=True) for p in request['prompts']]:
        raise ValueError('Tokenizer wrapper disagrees with reviewed preview')
    layer, site = request['layer'], request['activation_site']
    block = model.model.layers[layer]
    target = block if site == 'block' else block.self_attn if site == 'attention' else block.mlp
    captured = []
    started = time.perf_counter()
    for side, key in enumerate(('a', 'b')):
        record({'type': 'prefill', 'mode': 'prompt_pair', 'comparison_phase': 'prompt '+key.upper()})
        positions = sorted({pair[key] for pair in request['positions']})
        selected = {}
        def capture(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for position in positions:
                selected[position] = hidden[0, position].detach().float().tolist()
        hook = target.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                output = model.model(torch.tensor([ids[side]]), use_cache=False)
                del output
        finally:
            hook.remove()
        if set(selected) != set(positions):
            raise ValueError('Selected prompt activations missing')
        captured.append(selected)
    elapsed = (time.perf_counter()-started)*1000
    for index, positions in enumerate(request['positions']):
        a, b = (captured[i][positions[key]] for i, key in enumerate(('a', 'b')))
        delta, metrics = difference(a, b)
        pair = {'metrics': metrics, 'token_equal': ids[0][positions['a']] == ids[1][positions['b']],
                'prefix_equal': ids[0][:positions['a']+1] == ids[1][:positions['b']+1]}
        for i, key in enumerate(('a', 'b')):
            position = positions[key]
            pair[key] = {'position': position, 'token_id': ids[i][position], 'token_piece': preview['tokens'][i][position]['piece'],
                         'activation': captured[i][position]}
        step = {'type': 'step', 'index': index, 'mode': 'prompt_pair', 'layer': layer, 'activation_site': site,
                'activation_kind': capture_kinds[site]+'; B minus A, difference computed in binary64',
                'activation': delta, 'prompt_pair': pair, 'compute_ms': elapsed if index == 0 else 0., 'compute_total_ms': elapsed}
        validate_step(step, request);record(step)
    record({'type': 'done', 'mode': 'prompt_pair', 'record_count': len(request['positions']), 'reason': 'prompt_pair_complete', 'compute_total_ms': elapsed,
            'coverage': 'exactly the selected layer/site/position pairs; two uncached prefills, no generation'})
