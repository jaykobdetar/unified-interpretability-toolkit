"""Closed, bounded observational request/trace contracts; no numerical imports."""
import math
from inference_architecture import ARCH, WIDTH, LAYERS, HEADS, KV_HEADS, HEAD_DIM, VOCAB, CAPTURE_SITES

ATTENTION_SEMANTICS = 'selected query head, last consumed query; post-causal-mask softmax probabilities, eval dropout zero'
LENS_SEMANTICS = 'selected post-block residual through final RMSNorm and tied head; diagnostic readout, not early-exit inference or causal effect'


def schema():
    return {'kinds': ['attention', 'logit_lens'], 'query_heads': HEADS, 'kv_heads': KV_HEADS,
            'head_dim': HEAD_DIM, 'max_attention_keys': 160, 'max_lens_candidates': 10,
            'attention_scope': 'one layer/head, last consumed query per step',
            'attention_backend': 'eager', 'lens_score_kind': 'raw FP32 logits'}


def validate_observation(value, site):
    if type(value) is not dict:
        raise ValueError('Observation must be an object')
    kind = value.get('kind')
    if kind == 'attention':
        if set(value) != {'kind', 'head'} or type(value['head']) is not int or not 0 <= value['head'] < HEADS:
            raise ValueError(f'Select exactly one query head 0–{HEADS-1}')
        if site != 'attention':
            raise ValueError('Attention observation requires the attention capture site')
    elif kind == 'logit_lens':
        if set(value) != {'kind'} or site != 'block':
            raise ValueError('Logit lens requires the block capture site and no extra options')
    else:
        raise ValueError('Unknown observation kind')
    return dict(value)


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_record(step, selected=None, layer=None):
    """Accept only the requested observation attached to its actual activation."""
    present = [name for name in ('attention', 'logit_lens') if name in step]
    if selected is None:
        if present:
            raise ValueError('Unrequested observation')
        return
    if present != [selected['kind']] or type(step.get('layer')) is not int or step.get('layer') != layer:
        raise ValueError('Observation mode/layer differs from accepted request')
    kind = selected['kind']
    site = 'attention' if kind == 'attention' else 'block'
    if step.get('activation_site') != site or not _integer(step.get('position'), 0, 158):
        raise ValueError('Invalid observation site/position')
    if 'baseline' in step:
        expected = 'edited' if step.get('edited') is not None else 'baseline'
        if step.get('activation_branch') != expected:
            raise ValueError('Observation branch disagrees with activation')
    value = step[kind]
    if type(value) is not dict or type(value.get('layer')) is not int or value.get('layer') != layer:
        raise ValueError('Invalid observation record')
    if kind == 'attention':
        if set(value) != {'layer', 'query_head', 'kv_head', 'head_dim', 'query_position', 'key_positions', 'key_token_ids', 'probabilities', 'semantics'}:
            raise ValueError('Invalid attention fields')
        head = selected['head']
        if (type(value['query_head']) is not int or value['query_head'] != head or
                type(value['kv_head']) is not int or value['kv_head'] != head // (HEADS//KV_HEADS) or
                type(value['head_dim']) is not int or value['head_dim'] != HEAD_DIM or
                type(value['query_position']) is not int or value['query_position'] != step['position'] or
                value['semantics'] != ATTENTION_SEMANTICS):
            raise ValueError('Invalid attention geometry/semantics')
        n = step['position'] + 1
        keys, ids, probabilities = (value[k] for k in ('key_positions', 'key_token_ids', 'probabilities'))
        if any(type(x) is not list or len(x) != n for x in (keys, ids, probabilities)):
            raise ValueError('Invalid attention vector length')
        if any(type(p) is not int or p != i for i, p in enumerate(keys)) or any(not _integer(i, 0, VOCAB-1) for i in ids):
            raise ValueError('Invalid attention key address')
        if ids[-1] != step.get('input_token_id'):
            raise ValueError('Attention last key is not the consumed token')
        if any(not _finite(p) or not 0 <= p <= 1 for p in probabilities) or abs(sum(probabilities) - 1) > 1e-5:
            raise ValueError('Invalid attention probability distribution')
    else:
        if set(value) != {'layer', 'position', 'score_kind', 'lens_argmax_id', 'final_argmax_id', 'candidates', 'semantics'}:
            raise ValueError('Invalid logit-lens fields')
        if (type(value['position']) is not int or value['position'] != step['position'] or
                value['score_kind'] != 'raw FP32 logits' or value['semantics'] != LENS_SEMANTICS or
                not _integer(value['lens_argmax_id'], 0, VOCAB-1) or not _integer(value['final_argmax_id'], 0, VOCAB-1) or
                value['final_argmax_id'] != step.get('token_id')):
            raise ValueError('Invalid logit-lens semantics')
        candidates = value['candidates']
        if type(candidates) is not list or not 1 <= len(candidates) <= 10:
            raise ValueError('Invalid lens candidate count')
        seen = set()
        for candidate in candidates:
            if type(candidate) is not dict or set(candidate) != {'id', 'piece', 'lens_logit', 'final_logit', 'delta_lens_minus_final'}:
                raise ValueError('Invalid lens candidate fields')
            token = candidate['id']
            if not _integer(token, 0, VOCAB-1) or token in seen or type(candidate['piece']) is not str or len(candidate['piece']) > 1024:
                raise ValueError('Invalid lens candidate token')
            seen.add(token)
            a, b, delta = (candidate[k] for k in ('lens_logit', 'final_logit', 'delta_lens_minus_final'))
            if not all(_finite(n) for n in (a, b, delta)) or delta != a - b:
                raise ValueError('Invalid lens candidate score/delta')
        if not {value['lens_argmax_id'], value['final_argmax_id']} <= seen:
            raise ValueError('Lens candidate union omits an argmax')


def lens_record(torch, tokenizer, residual, model, final_logits, layer, position):
    """Only current-step full logits exist; return a small union with both values."""
    logits = model.lm_head(model.model.norm(residual))
    if tuple(logits.shape) != (VOCAB,) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Invalid logit-lens output')
    def top_ids(scores):
        ids = torch.topk(scores, 5).indices.tolist()
        best = int(torch.argmax(scores))
        # argmax chooses the first tied maximum; topk's boundary tie order may differ.
        if best not in ids:
            ids[-1] = best
        return ids
    ids = sorted(set(top_ids(logits)) | set(top_ids(final_logits)))
    candidates = []
    for token in ids:
        a, b = float(logits[token]), float(final_logits[token])
        candidates.append({'id': token, 'piece': tokenizer.decode([token], skip_special_tokens=False),
                           'lens_logit': a, 'final_logit': b, 'delta_lens_minus_final': a - b})
    return {'layer': layer, 'position': position, 'score_kind': 'raw FP32 logits',
            'lens_argmax_id': int(torch.argmax(logits)), 'final_argmax_id': int(torch.argmax(final_logits)),
            'candidates': candidates, 'semantics': LENS_SEMANTICS}
