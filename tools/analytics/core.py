"""Pure raw-value analytics. Inputs are native row-major bounded regions."""
import hashlib
import json
import math
import re

SCHEMA = 'weight-atlas.analytics.v1'
MAX_VALUES = 65536
MAX_AXIS = 4096
MAX_TOP = 32
MAX_TENSORS = 512


def integer(value, lo, hi, label):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f'{label} must be an integer in [{lo}, {hi}]')
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def geometry(shape, region):
    if not isinstance(shape, (list, tuple)) or len(shape) not in (1, 2):
        raise ValueError('Only native vectors and matrices are supported')
    for n in shape:
        integer(n, 1, 200000, 'shape')
    rows, cols = (1, shape[0]) if len(shape) == 1 else shape
    r, c, h, w = (region[k] for k in ('row', 'col', 'rows', 'cols'))
    integer(r, 0, rows - 1, 'row'); integer(c, 0, cols - 1, 'col')
    integer(h, 1, min(rows - r, MAX_AXIS), 'rows')
    integer(w, 1, min(cols - c, MAX_AXIS), 'cols')
    if h * w > MAX_VALUES:
        raise ValueError('Region exceeds 65536-value cap; select a smaller window')
    return r, c, h, w


def checked_values(values, shape, region):
    _, _, h, w = geometry(shape, region)
    if len(values) != h * w:
        raise ValueError('Region value count mismatch')
    # BF16's finite dynamic range prevents accumulation overflow. Reject arbitrary
    # finite doubles outside the supported source range, rather than overflow JSON.
    if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 3.39e38 for x in values):
        raise ValueError('Analytics require finite BF16-range raw values')


def permutation(n, seed):
    """xorshift32 + rejection-sampled Fisher–Yates; stable across runtimes."""
    integer(n, 0, MAX_VALUES, 'count'); integer(seed, 0, 2**32 - 1, 'seed')
    # Injective seeds except documented zero alias (avoid xorshift absorbing state).
    state = seed or 0x6D2B79F5
    indices = list(range(n))
    for i in range(n - 1, 0, -1):
        bound = i + 1
        limit = 2**32 - (2**32 % bound)
        while True:
            state ^= (state << 13) & 0xffffffff
            state ^= state >> 17
            state ^= (state << 5) & 0xffffffff
            state &= 0xffffffff
            if state < limit:
                break
        j = state % bound
        indices[i], indices[j] = indices[j], indices[i]
    return indices


def ranked(items, limit, score='mean_abs'):
    return sorted(items, key=lambda x: (-x[score], x['index']))[:limit]


def statistics(values, shape, region, top=16):
    checked_values(values, shape, region)
    integer(top, 1, MAX_TOP, 'top')
    r, c, h, w = geometry(shape, region)
    rows = [{'index': r+i, 'mean_abs': math.fsum(abs(x) for x in values[i*w:(i+1)*w]) / w,
             'count': w} for i in range(h)]
    cols = [{'index': c+j, 'mean_abs': math.fsum(abs(values[i*w+j]) for i in range(h)) / h,
             'count': h} for j in range(w)]
    positions = sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))[:top]
    outliers = [{'index': i, 'row': r+i//w, 'col': c+i%w,
                 'native_indices': [c+i] if len(shape) == 1 else [r+i//w, c+i%w],
                 'value': values[i], 'abs': abs(values[i])} for i in positions]
    return {'rows': rows, 'columns': cols,
            'row_order': [x['index'] for x in ranked(rows, len(rows))],
            'column_order': [x['index'] for x in ranked(cols, len(cols))],
            'top_values': outliers, 'top_rows': ranked(rows, top), 'top_columns': ranked(cols, top),
            'vector': [{'index': c+i, 'value': x} for i, x in enumerate(values)] if len(shape) == 1 else None}


def resolve_heads(tensor, shape, config, evidence, reviewed_profiles):
    """Profiles are trusted application data, NEVER client-supplied request data.

    Profile keys bind exact config + implementation digests. A reviewer must
    confirm the separate contiguous Linear [out,in] layout before adding a key.
    No production profile is guessed from a model name or dimension divisibility.
    """
    excluded = lambda why: {'available': False, 'reason': why}
    if not config or not evidence:
        return excluded('Verified configuration and implementation layout are required')
    key = (evidence.get('config_sha256'), evidence.get('implementation_sha256'))
    profile = reviewed_profiles.get(key)
    if evidence.get('config_canonical_sha256') != digest(config) or profile != 'separate-contiguous-linear-out-in-v1':
        return excluded('Configuration/implementation identity lacks a reviewed layout profile')
    if len(shape) != 2:
        return excluded('Head layout requires a native matrix')
    match = re.fullmatch(r'model\.layers\.(\d+)\.self_attn\.([qkvo])_proj\.weight', tensor)
    if not match:
        return excluded('Unrecognized, fused, or ambiguous projection layout')
    try:
        hidden = integer(config.get('hidden_size'), 1, 2**20, 'hidden_size')
        q = integer(config.get('num_attention_heads'), 1, MAX_AXIS, 'query heads')
        kv = integer(config.get('num_key_value_heads'), 1, MAX_AXIS, 'KV heads')
        layers = integer(config.get('num_hidden_layers'), 1, MAX_AXIS, 'layers')
        if int(match[1]) >= layers or q % kv:
            return excluded('Layer or grouped-query configuration mismatch')
        dim = config.get('head_dim')
        if dim is None:
            if hidden % q:
                return excluded('No unambiguous head dimension')
            dim = hidden // q
        dim = integer(dim, 1, MAX_AXIS, 'head dimension')
    except ValueError as exc:
        return excluded(str(exc))
    projection = match[2]
    count = kv if projection in 'kv' else q
    axis = 'column' if projection == 'o' else 'row'
    expected = [hidden, q*dim] if projection == 'o' else [count*dim, hidden]
    if list(shape) != expected:
        return excluded('Actual tensor shape does not match reviewed projection layout')
    label = {'q': 'Q heads', 'k': 'KV heads (K)', 'v': 'KV heads (V)',
             'o': 'Output-projection input columns (Q-head groups)'}[projection]
    return {'available': True, 'label': label, 'axis': axis, 'head_count': count, 'head_dim': dim,
            'boundaries': [i*dim for i in range(count+1)], 'evidence': evidence,
            'fold_statistic': 'mean absolute value by within-head offset'}


def fold(values, region, heads):
    if not heads['available']:
        return None
    h, w = region['rows'], region['cols']
    dim = heads['head_dim']
    groups = [[] for _ in range(dim)]
    covered = set()
    for i, value in enumerate(values):
        coord = region['row']+i//w if heads['axis'] == 'row' else region['col']+i%w
        groups[coord % dim].append(abs(value)); covered.add(coord // dim)
    start, length = (region['row'], h) if heads['axis'] == 'row' else (region['col'], w)
    # Preserve the other matrix axis when folding across heads. The offset
    # profile above is a separate magnitude summary, not a folded matrix.
    folded_rows, folded_cols = (dim, w) if heads['axis'] == 'row' else (h, dim)
    matrix = {'available': False, 'reason': 'Folded matrix exceeds 4096 output cells; offset profile remains available'}
    if folded_rows * folded_cols <= 4096:
        cells = [[] for _ in range(folded_rows * folded_cols)]
        for i, value in enumerate(values):
            local_row, local_col = divmod(i, w)
            row = (region['row'] + local_row) % dim if heads['axis'] == 'row' else local_row
            col = (region['col'] + local_col) % dim if heads['axis'] == 'column' else local_col
            cells[row * folded_cols + col].append(value)
        matrix = {'available': True, 'rows': folded_rows, 'cols': folded_cols,
                  'mean': [math.fsum(cell)/len(cell) if cell else None for cell in cells],
                  'mean_abs': [math.fsum(abs(x) for x in cell)/len(cell) if cell else None for cell in cells],
                  'counts': [len(cell) for cell in cells],
                  'row_axis': 'within-head offset' if heads['axis'] == 'row' else 'native window rows',
                  'column_axis': 'within-head offset' if heads['axis'] == 'column' else 'native window columns'}
    return {'offsets': [{'index': i, 'mean_abs': math.fsum(g)/len(g) if g else None,
                         'count': len(g)} for i, g in enumerate(groups)],
            'matrix': matrix, 'covered_heads': sorted(covered),
            'full_head_axis': start == 0 and length == heads['head_count']*dim,
            'region': region}



def analyze(values, *, source_identity, tensor, shape, region, seed=1, top=16,
            config=None, evidence=None, reviewed_profiles=None):
    if not isinstance(source_identity, str) or not source_identity or not isinstance(tensor, str) or not tensor:
        raise ValueError('Source identity and tensor name are required')
    checked_values(values, shape, region)
    order = permutation(len(values), seed)
    shuffled = [values[i] for i in order]
    heads = resolve_heads(tensor, shape, config, evidence, reviewed_profiles or {})
    original_stats = statistics(values, shape, region, top)
    shuffled_stats = statistics(shuffled, shape, region, top)
    original_stats['folded'] = fold(values, region, heads)
    shuffled_stats['folded'] = fold(shuffled, region, heads)
    for item in shuffled_stats['top_values']:
        src = order[item['index']]
        item['control_position'] = item.pop('native_indices')
        item['source_native_indices'] = ([region['col']+src] if len(shape) == 1 else
                                        [region['row']+src//region['cols'], region['col']+src%region['cols']])
    total = math.prod(shape)
    key = {'schema': SCHEMA, 'source': source_identity, 'tensor': tensor, 'shape': shape,
           'region': region, 'seed': seed, 'top': top, 'heads': heads}
    return {'schema': SCHEMA, 'source_identity': source_identity, 'tensor': tensor,
            'shape': shape, 'region': region, 'cache_key': digest(key),
            'coverage': {'visited_values': len(values), 'total_tensor_values': total,
                         'full_tensor': len(values) == total, 'full_model': False,
                         'scope': 'native region', 'coordinate_source': 'direct source values'},
            'control': {'kind': 'same-region-exact-multiset', 'seed': seed,
                        'algorithm': 'xorshift32-rejection-fisher-yates-v1',
                        'zero_seed_alias': 0x6D2B79F5, 'position_to_source': order,
                        'statistics': 'refit separately for original and control',
                        'calibration': 'raw values; shared display scale; no fitted transform',
                        'caution': 'A shuffle alone does not establish pattern meaning. Sorting can manufacture gradients.'},
            'original': original_stats, 'shuffled': shuffled_stats, 'heads': heads,
            'svd': {'available': False, 'reason': 'Optional isolated NumPy worker has not run'}}
