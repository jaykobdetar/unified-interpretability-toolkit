"""One analysis child: bounded source reads and optional NumPy, no inference."""
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys

from .core import geometry, integer
from .source import Catalog, Tensor, fingerprint

MAX_INPUT = 512*1024
MAX_OUTPUT = 2*1024*1024 - 2048  # Reserve space for HTTP job envelope.


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def parse_json(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def validate_request(data, catalog):
    if not isinstance(data, dict) or set(data) - {'scope', 'tensor', 'region', 'seed', 'svd'}:
        raise ValueError('Unknown analytics request field')
    integer(data.get('seed', 1), 0, 2**32-1, 'seed')
    if type(data.get('svd', False)) is not bool:
        raise ValueError('svd must be boolean')
    scope = data.get('scope', 'region')
    if scope == 'model':
        if set(data) - {'scope', 'seed'}:
            raise ValueError('Model ranking accepts only scope and seed')
        return None
    if scope == 'svd_summary':
        if set(data) != {'scope', 'tensor', 'region', 'seed'}:
            raise ValueError('SVD summary accepts only scope, tensor, region and seed')
    elif scope != 'region':
        raise ValueError('Unknown analytics scope')
    tensor_id = integer(data.get('tensor'), 0, 511, 'tensor')
    tensor = next((t for t in catalog if t['id'] == tensor_id), None)
    if tensor is None:
        raise ValueError('Tensor is not in the trusted catalog')
    region = data.get('region')
    if not isinstance(region, dict) or set(region) != {'row', 'col', 'rows', 'cols'}:
        raise ValueError('Exact native region fields required')
    geometry(tensor['shape'], region)
    if scope == 'svd_summary':
        from .svd_summary import check_window
        check_window(tensor['shape'], region)
    return tensor


def catalog_from_payload(payload):
    metadata = payload['model']
    raw_tensors = metadata['catalog']
    if not isinstance(raw_tensors, list) or not 1 <= len(raw_tensors) <= 512:
        raise ValueError('Catalog size exceeds bound')
    ids = [t['id'] for t in raw_tensors]
    if sorted(ids) != list(range(len(ids))) or any(type(x) is not int for x in ids):
        raise ValueError('Catalog IDs must be a complete native sequence')
    if any(t['dtype'] != 'BF16' for t in raw_tensors):
        raise ValueError('Only original BF16 values supported')
    tensors = [Tensor(t['name'], tuple(t['shape']), t['shard'], t['byte_offset']) for t in raw_tensors]
    catalog = Catalog(payload['root'], tensors, metadata['source_identity'])
    expected = {name: tuple(fp) for name, fp in payload['fingerprints'].items()}
    index_expected = payload.get('index_fingerprint')
    if index_expected is not None:
        index_expected = tuple(index_expected)
    if catalog.files != expected or catalog.index_fingerprint != index_expected:
        raise ValueError('Source identity changed after admission')
    # Cross-check native extents against actual bounded safetensors headers.
    total_header = 0
    for shard in catalog.files:
        fd = os.open(catalog.root/shard, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if fingerprint(os.fstat(fd)) != expected[shard]:
                raise ValueError('Source changed before header check')
            prefix = os.pread(fd, 8, 0)
            if len(prefix) != 8:
                raise ValueError('Short safetensors prefix')
            size = struct.unpack('<Q', prefix)[0]
            total_header += size
            if not 2 <= size <= 8*1024*1024 or total_header > 32*1024*1024:
                raise ValueError('Header budget exceeded')
            raw = os.pread(fd, size, 8)
            if len(raw) != size:
                raise ValueError('Short header')
            header = parse_json(raw)
            selected = [t for t in tensors if t.shard == shard]
            if set(header)-{'__metadata__'} != {t.name for t in selected}:
                raise ValueError('Header tensor catalog changed')
            for t in selected:
                entry = header[t.name]
                offsets = entry['data_offsets']
                if entry['dtype'] != 'BF16' or entry['shape'] != list(t.shape) or len(offsets) != 2 or offsets != [t.byte_offset-8-size, t.byte_offset-8-size+2*math.prod(t.shape)]:
                    raise ValueError('Header layout differs from validated native catalog')
            if fingerprint(os.fstat(fd)) != expected[shard]:
                raise ValueError('Source changed during header check')
        finally:
            os.close(fd)
    catalog.verify()
    return catalog


def analyze_request(payload):
    catalog = catalog_from_payload(payload)
    data = payload['request']
    chosen = validate_request(data, payload['model']['catalog'])
    seed = data.get('seed', 1)
    if chosen is None:
        return catalog.model_outliers(seed=seed)
    if data.get('scope') == 'svd_summary':
        from .svd_summary import compute_with_numpy, binding
        binding(payload['model'], chosen, data['region'], seed)
        tensor = next(t for t in catalog.tensors if t.name == chosen['name'])
        values = catalog.read(tensor, data['region'])
        import numpy as np  # Same owned worker, BLAS/process caps; no general dense report.
        result = compute_with_numpy(np, values, model=payload['model'], tensor=chosen, region=data['region'], seed=seed)
        catalog.verify()
        return result
    from .profiles import resolve_local
    options, unavailable = resolve_local(catalog.root)
    tensor = next(t for t in catalog.tensors if t.name == chosen['name'])
    region = data['region']
    values = catalog.read(tensor, region)
    from .core import analyze
    result = analyze(values, source_identity=catalog.identity, tensor=tensor.name,
                     shape=list(tensor.shape), region=region, seed=seed, **options)
    result['tensor_id'] = chosen['id']
    result['host_source_identity'] = payload['model']['source_identity']
    if unavailable:
        result['heads']['reason'] = unavailable
    if data.get('svd'):
        from .svd import MAX_SVD_AXIS, MAX_SVD_VALUES, compute_with_numpy
        if region['rows'] > MAX_SVD_AXIS or region['cols'] > MAX_SVD_AXIS or len(values) > MAX_SVD_VALUES:
            result['svd'] = {'available': False, 'reason': 'Excluded: select at most 64 × 64 / 4096 values for SVD'}
        else:
            import numpy as np  # BLAS caps and process limits already established.
            result['svd'] = {'available': True, 'scope': 'full tensor' if len(values) == math.prod(tensor.shape) else 'selected native window only',
                             'region': region, 'seed': seed, 'centered': False,
                             'control': 'same-region exact multiset; each SVD fitted separately',
                             'results': compute_with_numpy(np, values, region['rows'], region['cols'], seed)}
    catalog.verify()
    return result


def configure():
    import signal
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(5)  # Kernel-enforced wall timer even while native LAPACK runs.
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(10)
    for kind, ceiling in ((resource.RLIMIT_AS, 768*1024**2), (resource.RLIMIT_CPU, 4)):
        soft, hard = resource.getrlimit(kind)
        cap = min(n for n in (ceiling, soft, hard) if n != resource.RLIM_INFINITY)
        resource.setrlimit(kind, (cap, cap))
    mem = int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    if mem < 3.25*1024**3:
        raise ValueError('Memory reserve reached')


def main():
    configure()
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT+1)
        if len(raw) > MAX_INPUT:
            raise ValueError('Worker input exceeds budget')
        result = {'ok': True, 'result': analyze_request(parse_json(raw))}
        output = json.dumps(result, allow_nan=False, separators=(',', ':')).encode()
        if len(output) > MAX_OUTPUT:
            raise ValueError('Result exceeds 2 MiB output cap; select a smaller window')
    except Exception as exc:
        # No stack trace, paths, prompt, or source contents leave the child.
        output = json.dumps({'ok': False, 'error': str(exc) if isinstance(exc, ValueError) else 'Analysis failed; source or optional runtime unavailable'}).encode()
    sys.stdout.buffer.write(output+b'\n')
    sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()
