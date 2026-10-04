#!/usr/bin/env python3
"""Independent stdlib oracle; tiny generated safetensors only, no sockets/models.

Run only in an assigned heavy slot after the guarded offline release build:
python3 tests/data_view_reference.py --binary target/release/weight-atlas-rust
"""
import argparse
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import resource
import struct
import subprocess
import tempfile


def quantile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    x = q * (len(values)-1)
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi]-values[lo])*(x-lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (768*1024**2, 768*1024**2))
    available = int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')).split()[1])*1024
    if available < 5*1024**3:
        raise SystemExit('Qualification paused: need 5 GiB available; no fixture/binary work started')
    cases = 0
    with tempfile.TemporaryDirectory(prefix='atlas-data-reference-') as tmp:
        root = Path(tmp); model = root/'model'; model.mkdir()
        raw = bytearray(); header = {}; decoded = {}; words = {}
        for dtype in ['BF16', 'F16', 'F32']:
            values = [-0.0, 0.0, -2.0, 2.0, -0.5, 0.5, 1.0, 1.0, -4.0, 4.0, 0.125, 32.0]*4
            start = len(raw); decoded[dtype] = []; words[dtype] = []
            for value in values:
                packed = struct.pack('<e' if dtype == 'F16' else '<f', value)
                if dtype == 'BF16':
                    packed = packed[2:]
                    value = struct.unpack('<f', b'\0\0'+packed)[0]
                else:
                    value = struct.unpack('<e' if dtype == 'F16' else '<f', packed)[0]
                decoded[dtype].append(value); words[dtype].append(packed.hex()); raw.extend(packed)
            header[dtype] = {'dtype': dtype, 'shape': [2, 2, 3, 4], 'data_offsets': [start, len(raw)]}
        start = len(raw); raw.extend(struct.pack('<i', 17))
        header['packed'] = {'dtype': 'I32', 'shape': [1], 'data_offsets': [start, len(raw)]}
        encoded = json.dumps(header, separators=(',', ':')).encode()
        (model/'model.safetensors').write_bytes(struct.pack('<Q', len(encoded))+encoded+raw)

        def run(command, **opts):
            available = int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                                 if line.startswith('MemAvailable:')).split()[1])*1024
            if available < 3.25*1024**3:
                raise RuntimeError('Qualification stopped below 3.25 GiB reserve; no retry')
            cmd = [str(binary), command, '--model', str(model), '--cache', str(root/'cache')]
            for key, value in opts.items(): cmd.extend(['--'+key, str(value)])
            process = subprocess.run(cmd, text=True, capture_output=True, timeout=20)
            if process.returncode: raise RuntimeError(f'{command} failed: {process.stderr}')
            return json.loads(process.stdout)

        metadata = run('metadata')
        assert len(metadata['catalog']) == 4
        assert next(t for t in metadata['catalog'] if t['name'] == 'packed')['available'] is False
        for dtype in decoded:
            tensor = next(t for t in metadata['catalog'] if t['name'] == dtype)
            run('calibrate', tensor=tensor['id'])
            values = decoded[dtype]
            s = quantile([abs(x) for x in values if x != 0], .5) or 1
            d = quantile([abs(x) for x in values], .99) or 1
            for leading in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                plane = leading[0]*2 + leading[1]
                for row, col in [(0, 0), (0, 1), (2, 3)]:
                    index = plane*12 + row*4 + col
                    result = run('inspect', tensor=tensor['id'], slice=','.join(map(str, leading)),
                                 row=row, col=col, left='tensor_magnitude_asinh', right='tensor_magnitude')
                    assert result['native_indices'] == [*leading, row, col]
                    assert result['raw_hex_le'] == words[dtype][index]
                    assert Decimal(result['raw_exact']) == Decimal.from_float(values[index])
                    if values[index] == 0: assert result['raw_exact'].startswith('-') == (math.copysign(1, values[index]) < 0)
                    assert result['byte_offset'] == tensor['byte_offset']+index*tensor['element_bytes']
                    assert result['source_binding']['slice'] == {'leading_indices': list(leading), 'display_axes': [2, 3]}
                    cases += 1
                for factor in [1, 2, 4]:
                    out = root/f'{dtype}-{plane}-{factor}'
                    result = run('tile', tensor=tensor['id'], slice=','.join(map(str, leading)),
                                 rules='tensor_magnitude_asinh', level=tensor['max_level']-int(math.log2(factor)),
                                 x=0, y=0, out=out)
                    field = [v[0] for v in struct.iter_unpack('<d', Path(str(out)+'-tensor_magnitude_asinh.f64le').read_bytes())]
                    expected = []
                    for row in range(0, 3, factor):
                        for col in range(0, 4, factor):
                            xs = [values[plane*12+r*4+c] for r in range(row, min(row+factor, 3)) for c in range(col, min(col+factor, 4))]
                            expected.append(sum(min(1., math.asinh(abs(x)/s)/math.asinh(d/s)) for x in xs)/len(xs))
                    assert len(field) == len(expected)
                    assert max(abs(a-b) for a, b in zip(field, expected)) < 1e-13
                    assert result['metrics']['max_raw_band_values']*tensor['element_bytes'] <= 2*1024**2
                    cases += 1
    print(json.dumps({'status': 'PASS', 'cases': cases, 'scope': 'Tiny BF16/F16/F32 rank-four exact source and independent magnitude/pooling oracle; no sockets, real models, or browser'}))


if __name__ == '__main__': main()
