"""Verify actual browser-downloaded fixture CSVs independently with NumPy."""
import csv
import decimal
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import numpy as np

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
root = Path(__file__).resolve().parents[1]
expected = json.loads((root / 'fixtures/tiny-bf16/expected.json').read_text())
directory = Path(sys.argv[1])
files = ['matrix-region.csv', 'vector-signed-zero.csv', 'matrix-recovered.csv']
reports = []
for filename in files:
    path = directory / filename
    assert path.stat().st_size <= 262144
    data = path.read_bytes()
    records = list(csv.reader(io.StringIO(data.decode('utf-8'))))
    assert records[0][0] == 'metadata'
    meta = json.loads(records[0][1])
    assert meta['revision'] == 'fixture-A'
    assert meta['dtype'] == meta['source_dtype'] == 'BF16'
    assert meta['values'] == 'original, untransformed'
    assert meta['scale'] == 'none; no normalization or color transform'
    assert len(meta['model_identity']) == len(meta['source_identity']) == 64
    assert records[1] == ['row', 'column', 'native_indices', 'raw_exact', 'bf16_hex_le']
    rows = records[2:]
    assert 1 <= len(rows) <= 256
    assert len(rows) == np.prod(meta['dimensions'])
    bits = np.frombuffer(b''.join(bytes.fromhex(r[4]) for r in rows), dtype='<u2')
    widened = (bits.astype('<u4') << 16).view('<f4')
    decimal_values = np.array([r[3] for r in rows], dtype='<f4')
    np.testing.assert_array_equal(decimal_values.view('<u4'), widened.view('<u4'))
    fixture_values = []
    for i, record in enumerate(rows):
        row, col = int(record[0]), int(record[1])
        assert meta['region'][0] <= row <= meta['region'][2]
        assert meta['region'][1] <= col <= meta['region'][3]
        assert json.loads(record[2]) == ([col] if len(meta['shape']) == 1 else [row, col])
        assert decimal.Decimal(record[3]) == decimal.Decimal.from_float(float(widened[i]))
        fixture_values.append(expected[meta['name']]['values'][row * meta['cols'] + col])
    np.testing.assert_array_equal(np.array(fixture_values, dtype='<f4').view('<u4'), widened.view('<u4'))
    reports.append({'file': filename, 'cells': len(rows), 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest()})
print(json.dumps({'status': 'PASS', 'checks': len(reports), 'numpy': np.__version__,
                  'scope': 'Actual browser CSV files match independent fixture values, exact decimals, native indices and original BF16 bits, including signed zero',
                  'files': reports}, indent=2))
