"""Independent NumPy oracle for bounded actual typed browser downloads only."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np

out = Path(sys.argv[1])
manifest = json.loads((out/'fixture/fixture-manifest.json').read_text())
binding = json.loads((out/'served-source-binding.json').read_text())
result = json.loads((out/'browser-result.json').read_text())
assert result['status'] == 'PASS' and result['phase'] == 'exports'
assert len(result['downloads']) == 18
source = out/'fixture'/manifest['file']
assert source.stat().st_size == 640 and hashlib.sha256(source.read_bytes()).hexdigest() == manifest['sha256']
reports = []
for dtype in ('BF16','F16','F32'):
    for kind in ('rank3','vector'):
        name = dtype.lower()+'_'+kind
        tensor = manifest['tensors'][name]
        expected_words = tensor['source_hex_le'][16:24] if kind == 'rank3' else tensor['source_hex_le']
        raw = b''.join(bytes.fromhex(h) for h in expected_words)
        expected = (np.frombuffer(raw,dtype='<u2').astype('<u4') << 16).view('<f4') if dtype == 'BF16' else np.frombuffer(raw,dtype='<f2' if dtype == 'F16' else '<f4')
        bits = '<u2' if dtype == 'F16' else '<u4'
        csv_file = out/(name+'-csv-weight-atlas-region.csv')
        npy_file = out/(name+'-npy-weight-atlas-region.npy')
        meta_file = out/(name+'-npy-weight-atlas-region.metadata.json')
        for file in (csv_file,npy_file,meta_file):
            assert file.stat().st_size <= 262144
        rows = list(csv.reader(io.StringIO(csv_file.read_text())))
        assert rows[0][0] == 'metadata'
        assert rows[1] == ['row','column','native_indices','raw_exact','bf16_hex_le' if dtype == 'BF16' else 'raw_hex_le']
        csv_meta = json.loads(rows[0][1])
        meta = json.loads(meta_file.read_text())
        for m in (csv_meta,meta):
            assert m['name'] == name and m['shape'] == tensor['shape']
            assert m['dtype'] == m['source_dtype'] == dtype
            assert m['source_identity'] == binding['source_identity'] and m['model_identity'] == binding['model_identity']
            assert m['revision'] == manifest['revision'] and m['values'] == 'original, untransformed'
            assert m['scale'] == 'none; no normalization or color transform'
            assert m.get('slice', []) == ([1] if kind == 'rank3' else [])
            assert m['dimensions'] == ([2,4] if kind == 'rank3' else [1,4])
        assert len(rows[2:]) == expected.size
        for i, r in enumerate(rows[2:]):
            row, col = (1+i//4,i%4) if kind == 'rank3' else (0,i)
            assert [int(r[0]),int(r[1])] == [row,col]
            assert json.loads(r[2]) == ([1,row,col] if kind == 'rank3' else [col])
            assert r[4] == expected_words[i]
        csv_values = np.array([r[3] for r in rows[2:]],dtype=expected.dtype)
        np.testing.assert_array_equal(csv_values.view(bits),expected.view(bits))
        values = np.load(npy_file,allow_pickle=False)
        assert values.shape == ((2,4) if kind == 'rank3' else (1,4)) and values.flags.c_contiguous
        assert values.dtype == expected.dtype and meta['numpy']['allow_pickle'] is False
        assert meta['numpy']['descr'] == ('<f2' if dtype == 'F16' else '<f4') and meta['numpy']['fortran_order'] is False
        np.testing.assert_array_equal(values.reshape(-1).view(bits),expected.view(bits))
        assert meta['source_hex_le'] == expected_words
        reports.append({'tensor':name,'cells':int(values.size),'dtype':str(values.dtype),'shape':list(values.shape),
                        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (csv_file,npy_file,meta_file)}})
print(json.dumps({'status':'PASS','numpy':np.__version__,'cases':reports,
                  'scope':'Actual browser CSV/NPY/sidecars match independently known source bits/native rank3 slice/vector, negative zero, subnormal and max finite; np.load allow_pickle=False'},indent=2))
