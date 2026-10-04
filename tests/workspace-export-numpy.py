"""Independent small-fixture CSV roundtrip; no model reads or inference."""
import csv
import io
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
script = r"""
const A=require('./web/atlas-tools.js');
const hex=['0000','0080','803f','80bf','0100','7f7f','0081','aa3e'];
const t={id:0,name:'=formula,\"quoted\"\nline',dtype:'BF16',shape:[2,4],rows:2,cols:4};
const m={model_identity:'d'.repeat(64),source_identity:'a'.repeat(64),revision:'tiny-fixture',catalog:[t]};
const v=hex.map((h,i)=>({source_binding:A.sourceBinding(m,t),tensor:0,row:Math.floor(i/4),col:i%4,native_indices:[Math.floor(i/4),i%4],bf16_hex_le:h,raw_exact:Object.is(A.decodeBF16(h),-0)?'-0':String(A.decodeBF16(h))}));
const matrix=A.boundedCSV(A.scope(m,t),[0,0,1,3],v,A.sourceBinding(m,t));
const vt={...t,shape:[4],rows:1,cols:4};
const vector=A.boundedCSV(A.scope({...m,catalog:[vt]},vt),[0,1,0,3],v.slice(1,4).map(x=>({...x,source_binding:A.sourceBinding(m,vt),native_indices:[x.col]})),A.sourceBinding(m,vt));
process.stdout.write(JSON.stringify({hex,matrix,vector}));
"""
fixture = json.loads(subprocess.check_output(['node', '-e', script], cwd=ROOT, text=True))
for name, selection in [('matrix', slice(None)), ('vector', slice(1, 4))]:
    rows = list(csv.reader(io.StringIO(fixture[name])))
    assert rows[0][0] == 'metadata'
    meta = json.loads(rows[0][1])
    assert meta['name'] == '=formula,"quoted"\nline'
    assert meta['source_dtype'] == 'BF16'
    assert meta['scale'] == 'none; no normalization or color transform'
    original_bytes = b''.join(bytes.fromhex(h) for h in fixture['hex'][selection])
    # Independent reconstruction by NumPy bit widening, not the JS decoder.
    expected = (np.frombuffer(original_bytes, dtype='<u2').astype('<u4') << 16).view('<f4')
    actual = np.array([r[3] for r in rows[2:]], dtype='<f4')
    np.testing.assert_array_equal(actual.view('<u4'), expected.view('<u4'))
    assert actual.size == np.prod(meta['dimensions'])
    for i, row in enumerate(rows[2:]):
        assert bytes.fromhex(row[4]) == original_bytes[i*2:i*2+2]
        r, c = int(row[0]), int(row[1])
        assert json.loads(row[2]) == ([r, c] if name == 'matrix' else [c])
print(json.dumps({'status': 'PASS', 'checks': 2, 'numpy': np.__version__,
                  'scope': 'BF16 matrix/vector CSV, signed zero, subnormal, max finite, exact original bytes, native indices and metadata'}))
