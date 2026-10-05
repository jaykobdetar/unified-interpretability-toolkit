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
fixture = json.loads(
    subprocess.check_output(["node", "-e", script], cwd=ROOT, text=True)
)
for name, selection in [("matrix", slice(None)), ("vector", slice(1, 4))]:
    rows = list(csv.reader(io.StringIO(fixture[name])))
    assert rows[0][0] == "metadata"
    meta = json.loads(rows[0][1])
    assert meta["name"] == '=formula,"quoted"\nline'
    assert meta["source_dtype"] == "BF16"
    assert meta["scale"] == "none; no normalization or color transform"
    original_bytes = b"".join(bytes.fromhex(h) for h in fixture["hex"][selection])
    # Independent reconstruction by NumPy bit widening, not the JS decoder.
    expected = (np.frombuffer(original_bytes, dtype="<u2").astype("<u4") << 16).view(
        "<f4"
    )
    actual = np.array([r[3] for r in rows[2:]], dtype="<f4")
    np.testing.assert_array_equal(actual.view("<u4"), expected.view("<u4"))
    assert actual.size == np.prod(meta["dimensions"])
    for i, row in enumerate(rows[2:]):
        assert bytes.fromhex(row[4]) == original_bytes[i * 2 : i * 2 + 2]
        r, c = int(row[0]), int(row[1])
        assert json.loads(row[2]) == ([r, c] if name == "matrix" else [c])
print(
    json.dumps(
        {
            "status": "PASS",
            "checks": 2,
            "numpy": np.__version__,
            "scope": "BF16 matrix/vector CSV, signed zero, subnormal, max finite, exact original bytes, native indices and metadata",
        }
    )
)

# Numeric NPY roundtrip uses NumPy's independent IEEE decoder with pickle disabled.
script_npy = r"""
const A=require('./web/atlas-tools.js');
const hex={BF16:['0080','0100','7f7f','803f'],F16:['0080','0100','ff7b','003c'],F32:['00000080','01000000','ffff7f7f','0000803f']};
const result={};
for(const [dtype,entries] of Object.entries(hex)){
 const t={id:0,name:'fixture',dtype,shape:[2,1,4],slice:[1],rows:1,cols:4},m={model_identity:'d'.repeat(64),source_identity:'a'.repeat(64),revision:'fixture',catalog:[t]},binding=A.sourceBinding(m,t);
 const values=entries.map((h,col)=>({source_binding:binding,tensor:0,row:0,col,native_indices:[1,0,col],raw_hex_le:h,raw_exact:Object.is(A.decodeSource(dtype,h),-0)?'-0':String(A.decodeSource(dtype,h))}));
 const output=A.boundedNPY(A.scope(m,t),[0,0,0,3],values,binding);result[dtype]={hex:entries,base64:Buffer.from(output.data).toString('base64'),metadata:output.metadata,csv:A.boundedCSV(A.scope(m,t),[0,0,0,3],values,binding)};
}
process.stdout.write(JSON.stringify(result));
"""
import base64

outputs = json.loads(
    subprocess.check_output(["node", "-e", script_npy], cwd=ROOT, text=True)
)
for dtype, output in outputs.items():
    raw = b"".join(bytes.fromhex(h) for h in output["hex"])
    expected = (
        (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
        if dtype == "BF16"
        else np.frombuffer(raw, dtype="<f2" if dtype == "F16" else "<f4")
    )
    actual = np.load(io.BytesIO(base64.b64decode(output["base64"])), allow_pickle=False)
    np.testing.assert_array_equal(
        actual.reshape(-1).view("<u2" if dtype == "F16" else "<u4"),
        expected.view("<u2" if dtype == "F16" else "<u4"),
    )
    csv_actual = np.array(
        [r[3] for r in list(csv.reader(io.StringIO(output["csv"])))[2:]],
        dtype=expected.dtype,
    )
    np.testing.assert_array_equal(
        csv_actual.view("<u2" if dtype == "F16" else "<u4"),
        expected.view("<u2" if dtype == "F16" else "<u4"),
    )
    assert actual.shape == (1, 4) and output["metadata"]["slice"] == [1]
print(
    "PASS: independent NumPy CSV/NPY bit roundtrip for BF16/F16/F32, allow_pickle=False, signed zero, subnormal, maximum finite and explicit rank-three slice."
)
