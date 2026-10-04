#!/usr/bin/env python3
"""Prepared synthetic comparison acceptance; only run in a granted heavy slot.
Uses installed NumPy and the newly built binary. Never opens real model weights.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import decimal,hashlib,json,math,resource,struct,subprocess
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/comparison-reference';OUT.mkdir(parents=True,exist_ok=True)
A=OUT/'a';B=OUT/'b';CACHE=OUT/'cache'
BIN=ROOT/'target/release/weight-atlas-rust'
records=[]
def bits_values(x,dtype):
 if dtype=='BF16':
  bits=(x.astype('<f4').view('<u4')>>16).astype('<u2');decoded=(bits.astype('<u4')<<16).view('<f4').astype('<f8')
 elif dtype=='F16':bits=x.astype('<f2').view('<u2');decoded=bits.view('<f2').astype('<f8')
 else:bits=x.astype('<f4').view('<u4');decoded=bits.view('<f4').astype('<f8')
 return bits,decoded

def write_model(path,entries):
 path.mkdir(exist_ok=True);header={};data=bytearray();decoded={}
 for name,(dtype,x) in entries.items():
  bits,values=bits_values(x,dtype);start=len(data);data.extend(bits.tobytes());header[name]={'dtype':dtype,'shape':list(bits.shape),'data_offsets':[start,len(data)]};decoded[name]=(bits,values)
 h=json.dumps(header).encode();(path/'fixture.safetensors').write_bytes(struct.pack('<Q',len(h))+h+data);return decoded

def run(command,*args,a=A,b=B,cache=CACHE,ok=True):
 p=subprocess.run([str(BIN),'compare-'+command,'--model',str(a),'--compare-model',str(b),'--cache',str(cache),*map(str,args)],capture_output=True,text=True)
 with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=command,args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),default=str)+'\n')
 if ok:assert p.returncode==0,p.stderr
 return json.loads(p.stdout) if p.returncode==0 else p

rng=np.random.default_rng(200032)
rawa={'matrix':('F16',rng.normal(size=(17,19))),'cancel':('BF16',np.zeros((3,5))),'vector':('F32',rng.normal(size=513)), 'zero':('BF16',np.zeros((3,7))), 'rounding':('F32',np.array([np.finfo(np.float32).max,-np.finfo(np.float32).max,-0.,np.float32(1e-45)]))}
rawb={'matrix':('F32',rawa['matrix'][1]+rng.normal(size=(17,19))*.25),'cancel':('F16',np.array([[-2.,2.,-2.,2.,1.],[2.,-2.,2.,-2.,3.],[1.,1.,1.,1.,4.]])),'vector':('BF16',rawa['vector'][1]*.8),'zero':('F32',np.zeros((3,7))), 'rounding':('BF16',np.array([1.,-1.,0.,0.]))}
va=write_model(A,rawa);vb=write_model(B,rawb)
# An immutable snapshot of synthetic source payloads must survive every command.
source_hash={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [A/'fixture.safetensors',B/'fixture.safetensors']}
model=run('metadata');identity=model['comparison_identity'];assert model['inference_editable'] is False
assert model['compatibility']['complete'] and model['coordinate_space']=='checkpoint-comparison-v1'
assert all(not t['calibration_complete'] for t in model['catalog'])
for t in model['catalog']:
 name=t['name'];xa=va[name][1].reshape(t['rows'],t['cols']);xb=vb[name][1].reshape(xa.shape);delta=xb-xa
 before=run('inspect','--tensor',t['id']);assert before['difference']['derived'] and before['inference_editable'] is False and 'tensor' not in before
 calibrated=run('calibrate','--tensor',t['id']);scales=calibrated['catalog'][t['id']]['scales'];rawmax=max(np.abs(xa).max(),np.abs(xb).max());diffmax=np.abs(delta).max();assert scales['shared_raw_max']==rawmax and scales['difference_max']==diffmax
 assert scales['nonzero_difference_count']==int((delta!=0).sum())
 for quantity,values in [('a',xa),('b',xb),('delta',delta),('abs_delta',np.abs(delta))]:
  bound=rawmax if quantity in ['a','b'] else diffmax
  for mapping in ['linear','asinh','magnitude']:
   if bound==0:transformed=np.zeros_like(values)
   elif mapping=='linear':transformed=values/bound
   elif mapping=='magnitude':transformed=np.abs(values)/bound
   else:scale=bound/100;transformed=np.arcsinh(values/scale)/np.arcsinh(bound/scale)
   transformed=np.clip(transformed,-1,1)
   for factor in sorted({1,2,min(16,2**t['max_level']),2**t['max_level']}):
    for tx,ty in sorted({(0,0),((t['cols']-1)//(256*factor),(t['rows']-1)//(256*factor))}):
     prefix=OUT/f"{t['id']}-{quantity}-{mapping}-{factor}-{tx}-{ty}"
     result=run('tile','--tensor',t['id'],'--quantity',quantity,'--mapping',mapping,'--level',t['max_level']-int(math.log2(factor)),'--x',tx,'--y',ty,'--out',prefix)
     sub=transformed[ty*256*factor:min(t['rows'],(ty+1)*256*factor),tx*256*factor:min(t['cols'],(tx+1)*256*factor)]
     want=np.array([[sub[r:r+factor,c:c+factor].mean() for c in range(0,sub.shape[1],factor)] for r in range(0,sub.shape[0],factor)])
     got=np.fromfile(str(prefix)+'.f64le','<f8').reshape(want.shape);np.testing.assert_allclose(got,want,rtol=2e-13,atol=2e-13)
     assert result['comparison_identity']==identity and result['inference_editable'] is False
     assert result['legend']['calibration_domain']==('shared_raw' if quantity in ['a','b'] else 'difference')
     width=sum(4 if d=='F32' else 2 for d in [t['dtype_a'],t['dtype_b']]);assert result['metrics']['max_raw_band_values']//2*width<=2*1024**2
     records.append(dict(name=name,quantity=quantity,mapping=mapping,factor=factor,tile=[tx,ty],error=float(np.abs(got-want).max())))
 for index in sorted({0,t['count']-1}):
  row,col=divmod(index,t['cols']);ins=run('inspect','--tensor',t['id'],'--row',row,'--col',col)
  for label,values,bits in [('a',xa,va[name][0]),('b',xb,vb[name][0])]:
   raw=ins['originals'][label];assert decimal.Decimal(raw['raw_exact'])==decimal.Decimal.from_float(float(values[row,col]));assert raw['raw_hex_le']==bits.ravel()[index].tobytes().hex()
   if values[row,col]==0:assert raw['raw_exact'].startswith('-')==bool(np.signbit(values[row,col]))
  assert ins['difference']['value']==delta[row,col]
# Ordered identities bind direction; reusing one cache after reversal discards old calibration.
reversed_model=run('metadata',a=B,b=A);assert reversed_model['comparison_identity']!=identity and all(not t['calibration_complete'] for t in reversed_model['catalog'])
# Refuse unsupported comparison mappings rather than approximating them.
for mapping in ['robust99','signed_percentile']:
 refusal=run('tile','--mapping',mapping,'--out',OUT/'unsupported',ok=False);assert refusal.returncode!=0 and 'Unsupported comparison mapping' in refusal.stderr
for filename,digest in source_hash.items():assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest
report=dict(passed=True,cases=len(records),max_error=max(r['error'] for r in records),source_hashes_unchanged=source_hash,records=records)
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
