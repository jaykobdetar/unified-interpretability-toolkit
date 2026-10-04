#!/usr/bin/env python3
"""Independent NumPy/struct/Decimal dtype acceptance. Run only in a granted heavy slot."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import decimal,json,math,resource,struct,subprocess
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/dtype-reference';OUT.mkdir(parents=True,exist_ok=True)
MODEL=OUT/'model';MODEL.mkdir(exist_ok=True)
CACHE=OUT/'cache';BIN=ROOT/'target/release/weight-atlas-rust'
RULES=['global_linear','global_asinh','tensor_linear','tensor_asinh','tensor_magnitude']
records=[]
def run(command,*args,model=MODEL,cache=CACHE,ok=True):
 p=subprocess.run([str(BIN),command,'--model',str(model),'--cache',str(cache),*map(str,args)],capture_output=True,text=True)
 with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(command=command,args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),default=str)+'\n')
 if ok:assert p.returncode==0,p.stderr
 return json.loads(p.stdout) if p.returncode==0 else p

def write_model(model,tensors):
 raw=bytearray();h={}
 for name,(dtype,bits) in tensors.items():
  start=len(raw);raw.extend(bits.tobytes());h[name]=dict(dtype=dtype,shape=list(bits.shape),data_offsets=[start,len(raw)])
 text=json.dumps(h).encode();(model/'fixture.safetensors').write_bytes(struct.pack('<Q',len(text))+text+raw)

def values(dtype,bits):
 if dtype=='BF16':return (bits.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
 return bits.view('<f2' if dtype=='F16' else '<f4').astype(np.float64)

rng=np.random.default_rng(160032)
half=np.arange(65536,dtype='<u2');half=half[(half&0x7c00)!=0x7c00].reshape(248,256)
f32=rng.integers(0,2**32,size=(257,259),dtype='<u4');f32[(f32&0x7f800000)==0x7f800000]=0
bf=rng.integers(0,65536,size=(17,19),dtype='<u2');bf[(bf&0x7f80)==0x7f80]=0
tensors={'half_all_finite':('F16',half),'float_signed_random':('F32',f32),'bf16':('BF16',bf),
 'float_edges':('F32',np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff],dtype='<u4')),
 'half_zero':('F16',np.zeros((3,7),dtype='<u2')),'float_zero':('F32',np.zeros((3,7),dtype='<u4')),
 'half_cancel':('F16',np.array([[0x3c00,0xbc00]],dtype='<u2')),'float_cancel':('F32',np.array([[0x3f800000,0xbf800000]],dtype='<u4')),
 'float_band':('F32',(rng.normal(size=(129,8193)).astype('<f4')).view('<u4'))}
write_model(MODEL,tensors)
meta=run('metadata');cal=run('calibrate')['model'];G=cal['global_max']
payload=json.loads(json.loads((CACHE/'calibration.json').read_text())['payload']);assert payload['version']==4
for t in meta['catalog']:
 dtype,bits=tensors[t['name']];x=values(dtype,bits).reshape(t['rows'],t['cols']);a=np.abs(x);M=float(a.max());positive=a[a>0];median=float(np.median(positive)) if positive.size else 0.
 st=payload['tensors'][str(t['id'])];assert st['dtype']==dtype and st['max_abs']==M and st['exact_zero_count']==int((x==0).sum())
 np.testing.assert_allclose(st['q99'],np.quantile(a,.99),rtol=3e-16,atol=0)
 np.testing.assert_allclose(st['median_nonzero_abs'],median,rtol=3e-16,atol=0)
 assert (st['unique_bit_patterns'] is None)==(dtype=='F32')
 width=4 if dtype=='F32' else 2;assert t['element_bytes']==width
 for factor in sorted({1,2,16,2**t['max_level']}):
  if factor>2**t['max_level']:continue
  for tx,ty in sorted({(0,0),((t['cols']-1)//(256*factor),(t['rows']-1)//(256*factor))}):
   r0,c0=ty*256*factor,tx*256*factor;v=x[r0:min(t['rows'],r0+256*factor),c0:min(t['cols'],c0+256*factor)]
   prefix=OUT/f"{t['id']}-{factor}-{tx}-{ty}"
   for batch in [RULES[:4],RULES[4:]]:
    result=run('tile','--tensor',t['id'],'--rules',','.join(batch),'--level',t['max_level']-int(math.log2(factor)),'--x',tx,'--y',ty,'--out',prefix)
    assert result['metrics']['max_raw_band_values']*width<=2*1024**2
    for rule in batch:
     bound=G if rule.startswith('global') else M
     scale=(G/100 if G else 1) if rule.startswith('global') else (median or 1)
     if rule.endswith('asinh'):transformed=np.arcsinh(v/scale)/(np.arcsinh(bound/scale) or 1)
     elif rule=='tensor_magnitude':transformed=np.abs(v)/(M or 1)
     else:transformed=v/(bound or 1)
     transformed=np.clip(transformed,-1,1)
     want=np.array([[transformed[r:r+factor,c:c+factor].mean() for c in range(0,v.shape[1],factor)] for r in range(0,v.shape[0],factor)])
     got=np.fromfile(str(prefix)+'-'+rule+'.f64le','<f8').reshape(want.shape)
     np.testing.assert_allclose(got,want,rtol=2e-13,atol=0 if factor==1 and not rule.endswith('asinh') else 2e-13)
     records.append(dict(tensor=t['name'],dtype=dtype,rule=rule,factor=factor,tile=[tx,ty],max_error=float(np.abs(got-want).max()),metrics=result['metrics']))
 for index in sorted({0,t['count']-1,*rng.integers(0,t['count'],size=8).tolist()}):
  row,col=divmod(index,t['cols']);r=run('inspect','--tensor',t['id'],'--row',row,'--col',col,'--left','tensor_magnitude','--right','tensor_linear')
  raw=bits.ravel()[index].tobytes();assert r['raw_hex_le']==raw.hex() and r['dtype']==dtype and r['element_bytes']==width
  assert r['byte_offset']==t['byte_offset']+width*index
  assert r['native_indices']==([col] if bits.ndim==1 else [row,col])
  number=float(x[row,col]);assert decimal.Decimal(r['raw_exact'])==decimal.Decimal.from_float(number)
  if number==0:assert r['raw_exact'].startswith('-')==bool(np.signbit(number))
  assert r['transformed']['left']==(abs(number)/(M or 1))
# IEEE special bit patterns remain inspectable; calibration refuses publication.
for dtype,words in [('F16',[0x7c00,0xfc00,0x7e01]),('F32',[0x7f800000,0xff800000,0x7fc01234])]:
 for word in words:
  d=OUT/f'nonfinite-{dtype}-{word}';d.mkdir(exist_ok=True);cache=OUT/f'nonfinite-cache-{dtype}-{word}'
  write_model(d,{'special':(dtype,np.array([word],dtype='<u2' if dtype=='F16' else '<u4'))})
  r=run('inspect',model=d,cache=cache);assert r['classification']!='finite' and not r['transforms_ready']
  p=run('calibrate',model=d,cache=cache,ok=False);assert p.returncode!=0 and 'Nonfinite' in p.stderr
  assert not (cache/'calibration.json').exists()
report=dict(passed=True,cases=len(records),max_error=max(r['max_error'] for r in records),records=records)
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
