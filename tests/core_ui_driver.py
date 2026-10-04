#!/usr/bin/env python3
"""Normal feature/API acceptance; owns its loopback servers and reaps them."""
import hashlib,json,os,socket,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(os.environ['ATLAS_EVIDENCE_DIR']);BIN=ROOT/'target/release/weight-atlas-rust'
servers=[];logs=[];checks=[]
def get(url):
 with urllib.request.urlopen(url,timeout=15) as r:return r.read(),dict(r.headers)
def value(url):return json.loads(get(url)[0])
def start(command,model,cache,extra=()):
 with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
 log=(OUT/(command+'.log')).open('w');logs.append(log)
 p=subprocess.Popen([str(BIN),command,'--model',str(model),'--cache',str(cache),'--port',str(port),*map(str,extra)],stdout=log,stderr=log);servers.append(p)
 base=f'http://127.0.0.1:{port}';suffix='/api/model' if command=='serve' else '/api/comparison/model'
 for _ in range(100):
  assert p.poll() is None,'Server exited during startup'
  try:return base,value(base+suffix)
  except OSError:time.sleep(.05)
 raise RuntimeError('Server startup deadline')

def cli(command,model,cache,*args):
 p=subprocess.run([str(BIN),command,'--model',str(model),'--cache',str(cache),*map(str,args)],capture_output=True,text=True)
 with (OUT/'cli.jsonl').open('a') as f:f.write(json.dumps({'command':command,'args':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},default=str)+'\n')
 assert p.returncode==0,p.stderr
 return json.loads(p.stdout)
try:
 source=ROOT/'results/dtype-reference/model';cache=ROOT/'results/dtype-reference/cache';cli_cache=OUT/'single-cli-cache'
 single,model=start('serve',source,cache)
 bf=next(t for t in model['catalog'] if t['name']=='bf16');cli('calibrate',source,cli_cache,'--tensor',bf['id'])
 # Global rules need complete checkpoint calibration in the independent CLI cache.
 cli('calibrate',source,cli_cache)
 for rule in ['global_linear','global_asinh','tensor_linear','tensor_asinh','tensor_magnitude','tensor_robust99','tensor_signed_percentile']:
  t=bf['id'];level=bf['max_level'];prefix=OUT/('single-'+rule)
  cli('tile',source,cli_cache,'--tensor',t,'--rules',rule,'--level',level,'--out',prefix)
  url=single+f'/tile?tensor={t}&rule={rule}&level={level}&x=0&y=0';png,h=get(url);assert png==Path(str(prefix)+'-'+rule+'.png').read_bytes();assert get(url)[1]['X-Atlas-Cache']=='hit'
  for side in ['left','right']:
   rules={'left':'tensor_linear','right':'tensor_linear',side:rule};v=value(single+f"/api/view?tensor={t}&left={rules['left']}&right={rules['right']}");assert v['legends'][side]['id']==rule
  inspected=value(single+f'/api/inspect?tensor={t}&row=0&col=0&left={rule}&right=tensor_linear');assert inspected==cli('inspect',source,cli_cache,'--tensor',t,'--row',0,'--col',0,'--left',rule,'--right','tensor_linear')
 # Delete only one generated tile to exercise normal stale-cache recovery.
 files=list((cache/'tiles').glob('*.png'));victim=next(p for p in files if p.read_bytes()==png);victim.unlink();assert get(url)[1]['X-Atlas-Cache']=='miss';assert get(url)[0]==png
 checks.append('Seven rules: CLI/API PNG and scalar equality, both-panel view legends, hits and missing-tile regeneration')
 pair_a=ROOT/'results/comparison-reference/a';pair_b=ROOT/'results/comparison-reference/b';pair_cache=ROOT/'results/comparison-reference/cache';pair_cli=OUT/'pair-cli-cache'
 pair,meta=start('compare-serve',pair_a,pair_cache,('--compare-model',pair_b));p=next(t for t in meta['catalog'] if t['name']=='matrix');pid=p['id'];cli('compare-calibrate',pair_a,pair_cli,'--compare-model',pair_b,'--tensor',pid)
 for quantity in ['a','b','delta','abs_delta']:
  for mapping in ['linear','asinh','magnitude']:
   prefix=OUT/f'pair-{quantity}-{mapping}';level=p['max_level']-1
   result=cli('compare-tile',pair_a,pair_cli,'--compare-model',pair_b,'--tensor',pid,'--quantity',quantity,'--mapping',mapping,'--level',level,'--out',prefix)
   url=pair+f'/api/comparison/tile?tensor={pid}&quantity={quantity}&mapping={mapping}&level={level}&x=0&y=0&comparison_identity={meta["comparison_identity"]}'
   png,h=get(url);assert png==Path(str(prefix)+'.png').read_bytes();assert h['X-Atlas-Inference-Editable']=='false';assert h['X-Atlas-Comparison-Identity']==meta['comparison_identity'];assert get(url)[1]['X-Atlas-Cache']=='hit'
 raw=value(pair+f'/api/comparison/inspect?tensor={pid}&row=1&col=2');assert raw==cli('compare-inspect',pair_a,pair_cli,'--compare-model',pair_b,'--tensor',pid,'--row',1,'--col',2)
 checks.append('Comparison four quantities × three mappings: CLI/API PNG parity, identity headers/cache hits and exact A/B/delta inspector parity')
 env={**os.environ,'ATLAS_SINGLE_URL':single,'ATLAS_PAIR_URL':pair}
 (OUT/'launch.json').write_text(json.dumps({'single':single,'comparison':pair,'owned_server_pids':[p.pid for p in servers],'binary_sha256':hashlib.sha256(BIN.read_bytes()).hexdigest()},indent=2)+'\n')
 (OUT/'api-report.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2)+'\n')
 with (OUT/'browser.stdout').open('w') as stdout,(OUT/'browser.stderr').open('w') as stderr:
  result=subprocess.run(['node','tests/core-browser.cjs'],env=env,stdout=stdout,stderr=stderr)
 assert result.returncode==0,(OUT/'browser.stderr').read_text()
finally:
 for p in reversed(servers):
  if p.poll() is None:p.terminate()
  p.wait(timeout=10)
 for f in logs:f.close()
 (OUT/'server-cleanup.json').write_text(json.dumps({'owned_server_pids':[p.pid for p in servers],'reaped':all(p.poll() is not None for p in servers)},indent=2)+'\n')
