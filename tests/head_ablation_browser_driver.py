#!/usr/bin/env python3
"""Own one normal loopback coordinator and separately guarded browser lifetime.
No resource limit changes; outer 120s lifetime contains both explicit UI cases.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from live_inference import available,GIB
spec=importlib.util.spec_from_file_location('qualification_guard',ROOT/'tools/guarded-core-ui.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
OUT=Path(os.environ['ATLAS_EVIDENCE_DIR']);OUT.mkdir(parents=True,exist_ok=True)
MODEL=Path(os.environ['ATLAS_MODEL_DIR']);PYTHON=os.environ['ATLAS_CPU_PYTHON']
NODE_PATH=os.environ.get('NODE_PATH', '')
CHROMIUM=os.environ['ATLAS_CHROMIUM']

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def port():
    with socket.socket() as handle:
        handle.bind(('127.0.0.1',0));return handle.getsockname()[1]

started=time.monotonic();coordinator=None;browser=None;logs=[];error=None;cleanup=None
own=os.getpid();owned={own:guard.table()[own][1]};result={}
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
try:
    # Coordinator rechecks its own existing 4.75 GiB gate on launch. Browser's
    # independent existing 5 GiB gate runs immediately before its process tree.
    if available()<4.75*GIB:raise RuntimeError('Coordinator launch refused: need 4.75 GiB available RAM')
    a,b=port(),port()
    if a==b:raise RuntimeError('Fresh port collision; no alternate launch')
    for value in (a,b):
        if value in (8774,8775,8785):raise RuntimeError('Reserved port selected; no alternate launch')
    result['ports']={'coordinator':a,'renderer':b};result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result['renderer_sha256']=hashlib.sha256((ROOT/'target/release/weight-atlas-rust').read_bytes()).hexdigest()
    command=[sys.executable,'tools/live_inference.py','--model',str(MODEL),'--python',PYTHON,'--port',str(a),'--atlas-port',str(b)]
    for name in ('coordinator.stdout.txt','coordinator.stderr.txt'):logs.append((OUT/name).open('w'))
    coordinator=subprocess.Popen(command,cwd=ROOT,stdout=logs[0],stderr=logs[1],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    owned[coordinator.pid]=guard.table()[coordinator.pid][1]
    base=f'http://127.0.0.1:{a}'
    while True:
        guard.discover(guard.table(),owned)
        if coordinator.poll() is not None:raise RuntimeError(f'Coordinator exited during startup: {coordinator.returncode}')
        if time.monotonic()-started>15:raise RuntimeError('Coordinator startup deadline')
        if available()<3.25*GIB:raise RuntimeError('Stop reserve reached during coordinator startup')
        try:
            with urllib.request.urlopen(base+'/api/model',timeout=.5) as response:
                metadata=json.load(response)
            break
        except (OSError,ValueError):time.sleep(.1)
    result['renderer_source_identity']=metadata['source_identity'];result['renderer_model_identity']=metadata['model_identity']
    result['pre_browser_available_gib']=available()/GIB
    browser_out=OUT/'browser';browser_out.mkdir(exist_ok=False)
    env={**os.environ,'ATLAS_TEST_URL':base,'ATLAS_EVIDENCE_DIR':str(browser_out),'ATLAS_MODEL_DIR':str(MODEL),'NODE_PATH':NODE_PATH,'ATLAS_CHROMIUM':CHROMIUM,'PYTHONDONTWRITEBYTECODE':'1'}
    logs.append((OUT/'browser-driver.stdout.txt').open('w'));logs.append((OUT/'browser-driver.stderr.txt').open('w'))
    browser=subprocess.Popen([sys.executable,'tools/guarded-core-ui.py','node','tests/head-ablation-browser.cjs'],cwd=ROOT,env=env,stdout=logs[2],stderr=logs[3])
    owned[browser.pid]=guard.table()[browser.pid][1]
    while browser.poll() is None:
        guard.discover(guard.table(),owned)
        if coordinator.poll() is not None:raise RuntimeError('Coordinator exited before browser completed')
        if available()<3.25*GIB:raise RuntimeError('Stop reserve reached during browser acceptance')
        if time.monotonic()-started>110:raise RuntimeError('Outer120s lifetime cleanup margin reached')
        time.sleep(.1)
    result['browser_guard_exit_code']=browser.returncode
    if browser.returncode:raise RuntimeError(f'Guarded browser did not pass: exit{browser.returncode}')
except BaseException as exc:
    error=f'{type(exc).__name__}: {exc}'
finally:
    try:
        if coordinator is not None and coordinator.poll() is None:
            coordinator.terminate()
            coordinator.wait(timeout=3)
        if coordinator is not None:result['coordinator_exit_code']=coordinator.poll()
    except BaseException as exc:error='; '.join(filter(None,[error,f'Coordinator cleanup: {exc}']))
    try:
        cleanup=guard.cleanup(browser,owned,own)
        if coordinator is not None and coordinator.poll() is None:coordinator.wait(timeout=1)
    except BaseException as exc:cleanup={'cleanup_verified':False,'error':str(exc)}
    if not cleanup.get('cleanup_verified') or cleanup.get('remaining_owned_pids') or cleanup.get('cleanup_errors'):
        error='; '.join(filter(None,[error,'Owned-tree cleanup not fully confirmed']))
    for handle in logs:handle.close()
    result.update(status='PASS' if error is None else 'FAIL_OR_REFUSED',error=error,elapsed_seconds=time.monotonic()-started,cleanup=cleanup,
        browser_guard_scope='Guard, Node and Chromium descendants only; coordinator/renderer are separately owned here and retain their existing guards; model1.5GiB/120s/90CPU guards remain in coordinator/worker',
        resource_policy=f'OneCPU, browser{guard.BROWSER_RSS_CAP_BYTES/1024**2:g}MiBtree/start5GiB, model1.5GiBRSS/start4.75GiB, stop3.25GiB; outer lifetime120s including cleanup; user-approved browser RSS policy; no other cap changes',
        no_retries=True)
    save('driver-result.json',result)
    print(json.dumps(result,indent=2))
raise SystemExit(0 if error is None else 1)
