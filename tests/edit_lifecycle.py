#!/usr/bin/env python3
"""One guarded real worker at a time; controlled post-edit cancellation/failure."""
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from inference_edits import SOURCE_MODEL
from live_inference import Session, verify_model
root = Path(__file__).resolve().parents[1]
directory, python = Path(sys.argv[1]), sys.argv[2]
request = {'prompt':'The capital of France is','max_new_tokens':2,'layer':0,'source_model':SOURCE_MODEL,
           'edits':[dict(tensor='model.layers.0.self_attn.q_proj.weight',shape=[576,576],kind='rows',operation='zero',start=0,end=64)]}
verify_model(directory)
results = []
session = Session(python,directory)
real_popen = subprocess.Popen
try:
    for mode in ('cancel','failure','pristine'):
        def controlled(args, **kwargs):
            return real_popen([python,'-B',str(root/'tests/controlled_edit_worker.py'),str(directory),mode], **kwargs)
        with patch('live_inference.subprocess.Popen', side_effect=controlled):
            snapshot = session.start({**request,'edits':[] if mode=='pristine' else request['edits']})
        owner, child = snapshot['session'], session.process
        started = time.monotonic()
        saw_applied = False
        while session.process is not None:
            session.last_seen = time.monotonic()
            session.tick()
            if session.details.get('comparison_phase') == 'edited':
                saw_applied = True
                if mode == 'cancel': session.stop()
            if time.monotonic()-started>90:
                raise AssertionError('Test deadline before normal session deadline')
            time.sleep(.03)
        assert child.poll() is not None and not session.snapshot()['worker_alive']
        if mode == 'cancel': assert saw_applied and session.status == 'cancelled'
        elif mode == 'failure':
            assert session.status=='error' and session.details['error']=='ControlledAfterEditFailure'
        else:
            assert session.status=='complete'
            assert session.details['baseline'] == session.details['edited']
            assert all(c['delta']==0 for step in session.steps for c in step['candidates'])
        verify_model(directory)
        results.append({'mode':mode,'status':session.status,'worker_reaped':True,'disk_hashes_unchanged':True,
                        'peak_rss_mib':session.snapshot()['peak_worker_rss_mib'],
                        'minimum_available_gib':session.snapshot()['minimum_available_gib']})
        old_owner=owner
        session.id='test-next-owner'
        assert not session.owns(old_owner)
    print(json.dumps({'status':'PASS','scope':'controlled actual model post-edit failure/cancellation, then fresh pristine session', 'runs':results},indent=2))
finally:
    session.stop()
