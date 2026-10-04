#!/usr/bin/env python3
"""Three normal sequential jobs: pristine control, owned cancel, fresh control.

No failed/malformed requests or artificial process faults. A raced completion is
reported as incomplete cancellation coverage, not a successful cancellation test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from acceptance.owner_client import OwnerClient, clean, parity
ROOT=Path(__file__).resolve().parents[1]

def verify_pins(directory):
    pins=json.loads((ROOT/'docs/models/smollm2-135m.json').read_text())['files']
    for name,expected in pins.items():
        digest=hashlib.sha256()
        with (directory/name).open('rb') as source:
            while chunk:=source.read(1024*1024):digest.update(chunk)
        assert digest.hexdigest()==expected,name
    return pins

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True);parser.add_argument('--model',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});args.out.mkdir(parents=True,exist_ok=False)
    client=None;report={'status':'NOT_COMPLETED','scope':'Normal owner cancellation; no deterministic post-edit cancellation or failure-injection claim'}
    try:
        before=verify_pins(args.model);client=OwnerClient(args.base);meta=client.api();source=meta['comparison']['source_model'];assert meta.get('sweep')
        control={'prompt':'The capital of France is','max_new_tokens':1,'layer':0,'activation_site':'block','source_model':source,'edits':[]}
        def progress(s):
            value={'status':s['status'],'cleanup_confirmed':clean(s),'received_records':len(s.get('steps',[])),
                   'coverage':s.get('details',{}).get('sweep_coverage'),'accepted_jobs':client.starts}
            (args.out/'progress.json').write_text(json.dumps(value,indent=2)+'\n')
        client.start(control);first_owner=client.session;before_control=parity(client.finish(progress));report['initial_empty_control']='PASS'
        fixture=json.loads((ROOT/'tests/fixtures/sweep-maximum-acceptance.json').read_text());request=fixture['request_without_digest'];assert request['source_model']==source
        plan=client.api('sweep-plan',request)['plan'];assert plan==fixture['resolved_plan']
        admitted=client.start({**request,'plan_digest':plan['digest']});cancel_owner=client.session;assert cancel_owner!=first_owner
        report['cancel_requested_status']=admitted['status'];report['records_before_cancel']=len(admitted['steps']);report['cancel_requested_phase']=admitted.get('details',{}).get('comparison_phase')
        # Cancel promptly through the ordinary owned endpoint; do not delay/spin to manufacture a post-edit race.
        client.cancel();cancelled=client.finish(progress);report['cancel_return_status']=cancelled['status'];report['cancel_cleanup_confirmed']=clean(cancelled);report['cancel_coverage']=cancelled.get('details',{}).get('sweep_coverage')
        assert clean(cancelled)
        client.start(control);assert client.session not in (first_owner,cancel_owner);after_control=parity(client.finish(progress));assert after_control==before_control
        report['fresh_control_exact_token_text_logits_activation_parity']=True;report['accepted_jobs']=client.starts;assert client.starts==3
        report['disk_hashes_unchanged']=verify_pins(args.model)==before;report['peak_worker_rss_mib']=client.peak_worker_rss_mib
        report['status']='PASS' if cancelled['status']=='cancelled' else 'PARTIAL_CANCELLATION_NOT_OBSERVED'
        report['no_retries_or_automatic_resume']=True
        exit_code=0 if report['status']=='PASS' else 3
    except Exception as exc:
        report.update(status='FAIL_OR_INTERRUPTED',message=str(exc));exit_code=1
    finally:
        if client is not None:
            report['final_owned_cleanup_confirmed']=client.cleanup()
            if not report['final_owned_cleanup_confirmed']:report['status']='FAIL_CLEANUP_UNCONFIRMED';exit_code=1
        if 'before' in locals():
            try:report['disk_hashes_unchanged']=verify_pins(args.model)==before
            except (OSError,AssertionError) as exc:report.update(status='FAIL_PIN_READBACK',pin_error=str(exc));exit_code=1
        (args.out/'sweep-owner-acceptance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

    return exit_code

if __name__=='__main__':raise SystemExit(main())
