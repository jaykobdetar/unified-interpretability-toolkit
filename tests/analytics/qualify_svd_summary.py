#!/usr/bin/env python3
"""One explicit synthetic128 qualification; run only after the parent assigns a slot.

--prepare-only writes a 32 KiB BF16 fixture and plan outside the checkout, with no
renderer, worker, or NumPy. Run mode uses existing renderer metadata and the real
owned analytics child, then one separately bounded symmetric-eigenvalue oracle.
No HTTP/browser qualification, public weights, installs, retries, or seed search.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
from analytics import svd_summary as summary


def independent_order(n, seed):
    # Independent spelling of the documented uint32 rejection/Fisher-Yates rule.
    state = seed if seed else 0x6D2B79F5
    out = list(range(n))
    for i in range(n-1, 0, -1):
        limit = (2**32//(i+1))*(i+1)
        while True:
            state ^= (state << 13) & 0xffffffff
            state ^= state >> 17
            state ^= (state << 5) & 0xffffffff
            state &= 0xffffffff
            if state < limit: break
        j = state % (i+1)
        out[i], out[j] = out[j], out[i]
    return out


def oracle(directory):
    from analytics.worker import configure
    configure()  # Before importing NumPy: one CPU, <=768 MiB /4 CPU sec/5 wall sec.
    import numpy as np
    report = json.loads((directory/'result.json').read_bytes())
    source = (directory/'fixture'/'synthetic.safetensors').read_bytes()
    size = struct.unpack('<Q',source[:8])[0]
    words = struct.unpack('<16384H',source[8+size:])
    values = [struct.unpack('<f',struct.pack('<I',word << 16))[0] for word in words]
    order = independent_order(16384,77)
    sha = hashlib.sha256(struct.pack('<16384I',*order)).hexdigest()
    assert sha == report['control']['permutation_sha256']
    assert report['control']['preview_position_to_source'] == [order[i] for i in report['preview']['positions']]
    assert sorted(struct.pack('<d',x) for x in values) == sorted(struct.pack('<d',values[i]) for i in order)
    energy = sum(i*i for i in range(1,129))
    expected_original = list(range(128,0,-1))
    a = report['results']['original']
    np.testing.assert_allclose(a['singular_values'],expected_original,rtol=1e-12,atol=1e-12)
    assert a['frobenius_energy'] == energy
    np.testing.assert_allclose(a['rank_one_residual_energy_fraction'],(energy-128**2)/energy,rtol=1e-12)
    # Distinct algorithm for control: symmetric Gram eigensystem, no np.linalg.svd.
    matrix = np.asarray([values[i] for i in order],dtype=np.float64).reshape(128,128)
    eigen, vectors = np.linalg.eigh(matrix.T @ matrix)
    assert eigen[-1]-eigen[-2] > 1e-8*eigen[-1], 'Leading eigenvalue must be unique for preview comparison'
    spectrum = np.sqrt(np.maximum(eigen[::-1],0))
    b = report['results']['shuffled']
    # Gram eigenvalues lose precision near zero; compare squared spectrum instead.
    np.testing.assert_allclose(np.asarray(b['singular_values'])**2,spectrum**2,rtol=1e-9,atol=1e-8)
    np.testing.assert_allclose(b['rank_one_residual_energy_fraction'],1-eigen[-1]/energy,rtol=1e-10,atol=1e-12)
    direction = vectors[:,-1]
    residual = matrix-np.outer(matrix @ direction,direction)
    np.testing.assert_allclose(b['rank_one_residual_preview'],[residual.ravel()[i] for i in report['preview']['positions']],rtol=1e-8,atol=1e-8)
    expected_preview = [0 if i//128 == i%128 == 0 else (128-i//128 if i//128 == i%128 else 0) for i in report['preview']['positions']]
    np.testing.assert_allclose(a['rank_one_residual_preview'],expected_preview,rtol=1e-12,atol=1e-12)
    assert len(json.dumps(report).encode()) <= 63488
    evidence = {'status':'PASS','numpy':np.__version__,'independent_original':'analytic diag(128,...,1)',
                'independent_control':'symmetric Gram eigh; independent uint32 permutation; raw-bit multiset',
                'full_selected_spectra_checked':True,'residual_preview_values_checked_per_side':256,
                'omitted_residual_positions_per_side':16128,'full_model':False,'http_browser_qualified':False}
    (directory/'oracle.json').write_text(json.dumps(evidence,indent=2)+'\n')


def prepare(directory):
    if directory == ROOT or ROOT in directory.parents:
        raise ValueError('Evidence and source fixture must be outside the checkout')
    directory.mkdir(parents=True,exist_ok=True)
    fixture = directory/'fixture'; fixture.mkdir(exist_ok=True)
    header = json.dumps({'synthetic':{'dtype':'BF16','shape':[128,128],'data_offsets':[0,32768]}},separators=(',',':')).encode()
    raw = b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',128-r if r == c else 0))[0] >> 16) for r in range(128) for c in range(128))
    (fixture/'synthetic.safetensors').write_bytes(struct.pack('<Q',len(header))+header+raw)
    request = {'scope':'svd_summary','tensor':0,'region':{'row':0,'col':0,'rows':128,'cols':128},'seed':77}
    plan = {'status':'PREPARED_NOT_RUN','schema':summary.SCHEMA,'request':request,
            'source_bytes':32768,'source_values':16384,'fixture_sha256':hashlib.sha256(raw).hexdigest(),
            'expected_spectrum':list(range(128,0,-1)), 'expected_energy':sum(i*i for i in range(1,129)),
            'children':'one actual owned analytics worker, then one independent bounded oracle; no retries',
            'limits_per_numeric_child':{'cpu_count':1,'address_space_mib':768,'cpu_seconds':4,'wall_seconds':5},
            'response_body_bytes':63488,'required_authorization':'assigned parent heavy slot',
            'metadata_scope':'existing Rust CLI source metadata plus explicit synthetic revision binding; no live HTTP/browser claim'}
    (directory/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return fixture, request


def run(args, directory, fixture, request):
    from analytics.service import AnalyticsJobs
    from live_inference import available, signal_and_reap
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    def metadata():
        result = subprocess.run([args.renderer,'metadata','--model',str(fixture)],capture_output=True,timeout=10,check=True)
        model = json.loads(result.stdout)
        # CLI metadata is source-only; declare the synthetic revision explicitly.
        model['revision'] = 'synthetic128-diagonal-seed77'
        model['model_identity'] = summary.digest(['weight-atlas-model-v1',model['source_identity'],model['revision']])
        return model
    jobs = AnalyticsJobs(args.python,fixture,inference_busy=lambda:False,fetch_model=metadata,available=available,reap=signal_and_reap)
    before = (fixture/'synthetic.safetensors').stat()
    started = time.monotonic(); observed = set(); limits = set()
    minimum = available()
    try:
        first = jobs.start(request); assert jobs.owns(first['job']) and not jobs.owns('foreign')
        while jobs.process is not None:
            if time.monotonic()-started > 8: raise TimeoutError('Owned worker harness deadline')
            minimum = min(minimum,available())
            try:
                raw = Path(f'/proc/{jobs.process.pid}/status').read_text()
                observed.add(next(x.split(':',1)[1].strip() for x in raw.splitlines() if x.startswith('Cpus_allowed_list:')))
                limits.update(x for x in Path(f'/proc/{jobs.process.pid}/limits').read_text().splitlines() if x.startswith(('Max cpu time','Max address space')))
            except (OSError,StopIteration): pass
            jobs.tick(); time.sleep(.01)
        final = jobs.snapshot(); assert final['status'] == 'complete', final['error']
        assert minimum >= 3.25*1024**3 and observed == {str(min(os.sched_getaffinity(0)))}
        assert any(x.startswith('Max address space') and x.split()[3:5] == ['805306368','805306368'] for x in limits) and any(x.startswith('Max cpu time') and x.split()[3:5] == ['4','4'] for x in limits)
        after = (fixture/'synthetic.safetensors').stat()
        assert (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) == (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
        result = final['result']; assert len(json.dumps(result).encode()) <= 63488
        bound = result['source_binding']; expected_model = metadata()
        assert bound['source_identity'] == expected_model['source_identity'] and bound['model_identity'] == expected_model['model_identity']
        assert bound['shape'] == [128,128] and bound['tensor'] == 0 and bound['dtype'] == 'BF16'
        assert bound['slice'] == {'leading_indices':[], 'display_axes':[0,1]} and result['region'] == request['region']
        (directory/'result.json').write_text(json.dumps(result,allow_nan=False)+'\n')
        evidence = {'status':'WORKER_PASS_ORACLE_PENDING','worker_seconds':time.monotonic()-started,
                    'peak_worker_rss_mib':final['peak_worker_rss_mib'],'minimum_available_gib':minimum/1024**3,
                    'observed_cpu_affinity':sorted(observed),'observed_limits':sorted(limits),
                    'source_stat_identity_unchanged':True,'source_slice_revision_binding_checked':True,
                    'response_body_bytes':len(json.dumps(result).encode()),'worker_reaped':not jobs.busy}
        (directory/'worker.json').write_text(json.dumps(evidence,indent=2)+'\n')
    finally:
        assert jobs.stop(), 'Owned worker cleanup remains pending'
    env = {**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    subprocess.run([args.python,'-B',str(Path(__file__).resolve()),'--oracle','--evidence-dir',str(directory)],env=env,timeout=6,check=True)
    print('PASS: one synthetic128 worker + analytic/Gram oracle; HTTP/browser unqualified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--oracle',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--renderer'); parser.add_argument('--python')
    args = parser.parse_args(); directory = args.evidence_dir.resolve()
    if directory == ROOT or ROOT in directory.parents: parser.error('Evidence must be outside the checkout')
    if args.oracle: oracle(directory); return
    if any((directory/name).exists() for name in ('result.json','worker.json','oracle.json')):
        parser.error('Existing run evidence is immutable; use a fresh external directory for an explicitly authorized attempt')
    if not args.prepare_only and (not args.renderer or not args.python):
        parser.error('Run requires explicit existing --renderer and --python, plus an assigned slot')
    fixture, request = prepare(directory)
    if args.prepare_only: print('PREPARED_NOT_RUN: synthetic128 fixture and qualification plan'); return
    run(args,directory,fixture,request)


if __name__ == '__main__': main()
