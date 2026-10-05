#!/usr/bin/env python3
"""Opt-in exact pinned CPU protocols; read held fixtures, never record baselines."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/cpu-protocols'


def manifest():
    data = json.loads((FIXTURES / 'manifest.json').read_text())
    assert data['schema'] == 'cpu-protocol-reference-v1'
    assert len(data['cases']) == 10
    for case in data['cases'].values():
        payload = (FIXTURES / case['golden']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == case['golden_sha256']
    return data


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, 'Duplicate protocol field'
        result[key] = value
    return result


def encoding(events):
    return ''.join(json.dumps(event, ensure_ascii=True, separators=(',', ':'),
                              allow_nan=False) + '\n' for event in events)


def stable_protocol(raw, case):
    events = [json.loads(line, object_pairs_hook=unique_object)
              for line in raw.splitlines()]
    assert encoding(events) == raw, 'Worker JSON byte layout changed'
    assert len(events) == len(case['clock_paths_by_event']), 'Event count changed'
    for event, paths in zip(events, case['clock_paths_by_event']):
        assert event.get('type') != 'error', 'Worker reported an error'
        for path in paths:
            owner = event
            keys = path.split('.')
            for key in keys[:-1]:
                owner = owner[key]
            value = owner.pop(keys[-1])
            assert type(value) in (int, float) and math.isfinite(value) and value >= 0
    return encoding(events)


def compare(raw, name, data=None):
    data = manifest() if data is None else data
    case = data['cases'][name]
    actual = stable_protocol(raw, case)
    expected = (FIXTURES / case['golden']).read_text()
    assert actual == expected, ('Exact CPU protocol changed: ' + name +
                                '; expected SHA-256 ' + case['golden_sha256'] +
                                '; actual ' + hashlib.sha256(actual.encode()).hexdigest())
    return case['golden_sha256']


def execute_case(name, case, python, model, output):
    # The existing guard observes the actual worker directly. Sweep exec retains
    # that PID and the worker's original total-job 120s wall / 90s CPU budget.
    worker = ROOT / 'tools/inference_worker.py'
    target = [str(python), '-B', str(worker), str(model)]
    if case['request'].get('mode') == 'sweep':
        target = [sys.executable, '-B', str(Path(__file__).resolve()),
                  '--exec-sweep', str(python), str(worker), str(model)]
    command = [sys.executable, '-B', str(ROOT / 'tools/guarded-inference-test.py'), *target]
    request = json.dumps(case['request'], separators=(',', ':')) + '\n'
    (output / 'request.json').write_text(request)
    (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    started = time.monotonic()
    process = None
    owned = {}
    spec = importlib.util.spec_from_file_location('protocol_cleanup', ROOT / 'tools/guarded-core-ui.py')
    cleanup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cleanup)
    try:
        with (output / 'stdout').open('w') as stdout, (output / 'stderr').open('w') as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            rows = cleanup.table()
            owned[process.pid] = rows[process.pid][1]
            process.communicate(request.encode(), timeout=130)
        (output / 'exit').write_text(str(process.returncode) + '\n')
        assert process.returncode == 0, 'Worker/guard failed; retain stdout/stderr/exit'
        return (output / 'stdout').read_text()
    finally:
        # Normal guard completion already kills/waits its worker. An outer stop
        # reuses the existing bounded PID/start-time cleanup rather than a new
        # supervisor or signals to unverified process-group members.
        interrupted = process is not None and process.poll() is None
        receipt = {'cleanup_verified': True, 'remaining_owned_pids': [], 'cleanup_errors': []}
        if interrupted:
            receipt = cleanup.cleanup(process, owned, os.getpid())
        if process is not None and process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        (output / 'lifecycle.json').write_text(json.dumps({
            'elapsed_seconds': time.monotonic() - started,
            'guard_reaped': process is not None and process.poll() is not None,
            'interrupted': interrupted,
            'worker_cleanup': 'existing guard kills and waits its direct worker',
            **receipt,
        }, indent=2) + '\n')
        assert receipt['cleanup_verified'] and not receipt['remaining_owned_pids'] and not receipt['cleanup_errors']


def main():
    if len(sys.argv) == 5 and sys.argv[1] == '--exec-sweep':
        python, worker, model = sys.argv[2:]
        os.execv(python, [python, '-B', worker, model, str(time.monotonic() + 120), '90'])
    data = manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=list(data['cases']))
    args = parser.parse_args()
    output = args.output.resolve()
    model = args.model.resolve()
    assert output != model and model not in output.parents, 'Evidence cannot modify model files'
    output.mkdir(parents=True, exist_ok=False)
    names = [args.case] if args.case else list(data['cases'])
    results = []
    for name in names:
        for repetition in (1, 2):
            target = output / (name + '-' + str(repetition))
            target.mkdir()
            # Preserve a venv executable's symlink path: resolving it selects the
            # base interpreter and loses that environment's pinned CPU packages.
            raw = execute_case(name, data['cases'][name], args.python.absolute(), model, target)
            checksum = compare(raw, name, data)
            results.append({'case': name, 'repetition': repetition, 'golden_sha256': checksum})
            print('PASS: ' + name + ' repetition ' + str(repetition), flush=True)
    (output / 'results.json').write_text(json.dumps({'status': 'PASS', 'runs': results,
                                                   'baseline_written': False}, indent=2) + '\n')


if __name__ == '__main__':
    # An outer stop must reach the owned guard/worker cleanup above.
    def stopped(_signal, _frame):
        raise KeyboardInterrupt('CPU protocol qualification interrupted')
    signal.signal(signal.SIGTERM, stopped)
    main()
