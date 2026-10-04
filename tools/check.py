#!/usr/bin/env python3
"""Portable CI checks; no weights, browser, GPU, or numerical Python packages."""
import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONPATH': os.pathsep.join([str(ROOT / 'tools'), str(ROOT / 'tests')])}
    subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=['lint', 'contracts'])
    args = parser.parse_args()
    if args.suite == 'lint':
        for base in ('tools', 'tests'):
            for file in sorted((ROOT / base).rglob('*.py')):
                ast.parse(file.read_text(), filename=str(file.relative_to(ROOT)))
        for base in ('web', 'tests'):
            for file in sorted((ROOT / base).rglob('*')):
                if file.suffix in ('.js', '.cjs', '.mjs') and 'vendor' not in file.parts:
                    run(['node', '--check', str(file.relative_to(ROOT))])
        run(['bash', '-n', 'run-atlas.sh'])
        return
    python = sorted((ROOT / 'tests').glob('*contracts.py'))
    python += [ROOT / 'tests' / name for name in (
        'host_profile_adapter.py', 'ui_polish_readiness.py',
        'profile_worker_primitives.py', 'profile_review_regressions.py',
        'profile_host_supervisor.py', 'profile_runtime_doubles.py',
        'profile_review_fixes.py', 'profile_platform_preflight.py',
        'profile_http_integration.py', 'profile_http_ownership.py',
        'profile_internal_diagnostics.py', 'profile_lifetime_policy.py',
        'profile_spawn_registration.py', 'profile_startup_diagnostics.py',
        'analytics/test_core.py', 'analytics/test_service.py')]
    javascript = sorted((ROOT / 'tests').glob('ui-*.cjs'))
    javascript = [p for p in javascript if 'browser' not in p.name]
    javascript += sorted((ROOT / 'tests').glob('profile_*.js'))
    javascript += [ROOT / 'tests' / name for name in (
        'acceptance-support.cjs', 'acceptance-pin-buffer.cjs', 'data-view-contracts.cjs',
        'host-client.cjs', 'inference-availability.cjs', 'experiment_logs.cjs',
        'workspace-tools.cjs', 'workspace-ui.cjs', 'analytics/mount_contract.mjs',
        'analytics/ui_contract.mjs')]
    for file in python:
        run([sys.executable, '-B', str(file.relative_to(ROOT))])
    variants = {
        'ui-combined-context.cjs': ['slice-change', 'model-refresh', 'host-reopen'],
        'ui-pending-inspection-ownership.cjs': ['restore', 'prior-pin', 'newer-complete', 'newer-pending', 'cancel-before', 'cancel-opening', 'tensor-change', 'binding-mismatch'],
        'ui-readiness-ordering.cjs': ['busy', 'failure', 'finally', 'session'],
        'ui-recolor-selection.cjs': ['successful', 'pending'],
        'ui-head-selection-calibration.cjs': ['', 'pending-first-read'],
    }
    for file in javascript:
        for variant in variants.get(file.name, ['']):
            run(['node', str(file.relative_to(ROOT)), *([variant] if variant else [])])
    print(f'PASS: {len(python)} Python and {len(javascript)} Node contract files')


if __name__ == '__main__':
    main()
