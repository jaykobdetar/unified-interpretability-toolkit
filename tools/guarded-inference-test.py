#!/usr/bin/env python3
"""Run a single direct test process under inference RSS/headroom/CPU bounds."""
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_inference import available, GIB
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
if available() < 4.75 * GIB:
    raise SystemExit('Need 4.75 GiB available RAM')
p = subprocess.Popen(sys.argv[1:], env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
start, peak, minimum = time.monotonic(), 0, available()
try:
    while p.poll() is None:
        minimum = min(minimum, available())
        try:
            rss = int(next((s.split()[1] for s in Path(f'/proc/{p.pid}/status').read_text().splitlines() if s.startswith('VmRSS:')), 0)) * 1024
            peak = max(peak, rss)
        except FileNotFoundError:
            pass
        if peak > 1.5 * GIB or minimum < 3.25 * GIB or time.monotonic() - start > 120:
            p.kill()
            raise RuntimeError('Stopped own test process at resource/time limit')
        time.sleep(.05)
    print(f'Guard: peak RSS {peak / 1024**2:.1f} MiB, min available {minimum / GIB:.3f} GiB', file=sys.stderr)
    raise SystemExit(p.returncode)
finally:
    if p.poll() is None:
        p.kill()
    p.wait()
