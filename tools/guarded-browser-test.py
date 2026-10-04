#!/usr/bin/env python3
"""Guard only our fresh browser-test process tree; never unrelated processes."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_inference import available, GIB
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
if available() < 5 * GIB:
    raise SystemExit('Browser test requires 5 GiB available RAM')
p = subprocess.Popen(sys.argv[1:], start_new_session=True)
start, peak, minimum = time.monotonic(), 0, available()
owned = {p.pid}
try:
    while p.poll() is None:
        # Chromium can create process groups; follow actual parent relationships.
        table = {}
        for f in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = f.read_text().rsplit(')', 1)[1].split()
                table[int(f.parent.name)] = (int(fields[1]), int(fields[21]) * os.sysconf('SC_PAGE_SIZE'))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        owned = {p.pid}
        while True:
            children = {pid for pid, (parent, _) in table.items() if parent in owned}
            if children <= owned:
                break
            owned |= children
        rss = sum(table.get(pid, (0, 0))[1] for pid in owned)
        peak = max(peak, rss)
        minimum = min(minimum, available())
        if rss > 768 * 1024**2 or minimum < 3.25 * GIB or time.monotonic() - start > 120:
            raise RuntimeError(f'Browser test stopped: tree RSS {rss/1024**2:.1f} MiB, available {minimum/GIB:.3f} GiB, elapsed {time.monotonic()-start:.1f}s')
        time.sleep(.1)
    print(f'Browser guard: peak tree RSS {peak / 1024**2:.1f} MiB; min available {minimum/GIB:.3f} GiB', file=sys.stderr)
    raise SystemExit(p.returncode)
finally:
    for pid in sorted(owned - {p.pid}, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if p.poll() is None:
        p.terminate()
    p.wait()
