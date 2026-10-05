#!/usr/bin/env python3
"""Qualification guard: one CPU, 768 MiB physical owned browser tree, no sandbox changes."""

import json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from live_inference import available, GIB

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
if available() < 5 * GIB:
    raise SystemExit("Browser launch requires 5 GiB available")
output = ROOT / "results/analytics-qualification/browser-guard.json"
p = subprocess.Popen(sys.argv[1:], start_new_session=True)
seen = {}
peak = 0
minimum = available()
started = time.monotonic()
failure = None


def table():
    result = {}
    for f in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = f.read_text().rsplit(")", 1)[1].split()
            result[int(f.parent.name)] = (
                int(fields[1]),
                int(fields[19]),
                int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
            )
        except (OSError, ValueError, IndexError):
            pass
    return result


try:
    while p.poll() is None:
        current = table()
        owned = {p.pid}
        while True:
            children = {
                pid for pid, (parent, _, _) in current.items() if parent in owned
            }
            if children <= owned:
                break
            owned |= children
        for pid in owned:
            if pid in current:
                seen[pid] = current[pid][1]
        # Count still-live known descendants even if reparented, with start-time identity.
        alive = {
            pid
            for pid, stamp in seen.items()
            if pid in current and current[pid][1] == stamp
        }
        rss = sum(current[pid][2] for pid in alive)
        peak = max(peak, rss)
        minimum = min(minimum, available())
        if (
            rss > 768 * 1024**2
            or minimum < 3.25 * GIB
            or time.monotonic() - started > 100
        ):
            raise RuntimeError(
                f"Guard stopped: tree={rss/1024**2:.1f} MiB available={minimum/GIB:.3f} GiB elapsed={time.monotonic()-started:.1f}s"
            )
        time.sleep(0.05)
    if p.returncode:
        failure = f"Browser harness exited {p.returncode}"
except Exception as exc:
    failure = str(exc)
finally:
    current = table()
    for pid, stamp in seen.items():
        if pid != p.pid and pid in current and current[pid][1] == stamp:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    if p.poll() is None:
        p.terminate()
    try:
        p.wait(timeout=2)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=2)
    time.sleep(0.1)
    current = table()
    remaining = []
    for pid, stamp in seen.items():
        if pid != p.pid and pid in current and current[pid][1] == stamp:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            remaining.append(pid)
    report = {
        "status": "FAIL" if failure else "PASS",
        "error": failure,
        "cap_mib": 768,
        "peak_owned_tree_rss_mib": round(peak / 1024**2, 2),
        "minimum_available_gib": round(minimum / GIB, 3),
        "seconds": round(time.monotonic() - started, 3),
        "sandbox_weakened": False,
        "owned_processes_seen": len(seen),
        "cleanup_extra_kill_pids": remaining,
        "one_cpu": list(os.sched_getaffinity(0)),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
raise SystemExit(bool(failure))
