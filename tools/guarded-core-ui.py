#!/usr/bin/env python3
"""Guard the fresh qualification tree, including servers, then verify cleanup.
Browser RSS is user-approved at 1 GiB; all other gates remain unchanged.
Only tracked PID/start-time identities are signaled; browser VAS is exempt.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

TERM_GRACE = 2.0
KILL_GRACE = 2.0
REAP_TIMEOUT = 1.0
BROWSER_RSS_CAP_BYTES = 1024**3
BROWSER_POLICY = 'browser-summed-rss-1gib-user-approved-0341utc'


def available():
    return int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')).split()[1]) * 1024


def table():
    data = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            data[int(path.parent.name)] = (int(fields[1]), int(fields[19]),
                                          int(fields[21]) * os.sysconf('SC_PAGE_SIZE'))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return data


def discover(rows, owned):
    while True:
        added = {pid: row[1] for pid, row in rows.items()
                 if pid not in owned and row[0] in owned
                 and rows.get(row[0], (0, 0, 0))[1] == owned[row[0]]}
        if not added:
            return
        owned.update(added)


def matching(rows, owned, own):
    return [pid for pid, stamp in owned.items()
            if pid != own and pid in rows and rows[pid][1] == stamp]


def cleanup(process, owned, own):
    """At most two bounded grace periods plus a bounded direct-child reap.

    A scan/signal/reap error fails qualification even if no final survivor is
    observed. Fresh identity checks precede every signal, including SIGKILL.
    """
    errors, signals = [], []

    def snapshot():
        rows = table()
        discover(rows, owned)
        return rows

    def signal_remaining(kind):
        for pid in matching(snapshot(), owned, own)[::-1]:
            # Never use an earlier process-table snapshot to authorize a signal.
            fresh = snapshot()
            if pid not in matching(fresh, owned, own):
                continue
            try:
                os.kill(pid, kind)
                signals.append({'pid': pid, 'start_time': owned[pid], 'signal': int(kind)})
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f'Signal {int(kind)} to owned PID {pid} failed: {error}')

    def wait_until(deadline):
        while True:
            if process is not None:
                process.poll()  # Reap our direct child without blocking.
            if not matching(snapshot(), owned, own) or time.monotonic() >= deadline:
                return
            time.sleep(min(.05, max(0., deadline - time.monotonic())))

    # Nested finally blocks retain a final survivor scan and a reap attempt even
    # when process polling, process-table access or signaling raises unexpectedly.
    try:
        signal_remaining(signal.SIGTERM)
        wait_until(time.monotonic() + TERM_GRACE)
        if matching(snapshot(), owned, own):
            signal_remaining(signal.SIGKILL)
            wait_until(time.monotonic() + KILL_GRACE)
    except BaseException as error:
        errors.append(f'Cleanup operation failed: {type(error).__name__}: {error}')
    finally:
        try:
            if process is not None:
                process.wait(timeout=REAP_TIMEOUT)
        except BaseException as error:
            errors.append(f'Direct-child reap failed: {type(error).__name__}: {error}')
            # A wait timeout must not escape or prevent a bounded fallback.
            try:
                signal_remaining(signal.SIGKILL)
                if process is not None:
                    process.wait(timeout=REAP_TIMEOUT)
            except BaseException as fallback_error:
                errors.append(f'Fallback reap failed: {type(fallback_error).__name__}: {fallback_error}')
        finally:
            try:
                remaining = matching(snapshot(), owned, own)
                verified = True
            except BaseException as error:
                errors.append(f'Final cleanup scan failed: {type(error).__name__}: {error}')
                remaining = [pid for pid in owned if pid != own]
                verified = False
    return {'remaining_owned_pids': remaining, 'cleanup_verified': verified,
            'cleanup_errors': errors, 'termination_attempts': signals}


def write_receipt(out, result):
    """Always attempt the file receipt, with a failure JSON fallback on stdout."""
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'guard-report.json').write_text(json.dumps(result, indent=2) + '\n')
    except BaseException as error:
        result['passed'] = False
        result['failure'] = '; '.join(filter(None, [result['failure'],
                                     f'Receipt write failed: {type(error).__name__}: {error}']))
    print(json.dumps(result, indent=2))


def run(command, out):
    own, owned, process = os.getpid(), {}, None
    start, peak, minimum, failure = time.monotonic(), 0, None, None
    cleanup_result = {'remaining_owned_pids': [], 'cleanup_verified': False,
                      'cleanup_errors': [], 'termination_attempts': []}
    try:
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        os.nice(10)
        out.mkdir(parents=True, exist_ok=True)
        minimum = available()
        assert minimum >= 5 * 1024**3, 'Browser launch refused: need 5 GiB available RAM'
        assert shutil.disk_usage(out).free >= 25 * 1024**3, 'Browser launch refused: need 25 GiB disk reserve'
        owned[own] = table()[own][1]
        process = subprocess.Popen(command)
        owned[process.pid] = table()[process.pid][1]
        while process.poll() is None:
            rows = table()
            discover(rows, owned)
            rss = sum(rows[pid][2] for pid, stamp in owned.items()
                      if pid in rows and rows[pid][1] == stamp)
            peak = max(peak, rss)
            minimum = min(minimum, available())
            if rss > BROWSER_RSS_CAP_BYTES or minimum < 3.25 * 1024**3 or time.monotonic() - start > 120:
                raise RuntimeError(f'Qualification guard refused: owned RSS {rss/1024**2:.2f} MiB; available {minimum/1024**3:.3f} GiB; elapsed {time.monotonic()-start:.1f}s')
            time.sleep(.1)
        if process.returncode:
            raise RuntimeError(f'Qualification child exited {process.returncode}')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
    finally:
        try:
            cleanup_result = cleanup(process, owned, own)
        except BaseException as error:
            # Even an unexpected cleanup defect must not bypass receipt writing.
            cleanup_result['remaining_owned_pids'] = [pid for pid in owned if pid != own]
            cleanup_result['cleanup_errors'].append(f'Unexpected cleanup failure: {type(error).__name__}: {error}')
        finally:
            problems = list(cleanup_result['cleanup_errors'])
            if not cleanup_result['cleanup_verified']:
                problems.append('Owned-process cleanup could not be verified')
            if cleanup_result['remaining_owned_pids']:
                problems.append(f"Owned processes remain: {cleanup_result['remaining_owned_pids']}")
            failure = '; '.join(filter(None, [failure, *problems])) or None
            result = {'passed': failure is None, 'failure': failure,
                      'owned_tree_peak_rss_mib': peak / 1024**2,
                      'minimum_available_gib': None if minimum is None else minimum / 1024**3,
                      'elapsed_seconds': time.monotonic() - start,
                      'rss_cap_mib': BROWSER_RSS_CAP_BYTES / 1024**2,
                      'browser_resource_policy': BROWSER_POLICY, 'includes_guard_servers_node_browser': True,
                      'one_cpu': True, 'owned_processes': list(owned), **cleanup_result}
            write_receipt(out, result)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1:], Path(os.environ['ATLAS_EVIDENCE_DIR'])))
