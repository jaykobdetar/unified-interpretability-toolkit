#!/usr/bin/env python3
"""Offline, single-job Linux build with the existing memory and disk limits."""
import argparse
import os
from pathlib import Path
import resource
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'test', 'clippy', 'check'))
    args, rest = parser.parse_known_args()
    if not hasattr(os, 'sched_setaffinity') or not Path('/proc/meminfo').is_file():
        parser.error('Linux CPU affinity and /proc/meminfo are required')
    if shutil.which('cargo') is None:
        parser.error('Cargo is missing; install Rust and its clippy component')
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    mem = int(next(s for s in Path('/proc/meminfo').read_text().splitlines()
                   if s.startswith('MemAvailable:')).split()[1]) * 1024
    if mem < 5 * 1024**3:
        parser.error('build paused: need 5 GiB available RAM')
    if shutil.disk_usage('.').free < 25 * 1024**3:
        parser.error('build paused: need 25 GiB free disk')
    return subprocess.call(['cargo', args.command, '--offline', '--locked', '-j', '1', *rest])


if __name__ == '__main__':
    raise SystemExit(main())
