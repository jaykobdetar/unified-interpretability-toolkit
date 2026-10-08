"""Bounded analytics child setup; imports start no timer or numerical work."""

import os
import resource
import signal

from atlas_host import limits as _limits
from atlas_host.memory import available_bytes


def configure() -> None:
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(5)  # Kernel-enforced wall timer even while native LAPACK runs.
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(10)
    for kind, ceiling in (
        (resource.RLIMIT_AS, 768 * 1024**2),
        (resource.RLIMIT_CPU, 4),
    ):
        soft, hard = resource.getrlimit(kind)
        cap = min(n for n in (ceiling, soft, hard) if n != resource.RLIM_INFINITY)
        resource.setrlimit(kind, (cap, cap))
    mem = available_bytes()
    if mem < _limits.MEMORY_RESERVE_GIB * 1024**3:
        raise ValueError("Memory reserve reached")


def configure_svd_worker() -> None:
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(10)
    # Do not raise inherited caps.
    for kind, ceiling in (
        (resource.RLIMIT_AS, 768 * 1024**2),
        (resource.RLIMIT_CPU, 4),
    ):
        soft, hard = resource.getrlimit(kind)
        cap = min(x for x in (ceiling, soft, hard) if x != resource.RLIM_INFINITY)
        resource.setrlimit(kind, (cap, cap))
