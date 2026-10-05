"""Cooperative total-work ledger for future bounded profile adapters.

No worker is started and no OS resource limit is changed. Runtime integration
must keep its independent CPU/memory enforcement, not rely on post-call sampling.
"""

import time

from .common import integer, require


class WorkGrant:
    def __init__(
        self,
        values,
        wall_ms,
        cpu_ms,
        *,
        clock=time.monotonic,
        cpu_clock=None,
        lease_ms=15000,
    ):
        self.maximum = integer(values, 1)
        self.wall_ms = integer(wall_ms, 1, 5000)
        self.cpu_ms = integer(cpu_ms, 1, 4000)
        integer(lease_ms, 1, 15000)
        require(callable(cpu_clock), "Supply a CPU clock covering all owned job work")
        self.clock, self.cpu_clock = clock, cpu_clock
        self.started, self.cpu_started = clock(), cpu_clock()
        self.deadline = self.started + wall_ms / 1000
        self.lease_seconds = lease_ms / 1000
        self.lease_end = self.started + self.lease_seconds
        self.visited = 0
        self.failed = False
        self.in_flight = False

    def remaining(self):
        wall = int(max(0, self.deadline - self.clock()) * 1000)
        cpu = int(max(0, self.cpu_ms - (self.cpu_clock() - self.cpu_started) * 1000))
        require(not self.failed, "Grant failed; explicit new grant required")
        require(self.clock() < self.lease_end, "Owner lease expired")
        require(wall > 0 and cpu > 0, "Total profile budget exhausted")
        return {"values": self.maximum - self.visited, "wall_ms": wall, "cpu_ms": cpu}

    def heartbeat(self):
        # A late heartbeat cannot resurrect an expired grant.
        self.remaining()
        self.lease_end = self.clock() + self.lease_seconds

    def advance(self, chunk_values, advance_chunk):
        """Invoke one task15 adapter chunk with (max_values, absolute_deadline).

        Wall and CPU include waits/coordinator work since admission. On uncertain
        outcome, refuse any continuation; the owner must reconcile/drop state.
        Caller is responsible for capability, source binding and global heavy lease.
        """
        require(not self.in_flight, "A profile chunk is already active")
        allowance = self.remaining()
        count = min(integer(chunk_values, 1), allowance["values"])
        require(count > 0, "No authorized values remain")
        self.in_flight = True
        try:
            visited = advance_chunk(count, min(self.deadline, self.lease_end))
            integer(visited, 0, count)
            self.visited += visited
            self.remaining()
            return visited
        except BaseException:
            # Cancellation/interrupt may arrive after partial native work too.
            # Preserve the original exception and poison this grant rather than
            # authorizing its uncertain values again after in_flight is cleared.
            self.failed = True
            raise
        finally:
            self.in_flight = False
