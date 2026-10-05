"""Bounded normal-owner acceptance client; import starts no I/O/process."""

import json
import time
import urllib.request
from urllib.parse import urlsplit

TERMINAL = {
    "idle",
    "complete",
    "cancelled",
    "error",
    "resource_limit",
    "time_limit",
    "client_timeout",
}


def clean(snapshot):
    return (
        snapshot.get("worker_alive") is False
        and not snapshot.get("details", {}).get("cleanup_pending")
        and snapshot.get("status") in TERMINAL
    )


def parity(snapshot):
    assert clean(snapshot) and snapshot["status"] == "complete"
    assert snapshot["details"]["baseline"] == snapshot["details"]["edited"]
    assert len(snapshot["steps"]) == 1
    step = snapshot["steps"][0]
    assert step["alignment"] == "matched_prefix"
    assert all(
        c["delta"] == 0 and c["baseline_logit"] == c["edited_logit"]
        for c in step["candidates"]
    )
    # Do not compare nondeterministic compute timing or capabilities.
    return {
        "baseline": snapshot["details"]["baseline"],
        "edited": snapshot["details"]["edited"],
        "candidates": step["candidates"],
        "activation": step["activation"],
    }


class OwnerClient:
    def __init__(
        self,
        base,
        *,
        seconds=105,
        clock=time.monotonic,
        sleep=time.sleep,
        transport=None,
    ):
        url = urlsplit(base)
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or not url.port
            or url.port in (8774, 8775, 8785)
            or url.path
            or url.query
            or url.fragment
            or url.username
        ):
            raise ValueError("Explicit fresh loopback coordinator URL required")
        self.base, self.clock, self.sleep, self.transport = (
            base,
            clock,
            sleep,
            transport,
        )
        self.deadline = clock() + seconds
        self.session = None
        self.snapshot = None
        self.starts = 0
        self.peak_worker_rss_mib = 0
        self.admission_uncertain = False

    def api(self, action=None, data=None, *, cleanup=False):
        remaining = 5 if cleanup else self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError("Acceptance total budget exhausted; no further start")
        if self.transport is not None:
            return self.transport(action, data)
        body = None if data is None else json.dumps(data, allow_nan=False).encode()
        if body is not None and len(body) > 8192:
            raise ValueError("Acceptance request exceeds existing cap")
        req = urllib.request.Request(
            self.base + "/api/inference" + ("/" + action if action else ""),
            data=body,
            headers={
                "Origin": self.base,
                "X-Atlas-Local": "1",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=min(10, remaining)) as response:
            raw = response.read(2 * 1024**2 + 1)
        if len(raw) > 2 * 1024**2:
            raise ValueError("Acceptance response exceeds bounded buffer")
        return json.loads(raw)

    def accept(self, snapshot):
        assert snapshot["session"] == self.session
        self.snapshot = snapshot
        self.peak_worker_rss_mib = max(
            self.peak_worker_rss_mib, snapshot.get("peak_worker_rss_mib", 0)
        )
        return snapshot

    def start(self, request):
        if self.session is not None and not clean(self.snapshot or {}):
            raise RuntimeError("Owned cleanup unresolved; replacement forbidden")
        self.admission_uncertain = True
        snapshot = self.api("start", request)
        assert isinstance(snapshot.get("session"), str)
        self.admission_uncertain = False
        self.session = snapshot["session"]
        self.starts += 1
        return self.accept(snapshot)

    def finish(self, observe=lambda snapshot: None):
        observe(self.snapshot)
        while not clean(self.snapshot):
            if self.clock() >= self.deadline:
                raise TimeoutError("Acceptance total budget exhausted")
            self.sleep(0.25)
            self.accept(self.api("poll", {"session": self.session}))
            observe(self.snapshot)
        return self.snapshot

    def cancel(self):
        assert self.session is not None
        return self.accept(self.api("cancel", {"session": self.session}))

    def cleanup(self):
        if self.admission_uncertain:
            return False  # No owner capability arrived; supervisor must reap its coordinator.
        if self.session is None or clean(self.snapshot or {}):
            return True
        try:
            self.accept(self.api("cancel", {"session": self.session}, cleanup=True))
            deadline = self.clock() + 4
            while not clean(self.snapshot) and self.clock() < deadline:
                self.sleep(0.25)
                self.accept(self.api("poll", {"session": self.session}, cleanup=True))
        except (OSError, ValueError, TimeoutError):
            return False
        return clean(self.snapshot)
