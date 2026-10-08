#!/usr/bin/env python3
"""Add receipt witnesses to the unchanged opt-in tiny profile-worker oracle.

Uses the original grants, limits, sealed frames and bitwise comparisons. This
does not qualify aggregate host accounting or watchdog behavior.
"""

import hashlib
import json
import os
import subprocess
from typing import Any
from unittest.mock import patch

import profile_worker_oracle as oracle


def main() -> None:
    original_run = subprocess.run
    visited: dict[int, int] = {}
    grants: list[tuple[str, int, int]] = []

    def observed(*args: Any, **kwargs: Any) -> Any:
        result = original_run(*args, **kwargs)
        command = args[0]
        if len(command) < 2 or command[1] != "profile-worker":
            return result
        options = dict(zip(command[2::2], command[3::2]))
        receipt = json.loads(result.stdout)
        binding = json.loads(options["--binding"])
        output_fd = int(options["--output-fd"])
        previous = visited[int(options["--input-fd"])] if "--input-fd" in options else 0
        requested = int(options["--values"])
        count = previous + requested
        size = os.fstat(output_fd).st_size
        assert 0 < size <= 4096, "tiny fixture frame read bound"
        raw = os.pread(output_fd, size + 1, 0)
        assert len(raw) == size, "exact tiny sealed frame"
        # Independent fixed snapshot ABI: 24-byte sums, one 8-byte axis slot,
        # both frame generations and the original 2-MiB + 128-KiB reserve.
        axes = binding["rows"] + binding["cols"]
        expected = {
            "schema": "weight-atlas.profile-candidate.v1",
            "revision": hashlib.sha256(raw).hexdigest(),
            "visited_values": count,
            "new_values": requested,
            "frame_bytes": size,
            "live_bytes": 56 * axes + 2 * size + 2 * 1024**2 + 128 * 1024,
        }
        assert receipt == expected, ("exact profile receipt", receipt, expected)
        for key in ["visited_values", "new_values", "frame_bytes", "live_bytes"]:
            assert type(receipt[key]) is int, ("integer receipt field", key)
        visited[output_fd] = count
        grants.append((binding["dtype"], requested, count))
        return result

    with patch.object(oracle.subprocess, "run", observed):
        oracle.main()
    assert grants == [
        (dtype, requested, count)
        for dtype in ["BF16", "F16", "F32"]
        for requested, count in [(12, 12), (5, 5), (7, 12)]
    ], "all nine original grants observed"
    print(
        json.dumps(
            {
                "status": "PASS",
                "receipt_grants": len(grants),
                "original_oracle_unchanged": True,
                "scope": "Exact tiny candidate receipts; aggregate host/watchdog unqualified",
            }
        )
    )


if __name__ == "__main__":
    main()
