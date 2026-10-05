#!/usr/bin/env python3
"""PREPARED, UNRUN: independent tiny worker/codec numerical oracle.

Only run in a separately assigned guarded runtime slot after a reviewed build.
This is not the host provider's aggregate accounting/watchdog qualification.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import sys
import tempfile
import time

from profile_worker_primitives import frame, selected
from atlas_host import profile_snapshot as snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    binary = parser.parse_args().binary.resolve(strict=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))

    def available():
        return (
            int(
                next(
                    s
                    for s in Path("/proc/meminfo").read_text().splitlines()
                    if s.startswith("MemAvailable:")
                ).split()[1]
            )
            * 1024
        )

    if available() < 5 * 1024**3 or shutil.disk_usage("/tmp").free < 25 * 1024**3:
        raise SystemExit("Qualification gate refused; no retry")
    runs = 0
    with tempfile.TemporaryDirectory(prefix="atlas-profile-oracle-") as temporary:
        for dtype in ["BF16", "F16", "F32"]:
            root = Path(temporary) / dtype
            root.mkdir()
            values = [0.0] * 12 + [
                (-1 if i % 3 else 1) * (i + 1) / 8 for i in range(12)
            ]
            raw = bytearray()
            for value in values:
                encoded = struct.pack("<e" if dtype == "F16" else "<f", value)
                raw.extend(encoded[2:] if dtype == "BF16" else encoded)
            header = json.dumps(
                {
                    "weights": {
                        "dtype": dtype,
                        "shape": [2, 3, 4],
                        "data_offsets": [0, len(raw)],
                    }
                },
                separators=(",", ":"),
            ).encode()
            (root / "tiny.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + raw
            )
            metadata = subprocess.run(
                [str(binary), "metadata", "--model", str(root)],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            binding = selected()
            binding["dtype"] = dtype
            binding["source_identity"] = json.loads(metadata.stdout)["source_identity"]
            revision = "fixture-v1"
            binding["model_identity"] = hashlib.sha256(
                snapshot.canonical(
                    ["weight-atlas-model-v1", binding["source_identity"], revision]
                )
            ).hexdigest()

            def worker(values, previous=None):
                nonlocal runs
                if available() < 3.25 * 1024**3:
                    raise RuntimeError("Stop reserve reached; no retry")
                output = snapshot.new_memfd()
                try:
                    args = [
                        str(binary),
                        "profile-worker",
                        "--model",
                        str(root),
                        "--revision",
                        revision,
                        "--tensor",
                        "0",
                        "--slice",
                        "1",
                        "--seed",
                        "17",
                        "--values",
                        str(values),
                        "--wall-ms",
                        "2500",
                        "--cpu-ms",
                        "2000",
                        "--binding",
                        snapshot.canonical(binding).decode(),
                        "--output-fd",
                        str(output),
                    ]
                    fds = [output]
                    if previous:
                        args += [
                            "--input-fd",
                            str(previous[0]),
                            "--input-sha",
                            previous[1],
                        ]
                        fds.append(previous[0])
                    result = subprocess.run(
                        args,
                        pass_fds=fds,
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=True,
                    )
                    assert len(result.stdout.encode()) <= 16384
                    receipt = json.loads(result.stdout)
                    info = snapshot.validate(
                        output,
                        binding,
                        17,
                        receipt["revision"],
                        deadline=time.monotonic() + 1,
                    )
                    expected, *_ = frame(binding, 17, info.visited)
                    assert os.pread(output, len(expected) + 1, 0) == expected
                    runs += 1
                    return output, info.revision
                except BaseException:
                    os.close(output)
                    raise

            owned = []
            try:
                full = worker(12)
                owned.append(full[0])
                partial = worker(5)
                owned.append(partial[0])
                resumed = worker(7, partial)
                owned.append(resumed[0])
                assert full[1] == resumed[1]
            finally:
                for fd in owned:
                    os.close(fd)
    print(
        json.dumps(
            {
                "status": "PASS",
                "worker_grants": runs,
                "formats": ["BF16", "F16", "F32"],
                "scope": "Tiny sealed frames and independent bitwise reference; host aggregate watchdog/provider unqualified",
            }
        )
    )


if __name__ == "__main__":
    main()
