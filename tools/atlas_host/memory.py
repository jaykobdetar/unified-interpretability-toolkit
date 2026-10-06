"""Linux memory observations; imports perform no reads or resource changes."""

from pathlib import Path


def available_bytes() -> int:
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


def guard() -> None:
    available = available_bytes()
    if available < 3 * 1024**3:
        raise RuntimeError("Paused: fewer than 3 GiB available RAM")
