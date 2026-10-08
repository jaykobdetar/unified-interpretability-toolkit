"""Session-only sealed memfd snapshots. No path fallback, job admission or routes.

All expected bindings/revisions are owner-retained, never visitor credentials.
The 32 MiB estimate includes BOTH frame generations even for a fresh start.
"""

from atlas_host import limits as _limits

import array
import fcntl
import hashlib
import math
import os
import stat
import struct
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from .cache import binding as validate_binding
from .startup_diagnostics import diagnose, diagnosed
from .common import canonical, digest, integer, require

from collections.abc import Callable, Generator, Iterator
from typing import Any, NotRequired, TypedDict, Unpack, cast


class PublicationOptions(TypedDict):
    minimum_visited: int
    maximum_visited: int
    deadline: float
    source_check: Callable[[], object]
    final_check: Callable[[], object]
    clock: NotRequired[Callable[[], float]]


class SnapshotPageRow(TypedDict):
    index: int
    sum_abs: float
    visited_count: int
    expected_count: int
    mean_abs: float | None
    complete: bool


class SnapshotPage(TypedDict):
    revision: str
    binding: dict[str, Any]
    axis: str
    start: int
    end: int
    axis_length: int
    visited_values: int
    total_values: int
    original: list[SnapshotPageRow]
    control: list[SnapshotPageRow]


HEADER = struct.Struct("<8sII7Q32s32s32s")
RECORD = struct.Struct("<ddQ")
ALGORITHM = b"weight-atlas-strength-snapshot-v1:kahan-f64-abs:swap-or-not-8-v1"
MAX_STATE = _limits.PROFILE_MAX_STATE
FIXED_RESERVE = 2 * 1024**2 + 128 * 1024
SEALS = (
    getattr(fcntl, "F_SEAL_WRITE", 8)
    | getattr(fcntl, "F_SEAL_GROW", 4)
    | getattr(fcntl, "F_SEAL_SHRINK", 2)
    | getattr(fcntl, "F_SEAL_SEAL", 1)
)
MASK = 2**64 - 1
FINITE_MAX = {
    "F16": 65504.0,
    "BF16": float.fromhex("0x1.fep+127"),
    "F32": float.fromhex("0x1.fffffep+127"),
}


def layout(selected: dict[str, Any]) -> tuple[int, int, bytes]:
    selected = validate_binding(selected)
    rows, cols = selected["rows"], selected["cols"]
    integer(rows, 1, 200000)
    integer(cols, 1, 200000)
    raw = canonical(selected)
    require(0 < len(raw) <= 16384, "Snapshot binding too large")
    axes = rows + cols
    frame = HEADER.size + len(raw) + 48 * axes
    live = 56 * axes + 2 * frame + FIXED_RESERVE
    require(live <= MAX_STATE, "Snapshot overlap and restore exceed 32 MiB")
    return frame, live, raw


def profile_identity(selected: dict[str, Any], seed: int) -> bytes:
    integer(seed, 0, 2**32 - 1)
    return hashlib.sha256(
        canonical(["weight-atlas-strength-v1", selected, seed, "swap-or-not-8-v1"])
    ).digest()


def _mix(x: int) -> int:
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def destination(index: int, total: int, seed: int) -> int:
    for round_index in range(8):
        key = _mix(seed ^ ((round_index * 0x9E3779B97F4A7C15) & MASK))
        other = (key % total - index) % total
        if _mix(key ^ min(index, other)) & 1:
            index = other
    return index


def _deadline(deadline: float, clock: Callable[[], float]) -> None:
    require(clock() < deadline, "Snapshot deadline exhausted; Restart may be required")


def new_memfd() -> int:
    require(
        hasattr(os, "memfd_create") and hasattr(fcntl, "F_ADD_SEALS"),
        "Sealed memfd is unavailable; no disk fallback",
    )
    return os.memfd_create("atlas-profile", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)


def seal(fd: int) -> None:
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)


def check_sealed(fd: int, size: int) -> None:
    metadata = os.fstat(fd)
    require(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 0
        and metadata.st_size == size,
        "Snapshot is not exact-sized anonymous storage",
    )
    require(
        fcntl.fcntl(fd, fcntl.F_GET_SEALS) & SEALS == SEALS,
        "Snapshot must have immutable write/grow/shrink/seal seals",
    )


def _read(fd: int, count: int, offset: int) -> bytes:
    data = os.pread(fd, count, offset)
    require(len(data) == count, "Truncated snapshot")
    return data


@dataclass(frozen=True)
class SnapshotInfo:
    revision: str
    visited: int
    total: int
    frame_bytes: int
    live_bytes: int


def validation_steps(
    fd: int,
    selected: dict[str, Any],
    seed: int,
    expected_revision: str,
    *,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> Generator[None, None, SnapshotInfo]:
    """Streaming full validation; no allocation from frame-provided dimensions.

    Control prefix verification is time bounded, potentially O(visited). It may
    refuse a valid large partial snapshot rather than weaken its checks.
    """
    _deadline(deadline, clock)
    frame, live, binding_bytes = layout(selected)
    integer(seed, 0, 2**32 - 1)
    digest(expected_revision)
    check_sealed(fd, frame)
    header_raw = _read(fd, HEADER.size, 0)
    (
        magic,
        version,
        binding_len,
        rows,
        cols,
        total,
        visited,
        stored_seed,
        records,
        record_bytes,
        binding_sha,
        algorithm_sha,
        identity_sha,
    ) = HEADER.unpack(header_raw)
    require(
        (
            magic,
            version,
            binding_len,
            rows,
            cols,
            total,
            stored_seed,
            records,
            record_bytes,
        )
        == (
            b"WAPROF01",
            1,
            len(binding_bytes),
            selected["rows"],
            selected["cols"],
            selected["rows"] * selected["cols"],
            seed,
            2 * (selected["rows"] + selected["cols"]),
            48 * (selected["rows"] + selected["cols"]),
        ),
        "Snapshot header mismatch",
    )
    integer(visited, 0, total)
    require(
        binding_sha == hashlib.sha256(binding_bytes).digest()
        and algorithm_sha == hashlib.sha256(ALGORITHM).digest()
        and identity_sha == profile_identity(selected, seed),
        "Snapshot identity mismatch",
    )
    require(
        _read(fd, binding_len, HEADER.size) == binding_bytes,
        "Snapshot binding mismatch",
    )
    h = hashlib.sha256(header_raw)
    h.update(binding_bytes)
    # Compact numeric array, not a Python integer list proportional to axes.
    counts = array.array("Q", [0]) * (rows + cols)
    require(counts.itemsize == 8, "Unsupported native count representation")
    if visited == total:
        for i in range(rows + cols):
            if i % 256 == 0:
                _deadline(deadline, clock)
                yield None
            counts[i] = cols if i < rows else rows
    else:
        for i in range(visited):
            if i % 256 == 0:
                _deadline(deadline, clock)
                yield None
            j = destination(i, total, seed)
            counts[j // cols] += 1
            counts[rows + j % cols] += 1
    offset = HEADER.size + binding_len
    totals = []
    for group, length in enumerate((rows, cols, rows, cols)):
        aggregate = compensation = 0.0
        counted = 0
        for start in range(0, length, 256):
            _deadline(deadline, clock)
            yield None
            raw = _read(fd, min(256, length - start) * 24, offset)
            offset += len(raw)
            h.update(raw)
            for local, (value, correction, count) in enumerate(RECORD.iter_unpack(raw)):
                i = start + local
                require(
                    math.isfinite(value) and value >= 0 and math.isfinite(correction),
                    "Nonfinite or negative snapshot record",
                )
                require(
                    abs(correction) <= 64 * 2**-52 * value,
                    "Invalid snapshot compensation",
                )
                require(
                    count != 0 or (value == 0 and correction == 0),
                    "Nonempty zero-count record",
                )
                require(
                    value <= count * FINITE_MAX[selected["dtype"]],
                    "Sum exceeds dtype bound",
                )
                expected = (
                    min(max(visited - i * cols, 0), cols)
                    if group == 0
                    else (
                        visited // cols + int(i < visited % cols)
                        if group == 1
                        else counts[i] if group == 2 else counts[rows + i]
                    )
                )
                require(count == expected, "Snapshot axis prefix mismatch")
                counted += count
                y = value - compensation
                next_sum = aggregate + y
                compensation = (next_sum - aggregate) - y
                aggregate = next_sum
        require(counted == visited, "Snapshot paired count mismatch")
        totals.append(aggregate)
    require(
        offset == frame and h.hexdigest() == expected_revision,
        "Snapshot revision mismatch",
    )
    for total_sum in totals:
        require(
            math.isfinite(total_sum)
            and abs(total_sum - totals[0]) <= 128 * 2**-52 * max(total_sum, totals[0]),
            "Snapshot paired sums inconsistent",
        )
    check_sealed(fd, frame)
    _deadline(deadline, clock)
    return SnapshotInfo(expected_revision, visited, total, frame, live)


def validate(
    fd: int,
    selected: dict[str, Any],
    seed: int,
    expected_revision: str,
    *,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> SnapshotInfo:
    steps = validation_steps(
        fd, selected, seed, expected_revision, deadline=deadline, clock=clock
    )
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return cast(SnapshotInfo, done.value)


class SnapshotStore:
    """One owner-session latest handle and one pending candidate, serial use only.

    begin() reserves overlap before creating storage. publish() is called only
    after clean worker reap and within the same total job deadline; source_check
    and final_check are mandatory trusted callbacks. No visitor-supplied FDs.
    """

    def __init__(
        self, selected: dict[str, Any], seed: int, *, diagnostics: object = None
    ) -> None:
        self.diagnostics = diagnostics
        self.selected = validate_binding(selected)
        integer(seed, 0, 2**32 - 1)
        self.seed = seed
        self.frame_bytes, self.live_bytes, _ = layout(self.selected)
        self.latest: int | None
        self.pending: int | None
        self.latest = self.pending = None
        self.info: SnapshotInfo | None = None
        self.readers = 0
        self.lock = threading.RLock()
        self.publication_serial = 0
        # Role-independent immutable tuple: watchdog reads never acquire lock,
        # and storage stays charged during a delayed or uncertain close.
        self._owned: tuple[int, ...] = ()
        self._uncertain: frozenset[int] = frozenset()

    @property
    def owned_storage_bytes(self) -> int:
        return len(self._owned) * self.frame_bytes

    def _close_owned(self, fd: int) -> None:
        require(
            fd in self._owned and fd not in self._uncertain,
            "Snapshot close disposition uncertain; do not retry descriptor",
        )
        try:
            os.close(fd)
        except BaseException as diagnostic_error:
            diagnose(
                self,
                "profile_snapshot.SnapshotStore._close_owned.catch216",
                diagnostic_error,
            )
            # close errors may follow kernel descriptor release. Retain charge
            # and poison cleanup; blindly retrying could close a recycled FD.
            self._uncertain = self._uncertain | {fd}
            raise
        self._owned = tuple(owned for owned in self._owned if owned != fd)

    def require_settled(self) -> None:
        require(
            not self._uncertain
            and set(self._owned)
            == {fd for fd in (self.latest, self.pending) if fd is not None},
            "Uncertain retiring snapshot storage still owned",
        )

    def begin(self) -> int:
        with self.lock:
            self.require_settled()
            require(
                self.pending is None and self.readers == 0,
                "Snapshot candidate/read already exists",
            )
            self.pending = new_memfd()
            self._owned = self._owned + (self.pending,)
            return self.pending

    def abort_candidate(self) -> None:
        with self.lock:
            if self.pending is not None:
                fd, self.pending = self.pending, None
                self._close_owned(fd)

    def publication_steps(
        self,
        revision: str,
        *,
        minimum_visited: int,
        maximum_visited: int,
        deadline: float,
        source_check: Callable[[], object],
        final_check: Callable[[], object],
        clock: Callable[[], float] = time.monotonic,
    ) -> Generator[None, None, SnapshotInfo]:
        require(self.pending is not None, "No owned candidate")
        try:
            source_check()
            _deadline(deadline, clock)
            info = yield from validation_steps(
                cast(int, self.pending),
                self.selected,
                self.seed,
                revision,
                deadline=deadline,
                clock=clock,
            )
            integer(minimum_visited, 0, info.total)
            integer(maximum_visited, minimum_visited, info.total)
            integer(info.visited, minimum_visited, maximum_visited)
            source_check()
            final_check()
            _deadline(deadline, clock)
            with self.lock:
                require(self.readers == 0, "Page readers prevent publication")
                old = self.latest
                self.publication_serial += 1
                self.latest, self.pending, self.info = self.pending, None, info
                if old is not None:
                    self._close_owned(old)
            return info
        except BaseException as diagnostic_error:
            diagnose(
                self,
                "profile_snapshot.SnapshotStore.publication_steps.catch260",
                diagnostic_error,
            )
            self.abort_candidate()
            raise

    def publish(
        self, revision: str, **kwargs: Unpack[PublicationOptions]
    ) -> SnapshotInfo:
        steps = self.publication_steps(revision, **kwargs)
        while True:
            try:
                next(steps)
            except StopIteration as done:
                return cast(SnapshotInfo, done.value)

    @contextmanager
    def page_handle(
        self, revision: str, *, expected_publication: int | None = None
    ) -> Iterator[tuple[int, SnapshotInfo]]:
        # Serial host callback. No next admission/publication until reader exits;
        # no extra generation or duplicated FD escapes this context manager.
        with self.lock:
            require(
                expected_publication is None
                or expected_publication == self.publication_serial,
                "Snapshot publication has not been accepted",
            )
            require(
                self.latest is not None
                and cast(SnapshotInfo, self.info).revision == revision,
                "Stale snapshot revision",
            )
            require(self.readers == 0, "One bounded page reader at a time")
            self.readers += 1
            held = (cast(int, self.latest), cast(SnapshotInfo, self.info))
        try:
            yield held
        finally:
            with self.lock:
                self.readers -= 1

    def page(
        self,
        revision: str,
        axis: str,
        start: int,
        count: int,
        *,
        source_check: Callable[[], object],
        expected_publication: int | None = None,
    ) -> SnapshotPage:
        require(axis in ("rows", "columns"), "Invalid profile page axis")
        rows, cols = self.selected["rows"], self.selected["cols"]
        length, expected = (rows, cols) if axis == "rows" else (cols, rows)
        integer(start, 0, length - 1)
        integer(count, 1, 1024)
        end = min(start + count, length)
        base = HEADER.size + len(canonical(self.selected))
        positions = (0, rows + cols) if axis == "rows" else (rows, 2 * rows + cols)
        with self.page_handle(revision, expected_publication=expected_publication) as (
            fd,
            info,
        ):
            source_check()
            check_sealed(fd, self.frame_bytes)
            paired: list[list[SnapshotPageRow]] = []
            for position in positions:
                raw = _read(fd, (end - start) * 24, base + 24 * (position + start))
                records: list[SnapshotPageRow] = []
                for index, (value, _, visited) in enumerate(
                    RECORD.iter_unpack(raw), start
                ):
                    records.append(
                        {
                            "index": index,
                            "sum_abs": value,
                            "visited_count": visited,
                            "expected_count": expected,
                            "mean_abs": None if visited == 0 else value / visited,
                            "complete": visited == expected,
                        }
                    )
                paired.append(records)
            source_check()
            result: SnapshotPage = {
                "revision": info.revision,
                "binding": self.selected,
                "axis": axis,
                "start": start,
                "end": end,
                "axis_length": length,
                "visited_values": info.visited,
                "total_values": info.total,
                "original": paired[0],
                "control": paired[1],
            }
            require(
                len(canonical(result)) <= 2 * 1024**2,
                "Profile page output cap exceeded",
            )
            # Binding is copied so callers cannot mutate authoritative selection.
            result["binding"] = validate_binding(self.selected)
            return result

    def close(self) -> None:
        with self.lock:
            require(self.readers == 0, "Close deferred until page reader exits")
            self.latest = self.pending = self.info = None
            # Retire all known descriptors, including one whose role changed.
            # Uncertain descriptors remain charged and are never retried.
            for fd in self._owned:
                if fd not in self._uncertain:
                    try:
                        self._close_owned(fd)
                    except BaseException as diagnostic_error:
                        diagnose(
                            self,
                            "profile_snapshot.SnapshotStore.close.catch327",
                            diagnostic_error,
                        )
                        pass
            self.require_settled()
