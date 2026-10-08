"""Owner-only pinned data acquisition; no browser routes or model execution.

Network work requires a reviewed plan digest and explicit CLI license acceptance.
Tests inject streams. Importing/planning never opens a connection or creates files.
"""

from contextlib import contextmanager
from http.client import HTTPMessage
from os import PathLike
from typing import (
    Any,
    Callable,
    ContextManager,
    IO,
    Iterator,
    Protocol,
    TypedDict,
    TypeAlias,
    cast,
)
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .common import digest, identity, integer, label, require
from .config import LOCAL_LIMITS
from .registry import (
    MAX_REGISTRY_BYTES,
    Registry,
    content_digest,
    reservation,
    validate_manifest,
)

MAX_ACQUIRE_BYTES = 64 * 1024**3


class PlanFields(TypedDict):
    version: int
    name: str
    manifest: dict[str, Any]
    content_digest: str
    max_bytes: int
    payload_bytes: int
    cache_growth_bytes: int
    metadata_bytes: int
    disk_reserve_bytes: int
    installation: str
    deadline_enforcement: str
    enabled_after_acquisition: bool
    inference_ready: bool


class AcquisitionPlan(PlanFields):
    plan_digest: str


class AtomicRename(Protocol):
    argtypes: list[type[ctypes.c_int] | type[ctypes.c_char_p] | type[ctypes.c_uint]]
    restype: type[ctypes.c_int]

    def __call__(
        self,
        source_directory: int,
        source: bytes,
        destination_directory: int,
        destination: bytes,
        flags: int,
        /,
    ) -> int: ...


class AcquisitionStream(Protocol):
    def read(self, count: int, /) -> bytes: ...


Fetch: TypeAlias = Callable[[str, float], ContextManager[AcquisitionStream]]


class AcquisitionDeadline(ValueError):
    pass


def plan(
    manifest: dict[str, Any], name: str, *, max_bytes: int, cache_growth: int = 0
) -> AcquisitionPlan:
    manifest = validate_manifest(manifest)
    require(
        manifest["provenance"] == "owner_expected",
        "Only explicitly pinned owner data can be acquired",
    )
    label(name, 128)
    integer(max_bytes, 1, MAX_ACQUIRE_BYTES)
    integer(cache_growth, 0, 8 * 1024**3)
    total = sum(file["bytes"] for file in manifest["files"])
    require(total <= max_bytes, "Pinned files exceed explicit acquisition byte budget")
    result: PlanFields = {
        "version": 1,
        "name": name,
        "manifest": manifest,
        "content_digest": content_digest(manifest),
        "max_bytes": max_bytes,
        "payload_bytes": total,
        "cache_growth_bytes": cache_growth,
        "metadata_bytes": MAX_REGISTRY_BYTES + 65536,
        "disk_reserve_bytes": LOCAL_LIMITS["disk_reserve_bytes"],
        "installation": "same-filesystem staging then rename; no duplicate payload copy",
        "deadline_enforcement": "cooperative checks; buffered network I/O requires external hard supervision",
        "enabled_after_acquisition": False,
        "inference_ready": False,
    }
    return {**result, "plan_digest": identity("weight-atlas-acquire-plan-v1", result)}


def _url(manifest: dict[str, Any], filename: str) -> str:
    # Both components passed the closed manifest name/full-revision allowlist.
    return f'https://huggingface.co/{manifest["repository"]}/resolve/{manifest["revision"]}/{filename}'


def _allowed_url(url: str) -> None:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    require(
        parsed.scheme == "https"
        and parsed.port in (None, 443)
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
        and (
            host
            in ("huggingface.co", "cdn-lfs.huggingface.co", "cas-bridge.xethub.hf.co")
            or host.endswith(".cdn.hf.co")
            or host in ("cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co")
        ),
        "Unapproved acquisition redirect",
    )


class _Redirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        _allowed_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_public_data(url: str, timeout: float) -> ContextManager[AcquisitionStream]:
    """timeout limits idle socket operations, not cumulative buffered I/O time."""
    _allowed_url(url)
    # No account credentials, environment proxy authorization, or custom TLS.
    return cast(
        ContextManager[AcquisitionStream],
        build_opener(ProxyHandler({}), _Redirects()).open(url, timeout=timeout),
    )


def _install_new_directory(
    source: str | PathLike[str], destination: str | PathLike[str]
) -> None:
    """Linux atomic no-replace rename. Fail closed on unsupported libc/filesystem."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    require(rename is not None, "Atomic no-replace install unavailable")
    cast(AtomicRename, rename).argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    cast(AtomicRename, rename).restype = ctypes.c_int
    if (
        cast(AtomicRename, rename)(
            -100, os.fsencode(source), -100, os.fsencode(destination), 1
        )
        != 0
    ):
        number = ctypes.get_errno()
        raise OSError(number, "Atomic no-replace model installation failed")


@contextmanager
def _slot(registry: Registry) -> Iterator[None]:
    # All acquisitions for this registry serialize their disk reservations.
    registry._disk_guard(MAX_REGISTRY_BYTES + 65536)
    registry.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = registry.path.with_name(registry.path.name + ".acquire.lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), "Acquisition lock must be regular")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Another owner acquisition holds the disk reservation"
            ) from error
        yield
    finally:
        os.close(fd)


def acquire(
    registry: Registry,
    destination: str | PathLike[str],
    manifest: dict[str, Any],
    name: str,
    *,
    max_bytes: int,
    plan_digest: str,
    accept_license: bool = False,
    timeout_ms: int = 120000,
    cache_growth: int = 0,
    fetch: Fetch = open_public_data,
    clock: Callable[[], float] = time.monotonic,
    free_bytes: Callable[[Path], int] | None = None,
) -> dict[str, Any]:
    """Stream exact bounded files, verify hashes, atomically install, register off.

    No automatic resume/enable/eviction. If registration fails, verified installed
    data is retained for owner re-verification and is not selectable.
    """
    require(
        type(accept_license) is bool and accept_license,
        "Explicit CLI owner license acceptance required",
    )
    integer(timeout_ms, 100, 600000)
    reviewed = plan(manifest, name, max_bytes=max_bytes, cache_growth=cache_growth)
    digest(plan_digest)
    require(
        plan_digest == reviewed["plan_digest"],
        "Acquisition plan changed; review it again",
    )
    manifest = reviewed["manifest"]
    destination = Path(destination).absolute()
    require(
        destination.parent.is_dir()
        and not destination.exists()
        and not destination.is_symlink(),
        "Choose a new directory inside an existing owner destination",
    )
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    require(
        not registry.path.is_relative_to(destination),
        "Registry must stay outside model source",
    )
    deadline = clock() + timeout_ms / 1000
    disk: Callable[[Path], int] = free_bytes or (
        lambda path: shutil.disk_usage(path).free
    )

    def check(remaining: int) -> dict[str, int]:
        if clock() >= deadline:
            raise AcquisitionDeadline("Owner acquisition time allowance exhausted")
        return reservation(
            disk(parent),
            remaining_download=remaining,
            cache_growth=cache_growth,
            metadata=reviewed["metadata_bytes"],
        )

    with _slot(registry):
        check(reviewed["payload_bytes"])
        temporary = Path(tempfile.mkdtemp(prefix=".atlas-acquire-", dir=parent))
        installed = False
        try:
            remaining = reviewed["payload_bytes"]
            for file in manifest["files"]:
                check(remaining)
                measured = hashlib.sha256()
                left = file["bytes"]
                with fetch(
                    _url(manifest, file["name"]),
                    min(10, max(0.001, deadline - clock())),
                ) as source:
                    fd = os.open(
                        temporary / file["name"],
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                    )
                    with os.fdopen(fd, "wb") as output:
                        while left:
                            check(remaining)
                            block = source.read(min(65536, left))
                            check(remaining)
                            require(
                                type(block) is bytes
                                and 0 < len(block) <= min(65536, left),
                                "Pinned data ended early or exceeded read bound",
                            )
                            measured.update(block)
                            output.write(block)
                            left -= len(block)
                            remaining -= len(block)
                        require(
                            not source.read(1), "Pinned data exceeded expected size"
                        )
                        require(
                            measured.hexdigest() == file["sha256"],
                            "Pinned data hash mismatch",
                        )
                        output.flush()
                        os.fsync(output.fileno())
                check(remaining)
            # Destination must still be unused; never replace an owner directory.
            require(
                not destination.exists() and not destination.is_symlink(),
                "Destination appeared during acquisition",
            )
            _install_new_directory(temporary, destination)
            installed = True
            durable = False
            registration_attempted = False
            try:
                directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                    durable = True
                finally:
                    os.close(directory)
                check(0)
                registration_attempted = True
                receipt = registry.register(
                    destination,
                    manifest,
                    name,
                    max_bytes=max_bytes,
                    timeout_ms=min(5000, max(1, int((deadline - clock()) * 1000))),
                    publish_disabled=True,
                )
                return {
                    **receipt,
                    "installed": True,
                    "registered": True,
                    "inference_ready": False,
                    "plan_digest": plan_digest,
                    "installation_durability": "parent_directory_fsync_confirmed",
                    "registry_publication": "confirmed",
                }
            except AcquisitionDeadline:
                reason = "deadline_after_install_data_retained"
                publication = "not_attempted"
            except OSError:
                # Registry I/O may fail after its atomic replacement. Never call
                # an unconfirmed publication definitely absent or disabled.
                reason = (
                    "registry_publication_unconfirmed"
                    if registration_attempted
                    else "post_install_io_error_data_retained"
                )
                publication = (
                    "unconfirmed" if registration_attempted else "not_attempted"
                )
            except ValueError:
                reason = (
                    "verified_data_retained_for_owner_registration"
                    if registration_attempted
                    else "post_install_admission_failed_data_retained"
                )
                publication = (
                    "not_published" if registration_attempted else "not_attempted"
                )
            return {
                "installed": True,
                "registered": None if publication == "unconfirmed" else False,
                "enabled": None if publication == "unconfirmed" else False,
                "inference_ready": False,
                "plan_digest": plan_digest,
                "reason": reason,
                "installation_durability": (
                    "parent_directory_fsync_confirmed" if durable else "unconfirmed"
                ),
                "registry_publication": publication,
            }
        finally:
            if not installed:
                shutil.rmtree(temporary)
