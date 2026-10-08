"""Owner-bound tiny-fixture admission; no policy selector is exposed to HTTP.

The approval digest is an out-of-band owner/reviewer input. A recipe name alone,
request payload, command label or measured-peak claim cannot enable this policy.
"""

from . import limits as _limits

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .common import digest, fields, integer, require, read_json
from .registry import RegistryEntry, fingerprint
from .fixture_source import check_fixture

from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias

FileIdentity: TypeAlias = tuple[str, tuple[tuple[str, int], ...]]


class ValidationBinary(Protocol):
    @property
    def path(self) -> Path: ...
    def check(self) -> object: ...


MIB = 1024**2
GIB = 1024**3
DEFAULT_START = _limits.HOSTED_DEFAULT_START_BYTES
STOP_RESERVE = _limits.HOSTED_STOP_RESERVE_BYTES
TREE_CEILING = _limits.HOSTED_TREE_CEILING_BYTES
SNAPSHOT_MAX = _limits.HOSTED_SNAPSHOT_MAX_BYTES
DISK_RESERVE = _limits.HOSTED_DISK_RESERVE_BYTES
RECIPE = "tiny_bf16_provider_v1"
BINARY_SHA = "77831afc4288deae1e14d56833dd8bb6ddb5561885ed1ae88bb210934b676dd2"
FIXTURE_SHA = "c0075bfc55f9e51ccac3c5511ea55a5ca19744b002921e8d2e4ae3f60d321be3"
TOPOLOGY = ["coordinator", "resident_renderer", "one_disposable_worker"]
_SEAL = object()


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_names(root: Path) -> list[str]:
    """Only the runtime package, native sources, Cargo and the exact fixture."""
    names = {
        "Cargo.toml",
        "Cargo.lock",
        "fixtures/tiny-bf16/tiny.safetensors",
        "fixtures/tiny-bf16/host-manifest.json",
        "tools/make_fixture.py",
    }
    names.update(
        str(p.relative_to(root)) for p in (root / "tools/atlas_host").glob("*.py")
    )
    names.update(str(p.relative_to(root)) for p in (root / "src").rglob("*.rs"))
    require(1 <= len(names) <= 128, "Recipe source inventory bound")
    return sorted(names)


def _verified_file(path: Path, expected: str, limit: int) -> FileIdentity:
    digest(expected)
    require(not path.is_symlink(), "Recipe symlink unavailable")
    before = fingerprint(path.stat())
    require(before["bytes"] <= limit, "Recipe file exceeds bound")
    with path.open("rb") as stream:
        require(
            hashlib.file_digest(stream, "sha256").hexdigest() == expected,
            "Reviewed recipe file mismatch",
        )
    require(fingerprint(path.stat()) == before, "Recipe file changed")
    return (str(path), tuple(sorted(before.items())))


@dataclass(frozen=True)
class BoundValidationPolicy:
    receipt_sha256: str
    binary_sha256: str
    source_inventory: tuple[str, ...]
    files: tuple[FileIdentity, ...]
    monitor_path: str
    harness_path: str
    binary_path: str
    _seal: object

    @property
    def start_bytes(self) -> int:
        # Reviewed sample (25.28125 MiB + 1720 B) leaves the 256 MiB margin floor.
        return STOP_RESERVE + TREE_CEILING + 256 * MIB

    def check(self) -> None:
        require(self._seal is _SEAL, "Owner-reviewed recipe required")
        require(
            tuple(source_names(package_root())) == self.source_inventory,
            "Bound source inventory changed",
        )
        for name, expected in self.files:
            path = Path(name)
            require(
                not path.is_symlink()
                and tuple(sorted(fingerprint(path.stat()).items())) == expected,
                "Bound recipe changed",
            )

    def check_binary(self, binary: ValidationBinary) -> None:
        self.check()
        require(str(binary.path) == self.binary_path, "Recipe binary path mismatch")
        binary.check()

    def check_entry(self, entry: RegistryEntry) -> None:
        self.check()
        check_fixture(entry, hash_bytes=True)

    def command(self, interpreter: str, root: str | Path) -> list[str]:
        self.check()
        require(
            interpreter == "/usr/bin/python3"
            and Path(root).resolve() == package_root(),
            "Recipe command unavailable",
        )
        return [
            interpreter,
            "-B",
            self.harness_path,
            str(package_root()),
            self.receipt_sha256,
        ]


def bind_reviewed_recipe(
    receipt_path: str | Path,
    approved_sha256: str,
    *,
    monitor_path: str | Path,
    harness_path: str | Path,
    binary_path: str | Path,
) -> BoundValidationPolicy:
    """Called only by trusted owner bootstrap, before choosing a cheaper gate.

    Review receipt digest and exact script paths are never obtained from HTTP.
    The external receipt avoids a self-hashing source manifest cycle.
    """
    digest(approved_sha256)
    receipt_path = Path(receipt_path).resolve()
    receipt_identity = _verified_file(receipt_path, approved_sha256, 32768)
    data = read_json(receipt_path, 32768)
    fields(
        data,
        (
            "version",
            "recipe",
            "binary_sha256",
            "fixture_sha256",
            "topology",
            "source_sha256",
            "monitor_sha256",
            "harness_sha256",
        ),
    )
    require(
        data["version"] == 1
        and type(data["version"]) is int
        and data["recipe"] == RECIPE,
        "Unknown validation recipe",
    )
    require(
        data["binary_sha256"] == BINARY_SHA
        and data["fixture_sha256"] == FIXTURE_SHA
        and data["topology"] == TOPOLOGY,
        "Recipe workload differs from reviewed fixture",
    )
    root = package_root()
    require(
        type(data["source_sha256"]) is dict
        and sorted(data["source_sha256"]) == source_names(root),
        "Recipe source inventory mismatch",
    )
    files = [receipt_identity]
    for name, expected in data["source_sha256"].items():
        require(
            (root / name).resolve().is_relative_to(root),
            "Recipe source path unavailable",
        )
        files.append(_verified_file(root / name, expected, 512 * 1024))
    monitor = Path(monitor_path).resolve()
    harness = Path(harness_path).resolve()
    binary = Path(binary_path).resolve()
    require(
        binary == root / "target/release/weight-atlas-rust",
        "Recipe executable unavailable",
    )
    files.extend(
        (
            _verified_file(monitor, data["monitor_sha256"], 128 * 1024),
            _verified_file(harness, data["harness_sha256"], 128 * 1024),
            _verified_file(binary, BINARY_SHA, 32 * MIB),
        )
    )
    # Hash the actual fixture too; a caller cannot substitute a source-map digest.
    files.append(
        _verified_file(root / "fixtures/tiny-bf16/tiny.safetensors", FIXTURE_SHA, 244)
    )
    policy = BoundValidationPolicy(
        approved_sha256,
        BINARY_SHA,
        tuple(sorted(data["source_sha256"])),
        tuple(files),
        str(monitor),
        str(harness),
        str(binary),
        _SEAL,
    )
    policy.check()
    return policy


def check_start(
    policy: BoundValidationPolicy | None, available: int, disk: int
) -> None:
    integer(available)
    integer(disk)
    if policy is None:
        threshold = DEFAULT_START
    else:
        require(type(policy) is BoundValidationPolicy, "Owner-reviewed recipe required")
        policy.check()
        threshold = policy.start_bytes
    require(
        available >= threshold and disk >= DISK_RESERVE, "Hosted launch gate refused"
    )


def check_memory(
    observed: Mapping[str, Any], snapshot_bytes: int, *, allow_pending: bool = False
) -> None:
    fields(
        observed,
        ("rss_bytes", "available_bytes", "all_owned_accounted", "descendants_clear"),
    )
    integer(observed["rss_bytes"])
    integer(observed["available_bytes"])
    integer(snapshot_bytes)
    require(
        (observed["all_owned_accounted"] is True or allow_pending)
        and observed["descendants_clear"] is True,
        "Hosted resource observation uncertain",
    )
    require(
        observed["rss_bytes"] + snapshot_bytes <= TREE_CEILING,
        "Hosted charged memory ceiling exceeded",
    )
    require(observed["available_bytes"] >= STOP_RESERVE, "Hosted stop reserve breached")


class QualificationBoundary:
    """Trusted outer monitor adapter; never a replacement for provider inventory."""

    def __init__(self, policy: BoundValidationPolicy) -> None:
        require(type(policy) is BoundValidationPolicy, "Owner-reviewed recipe required")
        policy.check()
        self.policy = policy

    def preflight(self, available: int, disk: int) -> None:
        check_start(self.policy, available, disk)

    def check_sample(self, observed: Mapping[str, Any]) -> None:
        # Conservatively charge the full supported two-frame maximum, including
        # idle results. The provider's independent guard checks actual inventory.
        check_memory(observed, SNAPSHOT_MAX)
