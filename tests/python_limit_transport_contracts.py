"""Admission and metadata transport edges with temporary files and inert channels."""

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
import profile_runtime_doubles as runtime
import static_host_contracts as static_fixture
from atlas_host import profile_platform, static_models, validation_policy
from atlas_host.common import canonical
from atlas_host.registry import fingerprint
from atlas_host.runtime_adapter import HostError

MIB, GIB = 1024**2, 1024**3


class ReplySocket(runtime.Socket):
    def __init__(self, body_bytes=2, header_bytes=None):
        super().__init__()
        self.body_bytes, self.header_bytes = body_bytes, header_bytes

    def send(self, data):
        self.sent.extend(data)
        return len(data)

    def recv(self, count):
        if self.out is None:
            command = json.loads(self.sent[4:])
            header = canonical(
                {
                    "ack": {
                        "version": 1,
                        "operation_id": command["operation_id"],
                        "complete": True,
                        "numeric_idle": True,
                    },
                    "status": 200,
                    "mime": "application/json",
                    "body_bytes": self.body_bytes,
                }
            )
            if self.header_bytes is not None:
                header += b" " * max(0, self.header_bytes - len(header))
            advertised = len(header) if self.header_bytes is None else self.header_bytes
            self.out = bytearray(
                struct.pack("!I", advertised) + header + b"x" * max(0, self.body_bytes)
            )
        part = bytes(self.out[:count])
        del self.out[:count]
        return part


class TransportLimits(unittest.TestCase):
    def edges(self, operation, low, high):
        for number, accepted in (
            (low - 1, False),
            (low, True),
            (high, True),
            (high + 1, False),
        ):
            with self.subTest(number=number):
                if accepted:
                    operation(number)
                else:
                    with self.assertRaises(ValueError):
                        operation(number)

    def observed(self, **changes):
        return {
            "rss_bytes": 0,
            "available_bytes": 13 * GIB // 4,
            "all_owned_accounted": True,
            "descendants_clear": True,
            **changes,
        }

    def test_default_launch_available_disk_and_integer_ranges(self):
        self.assertEqual(
            (
                validation_policy.DEFAULT_START,
                validation_policy.STOP_RESERVE,
                validation_policy.TREE_CEILING,
                validation_policy.SNAPSHOT_MAX,
                validation_policy.DISK_RESERVE,
            ),
            (5 * GIB, 13 * GIB // 4, 768 * MIB, 32 * MIB, 25 * GIB),
        )
        self.edges(
            lambda memory: validation_policy.check_start(None, memory, 25 * GIB),
            5 * GIB,
            2**63 - 1,
        )
        self.edges(
            lambda disk: validation_policy.check_start(None, 5 * GIB, disk),
            25 * GIB,
            2**63 - 1,
        )

    def test_optional_reviewed_policy_gate_isolated_from_receipt_verification(self):
        # No receipt is bound or activated. A check double isolates this gate;
        # this does not qualify a source, executable or reduced launch policy.
        policy = validation_policy.BoundValidationPolicy(
            "a" * 64, "b" * 64, (), (), "synthetic", "synthetic", "synthetic", object()
        )
        with patch.object(validation_policy.BoundValidationPolicy, "check"):
            self.assertEqual(policy.start_bytes, 17 * GIB // 4)
            self.edges(
                lambda memory: validation_policy.check_start(policy, memory, 25 * GIB),
                17 * GIB // 4,
                2**63 - 1,
            )

    def test_charged_memory_and_available_floor_edges(self):
        for reserved in (0, 32 * MIB):
            self.edges(
                lambda rss: validation_policy.check_memory(
                    self.observed(rss_bytes=rss), reserved
                ),
                0,
                768 * MIB - reserved,
            )
        self.edges(
            lambda snapshot: validation_policy.check_memory(
                self.observed(rss_bytes=1), snapshot
            ),
            0,
            768 * MIB - 1,
        )
        self.edges(
            lambda available: validation_policy.check_memory(
                self.observed(available_bytes=available), 0
            ),
            13 * GIB // 4,
            2**63 - 1,
        )
        for field in ("rss_bytes", "available_bytes"):
            with self.assertRaisesRegex(ValueError, "Integer outside"):
                validation_policy.check_memory(self.observed(**{field: 2**63}), 0)

    def test_profile_resource_frame_and_reservation_coupling(self):
        platform = object.__new__(profile_platform.ProfilePlatform)
        platform.diagnostics = None
        platform.supervisor = SimpleNamespace(snapshot_reservation=0)
        platform.hooks = SimpleNamespace(resources=lambda: self.observed())
        self.edges(
            lambda frame: self.assertTrue(platform.resources_ok(frame)), 0, 32 * MIB
        )
        for reserve in (0, 32 * MIB):
            platform.supervisor.snapshot_reservation = reserve
            for rss, okay in (
                (768 * MIB - reserve, True),
                (768 * MIB - reserve + 1, False),
            ):
                platform.hooks.resources = lambda: self.observed(rss_bytes=rss)
                self.assertEqual(platform.resources_ok(0), okay)
        for available, okay in ((13 * GIB // 4, True), (13 * GIB // 4 - 1, False)):
            platform.hooks.resources = lambda: self.observed(available_bytes=available)
            self.assertEqual(platform.resources_ok(0), okay)

    def test_recipe_source_inventory_reachable_upper_bound(self):
        root = Path("/synthetic/package")
        for extra, accepted in ((0, True), (123, True), (124, False)):
            # Five mandatory names mean the declared lower count one is masked.
            with (
                patch.object(
                    Path,
                    "glob",
                    return_value=iter(
                        [root / f"tools/atlas_host/x{n}.py" for n in range(extra)]
                    ),
                ),
                patch.object(Path, "rglob", return_value=iter([])),
            ):
                if accepted:
                    self.assertEqual(
                        len(validation_policy.source_names(root)), 5 + extra
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "inventory bound"):
                        validation_policy.source_names(root)

    def test_verified_recipe_file_size_exact_and_one_past(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe"
            for size, accepted in ((0, True), (32768, True), (32769, False)):
                raw = b"x" * size
                path.write_bytes(raw)
                digest = hashlib.sha256(raw).hexdigest()
                if accepted:
                    validation_policy._verified_file(path, digest, 32768)
                else:
                    with self.assertRaisesRegex(ValueError, "file exceeds bound"):
                        validation_policy._verified_file(path, digest, 32768)

    def static_read(self, size, *, header=False, budget=0, maximum=2097152):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata"
            payload = b"x" * max(0, size)
            path.write_bytes(struct.pack("<Q", size) + payload if header else payload)
            entry = {
                "root": directory,
                "fingerprints": {path.name: fingerprint(path.stat())},
            }
            charged = [budget]
            raw = static_models._read(entry, path.name, maximum, charged, header=header)
            self.assertEqual(raw, payload)
            self.assertEqual(charged[0], budget + size + (8 if header else 0))

    def test_static_metadata_real_read_header_aux_and_cumulative_caps(self):
        self.assertEqual(
            (static_models.MAX_HEADER_BYTES, static_models.MAX_ACTIVATION_METADATA),
            (2097152, 8388608),
        )
        self.edges(lambda size: self.static_read(size), 1, 2097152)
        self.edges(lambda size: self.static_read(size, header=True), 2, 2097152)
        self.static_read(2, header=True, budget=8388608 - 10)
        with self.assertRaisesRegex(ValueError, "metadata budget"):
            self.static_read(2, header=True, budget=8388608 - 9)

    def test_native_transport_actual_header_and_body_caps(self):
        for body, header, accepted in (
            (0, None, True),
            (2097152, None, True),
            (-1, None, False),
            (2097153, None, False),
            (2, 16384, True),
            (2, 16385, False),
            (2, 0, False),
        ):
            case = runtime.NativeTests(methodName="runTest")
            channel = case.make()
            channel.stream = ReplySocket(body, header)
            if accepted:
                status, result, mime = channel.read("/api/model", "ctx")
                self.assertEqual(
                    (status, len(result), mime), (200, body, "application/json")
                )
                self.assertFalse(case.s.busy())
            else:
                with self.assertRaises(ValueError):
                    channel.read("/api/model", "ctx")
                self.assertTrue(case.child.stopped)
                case.child.reaped = True
                case.watch.pulse()
                self.assertFalse(case.s.busy())
        # Complete header JSON cannot fit the declared one-byte lower edge.

    def test_renderer_body_caps_for_status_and_tile_routes(self):
        for route, high in (
            ("progress", 16384),
            ("tensor-status", 16384),
            ("tile", 2097152),
        ):
            for size, accepted in ((high, True), (high + 1, False)):
                case = static_fixture.StaticHostContracts(methodName="runTest")
                case.setUp()
                self.addCleanup(case.doCleanups)
                lease = case.activate()
                if route == "tile":
                    raw, mime = b"x" * size, "image/png"
                else:
                    model = {
                        "source_identity": case.host.source_identity,
                        "model_identity": case.host.model_identity,
                        "padding": "",
                    }
                    model["padding"] = "x" * (size - len(canonical(model)))
                    raw, mime = canonical(model), "application/json"
                self.assertEqual(len(raw), size)
                case.host.reader.read = Mock(return_value=(200, raw, mime))
                query = {"context": [lease["context_id"]]}
                if route == "tensor-status":
                    query["tensor"] = ["0"]
                if accepted:
                    self.assertEqual(
                        case.host.read(lease["model_id"], route, query)[0], 200
                    )
                else:
                    with self.assertRaisesRegex(
                        HostError,
                        "Renderer receipt correspondence could not be verified",
                    ) as error:
                        case.host.read(lease["model_id"], route, query)
                    self.assertEqual(
                        (error.exception.status, error.exception.code),
                        (409, "source_changed"),
                    )


if __name__ == "__main__":
    unittest.main()
