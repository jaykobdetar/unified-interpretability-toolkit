"""Source-only receipt, diagnostic and grant-clock boundaries; no execution."""

import hashlib
import io
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import profile_worker_primitives as primitive
from atlas_host import (
    dense_static_admission as dense,
    profile_os,
    profile_snapshot as snapshot,
    startup_diagnostics as diagnostic,
)
from atlas_host.supervisor import AdmissionGrant, CpuLedger


class TraceError(ValueError):
    @property
    def __traceback__(self):
        return self.trace


class SourceClockLimits(unittest.TestCase):
    def test_dense_runtime_inventory_effective_minimum_and_128_names(self):
        root = Path("/synthetic/package")
        with (
            patch.object(dense, "source_names", return_value=["mandatory"]),
            patch.object(Path, "glob", return_value=[]),
        ):
            minimum = len(dense.runtime_names(root))
        self.assertGreater(minimum, 1)
        for count, accepted in ((minimum, True), (128, True), (129, False)):
            paths = [
                root / f"tools/inference_extra{n}.py" for n in range(count - minimum)
            ]
            with (
                patch.object(dense, "source_names", return_value=["mandatory"]),
                patch.object(Path, "glob", return_value=paths),
            ):
                if accepted:
                    self.assertEqual(len(dense.runtime_names(root)), count)
                else:
                    with self.assertRaisesRegex(ValueError, "inventory exceeds"):
                        dense.runtime_names(root)
        # Fixed bundled names mask the declared minimum of one, even with this
        # reduced source-name double. No real source registration is performed.

    def test_git_stdout_byte_and_two_line_guards_with_inert_command(self):
        for size, accepted in ((128, True), (129, False)):
            raw = b"a\n" + b"b" * (size - 2)
            with patch.object(
                dense.subprocess, "run", return_value=NS(stdout=raw)
            ) as run:
                if accepted:
                    self.assertEqual(len(dense._git_version("/synthetic")), 2)
                else:
                    with self.assertRaisesRegex(ValueError, "metadata exceeds"):
                        dense._git_version("/synthetic")
                self.assertEqual(run.call_args.kwargs["timeout"], 1)
        for raw, accepted in (
            (b"", False),
            (b"a", False),
            (b"a\nb", True),
            (b"a\nb\nc", False),
        ):
            with patch.object(dense.subprocess, "run", return_value=NS(stdout=raw)):
                if accepted:
                    dense._git_version("/synthetic")
                else:
                    with self.assertRaisesRegex(ValueError, "identity unavailable"):
                        dense._git_version("/synthetic")
        # The real supporting recipe separately requires two exact 40-hex IDs,
        # making its complete Git result 82 bytes; 128 isolates this read guard.

    def test_mount_metadata_one_mebibyte_guard(self):
        prefix = b"1 0 1:1 / / rw - ext4 none "
        for size, accepted in (
            (len(prefix) + 1, True),
            (1048576, True),
            (1048577, False),
        ):
            raw = prefix + b"x" * (size - len(prefix))
            with (
                patch.object(Path, "is_dir", return_value=True),
                patch.object(Path, "is_symlink", return_value=False),
                patch.object(Path, "resolve", return_value=Path("/synthetic")),
                patch.object(Path, "stat", return_value=NS(st_dev=1, st_ino=1)),
                patch.object(Path, "open", return_value=io.BytesIO(raw)),
            ):
                if accepted:
                    self.assertEqual(
                        dense._root_identity("/synthetic")["filesystem_type"], "ext4"
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "Mount metadata exceeds"):
                        dense._root_identity("/synthetic")
        # Empty/one-byte mountinfo cannot describe a complete matching mount.

    def test_verified_non_json_bytes_and_frozen_executable_size_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inert-data"
            for limit in (524288, 33554432):
                for size, accepted in ((0, True), (limit, True), (limit + 1, False)):
                    with path.open("wb") as stream:
                        stream.truncate(size)
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if accepted:
                        dense._verified(path, digest, limit, json_artifact=False)
                    else:
                        with self.assertRaisesRegex(ValueError, "evidence exceeds"):
                            dense._verified(path, digest, limit, json_artifact=False)
            for size, accepted in ((0, True), (33554432, True), (33554433, False)):
                with path.open("wb") as stream:
                    stream.truncate(size)
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if accepted:
                    profile_os.FrozenBinary(path, digest).check()
                else:
                    with self.assertRaisesRegex(ValueError, "Executable exceeds"):
                        profile_os.FrozenBinary(path, digest)
        # All files are inert regular data. No execution occurs. A negative
        # physical file length is inapplicable to the filesystem API.

    def test_diagnostic_frame_inventory_location_name_and_line_clamps(self):
        for count in (0, 1, 64, 65):
            error = TraceError("inert")
            error.trace = None
            for _ in range(count):
                error.trace = NS(
                    tb_next=error.trace,
                    tb_lineno=1,
                    tb_frame=NS(
                        f_code=NS(
                            co_filename="/synthetic/external.py", co_name="fixture"
                        )
                    ),
                )
            node = diagnostic.error_chain(error)["exceptions"][0]
            self.assertEqual(len(node["frames"]), min(count, 64))
            self.assertEqual(node["frames_truncated"], count > 64)
        for length, accepted in ((0, False), (1, True), (128, True), (129, False)):
            error = TraceError("inert")
            error.trace = NS(
                tb_next=None,
                tb_lineno=1,
                tb_frame=NS(
                    f_code=NS(co_filename="/synthetic/external.py", co_name="fixture")
                ),
            )
            frame = diagnostic.error_chain(
                error, locations={"/synthetic/external.py": "x" * length}
            )["exceptions"][0]["frames"][0]
            self.assertEqual(
                frame["file"], "x" * length if accepted else "<external-code>"
            )
        for length, accepted in ((0, False), (1, True), (64, True), (65, False)):
            error.trace.tb_frame.f_code.co_name = "x" * length
            frame = diagnostic.error_chain(
                error, locations={"/synthetic/external.py": "fixture"}
            )["exceptions"][0]["frames"][0]
            self.assertEqual(
                frame["function"], "x" * length if accepted else "<withheld>"
            )
        for number, output in ((-1, 0), (0, 0), (10**9, 10**9), (10**9 + 1, 10**9)):
            error.trace.tb_lineno = number
            frame = diagnostic.error_chain(error)["exceptions"][0]["frames"][0]
            self.assertEqual(frame["line"], output)

    def test_diagnostic_native_errno_digit_and_prefix_limits(self):
        for digits, accepted in ((0, False), (1, True), (10, True), (11, False)):
            text = (
                "ERROR: Hosted channel peer_addr failed: kind=InvalidInput; errno="
                + "1" * digits
            )
            result = diagnostic.sanitized_stderr(text.encode())
            self.assertEqual(result == text, accepted)
        text = "ERROR: Hosted channel peer_addr failed: kind=InvalidInput; errno=none"
        for size in (16384, 16385):
            result = diagnostic.sanitized_stderr((text + "\n").encode() * size)
            self.assertEqual(len(result), 16384)
        # A recognized kind's longest name is shorter than the regex's 40-byte
        # cap; a complete accepted kind at exactly 40 bytes is inapplicable.

    def test_admission_positive_submillisecond_rounding_and_child_cpu_domain(self):
        grant = AdmissionGrant(lambda: 0, NS(total=lambda: 0), wall_ms=1, cpu_ms=1)
        self.assertEqual(grant.remaining(), {"wall_ms": 1, "cpu_ms": 1})
        grant.clock = lambda: 0.0000001
        self.assertEqual(grant.remaining(), {"wall_ms": 0, "cpu_ms": 1})
        grant.clock = lambda: 0.001
        with self.assertRaisesRegex(ValueError, "grant exhausted"):
            grant.remaining()
        for cpu, accepted in (
            (-math.ulp(0.0), False),
            (0.0, True),
            (sys.float_info.max, True),
            (math.inf, False),
            (math.nan, False),
        ):
            grant = AdmissionGrant(lambda: 0, NS(total=lambda: 0))
            if accepted:
                grant.sample_child(cpu)
            else:
                with self.assertRaisesRegex(ValueError, "CPU clock regressed"):
                    grant.sample_child(cpu)
        for cpu, accepted in (
            (0, True),
            (sys.float_info.max, True),
            (-1, False),
            (math.inf, False),
            (math.nan, False),
        ):
            ledger = CpuLedger(
                {"admission": lambda: cpu, "owner": lambda: 0, "watchdog": lambda: 0}
            )
            if accepted:
                self.assertEqual(ledger.total(), cpu)
            else:
                with self.assertRaisesRegex(ValueError, "clock regressed"):
                    ledger.total()

    def test_snapshot_publication_visit_interval_edges(self):
        for visited, minimum, maximum, accepted in (
            (0, 0, 0, True),
            (1, 1, 1, True),
            (1, 0, 1, True),
            (0, -1, 1, False),
            (1, 2, 2, False),
            (1, 0, 2, False),
            (0, 1, 1, False),
            (1, 0, 0, False),
            (0, 1, 0, False),
        ):
            case = primitive.Base(methodName="runTest")
            case.setUp()
            try:
                store = snapshot.SnapshotStore(primitive.selected(1, 1), 17)
                raw, *_ = primitive.frame(store.selected, visited=visited)
                fd = store.begin()
                case.os.files[fd] = [raw, snapshot.SEALS]
                operation = lambda: store.publish(
                    hashlib.sha256(raw).hexdigest(),
                    minimum_visited=minimum,
                    maximum_visited=maximum,
                    deadline=5,
                    source_check=lambda: None,
                    final_check=lambda: None,
                    clock=lambda: 0,
                )
                if accepted:
                    self.assertEqual(operation().visited, visited)
                else:
                    with self.assertRaises(ValueError):
                        operation()
                    self.assertIsNone(store.pending)
                store.close()
            finally:
                case.doCleanups()


if __name__ == "__main__":
    unittest.main()
