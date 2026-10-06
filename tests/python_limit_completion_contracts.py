"""Remaining caller limits with inert bytes, clocks and validated source doubles."""

import __future__
import ast
import contextlib
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import acquisition_contracts as acquisition_fixture
import inference_architecture as architecture
import inference_edits as edits
import inference_sweep as sweep
import inference_worker as worker
import profile_lifetime_policy as recipe_fixture
import profile_startup_diagnostics as diagnostic_fixture
import python_limit_inference_records_contracts as records
import python_limit_numeric_contracts as numeric
import python_limit_static_metadata_contracts as metadata
from analytics import svd_summary as svd, worker as analytics_worker
from atlas_host import acquisition, cli, common, dense_static_admission as dense
from atlas_host import static_models as static, validation_policy as policy
from atlas_host.budget import WorkGrant
from atlas_host.supervisor import AdmissionGrant, CpuLedger


class CompletionLimits(unittest.TestCase):
    def check(self, operation, accepted, message=None):
        if accepted:
            return operation()
        with self.assertRaisesRegex(ValueError, message or "."):
            operation()
        return None

    def fixture(self, cls):
        case = cls(methodName="runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def test_ascii_label_control_boundaries(self):
        for code, accepted in (
            (31, False),
            (32, True),
            (126, True),
            (127, False),
            (128, True),
        ):
            with self.subTest(code=code):
                self.check(lambda: common.label(chr(code)), accepted)

    def test_worker_header_size_and_cumulative_budget_edges(self):
        cap = 8 * 1024**2
        for sizes, accepted in (
            ([1], False),
            ([2], True),
            ([cap], True),
            ([cap + 1], False),
            ([cap] * 4, True),
            ([cap] * 4 + [2], False),
        ):
            files = {f"shard{n}": (1, 2, 3, 4, 5) for n in range(len(sizes))}
            payload = {
                "root": "/synthetic",
                "fingerprints": files,
                "model": {
                    "source_identity": "a" * 64,
                    "catalog": [
                        {
                            "id": 0,
                            "name": "x",
                            "shape": [1],
                            "shard": "validated-elsewhere",
                            "dtype": "BF16",
                            "byte_offset": 0,
                        }
                    ],
                },
            }
            catalog = NS(
                root=Path("/synthetic"),
                files=files,
                index_fingerprint=None,
                verify=Mock(),
            )

            # Trusted catalog double isolates header-byte admission from the
            # separately tested tensor/catalog correspondence. No file is opened.
            def pread(fd, count, offset):
                size = sizes[fd]
                return (
                    struct.pack("<Q", size)
                    if offset == 0
                    else b"{}" + b" " * (size - 2)
                )

            with self.subTest(sizes=sizes), contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch.object(analytics_worker, "Catalog", return_value=catalog)
                )
                stack.enter_context(
                    patch.object(
                        analytics_worker, "fingerprint", return_value=(1, 2, 3, 4, 5)
                    )
                )
                stack.enter_context(
                    patch.object(
                        analytics_worker.os, "open", side_effect=range(len(sizes))
                    )
                )
                stack.enter_context(patch.object(analytics_worker.os, "fstat"))
                stack.enter_context(patch.object(analytics_worker.os, "close"))
                stack.enter_context(
                    patch.object(analytics_worker.os, "pread", side_effect=pread)
                )
                self.check(
                    lambda: analytics_worker.catalog_from_payload(payload),
                    accepted,
                    "Header budget",
                )
        # An additional legal header occupies at least two bytes. 32 MiB + 1
        # cannot isolate the cumulative guard while respecting that minimum.

    def test_acquisition_fetch_and_registration_clamps(self):
        for remaining, fetch_limit, registration_limit in (
            (0.0005, 0.001, 1),
            (0.001, 0.001, 1),
            (5, 5, 5000),
            (10, 10, 5000),
            (11, 10, 5000),
        ):
            case = self.fixture(acquisition_fixture.Contracts)
            clock = Mock(
                side_effect=lambda: 0 if clock.call_count == 1 else 20 - remaining
            )
            streams = []

            class Stream(io.BytesIO):
                def read(self, count=-1):
                    self.counts.append(count)
                    return super().read(count)

            def fetch(url, timeout):
                self.assertAlmostEqual(timeout, fetch_limit)
                stream = Stream(case.data[url.rsplit("/", 1)[1]])
                stream.counts = []
                stream.expected_size = len(stream.getvalue())
                streams.append(stream)
                return stream

            register = Mock(return_value={"enabled": False})
            with (
                self.subTest(remaining=remaining),
                patch.object(case.registry, "register", register),
            ):
                result = case.run_acquire(fetch=fetch, clock=clock, timeout_ms=20000)
                self.assertTrue(result["registered"])
                self.assertEqual(
                    register.call_args.kwargs["timeout_ms"], registration_limit
                )
                self.assertTrue(register.call_args.kwargs["publish_disabled"])
                self.assertEqual(
                    [s.counts for s in streams],
                    [[s.expected_size, 1] for s in streams],
                )
        # The remaining-byte branch above is below the fixed quantum. Exercise
        # the 65536 quantum and final one-byte remainder with real inert bytes.
        case = self.fixture(acquisition_fixture.Contracts)
        case.data["model.safetensors"] = b"x" * 65537
        for entry in case.manifest["files"]:
            raw = case.data[entry["name"]]
            entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        case.reviewed = acquisition.plan(case.manifest, "Tiny", max_bytes=100000)
        reads = {}

        class Recorded(io.BytesIO):
            def read(self, count=-1):
                reads[self.name].append(count)
                return super().read(count)

        def fetch_large(url, _timeout):
            name = url.rsplit("/", 1)[1]
            source = Recorded(case.data[name])
            source.name = name
            reads[name] = []
            return source

        with patch.object(case.registry, "register", return_value={"enabled": False}):
            case.run_acquire(
                max_bytes=100000,
                fetch=fetch_large,
            )
        self.assertEqual(reads["model.safetensors"], [65536, 1, 1])

    def test_work_grant_chunk_visit_and_effective_time_edges(self):
        high = 2**63 - 1
        for chunk, accepted in ((0, False), (1, True), (high, True), (high + 1, False)):
            grant = WorkGrant(1, 5000, 4000, clock=lambda: 0, cpu_clock=lambda: 0)
            callback = Mock(return_value=0)
            self.check(lambda: grant.advance(chunk, callback), accepted)
            if accepted:
                self.assertEqual(callback.call_args.args, (1, 5.0))
            else:
                callback.assert_not_called()
        for visits, accepted in ((-1, False), (0, True), (1, True), (2, False)):
            grant = WorkGrant(1, 5000, 4000, clock=lambda: 0, cpu_clock=lambda: 0)
            self.check(lambda: grant.advance(1, lambda *_: visits), accepted)
            self.assertEqual(grant.failed, not accepted)
        for kind in ("wall", "cpu"):
            clock, cpu = Mock(return_value=0), Mock(return_value=0)
            grant = WorkGrant(1, 1, 1, clock=clock, cpu_clock=cpu)
            for elapsed, accepted in ((0, True), (0.0005, False), (0.001, False)):
                (clock if kind == "wall" else cpu).return_value = elapsed
                self.check(grant.remaining, accepted)
        clock = Mock(return_value=0)
        grant = WorkGrant(1, 5000, 4000, clock=clock, cpu_clock=lambda: 0, lease_ms=1)
        clock.return_value = math.nextafter(0.001, 0)
        grant.remaining()
        clock.return_value = 0.001
        self.check(grant.remaining, False, "lease expired")

    def test_all_cli_manifest_read_callers_actual_byte_boundaries(self):
        case = self.fixture(acquisition_fixture.Contracts)
        base = json.dumps(case.manifest).encode()
        for command in ("plan", "register", "acquire-plan", "acquire"):
            for size, accepted in ((32768, True), (32769, False)):
                raw = base + b" " * (size - len(base))
                registry = NS(register=Mock(return_value={}))
                args = ["--config", "config", command, "--manifest", "manifest"]
                if command != "plan":
                    args += ["--name", "Tiny", "--max-bytes", "100"]
                if command == "register":
                    args += ["--source", "source"]
                if command == "acquire":
                    args += [
                        "--destination",
                        "destination",
                        "--plan-digest",
                        "a" * 64,
                        "--accept-license",
                    ]
                with (
                    self.subTest(command=command, size=size),
                    patch.object(
                        cli,
                        "load_config",
                        return_value={"paths": {"registry": "/inert"}},
                    ),
                    patch.object(cli, "Registry", return_value=registry),
                    patch.object(
                        Path, "open", side_effect=lambda *_a, **_k: io.BytesIO(raw)
                    ),
                    patch.object(
                        acquisition, "acquire", return_value={"registered": True}
                    ) as acquire,
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(cli.main(args), 0 if accepted else 2)
                    if not accepted:
                        registry.register.assert_not_called()
                        acquire.assert_not_called()

    def test_dense_verified_json_and_recipe_read_caller_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            for size, accepted in ((32768, True), (32769, False)):
                raw = common.canonical({"x": "x" * (size - 8)})
                self.assertEqual(len(raw), size)
                path.write_bytes(raw)
                self.check(
                    lambda: dense._verified(
                        path, hashlib.sha256(raw).hexdigest(), 32768, json_artifact=True
                    ),
                    accepted,
                )
            path.write_bytes(b"")
            self.check(
                lambda: dense._verified(
                    path, hashlib.sha256(b"").hexdigest(), 32768, json_artifact=True
                ),
                False,
            )
        case = self.fixture(recipe_fixture.PolicyTests)
        base = json.dumps(case.data).encode()
        for size, accepted in ((32768, True), (32769, False)):
            raw = base + b" " * (size - len(base))
            case.receipt.write_bytes(raw)
            operation = lambda: policy.bind_reviewed_recipe(
                case.receipt,
                hashlib.sha256(raw).hexdigest(),
                monitor_path=case.root / "run.py",
                harness_path=case.root / "runtime_harness.py",
                binary_path=case.root / "target/release/weight-atlas-rust",
            )
            self.check(operation, accepted)
            # The same cap is checked again by read_json. Isolate its caller
            # without changing the other source/binary evidence checks.
            original = policy._verified_file

            def verified(path, expected, limit):
                if Path(path) == case.receipt:
                    return (
                        str(path),
                        tuple(sorted(policy.fingerprint(path.stat()).items())),
                    )
                return original(path, expected, limit)

            with patch.object(policy, "_verified_file", side_effect=verified):
                self.check(operation, accepted)

    def test_dense_zero_counters_and_calibration_correspondence(self):
        fixture = metadata.StaticMetadataLimits(methodName="runTest")
        prepared = NS(entry={"name": "Owner"})
        for key in (
            "sha_hashed_shards",
            "sha_expected_matched_shards",
            "sha_missing_expected_shards",
            "sha_verified_shards",
        ):
            for number, accepted in ((-1, False), (0, True), (1, False)):
                model = fixture.model()
                model["coverage"][key] = number
                self.check(lambda: dense.check_native_model(prepared, model), accepted)

        for key, exact in (("calibrated_tensors", 1), ("values_streamed", 2)):
            for number, accepted in (
                (exact - 1, False),
                (exact, True),
                (exact + 1, False),
            ):
                model = fixture.model()
                model["coverage"][key] = number
                self.check(lambda: dense.check_native_model(prepared, model), accepted)

    def test_reviewed_recipe_file_caps_for_each_distinct_caller(self):
        # Source inventory, monitor/harness, binary, pinned tiny fixture and
        # receipt have distinct bounds even though they share one helper.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inert-evidence"
            for limit in (244, 32768, 128 * 1024, 512 * 1024, 32 * 1024**2):
                for size, accepted in ((0, True), (limit, True), (limit + 1, False)):
                    with path.open("wb") as stream:
                        stream.truncate(size)
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    self.check(
                        lambda: policy._verified_file(path, digest, limit), accepted
                    )
        # Negative file lengths are physically inapplicable; zero-byte evidence
        # passes this hash/size helper, while callers enforce recipe identities.

    def test_hash_reader_fixed_quantum_and_eof_with_inert_bytes(self):
        tree = ast.parse((ROOT / "tools/check_model_hashes.py").read_text())
        loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.While) and "f.read" in ast.unparse(node)
        ]
        self.assertEqual(len(loops), 1)
        counts = []

        class Reader(io.BytesIO):
            def read(self, size=-1):
                counts.append(size)
                return super().read(size)

        raw = b"x" * (2 * 1024**2 + 1)
        namespace = {"f": Reader(raw), "guard": Mock(), "digest": hashlib.sha256()}
        exec(
            compile(
                ast.Module(body=loops, type_ignores=[]),
                "original-hash-reader-loop",
                "exec",
            ),
            namespace,
        )
        self.assertEqual(counts, [2 * 1024**2] * 3)
        self.assertEqual(
            namespace["digest"].hexdigest(), hashlib.sha256(raw).hexdigest()
        )
        self.assertEqual(namespace["guard"].call_count, 3)

    def test_diagnostic_fd_lower_edge_without_real_descriptors(self):
        case = self.fixture(diagnostic_fixture.Diagnostics)
        for fd, accepted in ((2, False), (3, True), (2**31 - 1, True)):
            argv = diagnostic_fixture.ARGV[:-1] + [str(fd)]
            self.check(
                lambda: case.collector.prepare(argv, (fd,)), accepted, "inherited FD"
            )
        # No declared upper FD limit exists; the OS descriptor probe is mocked.

    def test_static_payload_offset_coupled_lower_and_upper_edges(self):
        fixture = metadata.StaticMetadataLimits(methodName="runTest")
        original = static._read
        # Reuse the real prepared-header path with its reviewed descriptor double.
        for offsets, accepted in (
            ([-1, 1], False),
            ([0, 2], True),
            ([0, 3], False),
            ([1, 3], False),
            ([0, 0], False),
        ):
            header = common.canonical(
                {"tensor0": {"dtype": "BF16", "shape": [1, 1], "data_offsets": offsets}}
            )
            entry = {
                "root": "/synthetic",
                "manifest": {
                    "revision": "a" * 40,
                    "provenance": "owner_expected",
                    "files": [
                        {"name": "config.json", "bytes": 2},
                        {"name": "model.safetensors", "bytes": 8 + len(header) + 2},
                    ],
                },
                "fingerprints": {},
            }
            for f in entry["manifest"]["files"]:
                entry["fingerprints"][f["name"]] = {
                    "bytes": f["bytes"],
                    "device": 1,
                    "inode": 1,
                    "mtime_ns": 1,
                    "ctime_ns": 1,
                }
            with (
                patch.object(static, "_current"),
                patch.object(
                    static,
                    "pinned_descriptor",
                    return_value={
                        "parameter_shapes": {"tensor0": [1, 1]},
                        "parameter_count": 1,
                    },
                ),
                patch.object(
                    static,
                    "_read",
                    side_effect=lambda _entry, name, *_a, **_k: (
                        b"{}" if name == "config.json" else header
                    ),
                ),
            ):
                self.check(
                    lambda: static.StaticPolicy(
                        NS(owner_receipt=lambda _: entry)
                    ).prepare("synthetic"),
                    accepted,
                )
        # Individual start==payload cannot satisfy positive tensor extent;
        # end==0 cannot either. [0,2] witnesses both effective legal edges.
        self.assertIs(static._read, original)
        fixture.prepare(count=2)  # Exact adjacent spans cover the full payload.

    def test_cpu_ledger_monotonic_and_clock_edges(self):
        for number, accepted in (
            (-math.ulp(0.0), False),
            (0.0, True),
            (sys.float_info.max, True),
            (math.inf, False),
            (math.nan, False),
            (True, False),
        ):
            ledger = CpuLedger(
                {
                    "admission": lambda: number,
                    "owner": lambda: 0.0,
                    "watchdog": lambda: 0.0,
                }
            )
            self.check(ledger.total, accepted)
        clock = Mock(return_value=1.0)
        ledger = CpuLedger(
            {"admission": clock, "owner": lambda: 0.0, "watchdog": lambda: 0.0}
        )
        self.assertEqual(ledger.total(), 1.0)
        clock.return_value = math.nextafter(1.0, 0.0)
        self.check(ledger.total, False, "regressed")
        clock.return_value = 1.0
        self.assertEqual(ledger.total(), 1.0)
        grant = AdmissionGrant(lambda: 0, ledger)
        grant.sample_child(1.0)
        self.check(lambda: grant.sample_child(math.nextafter(1.0, 0)), False)
        grant.sample_child(1.0)

    def test_legacy_architecture_positive_dimensions_and_unbounded_upper(self):
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "head_dim",
        ):
            for number, accepted in ((0, False), (1, True), (2**1000, True)):
                config = {**architecture.CONFIG, "head_dim": 1, key: number}
                if key == "num_attention_heads":
                    config["num_key_value_heads"] = 1
                if key == "num_key_value_heads":
                    config["num_attention_heads"] = number
                self.check(lambda: architecture.describe(config), accepted)
        # describe is metadata-only and declares no upper dimension bound.

    def test_edit_pair_candidate_caller_bounds(self):
        baseline = next(
            item for item in records.rows("edit_empty") if "baseline" in item
        )
        template = baseline["candidates"][0]
        for count, accepted in ((0, False), (1, True), (10, True), (11, False)):
            step = deepcopy(baseline)
            step["candidates"] = [{**template, "id": n} for n in range(count)]
            self.check(lambda: edits.validate_pair(step), accepted)
        for number, accepted in ((-1, False), (0, True), (49151, True), (49152, False)):
            step = deepcopy(baseline)
            step["candidates"] = [{**template, "id": number}]
            self.check(lambda: edits.validate_pair(step), accepted)

    def test_sweep_record_index_and_candidate_id_edges(self):
        fixture = records.InferenceRecordLimits(methodName="runTest")
        baseline, plan = fixture.sweep()
        all_steps = [
            item for item in records.rows("sweep_head_zero") if "sweep" in item
        ]
        for index, accepted in (
            (-1, False),
            (0, True),
            (plan["records"] - 1, True),
            (plan["records"], False),
        ):
            step = deepcopy(
                next((row for row in all_steps if row["index"] == index), baseline)
            )
            step["index"] = index
            self.check(lambda: sweep.validate_step(step, plan), accepted)
        for number, accepted in ((-1, False), (0, True), (49151, True), (49152, False)):
            step = deepcopy(baseline)
            step["sweep"]["candidates"] = [
                {**step["sweep"]["candidates"][0], "id": number}
            ]
            self.check(lambda: sweep.validate_step(step, plan), accepted)

    def test_worker_capture_key_count_uses_existing_validated_record_bounds(self):
        functions = [
            node
            for path in (
                ROOT / "tools/inference_worker.py",
                ROOT / "tools/inference_generation.py",
            )
            if path.is_file()
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.FunctionDef) and node.name == "capture"
        ]
        self.assertEqual(len(functions), 1)

        # Execute the original hook in isolation. Its tensor double only exposes
        # shape/index/float conversion; no ML import or generation is performed.
        class Tensor:
            ndim = 4

            def __init__(self, count):
                self.shape = (1, worker.HEADS, 1, count)

            def __getitem__(self, _key):
                return self

            def detach(self):
                return self

            def float(self):
                return self

            def tolist(self):
                return [0.0] * self.shape[-1]

        namespace = {
            "selected": [],
            "observed": {},
            "observation": {"kind": "attention", "head": 0},
            "HEADS": worker.HEADS,
        }
        exec(
            compile(
                ast.Module(body=functions, type_ignores=[]),
                "original-capture-hook",
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        for count, accepted in ((0, False), (1, True), (160, True), (161, False)):
            self.check(
                lambda: namespace["capture"](None, None, (Tensor(1), Tensor(count))),
                accepted,
            )
            if accepted:
                self.assertEqual(len(namespace["observed"]["probabilities"]), count)

    def test_svd_array_order_fraction_and_singular_edges(self):
        fixture = numeric.NumericLimits(methodName="runTest")
        baseline, kwargs = fixture.report()
        for key in ("singular_values", "energy_fractions", "rank_one_residual_preview"):
            for count, accepted in ((0, False), (1, True), (2, False)):
                report = deepcopy(baseline)
                report["results"]["original"][key] *= count
                self.check(lambda: svd.validate(report, **kwargs), accepted, "length")
        # Legal complete two-value spectrum at a zero second singular value.
        kwargs["tensor"]["shape"] = [2, 2]
        kwargs["region"].update(rows=2, cols=2)
        report = svd.skeleton(kwargs["model"], kwargs["tensor"], kwargs["region"], 17)
        side = {
            **baseline["results"]["original"],
            "singular_values": [1.0, 0.0],
            "energy_fractions": [1.0, 0.0],
            "rank_one_residual_preview": [0.0] * 4,
        }
        report["results"] = {key: deepcopy(side) for key in ("original", "shuffled")}
        svd.validate(report, **kwargs)
        for key, values in (
            ("singular_values", [0.0, 1.0]),
            ("singular_values", [1.0, -math.ulp(0.0)]),
            ("energy_fractions", [1.0, -math.ulp(0.0)]),
            ("energy_fractions", [math.nextafter(1 + 1e-10, math.inf), 0.0]),
        ):
            bad = deepcopy(report)
            bad["results"]["original"][key] = values
            self.check(lambda: svd.validate(bad, **kwargs), False)
        maximum = math.sqrt(4 * svd.MAX_BF16**2) * (1 + 1e-10)
        for value, accepted in (
            (maximum, True),
            (math.nextafter(maximum, math.inf), False),
        ):
            bad = deepcopy(report)
            bad["results"]["original"]["singular_values"][0] = value
            # Energy accounting constrains a complete spectrum more tightly
            # than the direct numeric ceiling. Isolate only accounting here;
            # the normal complete records above retain all accounting checks.
            with patch.object(svd.math, "isclose", return_value=True):
                self.check(
                    lambda: svd.validate(bad, **kwargs), accepted, "numeric arrays"
                )
        for value, accepted in (
            (1 + 1e-10, True),
            (math.nextafter(1 + 1e-10, math.inf), False),
        ):
            bad = deepcopy(report)
            bad["results"]["original"]["energy_fractions"][0] = value
            with patch.object(svd.math, "isclose", return_value=True):
                self.check(
                    lambda: svd.validate(bad, **kwargs), accepted, "energy fractions"
                )


if __name__ == "__main__":
    unittest.main()
