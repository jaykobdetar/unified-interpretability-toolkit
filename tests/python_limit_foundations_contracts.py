"""Independent declared budget edges; pure validators and injected clocks only."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import viewer_resources
import inference_model_descriptor as descriptor
from atlas_host import cache, config, common
from atlas_host.budget import WorkGrant
from atlas_host.supervisor import AdmissionGrant, CpuLedger, Supervisor
import host_contracts
import model_descriptor_contracts

MIB, GIB = 1024**2, 1024**3


class Foundations(unittest.TestCase):
    def boundary(self, operation, low, high):
        for number, accepted in (
            (low - 1, False),
            (low, True),
            (high, True),
            (high + 1, False),
        ):
            with self.subTest(number=number):
                if accepted:
                    try:
                        operation(number)
                    except ValueError as error:
                        self.fail(f"Declared accepted edge refused: {error}")
                else:
                    with self.assertRaises(ValueError):
                        operation(number)

    def test_viewer_declared_ranges_and_workspace_coupling(self):
        ranges = {
            "version": (1, 1),
            "cpu_count": (1, 8),
            "address_space_bytes": (256 * MIB, 8 * GIB),
            "available_floor_bytes": (512 * MIB, 128 * GIB),
            "disk_reserve_bytes": (GIB, 1024 * GIB),
            "workspace_bytes": (64 * MIB, GIB),
            "tile_cache_bytes": (16 * MIB, 64 * GIB),
            "tile_cache_files": (64, 100000),
        }
        self.assertEqual(viewer_resources.RANGES, ranges)
        for field, (low, high) in ranges.items():
            with self.subTest(field=field):
                # The independent workspace coupling must not mask an edge.
                base = {"address_space_bytes": 8 * GIB}
                self.boundary(
                    lambda n: viewer_resources.validate({**base, field: n}), low, high
                )
        for workspace, accepted in ((128 * MIB, True), (128 * MIB + 1, False)):
            with self.subTest(workspace=workspace):
                value = {"address_space_bytes": 256 * MIB, "workspace_bytes": workspace}
                if accepted:
                    viewer_resources.validate(value)
                else:
                    with self.assertRaisesRegex(ValueError, "half"):
                        viewer_resources.validate(value)

    def test_exact_host_policy_values_refuse_adjacent_changes(self):
        expected = {
            "cpu_count": 1,
            "numeric_workers": 1,
            "heavy_jobs": 1,
            "rust_address_space_bytes": 768 * MIB,
            "rust_available_floor_bytes": 3 * GIB,
            "build_address_space_bytes": 2 * GIB,
            "build_jobs": 1,
            "build_start_available_bytes": 5 * GIB,
            "browser_tree_rss_bytes": 768 * MIB,
            "browser_start_available_bytes": 5 * GIB,
            "stop_available_bytes": 13 * GIB // 4,
            "inference_rss_bytes": 3 * GIB // 2,
            "inference_address_space_bytes": 3 * GIB,
            "inference_start_available_bytes": 19 * GIB // 4,
            "inference_wall_ms": 120000,
            "inference_cpu_ms": 90000,
            "prompt_tokens": 128,
            "new_tokens": 32,
            "client_lease_ms": 15000,
            "inference_queue": 0,
            "analytics_address_space_bytes": 768 * MIB,
            "analytics_rss_bytes": 768 * MIB,
            "analytics_start_available_bytes": 15 * GIB // 4,
            "analytics_wall_ms": 5000,
            "analytics_cpu_ms": 4000,
            "disk_reserve_bytes": 25 * GIB,
            "tile_disk_bytes": 2 * GIB,
            "tile_files": 1000,
            "pending_headers": 4,
            "header_bytes": 8192,
            "header_deadline_ms": 500,
            "dispatch_queue": 4,
            "numeric_queue": 8,
            "write_deadline_ms": 3000,
            "coordinator_body_bytes": 8192,
            "upstream_response_bytes": 2 * MIB,
        }
        self.assertEqual(config.LOCAL_LIMITS, expected)
        example = json.loads(
            (
                Path(__file__).resolve().parents[1] / "config/atlas-host.example.json"
            ).read_text()
        )
        for field, number in expected.items():
            for candidate, accepted in (
                (number - 1, False),
                (number, True),
                (number + 1, False),
            ):
                with self.subTest(field=field, candidate=candidate):
                    value = {**example, "limits": {field: candidate}}
                    if accepted:
                        result = config.validate_config(value, "/synthetic")
                        self.assertEqual(
                            config.capabilities(result)["limits"], expected
                        )
                    else:
                        with self.assertRaises(ValueError):
                            config.validate_config(value, "/synthetic")

    def test_host_port_and_path_label_boundaries(self):
        example = json.loads(
            (
                Path(__file__).resolve().parents[1] / "config/atlas-host.example.json"
            ).read_text()
        )
        self.boundary(
            lambda n: config.validate_config(
                {**example, "ports": {"coordinator": n, "renderer": 8797}}, "/synthetic"
            ),
            1,
            65535,
        )
        for length, accepted in ((0, False), (1, True), (4096, True), (4097, False)):
            value = {**example, "paths": {"registry": "r" * length, "cache": "cache"}}
            with (
                self.subTest(length=length),
                patch.object(Path, "resolve", lambda p: p),
            ):
                if accepted:
                    config.validate_config(value, "/synthetic")
                else:
                    with self.assertRaisesRegex(ValueError, "label"):
                        config.validate_config(value, "/synthetic")

    def test_work_and_admission_input_budgets_both_ends(self):
        bounds = {
            "values": (1, 2**63 - 1),
            "wall_ms": (1, 5000),
            "cpu_ms": (1, 4000),
            "lease_ms": (1, 15000),
        }
        for field, (low, high) in bounds.items():
            with self.subTest(field=field):
                self.boundary(
                    lambda n: WorkGrant(
                        **{
                            "values": 1,
                            "wall_ms": 5000,
                            "cpu_ms": 4000,
                            "lease_ms": 15000,
                            field: n,
                        },
                        clock=lambda: 0,
                        cpu_clock=lambda: 0,
                    ),
                    low,
                    high,
                )
        for field, (low, high) in {"wall_ms": (1, 5000), "cpu_ms": (1, 4000)}.items():
            ledger = CpuLedger(
                {name: lambda: 0 for name in ("admission", "owner", "watchdog")}
            )
            with self.subTest(admission=field):
                self.boundary(
                    lambda n: AdmissionGrant(lambda: 0, ledger, **{field: n}), low, high
                )

    def test_native_command_deadline_range(self):
        supervisor = Supervisor()
        token = supervisor.acquire("tile", "synthetic")
        try:
            self.boundary(
                lambda n: supervisor.native_command(token, {}, n), 1, 2**31 - 1
            )
        finally:
            supervisor.admission_aborted(token, no_child_created=True)
            token.release()

    def test_descriptor_dimensions_both_ends_with_valid_grouping(self):
        bounds = {
            "hidden_size": (1, 16384),
            "intermediate_size": (1, 131072),
            "num_hidden_layers": (1, 256),
            "num_attention_heads": (1, 256),
            "num_key_value_heads": (1, 256),
            "vocab_size": (1, 500000),
            "head_dim": (2, 1024),
            "max_position_embeddings": (1, 1048576),
        }
        for field, (low, high) in bounds.items():

            def describe(number):
                value = model_descriptor_contracts.config()
                value[field] = number
                if field == "num_key_value_heads":
                    value["num_attention_heads"] = 256
                return descriptor.describe_config(value)

            with self.subTest(field=field):
                self.boundary(describe, low, high)

    def test_descriptor_finite_positive_and_fixed_numeric_options(self):
        for field in ("rms_norm_eps", "rope_theta"):
            for number, accepted in (
                (math.nextafter(0, math.inf), True),
                (0, False),
                (math.nextafter(0, -math.inf), False),
                (sys.float_info.max, True),
                (math.inf, False),
            ):
                with self.subTest(field=field, number=number):
                    value = {**model_descriptor_contracts.config(), field: number}
                    if accepted:
                        descriptor.describe_config(value)
                    else:
                        with self.assertRaises(ValueError):
                            descriptor.describe_config(value)
        for field, number in (
            ("attention_dropout", 0.0),
            ("partial_rotary_factor", 1.0),
        ):
            for candidate, accepted in (
                (math.nextafter(number, -math.inf), False),
                (number, True),
                (math.nextafter(number, math.inf), False),
            ):
                with self.subTest(field=field, candidate=candidate):
                    value = {**model_descriptor_contracts.config(), field: candidate}
                    if accepted:
                        descriptor.describe_config(value)
                    else:
                        with self.assertRaises(ValueError):
                            descriptor.describe_config(value)
        self.boundary(
            lambda n: descriptor.describe_config(
                {**model_descriptor_contracts.config(), "pretraining_tp": n}
            ),
            1,
            1,
        )

    def test_pinned_config_bytes_at_upper_edge_and_empty_refusal(self):
        raw = json.dumps(model_descriptor_contracts.config()).encode()
        for length, accepted in ((65536, True), (65537, False)):
            padded = raw + b" " * (length - len(raw))
            with self.subTest(length=length):
                if accepted:
                    descriptor.pinned_descriptor(
                        padded, model_descriptor_contracts.receipt(padded)
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "byte bound"):
                        descriptor.pinned_descriptor(
                            padded, model_descriptor_contracts.receipt(padded)
                        )
        with self.assertRaisesRegex(ValueError, "byte bound"):
            descriptor.pinned_descriptor(b"", model_descriptor_contracts.receipt(b"x"))
        # One byte cannot encode a complete model configuration. Isolate byte
        # admission with a parser double; the upper edge above uses real JSON.
        with patch.object(
            descriptor.json, "loads", return_value=model_descriptor_contracts.config()
        ):
            descriptor.pinned_descriptor(b"x", model_descriptor_contracts.receipt(b"x"))

    def test_common_integer_labels_digests_and_json_byte_boundaries(self):
        self.boundary(common.integer, 0, 2**63 - 1)
        for length, accepted in ((0, False), (1, True), (128, True), (129, False)):
            with self.subTest(label_length=length):
                if accepted:
                    common.label("x" * length)
                else:
                    with self.assertRaises(ValueError):
                        common.label("x" * length)
        for length in (63, 64, 65):
            with self.subTest(digest_length=length):
                if length == 64:
                    common.digest("a" * length)
                else:
                    with self.assertRaises(ValueError):
                        common.digest("a" * length)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.json"
            for size, accepted in ((16384, True), (16385, False)):
                path.write_bytes(b"{}" + b" " * (size - 2))
                with self.subTest(json_bytes=size):
                    if accepted:
                        self.assertEqual(common.read_json(path, 16384), {})
                    else:
                        with self.assertRaisesRegex(ValueError, "byte limit"):
                            common.read_json(path, 16384)

    def test_source_binding_rank_dimension_product_and_display_edges(self):
        self.boundary(
            lambda n: cache.binding({**host_contracts.binding_value(), "tensor": n}),
            0,
            9999,
        )
        for rank, accepted in ((0, False), (1, True), (32, True), (33, False)):
            value = (
                host_contracts.binding_value([1] * rank, [0] * max(0, rank - 2))
                if rank
                else {**host_contracts.binding_value(), "shape": []}
            )
            with self.subTest(rank=rank):
                if accepted:
                    cache.binding(value)
                else:
                    with self.assertRaises(ValueError):
                        cache.binding(value)
        for axis in (0, 1):

            def binding(number):
                shape = [1, 1]
                shape[axis] = number
                return cache.binding(host_contracts.binding_value(shape))

            with self.subTest(axis=axis):
                self.boundary(binding, 1, 200000)
        for dimension, accepted in (
            (0, False),
            (1, True),
            (2**63 - 1, True),
            (2**63, False),
        ):
            with self.subTest(leading_dimension=dimension):
                value = host_contracts.binding_value([dimension, 1, 1], [0])
                if accepted:
                    cache.binding(value)
                else:
                    with self.assertRaises(ValueError):
                        cache.binding(value)
        with self.assertRaisesRegex(ValueError, "overflow"):
            cache.binding(host_contracts.binding_value([2**62, 2, 1], [0]))
        self.boundary(
            lambda n: cache.binding(host_contracts.binding_value([3, 1, 1], [n])), 0, 2
        )
        for length, accepted in ((0, False), (1, True), (512, True), (513, False)):
            value = {**host_contracts.binding_value(), "name": "t" * length}
            with self.subTest(name_length=length):
                if accepted:
                    cache.binding(value)
                else:
                    with self.assertRaises(ValueError):
                        cache.binding(value)

    def test_immutable_response_byte_cap_uses_actual_payload_and_digest(self):
        for length, accepted in ((0, True), (2 * MIB, True), (2 * MIB + 1, False)):
            payload = b"x" * length
            digest = hashlib.sha256(payload).hexdigest()
            with self.subTest(payload_bytes=length):
                if accepted:
                    self.assertEqual(
                        cache.immutable_headers(digest, payload, "image/png", "tile")[
                            "Content-Length"
                        ],
                        str(length),
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "response bound"):
                        cache.immutable_headers(digest, payload, "image/png", "tile")

    def test_derivation_coordinate_parameter_and_control_boundaries(self):
        fixture = host_contracts.CacheContracts()
        fixture.setUp()
        base = fixture.value
        for field in ("level", "x", "y"):
            with self.subTest(field=field):
                self.boundary(
                    lambda n: cache.derivation_identity({**base, field: n}),
                    0,
                    2**31 - 1,
                )
        for count, accepted in ((0, True), (16, True), (17, False)):
            value = {**base, "parameters": {f"p{i}": 1 for i in range(count)}}
            with self.subTest(parameters=count):
                if accepted:
                    cache.derivation_identity(value)
                else:
                    with self.assertRaises(ValueError):
                        cache.derivation_identity(value)
        for field in ("algorithm", "rule", "encoding"):
            for length, accepted in ((0, False), (1, True), (128, True), (129, False)):
                with self.subTest(field=field, length=length):
                    value = {**base, field: "x" * length}
                    if accepted:
                        cache.derivation_identity(value)
                    else:
                        with self.assertRaises(ValueError):
                            cache.derivation_identity(value)
        for key_length, accepted in ((0, False), (1, True), (64, True), (65, False)):
            with self.subTest(parameter_key_length=key_length):
                value = {**base, "parameters": {"k" * key_length: 1}}
                if accepted:
                    cache.derivation_identity(value)
                else:
                    with self.assertRaises(ValueError):
                        cache.derivation_identity(value)
        for length, accepted in ((0, False), (1, True), (128, True), (129, False)):
            with self.subTest(parameter_string_length=length):
                value = {**base, "parameters": {"p": "x" * length}}
                if accepted:
                    cache.derivation_identity(value)
                else:
                    with self.assertRaises(ValueError):
                        cache.derivation_identity(value)
        self.boundary(
            lambda n: cache.derivation_identity({**base, "parameters": {"p": n}}),
            -(2**63 - 1),
            2**63 - 1,
        )
        control = {"algorithm": "control", "seed": 0, "permutation_digest": "b" * 64}
        self.boundary(
            lambda n: cache.derivation_identity(
                {**base, "control": {**control, "seed": n}}
            ),
            0,
            2**64 - 1,
        )
        for length, accepted in ((8192, True), (8193, False)):
            with (
                self.subTest(descriptor_bytes=length),
                patch.object(cache, "canonical", return_value=b"x" * length),
            ):
                # The individually bounded fields constrain real descriptor size.
                # This isolates the final byte-admission guard, not serialization.
                if accepted:
                    cache.derivation_identity(deepcopy(base))
                else:
                    with self.assertRaisesRegex(ValueError, "too large"):
                        cache.derivation_identity(deepcopy(base))


if __name__ == "__main__":
    unittest.main()
