"""Trusted correspondence and job-ledger edges; no model or worker runtime."""

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import host_contracts as fixture
import profile_worker_primitives as primitive
from atlas_host.inference import head_layout_metadata
from atlas_host.profile_worker import ProfileJob


class HeadLedgerLimits(unittest.TestCase):
    def metadata(self, *, layer=0, **changes):
        descriptor = json.loads(
            (ROOT / "tests/fixtures/host-head-layout-v1.json").read_text()
        )
        descriptor.update(
            layers=1, query_heads=1, kv_heads=1, head_dim=1, width=1, queries_per_kv=1
        )
        descriptor.update(changes)
        descriptor["queries_per_kv"] = descriptor["query_heads"] // max(
            1, descriptor["kv_heads"]
        )
        shape = [
            max(1, descriptor["kv_heads"] * descriptor["head_dim"]),
            max(1, descriptor["width"]),
        ]
        descriptor["projection_mappings"]["k_proj"] = {
            "shape": shape,
            "axis": "rows",
            "heads": descriptor["kv_heads"],
            "head_dim": descriptor["head_dim"],
        }
        source = descriptor["source_model"]
        manifest = {
            **deepcopy(fixture.MANIFEST),
            "repository": source["repo"],
            "revision": source["revision"],
            "files": [
                {
                    "name": "model.safetensors",
                    "bytes": 1,
                    "sha256": source["weights_sha256"],
                },
                {"name": "config.json", "bytes": 1, "sha256": source["config_sha256"]},
            ],
        }
        binding = fixture.binding_value(shape)
        binding["name"] = f"model.layers.{layer}.self_attn.k_proj.weight"
        return head_layout_metadata(manifest, descriptor, binding)

    def test_declared_layer_count_and_selected_layer_edges(self):
        for count, layer, accepted in (
            (0, 0, False),
            (1, 0, True),
            (200000, 199999, True),
            (200001, 0, False),
            (200000, 200000, False),
        ):
            if accepted:
                self.assertIn("head_layout", self.metadata(layers=count, layer=layer))
            else:
                with self.assertRaises(ValueError):
                    self.metadata(layers=count, layer=layer)

    def test_query_and_queries_per_kv_signed_integer_edges_in_k_projection(self):
        for count, accepted in (
            (0, False),
            (1, True),
            (2**63 - 1, True),
            (2**63, False),
        ):
            if accepted:
                result = self.metadata(query_heads=count)
                self.assertEqual(result["head_layout"]["queries_per_kv"], count)
            else:
                with self.assertRaises(ValueError):
                    self.metadata(query_heads=count)

    def test_kv_head_dimension_and_width_effective_binding_edges(self):
        for field in ("kv_heads", "head_dim", "width"):
            for count, accepted in (
                (0, False),
                (1, True),
                (200000, True),
                (200001, False),
            ):
                changes = {field: count}
                if field == "kv_heads":
                    changes["query_heads"] = count
                if accepted:
                    self.metadata(**changes)
                else:
                    with self.assertRaises(ValueError):
                        self.metadata(**changes)
        # These correspondence guards accept trusted descriptors, not visitor
        # architectures. Display axes <=200000 mask the signed-63-bit upper
        # guards for KV heads, head dimension and width in every projection.
        # The real descriptor producer has its own smaller, separately tested
        # architecture caps; these inputs do not qualify arbitrary models.

    def test_profile_local_cpu_and_original_wall_budget_edges(self):
        platform = NS(clock=lambda: 0, owner_cpu=lambda: 0)
        job = ProfileJob(primitive.selected(1, 1), 17, "a" * 32, "context", platform)
        job.cpu_start = 0
        job.cpu_budget = 4.0
        job.deadline = 5.0
        for cpu, accepted in (
            (-math.ulp(0.0), False),
            (0.0, True),
            (math.nextafter(4.0, 0.0), True),
            (4.0, False),
            (math.inf, False),
            (math.nan, False),
        ):
            platform.owner_cpu = lambda: cpu
            if accepted:
                job._time_budget()
            else:
                with self.assertRaises(ValueError):
                    job._time_budget()
        platform.owner_cpu = lambda: 0
        for wall, accepted in ((math.nextafter(5.0, 0.0), True), (5.0, False)):
            platform.clock = lambda: wall
            if accepted:
                job._time_budget()
            else:
                with self.assertRaisesRegex(ValueError, "grant exhausted"):
                    job._time_budget()


if __name__ == "__main__":
    unittest.main()
