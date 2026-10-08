"""Local configuration, cache and progress contracts without starting services."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

# The local-v1 policy is a saved interface, rather than a tunable test budget.
LIMITS = {
    "cpu_count": 1,
    "numeric_workers": 1,
    "heavy_jobs": 1,
    "rust_address_space_bytes": 805306368,
    "rust_available_floor_bytes": 3221225472,
    "build_address_space_bytes": 2147483648,
    "build_jobs": 1,
    "build_start_available_bytes": 5368709120,
    "browser_tree_rss_bytes": 805306368,
    "browser_start_available_bytes": 5368709120,
    "stop_available_bytes": 3489660928,
    "inference_rss_bytes": 1610612736,
    "inference_address_space_bytes": 3221225472,
    "inference_start_available_bytes": 5100273664,
    "inference_wall_ms": 120000,
    "inference_cpu_ms": 90000,
    "prompt_tokens": 128,
    "new_tokens": 32,
    "client_lease_ms": 15000,
    "inference_queue": 0,
    "analytics_address_space_bytes": 805306368,
    "analytics_rss_bytes": 805306368,
    "analytics_start_available_bytes": 4026531840,
    "analytics_wall_ms": 5000,
    "analytics_cpu_ms": 4000,
    "disk_reserve_bytes": 26843545600,
    "tile_disk_bytes": 2147483648,
    "tile_files": 1000,
    "pending_headers": 4,
    "header_bytes": 8192,
    "header_deadline_ms": 500,
    "dispatch_queue": 4,
    "numeric_queue": 8,
    "write_deadline_ms": 3000,
    "coordinator_body_bytes": 8192,
    "upstream_response_bytes": 2097152,
}


def configuration():
    return {
        "version": 1,
        "profile": "local-v1",
        "bind": "127.0.0.1",
        "ports": {"coordinator": 8796, "renderer": 8797},
        "paths": {"registry": "state/registry.json", "cache": "cache"},
    }


def source_binding():
    return json.loads((ROOT / "tests/fixtures/host-source-binding-v2.json").read_text())


class PureTransitiveTests(unittest.TestCase):
    def setUp(self):
        try:
            from atlas_host import cache, config, progress
        except Exception as exc:
            self.fail("local record helpers must initialize: " + repr(exc))
        self.cache = cache
        self.config = config
        self.progress = progress

    def valid(self, function, *args):
        try:
            return function(*args)
        except Exception as exc:
            self.fail("valid local record must complete: " + repr(exc))

    def expected_config(self, value, base):
        expected = deepcopy(value)
        expected["paths"] = {
            "registry": str(base / "state/registry.json"),
            "cache": str(base / "cache"),
        }
        expected["limits"] = deepcopy(LIMITS)
        return expected

    def test_validate_config(self):
        value = configuration()
        before = deepcopy(value)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            actual = self.valid(self.config.validate_config, value, base)
            self.assertEqual(actual, self.expected_config(value, base))
        self.assertEqual(value, before)
        self.assertIsNot(actual, value)
        self.assertIsNot(actual["ports"], value["ports"])
        self.assertIsNot(actual["limits"], self.config.LOCAL_LIMITS)
        actual["ports"]["renderer"] = 8801
        self.assertEqual(value, before)

    def test_load_config(self):
        value = configuration()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve() / "configuration"
            base.mkdir()
            path = base / "local.json"
            raw = json.dumps(value).encode("utf-8")
            raw += b" " * (16384 - len(raw))
            path.write_bytes(raw)
            self.assertEqual(len(raw), 16384)
            actual = self.valid(self.config.load_config, path)
            self.assertEqual(actual, self.expected_config(value, base))
            self.assertEqual(path.read_bytes(), raw)

    def test_capabilities(self):
        value = {**configuration(), "limits": deepcopy(LIMITS)}
        before = deepcopy(value)
        actual = self.valid(self.config.capabilities, value)
        self.assertEqual(
            actual,
            {
                "version": 1,
                "profile": "local-v1",
                "limits": LIMITS,
                "downloads": False,
                "gpu": False,
                "resident_activation": False,
                "http_routes_active": False,
            },
        )
        self.assertIsNot(actual["limits"], value["limits"])
        actual["limits"]["prompt_tokens"] = 1
        self.assertEqual(value, before)

    def test_binding(self):
        value = source_binding()
        before = deepcopy(value)
        actual = self.valid(self.cache.binding, value)
        self.assertEqual(actual, before)
        self.assertEqual(value, before)
        self.assertIsNot(actual, value)
        self.assertIsNot(actual["shape"], value["shape"])
        self.assertIsNot(actual["slice"], value["slice"])
        self.assertIsNot(
            actual["slice"]["leading_indices"], value["slice"]["leading_indices"]
        )
        actual["shape"][0] = 9
        actual["slice"]["leading_indices"][0] = 0
        self.assertEqual(value, before)

    def test_derivation_identity(self):
        value = {
            "content_digest": "a" * 64,
            "binding": source_binding(),
            "algorithm": "renderer-v1",
            "calibration_digest": "c" * 64,
            "rule": "tensor_magnitude_asinh",
            "parameters": {"s": 0.25, "D": 1.0},
            "level": 1,
            "x": 0,
            "y": 0,
            "control": None,
            "encoding": "png-v1",
        }
        before = deepcopy(value)
        portable = deepcopy(value)
        portable["binding"] = {
            key: item
            for key, item in portable["binding"].items()
            if key not in ("model_identity", "source_identity")
        }
        raw = json.dumps(
            ["weight-atlas-derived-v1", portable],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        expected = hashlib.sha256(raw).hexdigest()
        self.assertEqual(self.valid(self.cache.derivation_identity, value), expected)
        self.assertEqual(value, before)
        relocated = deepcopy(value)
        relocated["binding"]["model_identity"] = "d" * 64
        relocated["binding"]["source_identity"] = "e" * 64
        self.assertEqual(
            self.valid(self.cache.derivation_identity, relocated), expected
        )

    def test_no_store_headers(self):
        self.assertEqual(
            self.valid(self.cache.no_store_headers), {"Cache-Control": "no-store"}
        )

    def test_immutable_headers(self):
        payload = b"small completed artifact fixture"
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(
            self.valid(
                self.cache.immutable_headers, digest, payload, "image/png", "tile"
            ),
            {
                "Cache-Control": "private, max-age=31536000, immutable",
                "ETag": '"' + digest + '"',
                "Content-Type": "image/png",
                "Content-Length": "32",
                "X-Content-Type-Options": "nosniff",
            },
        )

    def test_model_id(self):
        value = "m_" + "a" * 64
        self.assertEqual(self.valid(self.progress.model_id, value), value)
        try:
            self.progress.model_id("fixture")
        except ValueError as exc:
            self.assertEqual(str(exc), "Invalid model ID")
        except Exception as exc:
            self.fail("expected local ValueError: " + repr(exc))
        else:
            self.fail("invalid local model ID must be refused")

    def test_validate_progress(self):
        for kind, total in (("calibration", 24), ("profile", 12), ("overview", 12)):
            with self.subTest(kind=kind):
                value = {
                    "model_id": "m_" + "a" * 64,
                    "kind": kind,
                    "binding": source_binding(),
                    "state": "running",
                    "visited_values": 2,
                    "total_values": total,
                    "elapsed_active_ms": 7,
                    "remaining_authorized_work": {
                        "values": total - 2,
                        "wall_ms": 100,
                        "cpu_ms": 90,
                    },
                    "complete": False,
                    "error": None,
                }
                before = deepcopy(value)
                actual = self.valid(self.progress.validate_progress, value)
                self.assertEqual(actual, before)
                self.assertEqual(value, before)
                self.assertIsNot(actual, value)
                self.assertIsNot(actual["binding"], value["binding"])
                self.assertIsNot(
                    actual["remaining_authorized_work"],
                    value["remaining_authorized_work"],
                )
                actual["remaining_authorized_work"]["values"] = 0
                actual["binding"]["shape"][0] = 9
                self.assertEqual(value, before)


if __name__ == "__main__":
    unittest.main()
