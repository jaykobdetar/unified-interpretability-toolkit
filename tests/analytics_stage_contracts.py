"""Inert analytics stage boundaries; no NumPy, files, workers or listeners."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from analytics import worker


class AnalyticsStageContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.region = {"row": 0, "col": 0, "rows": 2, "cols": 3}
        self.tensor = {"id": 0, "name": "fixture", "shape": [2, 3]}
        self.values = [1.0, -2.0, 0.0, 4.0, 5.0, -6.0]
        self.source_tensor = SimpleNamespace(name="fixture", shape=(2, 3))
        self.catalog = SimpleNamespace(
            root=Path("inert-fixture"),
            identity="catalog-identity",
            tensors=[self.source_tensor],
            read=Mock(return_value=self.values),
            verify=Mock(),
        )
        self.payload = {
            "model": {"catalog": [self.tensor], "source_identity": "host-identity"},
            "request": {"tensor": 0, "region": self.region, "seed": 7},
        }
        self.backend = object()
        self.summary = {"held": "summary"}
        self.fits = {"held": "fits"}

    @contextmanager
    def adapters(self) -> Iterator[SimpleNamespace]:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(worker, "catalog_from_payload", return_value=self.catalog)
            )
            stack.enter_context(patch.dict(sys.modules, {"numpy": self.backend}))
            binding = stack.enter_context(
                patch("analytics.svd_summary.binding", return_value={})
            )
            summary = stack.enter_context(
                patch(
                    "analytics.svd_summary.compute_with_numpy",
                    return_value=self.summary,
                )
            )
            stack.enter_context(
                patch("analytics.profiles.resolve_local", return_value=({}, None))
            )
            analysis = stack.enter_context(
                patch(
                    "analytics.core.analyze",
                    return_value={"heads": {"available": False}},
                )
            )
            fits = stack.enter_context(
                patch("analytics.svd.compute_with_numpy", return_value=self.fits)
            )
            yield SimpleNamespace(
                binding=binding, summary=summary, analysis=analysis, fits=fits
            )

    def test_summary_binding_refusal_precedes_read_and_compute(self) -> None:
        self.payload["request"]["scope"] = "svd_summary"
        with self.adapters() as calls:
            calls.binding.side_effect = ValueError("held binding refusal")
            with self.assertRaisesRegex(ValueError, "^held binding refusal$"):
                worker.analyze_request(self.payload)
            calls.binding.assert_called_once_with(
                self.payload["model"], self.tensor, self.region, 7
            )
            self.catalog.read.assert_not_called()
            calls.summary.assert_not_called()
            self.catalog.verify.assert_not_called()

    def test_summary_final_source_refusal_follows_compute(self) -> None:
        self.payload["request"]["scope"] = "svd_summary"
        with self.adapters() as calls:
            self.catalog.verify.side_effect = ValueError("held final source refusal")
            with self.assertRaisesRegex(ValueError, "^held final source refusal$"):
                worker.analyze_request(self.payload)
            calls.summary.assert_called_once_with(
                self.backend,
                self.values,
                model=self.payload["model"],
                tensor=self.tensor,
                region=self.region,
                seed=7,
            )
            self.catalog.verify.assert_called_once_with()

    def test_region_final_source_refusal_follows_analysis(self) -> None:
        with self.adapters() as calls:
            self.catalog.verify.side_effect = ValueError("held final source refusal")
            with self.assertRaisesRegex(ValueError, "^held final source refusal$"):
                worker.analyze_request(self.payload)
            calls.analysis.assert_called_once_with(
                self.values,
                source_identity="catalog-identity",
                tensor="fixture",
                shape=[2, 3],
                region=self.region,
                seed=7,
            )
            self.catalog.verify.assert_called_once_with()

    def test_optional_svd_keeps_exact_derived_envelope(self) -> None:
        self.payload["request"]["svd"] = True
        with self.adapters() as calls:
            result = worker.analyze_request(self.payload)
            self.assertEqual(
                result["svd"],
                {
                    "available": True,
                    "scope": "full tensor",
                    "region": self.region,
                    "seed": 7,
                    "centered": False,
                    "control": "same-region exact multiset; each SVD fitted separately",
                    "results": self.fits,
                },
            )
            self.assertIs(result["svd"]["results"], self.fits)
            calls.fits.assert_called_once_with(self.backend, self.values, 2, 3, 7)
            self.catalog.verify.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
