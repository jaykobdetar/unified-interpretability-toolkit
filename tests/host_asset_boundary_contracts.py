"""Pin the existing Python asset catalogue and inert runtime inventory."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import host_atlas
from atlas_host import dense_static_admission as admission

ASSET_ITEMS = [
    ("/style.css", ("style.css", "text/css")),
    ("/atlas-tools.js", ("atlas-tools.js", "text/javascript")),
    ("/workspace-tools.js", ("workspace-tools.js", "text/javascript")),
    ("/inference.js", ("inference.js", "text/javascript")),
    ("/app.js", ("app.js", "text/javascript")),
    ("/host-client.js", ("host-client.js", "text/javascript")),
]
BUNDLE_ITEMS = [
    "vendor/openseadragon.min.js",
    "atlas-tools.js",
    "host-client.js",
    "app.js",
    "workspace-tools.js",
    "inference.js",
]
FIXED_NAMES = {
    "tools/host_atlas.py",
    "tools/live_inference.py",
    "tools/profile_atlas.py",
    "tools/static_atlas.py",
    "docs/models/smollm2-135m.json",
    "docs/models/smollm2-135m-config.json",
    "web/index.html",
    "web/profile-client.js",
}


class HostAssetBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "tools").mkdir()
        for name in ["inference_z.py", "inference_a.py", "ignored.py"]:
            (self.root / "tools" / name).touch()

    def names(self):
        try:
            return admission.runtime_names(self.root)
        except BaseException as error:
            self.fail(f"Valid inert inventory raised {type(error).__name__}: {error}")

    def test_exact_asset_values_order_and_container(self):
        self.assertIs(type(host_atlas.ASSETS), dict)
        self.assertEqual(list(host_atlas.ASSETS.items()), ASSET_ITEMS)
        self.assertTrue(
            all(type(value) is tuple for value in host_atlas.ASSETS.values())
        )

    def test_exact_bundle_order_and_container(self):
        self.assertIs(type(host_atlas.BUNDLE), list)
        self.assertEqual(host_atlas.BUNDLE, BUNDLE_ITEMS)

    def test_inventory_union_glob_prefix_and_order(self):
        with patch.object(
            admission, "source_names", return_value=["z-last", "a-first", "z-last"]
        ) as source:
            names = self.names()
        source.assert_called_once_with(self.root)
        expected = FIXED_NAMES | {
            "z-last",
            "a-first",
            "tools/inference_z.py",
            "tools/inference_a.py",
        }
        expected.update("web/" + value[0] for _, value in ASSET_ITEMS)
        expected.update("web/" + name for name in BUNDLE_ITEMS)
        self.assertIs(type(names), list)
        self.assertEqual(names, sorted(expected))

    def test_inventory_observes_existing_catalogue_objects(self):
        assets, bundle = host_atlas.ASSETS, host_atlas.BUNDLE
        saved_bundle = bundle[:]
        self.addCleanup(bundle.__setitem__, slice(None), saved_bundle)
        bundle[:] = ["custom-bundle.js", "custom-asset.js"]
        with patch.dict(
            assets, {"/custom": ("custom-asset.js", "text/plain")}, clear=True
        ):
            with patch.object(admission, "source_names", return_value=[]):
                names = self.names()
        self.assertIs(host_atlas.ASSETS, assets)
        self.assertIs(host_atlas.BUNDLE, bundle)
        self.assertEqual(
            names,
            sorted(
                FIXED_NAMES
                | {
                    "tools/inference_z.py",
                    "tools/inference_a.py",
                    "web/custom-asset.js",
                    "web/custom-bundle.js",
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
