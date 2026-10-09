"""Shared asset facts, generated declarations and actual coordinator file replies."""

from email.message import Message
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from atlas_host import host_assets as assets
import generate_page_assets as generator
import live_inference as live
from page_startup_fixture import native_viewer


class PageAssetCatalogContracts(unittest.TestCase):
    def test_native_viewer_catalogue(self) -> None:
        self.assertEqual(
            assets.VIEWER_ASSETS,
            {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/inference.js": ("inference.js", "text/javascript"),
                "/inference-import.js": ("inference-import.js", "text/javascript"),
                "/atlas-tools.js": ("atlas-tools.js", "text/javascript"),
                "/workspace-tools.js": ("workspace-tools.js", "text/javascript"),
                "/app.js": ("app.js", "text/javascript"),
                **(
                    {"/viewer-context.js": ("viewer-context.js", "text/javascript")}
                    if native_viewer(live.ROOT)
                    else {}
                ),
                "/style.css": ("style.css", "text/css"),
                "/vendor/openseadragon.min.js": (
                    "vendor/openseadragon.min.js",
                    "text/javascript",
                ),
                "/vendor/OpenSeadragon-LICENSE.txt": (
                    "vendor/OpenSeadragon-LICENSE.txt",
                    "text/plain",
                ),
            },
        )

    def test_comparison_catalogue_keeps_its_own_scope(self) -> None:
        self.assertEqual(
            assets.COMPARISON_ASSETS,
            {
                "/": ("comparison.html", "text/html; charset=utf-8"),
                "/comparison.html": ("comparison.html", "text/html; charset=utf-8"),
                "/comparison.js": ("comparison.js", "text/javascript"),
                "/comparison.css": ("comparison.css", "text/css"),
                "/vendor/openseadragon.min.js": (
                    "vendor/openseadragon.min.js",
                    "text/javascript",
                ),
                "/vendor/OpenSeadragon-LICENSE.txt": (
                    "vendor/OpenSeadragon-LICENSE.txt",
                    "text/plain",
                ),
            },
        )

    def test_viewer_startup_order_is_distinct_from_host(self) -> None:
        self.assertEqual(
            assets.VIEWER_BUNDLE,
            (
                "vendor/openseadragon.min.js",
                "atlas-tools.js",
                "app.js",
                "workspace-tools.js",
                "inference.js",
            ),
        )
        self.assertIs(type(assets.BUNDLE), list)
        self.assertEqual(assets.BUNDLE[2], "host-client.js")
        self.assertEqual(
            assets.BUNDLE[:2] + assets.BUNDLE[3:], list(assets.VIEWER_BUNDLE)
        )

    def test_native_declarations_are_current_without_tool_installation(self) -> None:
        self.assertEqual(generator.OUTPUT.read_text(), generator.render())

    def test_coordinator_replies_with_exact_existing_bytes_and_mime(self) -> None:
        expected = {
            "/analytics-panel.js": "text/javascript",
            "/analytics-mount.js": "text/javascript",
            "/analytics-panel.css": "text/css",
            "/app.js": "text/javascript",
        }
        self.assertEqual(
            assets.COORDINATOR_ASSETS,
            {route: (route[1:], mime) for route, mime in expected.items()},
        )
        for route, mime in expected.items():
            with self.subTest(route=route):
                handler = object.__new__(live.Handler)
                handler.command = "GET"
                handler.server = SimpleNamespace(session=object())
                handler.headers = Message()
                send = Mock()
                with (
                    patch.object(handler, "validated", return_value=(route, 0)),
                    patch.object(handler, "send", send),
                ):
                    handler.handle_action()
                send.assert_called_once_with(
                    200, (live.ROOT / "web" / route[1:]).read_bytes(), mime
                )

    def test_catalogue_files_are_local_existing_page_files(self) -> None:
        root = Path(__file__).resolve().parents[1] / "web"
        for file in assets.PAGE_TYPES:
            self.assertTrue((root / file).is_file(), file)
            self.assertEqual(Path(file).as_posix(), file)


if __name__ == "__main__":
    unittest.main()
