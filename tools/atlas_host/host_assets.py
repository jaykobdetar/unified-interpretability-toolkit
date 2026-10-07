"""Existing Python host asset catalogue; no host/runtime imports."""

ASSETS: dict[str, tuple[str, str]] = {
    "/style.css": ("style.css", "text/css"),
    "/atlas-tools.js": ("atlas-tools.js", "text/javascript"),
    "/workspace-tools.js": ("workspace-tools.js", "text/javascript"),
    "/inference.js": ("inference.js", "text/javascript"),
    "/app.js": ("app.js", "text/javascript"),
    "/host-client.js": ("host-client.js", "text/javascript"),
}
BUNDLE: list[str] = [
    "vendor/openseadragon.min.js",
    "atlas-tools.js",
    "host-client.js",
    "app.js",
    "workspace-tools.js",
    "inference.js",
]
