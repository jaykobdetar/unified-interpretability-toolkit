"""Page file facts and exact serving scopes; no host/runtime imports.

The native lookup table is generated from this catalogue, not loaded at runtime.
ASSETS and BUNDLE retain the existing host's mutable ordered container contract.
"""

from collections.abc import Mapping
import json
from pathlib import Path

PAGE_TYPES: dict[str, str] = {
    "index.html": "text/html; charset=utf-8",
    "comparison.html": "text/html; charset=utf-8",
    "style.css": "text/css",
    "comparison.css": "text/css",
    "analytics-panel.css": "text/css",
    "atlas-tools.js": "text/javascript",
    "workspace-tools.js": "text/javascript",
    "inference.js": "text/javascript",
    "inference-import.js": "text/javascript",
    "app.js": "text/javascript",
    "viewer-context.js": "text/javascript",
    "comparison.js": "text/javascript",
    "host-client.js": "text/javascript",
    "profile-client.js": "text/javascript",
    "analytics-panel.js": "text/javascript",
    "analytics-mount.js": "text/javascript",
    "vendor/openseadragon.min.js": "text/javascript",
    "vendor/OpenSeadragon-LICENSE.txt": "text/plain",
}


def asset_routes(
    files: tuple[str, ...], *, root: str | None = None
) -> dict[str, tuple[str, str]]:
    """Construct only the declared routes, preserving each owner's ordering."""
    routes = {} if root is None else {"/": (root, PAGE_TYPES[root])}
    routes.update(("/" + name, (name, PAGE_TYPES[name])) for name in files)
    return routes


VIEWER_ASSETS: dict[str, tuple[str, str]] = asset_routes(
    (
        "index.html",
        "inference.js",
        "inference-import.js",
        "atlas-tools.js",
        "workspace-tools.js",
        "app.js",
        "viewer-context.js",
        "style.css",
        "vendor/openseadragon.min.js",
        "vendor/OpenSeadragon-LICENSE.txt",
    ),
    root="index.html",
)
COMPARISON_ASSETS: dict[str, tuple[str, str]] = asset_routes(
    (
        "comparison.html",
        "comparison.js",
        "comparison.css",
        "vendor/openseadragon.min.js",
        "vendor/OpenSeadragon-LICENSE.txt",
    ),
    root="comparison.html",
)
ASSETS: dict[str, tuple[str, str]] = asset_routes(
    (
        "style.css",
        "atlas-tools.js",
        "workspace-tools.js",
        "inference.js",
        "app.js",
        "host-client.js",
        "profile-client.js",
        "viewer-context.js",
    )
)
COORDINATOR_ASSETS: dict[str, tuple[str, str]] = asset_routes(
    ("analytics-panel.js", "analytics-mount.js", "analytics-panel.css", "app.js")
)
VIEWER_BUNDLE: tuple[str, ...] = (
    "vendor/openseadragon.min.js",
    "atlas-tools.js",
    "app.js",
    "workspace-tools.js",
    "inference.js",
)
BUNDLE: list[str] = [
    "vendor/openseadragon.min.js",
    "atlas-tools.js",
    "host-client.js",
    "app.js",
    "workspace-tools.js",
    "inference.js",
]


def add_profile_client(bundle: list[str]) -> None:
    """Keep the existing opt-in insertion before the host client."""
    bundle.insert(bundle.index("host-client.js"), "profile-client.js")


def module_startup(bundle: tuple[str, ...] | list[str]) -> bytes:
    """Fetch modules sequentially, then mount in the original synchronous order."""
    lines = [
        "(async () => {",
        'const shared = (await import("./viewer-context.js")).default;',
    ]
    for name in bundle:
        if name == "vendor/openseadragon.min.js":
            continue
        imported = f"(await import({json.dumps('./' + name)}))"
        if name == "atlas-tools.js":
            lines.append(f"shared.AtlasTools = {imported}.default;")
        elif name == "profile-client.js":
            lines.append(f"shared.AtlasProfiles = {imported}.default;")
        elif name == "host-client.js":
            lines.append(
                f"shared.AtlasHost = {imported}.default.create({{"
                "fetchImpl: globalThis.fetch.bind(globalThis),"
                "document: globalThis.document,"
                "schedule: globalThis.setTimeout.bind(globalThis),"
                "cancel: globalThis.clearTimeout.bind(globalThis),"
                "window: globalThis});"
            )
        elif name == "app.js":
            lines.append(f"const app = {imported};")
        elif name == "workspace-tools.js":
            lines.append(f"const workspace = {imported};")
        elif name == "inference.js":
            lines.append(f"const inference = {imported};")
    lines.extend(
        ["app.initialize();", "workspace.mount(app);", "inference.mount();", "})();"]
    )
    return ("\n".join(lines) + "\n").encode()


def bundle_bytes(root: Path, bundle: list[str]) -> bytes:
    """Retain local file admission and the vendored classic script boundary."""
    parts = [(root / "web" / name).read_bytes() for name in bundle]
    return b"\n;\n".join([parts[0], module_startup(bundle)])


def rust_string(value: str) -> str:
    """Catalogue identifiers are ASCII; JSON quoting is also valid Rust here."""
    if not value.isascii() or "\\" in value or any(ord(c) < 32 for c in value):
        raise ValueError(
            "Asset declarations require printable ASCII without backslashes"
        )
    return json.dumps(value)


def lookup(name: str, routes: Mapping[str, tuple[str, str]]) -> str:
    groups: dict[tuple[str, str], list[str]] = {}
    for route, asset in routes.items():
        groups.setdefault(asset, []).append(route)
    lines = [
        f"pub(crate) fn {name}(path: &str) -> Option<(&'static str, &'static [u8])> {{",
        "    match path {",
    ]
    for (file, mime), paths in groups.items():
        pattern = " | ".join(rust_string(path) for path in paths)
        content = f'include_bytes!({rust_string("../web/" + file)})'
        inline = f"        {pattern} => Some(({rust_string(mime)}, {content})),"
        if len(inline) <= 100:
            lines.append(inline)
        else:
            lines.extend(
                [
                    f"        {pattern} => Some((",
                    f"            {rust_string(mime)},",
                    f"            {content},",
                    "        )),",
                ]
            )
    lines.extend(["        _ => None,", "    }", "}", ""])
    return "\n".join(lines)


def render() -> str:
    lines = [
        "// Generated by tools/generate_page_assets.py from atlas_host.host_assets.",
        "// Edit the catalogue, regenerate, and run the existing checks.",
        "",
        "pub(crate) const VIEWER_SCRIPTS: [&[u8]; 3] = [",
        '    include_bytes!("../web/vendor/openseadragon.min.js"),',
        '    b"\\n;\\n",',
    ]
    startup = module_startup(VIEWER_BUNDLE).decode()
    # JSON's ASCII escaping also preserves this trusted generated Rust byte literal.
    lines.append(f"    b{json.dumps(startup)},")
    lines.extend(["];", ""])
    return (
        "\n".join(lines)
        + "\n"
        + lookup("viewer", VIEWER_ASSETS)
        + "\n"
        + lookup("comparison", COMPARISON_ASSETS)
        + '\n#[cfg(test)]\n#[path = "page_asset_tests.rs"]\nmod tests;\n'
    )
