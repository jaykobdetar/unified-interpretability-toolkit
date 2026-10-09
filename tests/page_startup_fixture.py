"""Exact startup bytes for the classic and native page formats under test."""

import json
from pathlib import Path
import re


def native_viewer(root: Path) -> bool:
    return (
        re.search(
            r"^\s*(?:import|export)\s", (root / "web/app.js").read_text(), re.MULTILINE
        )
        is not None
    )


def expected_bundle(root: Path, names: list[str]) -> bytes:
    if not native_viewer(root):
        return b"\n;\n".join((root / "web" / name).read_bytes() for name in names)
    fixtures = json.loads(
        (Path(__file__).parent / "fixtures/viewer-startup.json").read_text()
    )
    kind = (
        "profiles"
        if "profile-client.js" in names
        else "host" if "host-client.js" in names else "viewer"
    )
    return b"\n;\n".join(
        [
            (root / "web/vendor/openseadragon.min.js").read_bytes(),
            fixtures[kind].encode(),
        ]
    )
