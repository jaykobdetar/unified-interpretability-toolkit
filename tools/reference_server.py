#!/usr/bin/env python3
"""Reference server with all generated cache writes redirected to this new project."""

import os, sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
REF = Path(os.environ["ATLAS_PYTHON_REFERENCE"])
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REF))
from atlas8.limits import configure

configure()
from atlas8 import server
from atlas8.cache import TileCache
from unittest.mock import patch

cache = (
    Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "cache-python-http"
) / "tiles"
cache.mkdir(parents=True, exist_ok=True)
# Optional read-only overview archive copy is deliberately omitted for cold generated-tile comparison.
with patch.object(server, "TileCache", lambda: TileCache(root=cache)):
    server.serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8776)
