#!/usr/bin/env python3
"""Owner-only local CLI entry point; explicit pinned acquisition, never HTTP."""

from atlas_host.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
