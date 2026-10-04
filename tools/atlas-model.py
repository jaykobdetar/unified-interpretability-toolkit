#!/usr/bin/env python3
"""Owner-only local CLI entry point; all acquisition is disabled."""
from atlas_host.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
