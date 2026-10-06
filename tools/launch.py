"""Compatibility alias for the ordinary atlas_host package implementation."""

import sys
from atlas_host import launch as _implementation
from atlas_host.launch import (
    argparse as argparse,
    json as json,
    os as os,
    Path as Path,
    shutil as shutil,
    subprocess as subprocess,
    encode_resources as encode_resources,
    ROOT as ROOT,
    BINARY as BINARY,
    FIELDS as FIELDS,
    unique_fields as unique_fields,
    main as main,
)

if __name__ == "__main__":
    raise SystemExit(main())
else:
    sys.modules[__name__] = _implementation
