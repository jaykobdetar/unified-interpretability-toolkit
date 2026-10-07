#!/usr/bin/env python3
"""Compatibility import and script for the ordinary atlas_host worker module."""

import sys
from atlas_host import inference_worker as _implementation
from atlas_host.inference_worker import (
    emit as emit,
    main as main,
    load_engine as load_engine,
    _contracts as _contracts,
    generate as generate,
    paired_step as paired_step,
    compare as compare,
    configure_worker_limits as configure_worker_limits,
)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never echo prompt text or model paths in the HTTP-visible error.
        emit(
            {
                "type": "error",
                "error": (
                    str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                ),
            }
        )
        raise SystemExit(1)
else:
    sys.modules[__name__] = _implementation
