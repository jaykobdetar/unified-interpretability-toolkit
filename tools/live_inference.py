#!/usr/bin/env python3
"""Compatibility import and script for the ordinary coordinator package module."""

import sys
from atlas_host import live_inference as _implementation
from atlas_host.memory import available_bytes as _available
from atlas_host.live_inference import (
    ROOT as ROOT,
    MANIFEST as MANIFEST,
    GIB as GIB,
    MAX_BODY as MAX_BODY,
    MAX_TRACE as MAX_TRACE,
    REQUEST_DEADLINE as REQUEST_DEADLINE,
    UPSTREAM_DEADLINE as UPSTREAM_DEADLINE,
    IO_TICK as IO_TICK,
    TickOwner as TickOwner,
    OwnedProcess as OwnedProcess,
    ClosableServer as ClosableServer,
    HashState as HashState,
    Fingerprint as Fingerprint,
    LayoutReceipt as LayoutReceipt,
    Snapshot as Snapshot,
    BackendError as BackendError,
    OperationDeadline as OperationDeadline,
    DeadlineReader as DeadlineReader,
    DeadlineSocket as DeadlineSocket,
    ProxyConnection as ProxyConnection,
    proxy_request as proxy_request,
    process_alive as process_alive,
    signal_and_reap as signal_and_reap,
    SweepAdmissionBudget as SweepAdmissionBudget,
    verify_model as verify_model,
    layout_fingerprints as layout_fingerprints,
    verified_layout_receipt as verified_layout_receipt,
    current_layout_binding as current_layout_binding,
    _contracts as _contracts,
    Session as Session,
    CoordinatorServer as CoordinatorServer,
    Handler as Handler,
    bind_inference_source as bind_inference_source,
    cleanup_owned as cleanup_owned,
    main as main,
)

available = _available

if __name__ == "__main__":
    main()
else:
    sys.modules[__name__] = _implementation
