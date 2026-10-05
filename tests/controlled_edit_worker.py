#!/usr/bin/env python3
"""Test-only worker controls after in-memory edits, never exposed by the server."""

import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from inference_worker import compare, emit, load_engine
from live_inference import verify_model

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
request = json.loads(sys.stdin.buffer.readline(8193))
directory = Path(sys.argv[1])
verify_model(directory)
torch, tokenizer, model = load_engine(directory)


def controlled_record(event):
    emit(event)
    if event.get("comparison_phase") == "edited":
        if sys.argv[2] == "cancel":
            time.sleep(60)  # Owner test cancels and reaps this exact child here.
        elif sys.argv[2] == "failure":
            raise RuntimeError("ControlledAfterEditFailure")


try:
    compare(torch, tokenizer, model, request, controlled_record)
except RuntimeError:
    emit({"type": "error", "error": "ControlledAfterEditFailure"})
    raise SystemExit(1)
