"""Actual owned analysis exclusion/cancel and kernel wall timer; no inference."""

import os

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import json, signal, subprocess, sys, time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from analytics.service import AnalyticsJobs
from live_inference import Session, available, signal_and_reap

PYTHON = os.environ.get("ATLAS_CPU_PYTHON", sys.executable)
if not os.environ.get("ATLAS_SMOL_MODEL"):
    raise SystemExit(
        "Set ATLAS_SMOL_MODEL to the complete local pinned SmolLM2 directory"
    )
MODEL = Path(os.environ["ATLAS_SMOL_MODEL"])


def metadata():
    return json.loads(
        subprocess.check_output(
            [
                str(ROOT / "target/release/weight-atlas-rust"),
                "metadata",
                "--model",
                str(MODEL),
            ],
            timeout=10,
        )
    )


meta = metadata()
t = next(
    t for t in meta["catalog"] if t["name"] == "model.layers.0.self_attn.q_proj.weight"
)
session = Session(PYTHON, MODEL)
jobs = AnalyticsJobs(
    PYTHON,
    MODEL,
    inference_busy=lambda: session.process is not None
    or session.status in ("loading", "running", "stopping"),
    fetch_model=metadata,
    available=available,
    reap=signal_and_reap,
)
session.analytics = jobs
request = {
    "tensor": t["id"],
    "region": {"row": 0, "col": 0, "rows": 128, "cols": 512},
    "seed": 77,
    "svd": False,
}
try:
    started = jobs.start(request)
    assert started["worker_alive"]
    # Invalid prompt is an additional fail-safe: even a broken mutual-exclusion gate
    # cannot cause model inference. The assertion specifically requires the gate.
    with patch(
        "live_inference.verify_model",
        side_effect=AssertionError("No inference source hash/load permitted"),
    ):
        try:
            session.start({})
        except ValueError as exc:
            assert "Analysis owns" in str(exc), str(exc)
        else:
            raise AssertionError("Inference admission was not rejected")
    assert jobs.stop() and jobs.process is None
    session.status = "loading"
    try:
        jobs.start(request)
    except ValueError as exc:
        assert "compute busy" in str(exc)
    else:
        raise AssertionError("Analysis accepted simulated inference ownership")
    session.status = "idle"
finally:
    assert jobs.stop()
start = time.monotonic()
p = subprocess.run(
    [
        PYTHON,
        "-B",
        "-c",
        "from analytics.worker import configure; import time; configure(); time.sleep(7)",
    ],
    cwd=ROOT / "tools",
    timeout=6,
    capture_output=True,
)
elapsed = time.monotonic() - start
assert p.returncode == -signal.SIGALRM, (p.returncode, p.stderr)
assert 5 <= elapsed < 6, elapsed
result = {
    "status": "PASS",
    "actual_owned_analysis_blocks_inference_before_verification": True,
    "inference_ownership_state_blocks_analysis": True,
    "inference_executed": False,
    "actual_owned_child_cancel_reaped": True,
    "kernel_wall_timer_returncode": p.returncode,
    "kernel_wall_timer_seconds": round(elapsed, 4),
    "cap_seconds": 5,
    "cpu_count": len(os.sched_getaffinity(0)),
}
(ROOT / "results/analytics-qualification/ownership-deadline.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
