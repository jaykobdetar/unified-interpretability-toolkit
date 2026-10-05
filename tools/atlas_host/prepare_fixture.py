"""Explicit owner-only fixture calibration; not callable through HTTP routes."""

from pathlib import Path
import subprocess

from .common import require
from .registry import Registry
from .runtime_adapter import check_fixture


def prepare(config, identifier, *, run=subprocess.run):
    registry = Registry(config["paths"]["registry"])
    entry = registry.owner_receipt(identifier)
    check_fixture(entry, hash_bytes=True)
    cache = Path(config["paths"]["cache"]) / identifier
    require(
        not cache.resolve().is_relative_to(Path(entry["root"])),
        "Cache must be outside source",
    )
    binary = Path(__file__).resolve().parents[2] / "target/release/weight-atlas-rust"
    # Rust retains its normal CPU/AS/memory/disk guards and cache-lock refusal.
    result = run(
        [
            str(binary),
            "calibrate",
            "--model",
            entry["root"],
            "--cache",
            str(cache),
            "--name",
            entry["name"],
            "--revision",
            entry["manifest"]["revision"],
        ],
        timeout=5,
        check=True,
    )
    return {
        "model_id": identifier,
        "fixture_calibrated": result.returncode == 0,
        "inference_ready": False,
        "download_enabled": False,
    }
