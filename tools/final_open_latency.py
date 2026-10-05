#!/usr/bin/env python3
"""Hash-bound same-workload loopback trials; terminates only owned child servers."""

import argparse, hashlib, json, os, platform, resource, shutil, socket, statistics
import subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, required=True)
p.add_argument("--calibration", type=Path, required=True)
p.add_argument("--python-reference", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--binary", type=Path, default=ROOT / "bin/weight-atlas-rust")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
os.nice(10)
resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(paths, root):
    entries = {str(x.relative_to(root)): digest(x) for x in sorted(paths)}
    aggregate = hashlib.sha256(
        "".join(f"{h}  {name}\n" for name, h in sorted(entries.items())).encode()
    ).hexdigest()
    return {"sha256": aggregate, "files": entries}


core = [x for name in ["src", "web"] for x in (ROOT / name).rglob("*") if x.is_file()]
core += [ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / ".cargo/config.toml"]
source_pin = pin(core, ROOT)
binary_pin = digest(a.binary)
reference_files = list((a.python_reference / "atlas8").rglob("*.py"))
reference_pin = pin(reference_files, a.python_reference)
minimum_ram = float("inf")


def guard():
    global minimum_ram
    ram = (
        int(
            next(
                x
                for x in Path("/proc/meminfo").read_text().splitlines()
                if x.startswith("MemAvailable:")
            ).split()[1]
        )
        * 1024
    )
    minimum_ram = min(minimum_ram, ram)
    assert ram >= 3 * 1024**3, "RAM reserve below 3 GiB"
    assert shutil.disk_usage(a.output).free >= 25 * 1024**3, "Disk reserve below 25 GiB"
    return ram


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read(), dict(r.headers)


records = []
result = {
    "status": "RUNNING",
    "executed_rust_binary": {
        "path": str(a.binary.resolve()),
        "sha256": binary_pin,
        "bytes": a.binary.stat().st_size,
    },
    "source_manifest": source_pin,
    "python_reference_manifest": reference_pin,
    "calibration_sha256": digest(a.calibration),
    "python_inventory_sha256": digest(a.python_reference / "results/inventory.json"),
    "benchmark_script_sha256": digest(Path(__file__)),
    "reference_server_script_sha256": digest(ROOT / "tools/reference_server.py"),
    "environment": {
        "platform": platform.platform(),
        "cpu": sorted(os.sched_getaffinity(0)),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "python": sys.version,
        "cpu_model": next(
            x.split(":", 1)[1].strip()
            for x in Path("/proc/cpuinfo").read_text().splitlines()
            if x.startswith("model name")
        ),
    },
    "scope": "Fresh processes, valid saved completed calibration, empty per-trial PNG caches. Same Q tensor id 11, global_linear and tensor_asinh, level 8/factor 16, x=y=0, two sequential overview PNGs. Three trials each in R/P/P/R/R/P order. OS/drive caches not cleared. HTTP payload readiness excludes browser JS, polling, layout and paint. Warm measurement is one repeated GET per rule per trial. Only owned child servers terminated. No full-source scan or model execution.",
    "records": records,
}


def save():
    result["minimum_sampled_available_gib"] = minimum_ram / 1024**3
    (a.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


try:
    guard()
    for backend in ["rust", "python", "python", "rust", "rust", "python"]:
        guard()
        idx = len(records)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cache = a.output / f"cache-{idx}-{backend}"
        cache.mkdir()
        if backend == "rust":
            shutil.copyfile(a.calibration, cache / "calibration.json")
            assert not (cache / "tiles").exists()
            assert digest(a.binary) == binary_pin
            cmd = [
                str(a.binary.resolve()),
                "serve",
                "--model",
                str(a.model.resolve()),
                "--cache",
                str(cache.resolve()),
                "--port",
                str(port),
            ]
        else:
            cmd = [
                "/usr/bin/python3",
                str(ROOT / "tools/reference_server.py"),
                str(port),
                str(cache.resolve()),
            ]
        logpath = a.output / f"{idx}-{backend}.log"
        with logpath.open("w") as log:
            start = time.perf_counter()
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=log,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "ATLAS_PYTHON_REFERENCE": str(a.python_reference.resolve()),
                },
            )
            base = f"http://127.0.0.1:{port}"
            try:
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"{backend} exited; inspect {logpath}")
                    try:
                        body, _ = get(base + "/api/model")
                        break
                    except urllib.error.URLError:
                        assert time.perf_counter() - start < 30, "Startup timeout"
                        time.sleep(0.003)
                metadata_time = time.perf_counter() - start
                model = json.loads(body)
                assert Path(model["source_directory"]).resolve() == a.model.resolve()
                assert (
                    model["calibration_complete"]
                    and model["parameter_count"] == 8190735360
                )
                assert model["global_max"] == 34
                assert (
                    model["catalog"][11]["name"]
                    == "model.layers.0.self_attn.q_proj.weight"
                )
                if backend == "rust":
                    assert model["coverage"]["sha_hashed_shards"] == 0
                    assert model["coverage"]["sha_verified_shards"] == 0
                get(base + "/api/view?tensor=11&left=global_linear&right=tensor_asinh")
                view_time = time.perf_counter() - start
                tiles = []
                for rule in ["global_linear", "tensor_asinh"]:
                    begin = time.perf_counter()
                    b, h = get(base + f"/tile?tensor=11&rule={rule}&level=8&x=0&y=0")
                    elapsed = time.perf_counter() - begin
                    assert h["X-Atlas-Cache"] == "miss"
                    tiles.append(
                        {
                            "rule": rule,
                            "seconds": elapsed,
                            "bytes": len(b),
                            "cache": h["X-Atlas-Cache"],
                            "png_sha256": hashlib.sha256(b).hexdigest(),
                        }
                    )
                first_pair = time.perf_counter() - start
                warm = []
                for rule in ["global_linear", "tensor_asinh"]:
                    begin = time.perf_counter()
                    b, h = get(base + f"/tile?tensor=11&rule={rule}&level=8&x=0&y=0")
                    warm.append(time.perf_counter() - begin)
                    assert h["X-Atlas-Cache"] == "hit"
                status = Path(f"/proc/{proc.pid}/status").read_text()
                memory = {
                    line.split(":")[0]: line.split(":")[1].strip()
                    for line in status.splitlines()
                    if line.startswith(
                        ("VmRSS:", "VmHWM:", "VmSize:", "Cpus_allowed_list:")
                    )
                }
                records.append(
                    {
                        "backend": backend,
                        "trial": idx,
                        "executed_binary_sha256": (
                            binary_pin if backend == "rust" else None
                        ),
                        "process_to_model_seconds": metadata_time,
                        "process_to_view_metadata_seconds": view_time,
                        "process_to_two_first_overview_pngs_seconds": first_pair,
                        "cold_generated_tiles": tiles,
                        "warm_tile_seconds": warm,
                        "process_memory": memory,
                        "source_identity": model.get("source_identity"),
                    }
                )
                guard()
                save()
            finally:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=10)
    assert digest(a.binary) == binary_pin and pin(core, ROOT) == source_pin
    summary = {}
    for backend in ["rust", "python"]:
        rows = [r for r in records if r["backend"] == backend]
        summary[backend] = {
            "metadata_median_ms": 1000
            * statistics.median(r["process_to_model_seconds"] for r in rows),
            "first_pair_median_ms": 1000
            * statistics.median(
                r["process_to_two_first_overview_pngs_seconds"] for r in rows
            ),
            "warm_tile_median_ms": 1000
            * statistics.median(t for r in rows for t in r["warm_tile_seconds"]),
            "peak_vm_hwm_mib": max(
                int(r["process_memory"]["VmHWM"].split()[0]) for r in rows
            )
            / 1024,
        }
    result["summary"] = summary
    result["status"] = "PASS"
    save()
    print(
        json.dumps(
            {
                "status": result["status"],
                "binary_sha256": binary_pin,
                "source_manifest_sha256": source_pin["sha256"],
                "summary": summary,
                "minimum_available_gib": result["minimum_sampled_available_gib"],
            },
            indent=2,
        )
    )
except BaseException as e:
    result["status"] = "FAILED"
    result["error"] = str(e)
    save()
    raise
