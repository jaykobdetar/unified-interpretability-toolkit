#!/usr/bin/env python3
"""Granted-slot numerical qualification; bounded actual children, no inference."""

import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import resource

resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
import json, struct, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from analytics.service import AnalyticsJobs
from live_inference import available, signal_and_reap
import numpy as np

OUT = ROOT / "results/analytics-qualification"
OUT.mkdir(parents=True, exist_ok=True)
PYTHON = os.environ.get("ATLAS_CPU_PYTHON", sys.executable)
BIN = ROOT / "target/release/weight-atlas-rust"
MODELS = {"fixture": ROOT / "fixtures/tiny-bf16"}
for label, variable in [("smol", "ATLAS_SMOL_MODEL"), ("qwen", "ATLAS_QWEN_MODEL")]:
    if os.environ.get(variable):
        MODELS[label] = Path(os.environ[variable])
records = []
failures = []
minimum = available()


def metadata(root):
    p = subprocess.run(
        [str(BIN), "metadata", "--model", str(root)], capture_output=True, timeout=10
    )
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[:500])
    return json.loads(p.stdout)


def snapshot(root, meta):
    names = {t["shard"] for t in meta["catalog"]} | {
        "config.json",
        "model.safetensors.index.json",
    }
    return {
        name: [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
        for name in sorted(names)
        if (root / name).exists()
        for s in [(root / name).stat()]
    }


def raw_region(root, t, region):
    # Independent direct BF16 decode, no production source/analytics computation.
    rows = []
    with (root / t["shard"]).open("rb") as f:
        for i in range(region["row"], region["row"] + region["rows"]):
            f.seek(t["byte_offset"] + 2 * (i * t["shape"][-1] + region["col"]))
            bits = (
                np.frombuffer(f.read(2 * region["cols"]), dtype="<u2").astype(np.uint32)
                << 16
            )
            rows.append(bits.view(np.float32).astype(np.float64))
    return np.stack(rows)


def compare(root, t, result):
    a = raw_region(root, t, result["region"])
    flat = a.ravel()
    order = np.asarray(result["control"]["position_to_source"])
    assert sorted(order.tolist()) == list(range(len(flat)))
    b = flat[order].reshape(a.shape)
    assert sorted(flat.view(np.uint64).tolist()) == sorted(
        b.ravel().view(np.uint64).tolist()
    )
    for side, array in [("original", a), ("shuffled", b)]:
        stats = result[side]
        np.testing.assert_allclose(
            [x["mean_abs"] for x in stats["rows"]],
            np.mean(np.abs(array), axis=1),
            rtol=1e-14,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            [x["mean_abs"] for x in stats["columns"]],
            np.mean(np.abs(array), axis=0),
            rtol=1e-14,
            atol=1e-15,
        )
        assert [x["index"] for x in stats["top_values"]] == sorted(
            range(array.size), key=lambda i: (-abs(array.ravel()[i]), i)
        )[:16]
        if stats["folded"] and stats["folded"]["matrix"]["available"]:
            heads = result["heads"]
            dim = heads["head_dim"]
            f = stats["folded"]["matrix"]
            cells = [[] for _ in f["mean"]]
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    row = (
                        (result["region"]["row"] + i) % dim
                        if heads["axis"] == "row"
                        else i
                    )
                    col = (
                        (result["region"]["col"] + j) % dim
                        if heads["axis"] == "column"
                        else j
                    )
                    cells[row * f["cols"] + col].append(array[i, j])
            for got, cell in zip(f["mean"], cells):
                if cell:
                    np.testing.assert_allclose(
                        got, np.mean(cell), rtol=1e-13, atol=1e-15
                    )
                else:
                    assert got is None
        if result["svd"]["available"]:
            singular = np.linalg.svd(array, compute_uv=False)
            energy = singular**2
            total = np.sum(array * array)
            got = result["svd"]["results"][side]
            np.testing.assert_allclose(
                got["singular_values"], singular, rtol=1e-12, atol=1e-12
            )
            if total:
                np.testing.assert_allclose(
                    got["energy_fractions"], energy / total, rtol=1e-12, atol=1e-12
                )
                np.testing.assert_allclose(
                    got["rank_one_residual_energy_fraction"],
                    1 - energy[0] / total,
                    rtol=1e-12,
                    atol=1e-12,
                )
            else:
                assert got["rank_one_residual_energy_fraction"] is None
    return a.size


def run_case(label, root, meta, data):
    global minimum
    jobs = AnalyticsJobs(
        PYTHON,
        root,
        inference_busy=lambda: False,
        fetch_model=lambda: metadata(root),
        available=available,
        reap=signal_and_reap,
    )
    start = time.monotonic()
    before = snapshot(root, meta)
    observed_affinity = set()
    cpu_limits = set()
    as_limits = set()
    try:
        first = jobs.start(data)
        assert jobs.owns(first["job"])
        assert not jobs.owns("not-owner")
        # A second actual admission must not replace the owned live child.
        try:
            jobs.start(data)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate job admitted")
        while jobs.process is not None:
            minimum = min(minimum, available())
            assert minimum >= 3.25 * 1024**3
            try:
                status = Path(f"/proc/{jobs.process.pid}/status").read_text()
                observed_affinity.add(
                    next(
                        x.split(":", 1)[1].strip()
                        for x in status.splitlines()
                        if x.startswith("Cpus_allowed_list:")
                    )
                )
                limits = (
                    Path(f"/proc/{jobs.process.pid}/limits").read_text().splitlines()
                )
                cpu_limits.add(next(x for x in limits if x.startswith("Max cpu time")))
                as_limits.add(
                    next(x for x in limits if x.startswith("Max address space"))
                )
            except (OSError, StopIteration):
                pass
            jobs.tick()
            time.sleep(0.01)
            assert time.monotonic() - start < 8, "Harness ownership deadline"
        final = jobs.snapshot()
        assert final["status"] == "complete", final
        result = final["result"]
        if data.get("scope") == "model":
            coverage = result["coverage"]
            assert coverage["visited_values"] <= 65536
            assert not coverage["full_model"] or label.startswith("fixture")
            for item in result["rankings"]["original"]["values"]:
                t = next(t for t in meta["catalog"] if t["name"] == item["tensor"])
                coord = item["native_indices"]
                rr = {
                    "row": coord[0] if len(coord) == 2 else 0,
                    "col": coord[-1],
                    "rows": 1,
                    "cols": 1,
                }
                assert float(raw_region(root, t, rr)[0, 0]) == item["value"]
            effect = None
        else:
            t = next(t for t in meta["catalog"] if t["id"] == data["tensor"])
            compare(root, t, result)
            coverage = result["coverage"]
            effect = (
                {
                    side: result["svd"]["results"][side][
                        "rank_one_residual_energy_fraction"
                    ]
                    for side in ("original", "shuffled")
                }
                if result["svd"]["available"]
                else None
            )
            if label.endswith("/q-band"):
                assert (
                    result["heads"]["axis"] == "row"
                    and result["heads"]["label"] == "Q heads"
                )
            if label.endswith("/o-band"):
                assert result["heads"]["axis"] == "column"
            if label.endswith("/kv-band"):
                assert result["heads"]["label"] == "KV heads (K)"
        assert before == snapshot(root, meta), "Source metadata changed"
        expected_cpu = str(min(os.sched_getaffinity(0)))
        assert observed_affinity == {expected_cpu}, observed_affinity
        assert all("805306368" in x for x in as_limits), as_limits
        record = {
            "case": label,
            "status": "PASS",
            "seconds": round(time.monotonic() - start, 4),
            "peak_worker_rss_mib": round(final["peak_worker_rss_mib"], 3),
            "one_cpu": expected_cpu,
            "observed_cpu_limits": sorted(cpu_limits),
            "observed_address_space_limits": sorted(as_limits),
            "coverage": coverage,
            "rank_one_residual_energy": effect,
            "control_scope": "same selected window exact multiset; each statistic/SVD refitted independently; diagnostic only",
            "source_stat_identity_unchanged": True,
        }
        records.append(record)
        # Bound public evidence; full actual response stays in task-owned ignored output.
        (OUT / (label.replace("/", "-") + ".json")).write_text(
            json.dumps(result, allow_nan=False)
        )
    except Exception as exc:
        failures.append({"case": label, "status": "FAIL", "error": str(exc)[:1500]})
    finally:
        assert jobs.stop(), "Owned child cleanup remained pending"


for name, root in MODELS.items():
    meta = metadata(root)

    def tensor(suffix):
        return next(t for t in meta["catalog"] if t["name"].endswith(suffix))

    def region_case(label, t, h, w, svd=False):
        run_case(
            name + "/" + label,
            root,
            meta,
            {
                "tensor": t["id"],
                "region": {"row": 0, "col": 0, "rows": h, "cols": w},
                "seed": 42,
                "svd": svd,
            },
        )

    if name == "fixture":
        for t in meta["catalog"]:
            region_case(t["name"], t, t["rows"], t["cols"], True)
    else:
        q = tensor("layers.0.self_attn.q_proj.weight")
        k = tensor("layers.0.self_attn.k_proj.weight")
        o = tensor("layers.0.self_attn.o_proj.weight")
        region_case("q-band", q, q["rows"], 8)
        region_case("kv-band", k, k["rows"], 8)
        region_case("o-band", o, 8, o["cols"])
        v = next(t for t in meta["catalog"] if len(t["shape"]) == 1)
        region_case("vector", v, 1, min(64, v["cols"]))
        region_case("svd64", q, 64, 64, True)
        run_case(name + "/model-prefix", root, meta, {"scope": "model", "seed": 42})
report = {
    "status": "PASS" if not failures else "FAIL",
    "numpy": np.__version__,
    "selected_models": list(MODELS),
    "unselected_models": [name for name in ("smol", "qwen") if name not in MODELS],
    "cases": records,
    "failures": failures,
    "minimum_available_gib": round(minimum / 1024**3, 3),
    "no_inference": True,
    "source_scope": "header metadata plus bounded requested native windows; no full model allocation or full SHA scan",
}
(OUT / "numeric-report.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n"
)
print(
    json.dumps(
        {
            "status": report["status"],
            "cases": len(records),
            "failures": failures,
            "minimum_available_gib": report["minimum_available_gib"],
        },
        indent=2,
    )
)
raise SystemExit(bool(failures))
