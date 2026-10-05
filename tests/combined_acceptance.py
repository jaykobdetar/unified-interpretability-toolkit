"""Benign sequential combined API acceptance on one explicitly owned coordinator."""

import argparse, hashlib, json, math, os, struct, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/acceptance"))
sys.path.insert(0, str(ROOT / "tools"))
from owner_client import OwnerClient, clean, parity
from render_contract import assert_render_rules
from inference_edits import SOURCE_MODEL
from live_inference import verify_model

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
client = OwnerClient(args.base)
owned_analysis = None
checks = {}
analysis_peak = 0


def api(route, data=None, expected=200):
    remaining = client.deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Combined acceptance total budget exhausted")
    body = None if data is None else json.dumps(data, allow_nan=False).encode()
    assert body is None or len(body) <= 8192
    request = urllib.request.Request(
        args.base + route,
        data=body,
        headers={
            "Origin": args.base,
            "X-Atlas-Local": "1",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=min(10, remaining)) as r:
            status, raw = r.status, r.read(2 * 1024**2 + 1)
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read(2 * 1024**2 + 1)
    assert len(raw) <= 2 * 1024**2
    assert status == expected, (route, status, raw[:300])
    return json.loads(raw)


def save():
    text = json.dumps(checks, indent=2) + "\n"
    assert len(text.encode()) < 1024**2
    (args.out / "progress.json").write_text(text)


try:
    verify_model(args.model)
    metadata = api("/api/model")
    assert metadata["inference_source_model"] == SOURCE_MODEL
    assert_render_rules(metadata)
    tensor = next(
        t
        for t in metadata["catalog"]
        if t["name"] == "model.layers.0.self_attn.q_proj.weight"
    )
    request = {
        "prompt": "The capital of France is",
        "max_new_tokens": 1,
        "layer": 7,
        "activation_site": "attention",
        "source_model": SOURCE_MODEL,
        "edits": [],
    }
    first = client.start(request)
    assert first["worker_alive"]
    analysis_request = {
        "tensor": tensor["id"],
        "region": {"row": 0, "col": 0, "rows": 8, "cols": 8},
        "seed": 7,
        "svd": True,
    }
    refused = api("/api/analytics/start", analysis_request, 409)
    assert "compute busy" in refused["error"]
    checks["live_inference_excludes_analytics"] = True
    initial = parity(client.finish())
    checks["initial_empty_exact_parity"] = True
    save()
    # First actual analytics job follows confirmed inference cleanup.
    analysis = api("/api/analytics/start", analysis_request, 202)
    owned_analysis = analysis["job"]
    while analysis["worker_alive"] or analysis.get("cleanup_pending"):
        time.sleep(0.1)
        analysis = api("/api/analytics/poll", {"job": owned_analysis})
        analysis_peak = max(analysis_peak, analysis.get("peak_worker_rss_mib", 0))
    assert analysis["status"] == "complete", analysis.get("error")
    owned_analysis = None
    report = analysis["result"]
    assert report["region"] == analysis_request["region"]
    assert report["svd"]["available"]
    # Independent original BF16 decode and row/column absolute means, no analytics computation imported.
    values = []
    with (args.model / tensor["shard"]).open("rb") as stream:
        for row in range(8):
            stream.seek(tensor["byte_offset"] + 2 * row * tensor["cols"])
            raw = stream.read(16)
            values.append(
                [
                    struct.unpack("<f", b"\0\0" + raw[i : i + 2])[0]
                    for i in range(0, 16, 2)
                ]
            )
    maximum_error = 0
    for axis, expected in [
        ("rows", [sum(abs(v) for v in row) / 8 for row in values]),
        ("columns", [sum(abs(values[r][c]) for r in range(8)) / 8 for c in range(8)]),
    ]:
        for got, want in zip(report["original"][axis], expected):
            maximum_error = max(maximum_error, abs(got["mean_abs"] - want))
    assert maximum_error < 1e-14
    order = report["control"]["position_to_source"]
    assert sorted(order) == list(range(64))
    checks["analytics"] = {
        "values": 64,
        "rows": 8,
        "columns": 8,
        "svd_requested_and_returned": True,
        "independent_mean_abs_error": maximum_error,
        "seeded_source_map_is_permutation": True,
        "peak_worker_rss_mib": analysis_peak,
    }
    save()
    edit = {
        "tensor": tensor["name"],
        "shape": [576, 576],
        "kind": "rows",
        "operation": "zero",
        "start": 0,
        "end": 64,
    }
    client.start({**request, "edits": [edit]})
    edited = client.finish()
    assert clean(edited) and edited["status"] == "complete"
    step = edited["steps"][0]
    assert step["alignment"] == "matched_prefix"
    assert all(
        c["delta"] == c["edited_logit"] - c["baseline_logit"]
        for c in step["candidates"]
    )
    checks["head0_zero_effect"] = {
        "prompt": request["prompt"],
        "native_rows_half_open": [0, 64],
        "top_union_max_abs_raw_logit_delta": max(
            abs(c["delta"]) for c in step["candidates"]
        ),
        "baseline_ids": edited["details"]["baseline"]["generated_ids"],
        "edited_ids": edited["details"]["edited"]["generated_ids"],
    }
    save()
    client.start(request)
    assert parity(client.finish()) == initial
    checks["fresh_empty_control_exact_parity_after_analytics_and_edit"] = True
    fixture = json.loads(
        (ROOT / "tests/fixtures/sweep-maximum-acceptance.json").read_text()
    )["request_without_digest"]
    sweep_request = {
        **fixture,
        "prompts": fixture["prompts"][:1],
        "targets": fixture["targets"][:1],
    }
    plan = api("/api/inference/sweep-plan", sweep_request)["plan"]
    assert plan["records"] == 3 and plan["prefills"] == 6
    client.start({**sweep_request, "plan_digest": plan["digest"]})
    result = client.finish()
    assert result["status"] == "complete" and len(result["steps"]) == 3
    coverage = result["details"]["sweep_coverage"]
    assert coverage["complete"] and not coverage["unrun_ids"]
    assert all(s["sweep"]["restoration_verified"] for s in result["steps"])
    checks["representative_sweep"] = {
        "records": 3,
        "prefills": 6,
        "coverage": coverage,
        "restoration_verified": True,
        "maximum_browser_remains_blocked": True,
    }
    save()
    verify_model(args.model)
    checks.update(
        status="PASS",
        disk_hashes_unchanged=True,
        inference_jobs=client.starts,
        peak_inference_worker_rss_mib=client.peak_worker_rss_mib,
        scope="One empty control,8x8 analytics/SVD,one query-head edit,fresh control,and three-record fixed-context sweep; no maximum-browser retry or broad resource-fit claim",
    )
    save()
finally:
    cleanup = client.cleanup()
    if owned_analysis:
        try:
            stopped = api("/api/analytics/cancel", {"job": owned_analysis})
            cleanup = (
                cleanup
                and not stopped["worker_alive"]
                and not stopped.get("cleanup_pending")
            )
        except Exception:
            cleanup = False
    checks["owned_cleanup_confirmed"] = cleanup
    verify_model(args.model)
    checks["final_disk_hashes_match"] = True
    save()
    if not cleanup:
        raise RuntimeError(
            "Combined owned cleanup unconfirmed; supervisor must reap coordinator"
        )
print(json.dumps(checks, indent=2))
