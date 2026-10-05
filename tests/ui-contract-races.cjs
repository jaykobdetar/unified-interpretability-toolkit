"use strict";
// Real frontend state machine with fresh DOM/OSD/transport doubles.
const { createFixture } = require("./support/ui-fixture.cjs");
const {
  fs,
  vm,
  assert,
  crypto,
  path,
  source,
  Element,
  elements,
  get,
  pending,
  viewers,
  frames,
  timers,
  OSD,
  context,
  run,
  copy,
  tick,
  take,
  tensor,
  catalog,
  model,
  statusUpdate,
  currentSettings,
  view,
  inspect,
  complete,
  activate,
  passed,
} = createFixture();
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    run("bind()");
    const init = run("refreshModel()");
    await tick();
    take("/api/model").resolve(model);
    await tick();
    await complete(take("/api/view"));
    await init;
    assert.deepEqual(currentSettings(), {
      tensor: 11,
      left: "global_linear",
      right: "tensor_asinh",
    });
    assert.equal(get("left-max").textContent, "34");
    assert.equal(get("right-max").textContent, "0.5");
    assert(
      get("right-units").textContent.includes("cannot compare magnitudes"),
    );
    assert.equal(run("state.current.tensor.id"), 11);
    passed.push(
      "Defaults compare identical source tensor: global linear versus independently labeled tensor asinh",
    );
    assert(get("global-scale-note").textContent.includes("k_norm.weight"));
    assert(
      get("global-scale-note").textContent.includes(
        "Largest matrix magnitude: 1",
      ),
    );
    assert(
      get("render-coverage-note").textContent.includes("not materialized"),
    );
    passed.push(
      "Global norm-vector maximum, matrix maximum, and incomplete pixel materialization stay explicit",
    );
    assert.equal(
      get("source-coverage").textContent,
      "Headers/index complete · 5 shards freshly hashed · 3 matched saved expectations · 2 hashed without expectations",
    );
    passed.push(
      "Freshly hashed shards, expected-hash matches, and missing expectations have distinct labels",
    );
    const old = run("loadView()"),
      oldReq = take("/api/view");
    get("left-rule").value = "tensor_linear";
    const latest = run("loadView()"),
      latestReq = take("/api/view");
    await complete(latestReq);
    oldReq.resolve(
      view({ tensor: 11, left: "global_linear", right: "tensor_asinh" }),
    );
    await Promise.all([old, latest]);
    assert.equal(run("state.current.settings.left"), "tensor_linear");
    passed.push(
      "Late view metadata cannot replace a newer selection even when transport ignores abort",
    );
    const oldOpen = run("loadView()"),
      oldOpenReq = take("/api/view"),
      start = viewers.length;
    oldOpenReq.resolve(view(currentSettings()));
    await tick();
    const abandoned = viewers.slice(start);
    get("right-rule").value = "global_asinh";
    const newOpen = run("loadView()");
    await complete(take("/api/view"));
    for (const v of abandoned) v.emit("open");
    await Promise.all([oldOpen, newOpen]);
    assert.equal(run("state.current.settings.right"), "global_asinh");
    assert(abandoned.every((v) => v.destroyed));
    assert.equal(run("state.loading"), false);
    passed.push(
      "Interrupted OpenSeadragon opens cannot mix old and new panels",
    );
    const p1 = run("inspectAt(0,0)"),
      r1 = take("/api/inspect"),
      p2 = run("inspectAt(1,1)"),
      r2 = take("/api/inspect");
    r2.resolve(inspect(catalog[0], 1, 1, "0.125"));
    await p2;
    r1.resolve(inspect(catalog[0], 0, 0, "stale"));
    await p1;
    assert.equal(get("inspection").children[1].textContent, "0.125");
    assert.deepEqual(copy(run("state.selected")), [1, 1]);
    passed.push(
      "One source lookup per address and only the latest address is displayed",
    );
    const stale = run("inspectAt(0,0)"),
      staleReq = take("/api/inspect");
    get("left-rule").value = "global_linear";
    const changed = run("loadView()");
    staleReq.resolve(inspect(catalog[0], 0, 0, "stale"));
    await stale;
    assert.deepEqual(copy(run("state.selected")), [1, 1]);
    assert.equal(get("inspection").children[1].textContent, "0.125");
    assert.equal(get("inspect-submit").disabled, true);
    await complete(take("/api/view"));
    await changed;
    passed.push(
      "Rule changes retain the last verified scalar and invalidate a late pending inspection before updating its color transforms",
    );
    const before = pending.length;
    await run("inspectAt(-1,0)");
    await run("inspectAt(4,0)");
    await run("inspectAt(0,0.1)");
    assert.equal(pending.length, before);
    get("row").value = "";
    get("col").value = "0";
    get("inspect-form").listeners.submit({ preventDefault() {} });
    assert.equal(pending.length, before);
    assert(get("error").textContent.includes("both"));
    passed.push(
      "Empty, fractional, negative, and out-of-bounds addresses make no source request",
    );
    let left = run("state.viewers.left"),
      right = run("state.viewers.right");
    left.bounds = { x: 0.1, y: 0.2, width: 0.25, height: 0.25 };
    left.emit("viewport-change");
    assert.deepEqual(right.bounds, left.bounds);
    right.bounds = { x: 0.3, y: 0.4, width: 0.125, height: 0.125 };
    right.emit("viewport-change");
    assert.deepEqual(left.bounds, right.bounds);
    passed.push(
      "Pan/zoom bounds synchronize both directions without a feedback loop",
    );
    const savedBounds = copy(left.bounds);
    const recolor1 = run("loadView()"),
      recolorReq1 = take("/api/view");
    get("left-rule").value = "tensor_asinh";
    const recolor2 = run("loadView()");
    await complete(take("/api/view"));
    recolorReq1.resolve(
      view({ tensor: 11, left: "global_linear", right: "global_asinh" }),
    );
    await Promise.all([recolor1, recolor2]);
    assert.deepEqual(copy(run("state.viewers.left.bounds")), savedBounds);
    assert.deepEqual(copy(run("state.selected")), [1, 1]);
    assert.equal(get("inspection").children[1].textContent, "0.125");
    passed.push(
      "Consecutive in-flight recolorings preserve the last active viewport and original scalar",
    );
    get("left-rule").value = "global_linear";
    await activate();
    left = run("state.viewers.left");
    right = run("state.viewers.right");
    left.item.lastDrawn = [{ tile: { level: 1 } }];
    right.item.lastDrawn = [{ tile: { level: 2 } }];
    run("updateResolution()");
    assert(get("left-resolution").textContent.includes("Pooled"));
    assert(get("right-resolution").textContent.includes("Scalar-level"));
    assert(get("right-magnification").textContent.includes("CSS px / scalar"));
    passed.push(
      "Actual drawn numerical levels are distinct from CSS magnification",
    );
    const staleViewer = left;
    await activate();
    get("error").textContent = "";
    staleViewer.emit("tile-load-failed");
    assert.equal(get("error").textContent, "");
    run("state.viewers.left.emit('tile-load-failed')");
    assert(get("error").textContent.includes("tile"));
    passed.push(
      "Stale tile failures are ignored; active failures remain visible",
    );
    const selectingVector = run("selectTensor(13)");
    await tick();
    take("/api/tensor-status").resolve(statusUpdate(catalog[2]));
    await tick();
    await complete(take("/api/view"));
    await selectingVector;
    await tick();
    assert.equal(run("state.current.tensor.rows"), 1);
    assert.equal(run("state.viewers.left.source.height"), 1);
    assert.equal(run("state.viewers.left.source.width"), 128);
    assert(
      get("tensor-shape").textContent.includes("one unwrapped vector row"),
    );
    const vector = run("inspectAt(0,127)");
    take("/api/inspect").resolve(inspect(catalog[2], 0, 127, "-0"));
    await vector;
    assert.equal(get("inspection").children[1].textContent, "-0");
    assert(get("inspection").children[2].textContent.includes("[127]"));
    passed.push(
      "Norm vector uses native 1×128 image, native [column] index, and exact raw string including signed zero",
    );
    get("tensor-search").value = "q_proj";
    get("layer-filter").value = "1";
    run("drawCatalog()");
    assert.equal(get("filtered-count").textContent, "1 of 3 tensors shown");
    get("tensor-search").value = "";
    get("layer-filter").value = "all";
    passed.push("Tensor name and layer filters combine correctly");
    const failed = run("loadView()");
    take("/api/view").resolve({ error: "initializing" }, 503);
    await failed;
    assert.equal(run("state.current"), null);
    assert.equal(get("inspect-submit").disabled, true);
    assert(get("status").textContent.includes("Initializing"));
    assert.equal(pending.length, 0);
    passed.push(
      "View 503 has no active image, scalar lookup, or automatic calibration fallback",
    );
    const refresh = run("refreshModel()");
    await tick();
    assert.equal(run("state.model"), null);
    run("selectTensor(11)");
    assert.equal(pending.filter((p) => p.url.includes("/api/view")).length, 0);
    take("/api/model").resolve({ error: "initializing full statistics" }, 503);
    await refresh;
    assert.equal(run("state.model"), null);
    assert.equal(get("left-rule").disabled, true);
    assert.equal(get("right-rule").disabled, true);
    assert.equal(pending.length, 0);
    passed.push(
      "Model refresh/503 disables stale catalog actions and both rules without substituting partial metadata",
    );
    const unavailable = run("refreshModel()");
    await tick();
    take("/api/model").resolve(
      { error: "Local renderer unavailable", code: "backend_unavailable" },
      503,
    );
    await unavailable;
    assert(get("error").textContent.includes("renderer unavailable"));
    assert(get("status").textContent.includes("Catalog unavailable"));
    assert.equal(pending.length, 0);
    passed.push(
      "Proxy503 is displayed as backend failure without claiming calibration is initializing",
    );
    const retry = run("refreshModel()");
    await tick();
    take("/api/model").resolve({
      ...model,
      global_max: null,
      calibration_complete: false,
      catalog: model.catalog.map((t) => ({
        ...t,
        calibration_complete: false,
        max_abs: null,
      })),
    });
    await tick();
    assert.equal(run("state.current"), null);
    assert.equal(get("inspect-submit").disabled, false);
    assert.equal(
      pending.filter(
        (p) => p.url.includes("/api/view") || p.url.includes("/tile"),
      ).length,
      0,
    );
    const local = take("/api/calibrate");
    assert.equal(local.options.method, "POST");
    assert.equal(local.options.headers["X-Atlas-Local"], "1");
    local.resolve({ api_version: 1, queued: 13 }, 202);
    await retry;
    assert.equal(
      get("left-rule").options.find((o) => o.value === "global_linear")
        .disabled,
      true,
    );
    assert.equal(get("left-rule").value, "tensor_linear");
    const raw = run("inspectAt(0,51)");
    const rawReq = take("/api/inspect");
    rawReq.resolve({
      ...inspect(catalog[2], 0, 51, "34"),
      transformed: { left: null, right: null },
      transforms_ready: false,
    });
    await raw;
    assert.equal(get("inspection").children[1].textContent, "34");
    assert.equal(run("state.current"), null);
    passed.push(
      "Incomplete calibration exposes raw BF16 inspection, queues explicit tensor-only calibration, disables global options, and never fabricates a color view",
    );
    const recover = run("refreshModel()");
    await tick();
    take("/api/model").resolve(model);
    await tick();
    await complete(take("/api/view"));
    await recover;
    assert.equal(get("inspect-submit").disabled, false);
    assert.equal(get("calibration-note").hidden, true);
    passed.push(
      "Explicit refresh recovers after initialization and retains the requested rules",
    );
    for (const side of ["left", "right"]) {
      get(side + "-rule").value = "tensor_magnitude";
      await activate();
      assert.equal(run("state.current.settings." + side), "tensor_magnitude");
      assert.equal(get(side + "-min").textContent, "0");
      assert.equal(get(side + "-mid").textContent, "");
      assert(get(side + "-gradient").classes.has("magnitude"));
      assert(get(side + "-parameter").textContent.includes("exact tensor M"));
      assert(
        run("state.viewers." + side + ".source.getTileUrl(0,0,0)").includes(
          "tensor_magnitude",
        ),
      );
    }
    const magInspect = run("inspectAt(0,0)");
    const magReq = take("/api/inspect");
    assert(
      magReq.url.includes("left=tensor_magnitude") &&
        magReq.url.includes("right=tensor_magnitude"),
    );
    magReq.resolve({
      ...inspect(catalog[2], 0, 0, "-0.5"),
      transformed: { left: 0.5 / 34, right: 0.5 / 34 },
    });
    await magInspect;
    assert.equal(get("inspection").children[1].textContent, "-0.5");
    run(
      "drawLegend('left',{id:'tensor_magnitude',title:'Tensor magnitude',min:0,max:0,scope:'complete original tensor',units:'magnitude',formula:'abs(x)/M'})",
    );
    assert(get("left-gradient").classes.has("all-zero"));
    get("left-rule").value = "tensor_linear";
    await activate();
    assert(!get("left-gradient").classes.has("magnitude"));
    assert.equal(get("left-mid").textContent, "0");
    passed.push(
      "Magnitude selects in either panel, sends rule on tile/inspect requests, preserves signed raw scalar, uses one-sided legend and switches back cleanly",
    );
    for (const [dtype, element_bytes, raw_hex_le, raw_exact] of [
      ["F16", 2, "0100", "0.000000059604644775390625"],
      ["F32", 4, "00000080", "-0.0"],
    ]) {
      const request = run("inspectAt(0,0)");
      take("/api/inspect").resolve({
        ...inspect(catalog[2], 0, 0, raw_exact),
        dtype,
        element_bytes,
        raw_hex_le,
        bf16_hex_le: null,
        classification: "finite",
      });
      await request;
      assert.equal(
        get("inspection").children[0].textContent,
        "Original " + dtype + " value",
      );
      assert.equal(get("inspection").children[1].textContent, raw_exact);
      assert.equal(
        get("inspection").children[3].children[1].textContent,
        raw_hex_le,
      );
    }
    const special = run("inspectAt(0,0)");
    take("/api/inspect").resolve({
      ...inspect(catalog[2], 0, 0, "NaN"),
      dtype: "F32",
      element_bytes: 4,
      raw_hex_le: "3412c07f",
      bf16_hex_le: null,
      classification: "nan",
      transforms_ready: false,
      transformed: { left: null, right: null },
    });
    await special;
    assert.equal(get("inspection").children[1].textContent, "NaN");
    assert.equal(
      get("inspection").children[4].children[0].children[1].textContent,
      "Unavailable: nonfinite source",
    );
    const wrongWidth = run("inspectAt(0,0)");
    take("/api/inspect").resolve({
      ...inspect(catalog[2], 0, 0, "1"),
      dtype: "F32",
      element_bytes: 2,
      raw_hex_le: "0000803f",
    });
    await wrongWidth;
    assert(get("error").textContent.includes("byte width"));
    passed.push(
      "F16/F32 inspector uses dtype-sized exact bytes and raw decimal, preserves signed zero, refuses width mismatch, and labels nonfinite transforms unavailable",
    );
    for (const [side, rule] of [
      ["left", "tensor_robust99"],
      ["right", "tensor_signed_percentile"],
    ]) {
      get(side + "-rule").value = rule;
      await activate();
      assert.equal(run("state.current.settings." + side), rule);
    }
    assert(get("left-parameter").textContent.includes("Q99"));
    assert(get("left-parameter").textContent.includes("clipped"));
    assert(get("right-units").textContent.includes("dimensionless"));
    assert.equal(get("right-max").textContent, "1");
    run(
      "drawLegend('left',{id:'tensor_robust99',title:'Robust',min:-1,max:1,q99:0,effective_divisor:1,zero_quantile_fallback:true,clipped_count:1,clipped_fraction:.001,scope:'complete original tensor',units:'clipped raw weight',formula:'clip(x/D,-1,1)'})",
    );
    assert(get("left-parameter").textContent.includes("Q99=0 fallback"));
    run(
      "state.tensor.dtype='F32';state.tensor.element_bytes=4;updateRuleAvailability()",
    );
    const beforeUnsupported = pending.length;
    await run("loadView()");
    assert.equal(pending.length, beforeUnsupported);
    assert.equal(run("state.current"), null);
    assert.equal(get("inspect-submit").disabled, false);
    assert(
      get("error").textContent.includes(
        "exact absolute-value rank index is pending",
      ),
    );
    assert(
      get("right-rule").options.find(
        (o) => o.value === "tensor_signed_percentile",
      ).disabled,
    );
    assert.equal(get("right-rule").value, "tensor_signed_percentile");
    const unsupportedRaw = run("inspectAt(0,0)");
    take("/api/inspect").resolve({
      ...inspect(catalog[2], 0, 0, "1"),
      dtype: "F32",
      element_bytes: 4,
      raw_hex_le: "0000803f",
      bf16_hex_le: null,
      classification: "finite",
      transforms_ready: false,
      transformed: { left: 1, right: null },
      transform_errors: {
        right: "Exact F32 absolute-value rank index pending",
      },
    });
    await unsupportedRaw;
    assert.equal(get("inspection").children[1].textContent, "1");
    assert(
      get(
        "inspection",
      ).children[4].children[1].children[1].textContent.includes(
        "rank index pending",
      ),
    );
    run(
      "state.tensor.dtype='BF16';state.tensor.element_bytes=2;updateRuleAvailability()",
    );
    await activate();
    assert(
      !get("right-rule").options.find(
        (o) => o.value === "tensor_signed_percentile",
      ).disabled,
    );
    passed.push(
      "Robust and signed-percentile selection, Q99 fallback/clipping legend, dimensionless rank units, and explicit F32 refusal without replacement field or loss of raw inspection",
    );
    const backendView = run("loadView()");
    take("/api/view").resolve(
      { error: "Local renderer unavailable", code: "backend_unavailable" },
      503,
    );
    await backendView;
    assert(get("status").textContent.includes("View unavailable"));
    assert.equal(pending.length, 0);
    passed.push(
      "View proxy503 is distinct from renderer calibration readiness",
    );
    const bad = run("loadView()"),
      badReq = take("/api/view"),
      wrong = view(currentSettings());
    wrong.legends.left.id = "unexpected";
    badReq.resolve(wrong);
    await bad;
    assert.equal(run("state.current"), null);
    assert(get("error").textContent.includes("legend rule"));
    passed.push(
      "Mismatched API legend is rejected before any image activation",
    );
    assert.equal(pending.length, 0);
    console.log(
      JSON.stringify(
        {
          status: "PASS",
          checks: passed.length,
          source_sha256: crypto
            .createHash("sha256")
            .update(source)
            .digest("hex"),
          passed,
        },
        null,
        2,
      ),
    );
  })().catch((e) => {
    console.error(e);
    process.exitCode = 1;
  }),
);
