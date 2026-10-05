"use strict";
// Actual app.js -> publishInferenceSelection -> inference.js DOM handoff.
// Valid calibration/inspection responses only; no browser, model or network.
const { createFixture } = require("./support/ui-fixture.cjs");
const path = require("node:path");
const appSource = path.resolve(
  process.env.ATLAS_TEST_APP_SOURCE || "web/app.js",
);
const {
  fs,
  vm,
  assert,
  crypto,
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
} = createFixture({ appSource });
const pendingFirstRead = process.argv[2] === "pending-first-read";
(async () => {
  Element.prototype.getContext = function () {
    return { clearRect() {}, fillRect() {} };
  };
  context.window.addEventListener = () => {};
  for (const [key, value] of Object.entries({
    task: "generation",
    rate: "2",
    limit: "4",
    layer: "0",
    site: "attention",
    mode: "step",
    prompt: "synthetic",
    kind: "element",
    operation: "zero",
    factor: ".5",
  }))
    get("infer-" + key).value = value;
  const sourceModel = {
    repo: "HuggingFaceTB/SmolLM2-135M",
    revision: "pinned",
    weights_sha256: "pinned",
  };
  const q = tensor(
    11,
    "model.layers.0.self_attn.q_proj.weight",
    [576, 576],
    0.5,
  );
  const o = {
    ...tensor(12, "model.layers.0.self_attn.o_proj.weight", [576, 576], 0.5),
    calibration_complete: false,
  };
  catalog.splice(0, catalog.length, q, o);
  model.parameter_count = 2 * 576 * 576;
  model.calibration_complete = false;
  model.inference_source_model = sourceModel;
  model.model_identity = "model-fixture";
  model.source_identity = "source-fixture";
  vm.runInContext(
    fs.readFileSync(path.join(__dirname, "../web/inference.js"), "utf8"),
    context,
  );
  take("/api/inference").resolve({
    model: "fixture",
    engine: "CPU",
    architecture: require("./fixtures/inference-architecture.json"),
    comparison: {
      source_model: sourceModel,
      max_edits: 8,
      tensors: catalog.map((t) => ({
        name: t.name,
        shape: t.shape,
        aliases: [],
      })),
    },
  });
  await tick();
  // Refresh and selection await the supported profile-cleanup boundary, even
  // when no hosted profile client is present. Observe the queued read afterward.
  run("bind()");
  const init = run("refreshModel()");
  await tick();
  take("/api/model").resolve(copy(model));
  await tick();
  await complete(take("/api/view"));
  await init;
  run("selectTensor(12)");
  await tick();
  take("/api/tensor-status").resolve({
    ...statusUpdate(o),
    calibration_complete: false,
    coverage: { ...statusUpdate(o).coverage, calibrated_tensors: 1 },
    tensor_status: { id: o.id, calibration_complete: false, max_abs: null },
  });
  await tick();
  take("/api/calibrate").resolve({ api_version: 1, queued: 12 }, 202);
  await tick();
  assert.equal(get("inspect-submit").disabled, false);
  assert.equal(run("state.current"), null);
  const inspectForm = async (t, raw) => {
    get("row").value = "70";
    get("col").value = "575";
    get("inspect-form").listeners.submit({ preventDefault() {} });
    take("/api/inspect").resolve({
      ...inspect(t, 70, 575, raw),
      transformed: { left: null, right: null },
      transforms_ready: false,
    });
    await tick();
  };
  let delayedRaw = null;
  if (pendingFirstRead) {
    get("row").value = "70";
    get("col").value = "575";
    get("inspect-form").listeners.submit({ preventDefault() {} });
    delayedRaw = take("/api/inspect");
    assert.equal(context.window.atlasInferenceSelection, null);
  } else {
    await inspectForm(o, "0.125");
    assert.equal(context.window.atlasInferenceSelection.tensor, o.name);
    assert.equal(get("infer-head").disabled, false);
    assert(get("infer-head").textContent.includes("output head (columns)"));
  }
  // Complete the normal first-calibration poll after the raw selection exists.
  timers.filter(Boolean).at(-1)();
  await tick();
  o.calibration_complete = true;
  take("/api/progress").resolve({
    ...statusUpdate(o),
    coverage: {
      ...statusUpdate(o).coverage,
      calibrated_tensors: 2,
      values_streamed: model.parameter_count,
    },
  });
  await tick();
  assert.equal(
    get("infer-head").disabled,
    true,
    "View transition must not expose a stale selection",
  );
  if (delayedRaw) {
    assert(delayedRaw.options.signal.aborted);
    delayedRaw.resolve({
      ...inspect(o, 70, 575, "0.125"),
      transformed: { left: null, right: null },
      transforms_ready: false,
    });
    await tick();
    assert.equal(
      context.window.atlasInferenceSelection,
      null,
      "Cancelled raw response must not publish",
    );
    const opening = take("/api/view"),
      before = viewers.length;
    opening.resolve(view(currentSettings()));
    await tick();
    for (const viewer of viewers.slice(before)) viewer.emit("open");
    await tick();
    assert(
      pending.some((r) => r.url.includes("/api/inspect")),
      "A valid raw address requested before calibration completion must be re-read after the first view opens",
    );
    take("/api/inspect").resolve(inspect(o, 70, 575, "0.125"));
    await tick();
  } else {
    await complete(take("/api/view"));
    await tick();
  }
  assert.equal(
    get("infer-head").disabled,
    false,
    "Completed same-source first calibration must restore the inspected O-projection handoff",
  );
  assert.equal(context.window.atlasInferenceSelection.tensor, o.name);
  assert.equal(context.window.atlasInferenceSelection.row, 70);
  assert.equal(context.window.atlasInferenceSelection.col, 575);
  get("infer-head").listeners.click();
  assert(
    get("infer-edits").children[0].textContent.includes("columns [512,576)"),
  );
  get("infer-clear-edits").listeners.click();
  run("selectTensor(11)");
  await tick();
  take("/api/tensor-status").resolve({
    ...statusUpdate(q),
    coverage: {
      ...statusUpdate(q).coverage,
      calibrated_tensors: 2,
      values_streamed: model.parameter_count,
    },
  });
  await tick();
  await complete(take("/api/view"));
  await tick();
  assert.equal(
    context.window.atlasInferenceSelection,
    null,
    "Tensor changes must clear the old O selection",
  );
  assert.equal(get("infer-head").disabled, true);
  await inspectForm(q, "0.25");
  assert(
    get("infer-head").textContent.includes("query rows (query intervention)"),
  );
  get("infer-head").listeners.click();
  assert(get("infer-edits").children[0].textContent.includes("rows [64,128)"));
  get("infer-clear-edits").listeners.click();
  run(
    "state.model.inference_source_model={...state.model.inference_source_model,weights_sha256:'different'}",
  );
  await inspectForm(q, "0.25");
  assert.equal(get("infer-head").disabled, true);
  assert(
    get("infer-selection").textContent.includes("no verified edit mapping"),
  );
  assert.equal(pending.length, 0);
  console.log(
    JSON.stringify(
      {
        status: "PASS",
        pending_first_read: pendingFirstRead,
        checks: 5,
        scope:
          "Actual raw-inspect form and callback across first calibration; O columns512:576; Q rows64:128; tensor switch clears; changed source remains disabled; DOM/transport only",
        app_sha256: crypto.createHash("sha256").update(source).digest("hex"),
      },
      null,
      2,
    ),
  );
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
