"use strict";
// Existing DOM/OSD/transport plumbing, with explicit interleaving of an
// analytics Jump and viewer open completion. No browser or analytics job.
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
const scenario = process.argv[2] || "successful";
if (!["successful", "pending"].includes(scenario))
  throw Error("Unknown scenario");
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    run("bind()");
    const init = run("refreshModel()");
    await tick();
    take("/api/model").resolve(model);
    await tick();
    await complete(take("/api/view"));
    await init;
    const pin = run("inspectAt(1,1)");
    take("/api/inspect").resolve(inspect(catalog[0], 1, 1, "0.125"));
    await pin;
    get("left-rule").value = "tensor_magnitude";
    const recolor = run("loadView()"),
      metadata = take("/api/view");
    context.window.atlasAnalyticsBridge.jump({ axis: "row", index: 2 });
    const jump = take("/api/inspect");
    if (scenario === "successful") {
      jump.resolve(inspect(catalog[0], 2, 1, "0.25"));
      await tick();
      assert.deepEqual(copy(run("state.selected")), [2, 1]);
    }
    const before = viewers.length;
    metadata.resolve(view(currentSettings()));
    await tick();
    for (const viewer of viewers.slice(before)) viewer.emit("open");
    await recolor;
    await tick();
    assert.equal(
      pending.filter((r) => r.url.includes("/api/inspect")).length,
      0,
      "Recolor must not request the old remembered pin after a newer explicit inspection",
    );
    assert.equal(
      jump.options.signal.aborted,
      false,
      "A pending explicit inspection must retain ownership",
    );
    if (scenario === "pending") {
      jump.resolve(inspect(catalog[0], 2, 1, "0.25"));
      await tick();
    }
    assert.deepEqual(copy(run("state.selected")), [2, 1]);
    assert.equal(get("row").value, "2");
    assert.equal(get("col").value, "1");
    assert.equal(get("inspection").children[1].textContent, "0.25");
    assert.equal(context.window.atlasInferenceSelection.row, 2);
    assert.equal(context.window.atlasInferenceSelection.col, 1);
    assert(
      viewers.slice(before).every((v) => v.overlays.length === 1),
      "Both new viewers must show the winning selection marker",
    );
    assert.equal(pending.length, 0);
    console.log(
      JSON.stringify({
        status: "PASS",
        scenario,
        selected: copy(run("state.selected")),
        scope:
          "Pure DOM/OSD/transport only; latest explicit selection owns raw value and edit handoff",
      }),
    );
  })().catch((e) => {
    console.error(e);
    process.exitCode = 1;
  }),
);
