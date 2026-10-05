"use strict";
// Combined source: pending raw selection must not cross a slice or model lease.
// Real app/binding code; deterministic DOM, OSD and transport doubles only.
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
const scenario = process.argv[2];
if (!["slice-change", "model-refresh", "host-reopen"].includes(scenario))
  throw Error("Unknown scenario");
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    vm.runInContext(
      fs.readFileSync(path.join(__dirname, "../web/atlas-tools.js"), "utf8"),
      context,
    );
    const high = {
      id: 21,
      name: "rank3",
      dtype: "BF16",
      element_bytes: 2,
      shape: [2, 3, 4],
      rows: 3,
      cols: 4,
      count: 24,
      max_level: 2,
      min_level: 0,
      available: true,
      calibration_complete: false,
      max_abs: 12,
      slice: [1],
    };
    const selected =
      scenario === "slice-change"
        ? high
        : {
            ...catalog[0],
            dtype: "BF16",
            element_bytes: 2,
            calibration_complete: false,
          };
    const initial = {
      ...model,
      host_context: { context_id: "old-context" },
      calibration_complete: false,
      catalog: [selected],
      parameter_count: selected.count,
    };
    context.initial = initial;
    run(
      "bind();populateModel(initial);state.tensor=initial.catalog[0];allowRawInspection();drawSlicePicker()",
    );
    const oldKey = run("tensorContextKey()"),
      binding = copy(run("AtlasTools.sourceBinding(state.model,state.tensor)"));
    const oldRead = run("inspectAt(1,1)"),
      raw = take("/api/inspect");
    const oldResponse = {
      ...inspect(selected, 1, 1, "0.125"),
      source_binding: binding,
      native_indices: scenario === "slice-change" ? [1, 1, 1] : [1, 1],
    };
    let opening;
    if (scenario === "slice-change") {
      // Actual slice-apply event invalidates the pending pin before opening slice 0.
      run("state.tensor.calibration_complete=true");
      const picker = get("slice-picker");
      picker.children[1].children[0].value = "0";
      picker.children.at(-1).listeners.click();
      await tick();
    } else {
      const next = {
        ...initial,
        catalog: [{ ...selected, calibration_complete: true }],
        calibration_complete: true,
      };
      if (scenario === "host-reopen")
        next.host_context = { context_id: "new-context" };
      opening = run("refreshModel()");
      await tick();
      take("/api/model").resolve(next);
      await tick();
    }
    assert.notEqual(
      run("tensorContextKey()"),
      oldKey,
      "Slice or model/lease epoch changes the remembered context",
    );
    assert(raw.options.signal.aborted);
    assert.equal(run("state.pendingInspection"), null);
    const request = take("/api/view"),
      t = copy(run("state.tensor")),
      s = currentSettings(),
      before = viewers.length;
    const legends = {};
    for (const side of ["left", "right"])
      legends[side] = {
        id: s[side],
        min: -12,
        max: 12,
        zero: 0,
        s: 1,
        formula: "fixture",
        scope: "original tensor",
        units: "raw weight",
      };
    request.resolve({
      api_version: 1,
      tensor: t,
      source_binding: copy(
        run("AtlasTools.sourceBinding(state.model,state.tensor)"),
      ),
      legends,
      tile_size: 256,
      overlap: 0,
      source_values_unchanged: true,
    });
    await tick();
    for (const v of viewers.slice(before)) v.emit("open");
    await opening;
    await tick();
    raw.resolve(oldResponse);
    await oldRead;
    await tick();
    assert.equal(run("state.inspectionData"), null);
    assert.equal(run("state.pendingInspection"), null);
    assert.equal(context.window.atlasInferenceSelection, null);
    assert.equal(pending.length, 0, "No old-context automatic reread");
    assert.equal(run("state.loading"), false);
    if (scenario === "slice-change")
      assert.match(viewers.at(-1).source.getTileUrl(2, 0, 0), /slice=0/);
    console.log(
      JSON.stringify({
        status: "PASS",
        scenario,
        scope:
          "Pending raw read cleared by actual slice apply or model refresh; late old response cannot restore a pin; model/lease epoch and native slice remain distinct",
      }),
    );
  })().catch((e) => {
    console.error(e);
    process.exitCode = 1;
  }),
);
