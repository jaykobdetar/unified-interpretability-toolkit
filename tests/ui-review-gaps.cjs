"use strict";
// Production app/workspace behavior with the existing deterministic DOM/OSD doubles.
const { createFixture } = require("./support/ui-fixture.cjs");
const assert = require("node:assert/strict");

async function ready() {
  const f = createFixture();
  Object.assign(f.catalog[0], {
    shape: [513, 769],
    rows: 513,
    cols: 769,
    count: 513 * 769,
    max_level: 10,
    dtype: "BF16",
    element_bytes: 2,
  });
  f.model.parameter_count = f.catalog.reduce(
    (total, tensor) => total + tensor.count,
    0,
  );
  f.model.source_bytes = 2 * f.model.parameter_count;
  f.context.fixtureModel = f.model;
  f.run("bind();populateModel(fixtureModel,{tensor:11})");
  await f.activate();
  return f;
}

require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    const tiles = await ready();
    for (const viewer of tiles.viewers) {
      assert.equal(viewer.source.width, 769);
      assert.equal(viewer.source.height, 513);
      for (const [x, y] of [
        [0, 0],
        [1, 0],
        [0, 1],
        [2, 1],
        [1, 2],
      ]) {
        const query = new URL(
          viewer.source.getTileUrl(10, x, y),
          "http://127.0.0.1",
        ).searchParams;
        assert.equal(query.get("level"), "10");
        assert.equal(query.get("x"), String(x));
        assert.equal(query.get("y"), String(y));
      }
    }

    const rules = await ready();
    rules.get("right-rule").value = "tensor_linear";
    const loading = rules.run("loadView()"),
      request = rules.take("/api/view");
    const requested = rules.currentSettings(),
      before = rules.viewers.length;
    request.resolve(rules.view(requested));
    await rules.tick();
    // Settings can change while the two independent viewer opens are pending.
    // Deliver both opens without a new epoch: this exercises the settings guard.
    rules.get("right-rule").value = "tensor_magnitude";
    for (const viewer of rules.viewers.slice(before)) viewer.emit("open");
    await loading;
    assert.equal(
      rules.run("state.current"),
      null,
      "Stale right-rule view must not become active",
    );
    const current = rules.run("loadView()");
    await rules.complete(rules.take("/api/view"));
    await current;
    assert.equal(rules.run("state.current.settings.right"), "tensor_magnitude");

    const inspection = await ready();
    const reading = inspection.run("inspectAt(300,600)");
    inspection
      .take("/api/inspect")
      .resolve(inspection.inspect(inspection.catalog[0], 300, 600, "-0.0"));
    await reading;
    const facts = inspection
      .get("inspection")
      .children.find((child) => child.tag === "dl");
    const address = facts.children.findIndex(
      (child) => child.textContent === "Display row / column",
    );
    assert(address >= 0);
    assert.equal(facts.children[address + 1].textContent, "300 / 600");
    assert.deepEqual(
      inspection.copy(inspection.run("state.selected")),
      [300, 600],
    );

    const region = await ready(),
      storage = new Map();
    Object.assign(region.context, {
      location: { pathname: "/", search: "", hash: "" },
      history: { pushState() {}, replaceState() {} },
      localStorage: {
        getItem: (key) => storage.get(key) || null,
        setItem: (key, value) => storage.set(key, value),
      },
      crypto: region.crypto,
      TextEncoder,
    });
    region.context.window.addEventListener = () => {};
    region.context.document.body = new region.Element("body");
    region.Element.prototype.focus = () => {};
    for (const id of ["region-r0", "region-c0", "region-r1", "region-c1"])
      region.get(id).value = "0";
    for (const file of ["atlas-tools.js", "workspace-tools.js"]) {
      region.vm.runInContext(
        region.fs.readFileSync(
          region.path.join(__dirname, "../web", file),
          "utf8",
        ),
        region.context,
      );
    }
    region.run("atlasWorkspace.model(state.model)");
    for (const [bounds, expected] of [
      [[300, 600, 305, 611], { x: 600, y: 300, width: 12, height: 6 }],
      [[320, 640, 320, 640], { x: 640, y: 320, width: 1, height: 1 }],
    ]) {
      ["region-r0", "region-c0", "region-r1", "region-c1"].forEach(
        (id, index) => {
          region.get(id).value = String(bounds[index]);
        },
      );
      region.get("region-focus").listeners.click();
      for (const side of ["left", "right"]) {
        assert.deepEqual(
          region.copy(region.run(`state.viewers.${side}.viewport.getBounds()`)),
          expected,
        );
      }
    }
    console.log(
      JSON.stringify({
        status: "PASS",
        checks: [
          "Asymmetric multi-tile coordinates",
          "Right-rule stale-open refusal and recovery",
          "Asymmetric inspector address",
          "Inclusive region and single-cell focus",
        ],
        scope: "DOM/OSD doubles; actual-browser qualification is separate",
      }),
    );
  })().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  }),
);
