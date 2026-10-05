"use strict";
// Real client and status handlers with fake DOM/transport; no browser or sockets.
const assert = require("node:assert/strict"),
  fs = require("node:fs"),
  vm = require("node:vm");
const { create } = require("../web/host-client.js");
const tools = require("../web/atlas-tools.js");
const A = "m_" + "1".repeat(64),
  B = "m_" + "2".repeat(64);
const capability = "9abcdef012345678".repeat(4).padEnd(64, "0");
const nodes = new Map();
const element = () => ({
  hidden: true,
  disabled: false,
  textContent: "",
  value: "",
  children: [],
  events: {},
  replaceChildren(...children) {
    this.children = children;
  },
  addEventListener(name, handler) {
    this.events[name] = handler;
  },
});
const document = {
  getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, element());
    return nodes.get(id);
  },
  createElement: element,
};
const calls = [],
  changes = [],
  scheduled = [];
let busy = false,
  serial = 0,
  selected = null;
const response = (body, status = 200) => ({
  ok: status === 200 || status === 202,
  status,
  json: async () => body,
});
async function fetchImpl(url, options = {}) {
  calls.push({ url, options });
  if (url === "/api/models")
    return response({
      api_version: 1,
      models: [
        { model_id: A, name: "First", fixture_eligible: true },
        { model_id: B, name: "Second", fixture_eligible: true },
      ],
    });
  if (url === "/api/view-contexts") {
    if (busy)
      return response(
        {
          api_version: 1,
          code: "reader_busy",
          error: "Another tab owns the reader",
        },
        409,
      );
    const body = JSON.parse(options.body);
    selected = {
      model_id: body.model_id,
      context_id: String(++serial).padStart(32, "0"),
      capability,
      lease_seconds: 15,
    };
    return response({ api_version: 1, ...selected }, 202);
  }
  if (url.includes("/heartbeat") || url.includes("/release"))
    return response({ api_version: 1 });
  return response({
    api_version: 1,
    host_context: {
      model_id: selected.model_id,
      context_id: selected.context_id,
      source_identity: "3".repeat(64),
      model_identity: "4".repeat(64),
    },
  });
}
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    const client = create({
      fetchImpl,
      document,
      schedule: (fn) => {
        scheduled.push(fn);
        return scheduled.length;
      },
      cancel: () => {},
    });
    await client.initialize(
      async (context) => changes.push(context),
      () => {},
    );
    assert.equal(changes.length, 1);
    assert.equal(document.getElementById("host-model-picker").hidden, false);
    assert.equal(
      document.getElementById("host-model-select").children.length,
      2,
    );
    assert.equal(client.snapshot().model_id, A);
    const first = client.bindRead("/api/inspect?tensor=0&row=1&col=1");
    assert.match(first.url, new RegExp("/api/models/" + A + "/inspect"));
    assert(!first.url.includes(capability));
    assert(!JSON.stringify(client.snapshot()).includes(capability));
    first.check(await (await fetchImpl(first.url)).json());
    busy = true;
    await assert.rejects(
      () => client.select(B),
      (error) => error.code === "reader_busy",
    );
    assert.equal(client.snapshot().model_id, A);
    assert.equal(changes.length, 1);
    busy = false;
    await client.select(B);
    assert.equal(client.snapshot().model_id, B);
    assert.throws(
      () => first.assertCurrent(),
      (error) => error.name === "AbortError",
    );
    assert.throws(
      () => first.check({ host_context: { model_id: A } }),
      (error) => error.name === "AbortError",
    );
    const active = client.bindRead("/api/progress?tensor=0");
    assert.throws(
      () =>
        active.check({
          host_context: { model_id: A, context_id: "0".repeat(32) },
        }),
      /another model/,
    );
    assert.throws(() => client.url("/api/calibrate"), /Read-only/);
    assert.throws(() => client.url("/api/model?context=override"), /override/);
    globalThis.AtlasHost = client;
    await tools.readJSON("/api/progress?tensor=0", { fetchImpl });
    await tools.readJSON("/api/tensor-status?tensor=0", { fetchImpl });
    assert(calls.some((call) => call.url.includes("/progress?")));
    delete globalThis.AtlasHost;
    await client.release();
    assert.equal(client.snapshot(), null);
    assert(calls.every((call) => !call.url.includes(capability)));

    const source = fs.readFileSync(require.resolve("../web/app.js"), "utf8");
    const statusStart = source.search(
        /function\s+applyStatus\s*\(\s*update\s*\)\s*\{/,
      ),
      statusEnd = source.indexOf("// Pointer reads", statusStart);
    assert(statusStart >= 0 && statusEnd > statusStart);
    const snippet = source.slice(statusStart, statusEnd);
    const model = {
      source_identity: "a".repeat(64),
      model_identity: "b".repeat(64),
      parameter_count: 26,
      global_max: null,
      calibration_complete: false,
      coverage: {},
      catalog: [{ id: 0, calibration_complete: false }, { id: 1 }, { id: 2 }],
    };
    const update = {
      api_version: 1,
      model_status_version: 1,
      source_identity: model.source_identity,
      model_identity: model.model_identity,
      global_max: 34,
      calibration_complete: true,
      coverage: {
        calibrated_tensors: 3,
        values_streamed: 26,
        all_requested: false,
        active_tensor: null,
        calibration_error: null,
      },
      tensor_status: {
        id: 0,
        calibration_complete: true,
        max_abs: 2,
        q99: 1.9,
      },
    };
    const reads = [];
    let views = 0;
    const context = {
      state: {
        model,
        tensor: model.catalog[0],
        pollController: null,
        viewEpoch: 1,
        modelEpoch: 1,
      },
      AbortController,
      assert: (condition, message) => assert(condition, message),
      drawCoverage: () => {},
      updateRuleAvailability: () => {},
      AtlasTools: { delay: async () => {} },
      json: async (url) => {
        reads.push(url);
        return structuredClone(update);
      },
      loadView: async () => {
        views++;
      },
      $: () => ({ hidden: false }),
      text: () => {},
      exact: String,
      showError: (message) => assert.fail(message),
      status: () => {},
    };
    vm.createContext(context);
    vm.runInContext(snippet, context);
    await context.pollStatus();
    assert.deepEqual(reads, ["/api/progress?tensor=0"]);
    assert.equal(views, 1);
    assert.equal(context.state.tensor.max_abs, 2);
    assert.equal(context.state.model.coverage.calibrated_tensors, 3);
    assert(
      !reads.includes("/api/model"),
      "Light status must not fetch the full catalog",
    );
    const before = JSON.stringify(context.state.model);
    assert.throws(
      () => context.applyStatus({ ...update, source_identity: "c".repeat(64) }),
      /status source/,
    );
    assert.equal(JSON.stringify(context.state.model), before);
    assert.throws(
      () =>
        context.applyStatus({
          ...update,
          tensor_status: { id: 1, calibration_complete: true },
        }),
      /another tensor/,
    );
    context.state.tensor = {
      ...context.state.tensor,
      shape: [2, 3, 4],
      slice: [1],
      rows: 3,
      cols: 4,
      count: 24,
    };
    context.applyStatus(update);
    assert.deepEqual(context.state.tensor.slice, [1]);
    assert.deepEqual(context.state.tensor.shape, [2, 3, 4]);
    assert.equal(
      context.state.tensor.count,
      24,
      "Status never replaces original native count with slice count",
    );
    const scopeStart = source.search(
        /function\s+calibrationScope\s*\(\s*model\s*\)\s*\{/,
      ),
      scopeEnd = source.search(
        /function\s+drawCoverage\s*\(\s*model\s*\)\s*\{/,
      );
    assert(scopeStart >= 0 && scopeEnd > scopeStart);
    const sourceScope = source.slice(scopeStart, scopeEnd);
    vm.runInContext(sourceScope, context);
    const mixed = {
      catalog: [{ id: 0 }, { id: 1 }, { id: 2, available: false }],
      coverage: { calibrated_tensors: 2 },
      global_calibration_supported: false,
    };
    const scope = context.calibrationScope(mixed);
    assert.equal(
      scope.pending,
      0,
      "Aggregate completion must not depend on stale unselected catalog entries",
    );
    assert.equal(scope.unavailable, 1);
    assert.equal(scope.globalSupported, false);
    console.log(
      JSON.stringify({
        status: "PASS",
        scope:
          "Real picker client and app status handlers; fake DOM/transport only",
        cases: [
          "picker initialization",
          "context URLs",
          "private lease bodies",
          "cross-tab busy preservation",
          "stale read rejection",
          "source response binding",
          "read-only allowlist",
          "selected status routing",
          "owned release",
          "light progress transition",
          "wrong-source rejection",
          "slice-preserving status",
          "mixed-catalog aggregate completion",
        ],
      }),
    );
  })().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  }),
);
