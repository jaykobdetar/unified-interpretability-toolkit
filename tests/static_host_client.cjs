"use strict";
// Real host client with inert DOM, timers, profile controls and transport only.
const assert = require("node:assert/strict");
const { create } = require("../web/host-client.js");
const A = "m_" + "a".repeat(64),
  B = "m_" + "b".repeat(64),
  F = "m_" + "f".repeat(64);
const source = "1".repeat(64),
  model = "2".repeat(64),
  capability = "3".repeat(64);
function harness({ enabled = true, fixture = false, admitted = true } = {}) {
  const nodes = new Map(),
    calls = [],
    configs = [],
    changes = [],
    scheduled = [];
  const element = () => ({
    hidden: true,
    disabled: false,
    value: "",
    textContent: "",
    children: [],
    events: {},
    replaceChildren(...items) {
      this.children = items;
    },
    addEventListener(name, fn) {
      this.events[name] = fn;
    },
  });
  const document = {
    getElementById(id) {
      if (!nodes.has(id)) nodes.set(id, element());
      return nodes.get(id);
    },
    createElement: element,
  };
  let selected = null,
    serial = 0,
    alterLease = (x) => x;
  const response = (body) => ({
    ok: true,
    status: 200,
    json: async () => body,
  });
  const entries = [
    {
      model_id: A,
      name: "Dense A",
      static_view_candidate: true,
      static_activation_allowed: admitted,
      static_view_ready: false,
    },
    {
      model_id: B,
      name: "Dense B",
      static_view_candidate: true,
      static_activation_allowed: admitted,
      static_view_ready: false,
    },
    ...(fixture
      ? [{ model_id: F, name: "Fixture", fixture_eligible: true }]
      : []),
  ];
  async function fetchImpl(url, options = {}) {
    calls.push({ url, options });
    if (url === "/api/models")
      return response({
        api_version: 1,
        static_views_enabled: enabled,
        profiles_enabled: true,
        models: entries,
      });
    if (url === "/api/view-contexts") {
      const request = JSON.parse(options.body);
      assert.deepEqual(
        Object.keys(request).sort(),
        selected ? ["capability", "context_id", "model_id"] : ["model_id"],
      );
      selected = {
        api_version: 1,
        model_id: request.model_id,
        context_id: String(++serial).padStart(32, "0"),
        capability,
        view_kind: request.model_id === F ? "fixture" : "static",
        profiles_enabled: request.model_id === F,
      };
      return response(alterLease({ ...selected }));
    }
    if (url.includes("/binding?"))
      return response({
        api_version: 1,
        source_binding: { tensor: 0, slice: { leading_indices: [] } },
      });
    return response({ api_version: 1 });
  }
  const client = create({
    fetchImpl,
    document,
    schedule: (fn) => {
      scheduled.push(fn);
      return scheduled.length;
    },
    cancel: () => {},
  });
  const profile = {
    configure: (...args) => configs.push(args),
    suspend: () => {},
    reset: async () => {},
    client: { snapshot: () => ({ cleanup_pending: false }) },
    unavailable: () => {},
  };
  client.mountProfiles({}, { mount: () => profile });
  function metadata(overrides = {}) {
    const saved = client.snapshot();
    return {
      api_version: 1,
      view_kind: "static",
      static_view_ready: true,
      profiles_enabled: false,
      inference_ready: false,
      inference_enabled: false,
      fit_verified: false,
      host_context: {
        model_id: saved.model_id,
        context_id: saved.context_id,
        source_identity: source,
        model_identity: model,
      },
      static_binding: {
        schema: "weight-atlas-static-binding-v1",
        model_id: saved.model_id,
        source_identity: source,
        model_identity: model,
      },
      ...overrides,
    };
  }
  return {
    client,
    document,
    calls,
    configs,
    changes,
    metadata,
    setLease: (fn) => {
      alterLease = fn;
    },
    initialize: () =>
      client.initialize(
        async (saved) => changes.push(saved),
        () => {},
      ),
  };
}
async function main() {
  let cases = 0;
  const disabled = harness({ enabled: false });
  await disabled.initialize();
  assert.equal(disabled.client.snapshot(), null);
  assert(
    disabled.document
      .getElementById("host-model-select")
      .children.every((item) => item.disabled),
  );
  await assert.rejects(() => disabled.client.select(A), /unavailable/);
  assert.equal(disabled.calls.length, 1);
  cases++;

  const candidate = harness({ admitted: false });
  await candidate.initialize();
  assert(
    candidate.document
      .getElementById("host-model-select")
      .children.every((item) => item.disabled),
  );
  await assert.rejects(() => candidate.client.select(A), /unavailable/);
  assert.equal(candidate.calls.length, 1);
  cases++;

  const h = harness();
  await h.initialize();
  assert.equal(h.client.snapshot(), null);
  assert(!h.document.getElementById("host-model-select").children[0].disabled);
  assert.equal(h.calls.length, 1);
  cases++;
  await h.client.select(A);
  assert.equal(h.client.snapshot().model_id, A);
  assert.throws(
    () => h.client.bindRead("/api/inspect?tensor=0&row=0&col=0"),
    /metadata/,
  );
  assert.throws(() => h.client.bindRead("/tile?tensor=0"), /metadata/);
  cases++;
  const original = h.client.bindRead("/api/model");
  original.check(h.metadata());
  const numeric = h.client.bindRead("/api/inspect?tensor=0&row=0&col=0");
  numeric.check(h.metadata());
  assert(!numeric.url.includes(capability));
  assert(!JSON.stringify(h.client.snapshot()).includes(capability));
  cases++;
  await h.client.setProfileSelection({
    tensor: 0,
    slice: { leading_indices: [] },
  });
  assert(h.configs.length > 0);
  assert(h.configs.every((args) => args[0].profiles_enabled === false));
  assert(
    !h.calls.some(
      (call) =>
        call.url.includes("/binding?") || call.url.startsWith("/api/profiles/"),
    ),
  );
  cases++;
  const rebound = h.client.bindRead("/api/model");
  assert.throws(
    () => rebound.check(h.metadata({ static_view_ready: false })),
    /not ready/,
  );
  assert.throws(() => h.client.bindRead("/api/progress"), /metadata/);
  rebound.check(h.metadata());
  cases++;
  const old = h.client.bindRead("/api/inspect?tensor=0");
  await h.client.select(B);
  assert.throws(
    () => old.check(h.metadata()),
    (error) => error.name === "AbortError",
  );
  assert.throws(() => h.client.bindRead("/api/inspect?tensor=0"), /metadata/);
  h.client.bindRead("/api/model").check(h.metadata());
  cases++;
  const foreign = h.client.bindRead("/api/model");
  assert.throws(
    () =>
      foreign.check(
        h.metadata({
          host_context: { model_id: A, context_id: "0".repeat(32) },
        }),
      ),
    /another/,
  );
  assert.throws(() => h.client.bindRead("/api/view?tensor=0"), /metadata/);
  cases++;

  const invalid = harness();
  await invalid.initialize();
  invalid.setLease((next) => ({ ...next, profiles_enabled: true }));
  await assert.rejects(() => invalid.client.select(A), /Invalid static/);
  assert.equal(invalid.client.snapshot(), null);
  await assert.rejects(() => invalid.client.select(A), /unresolved/);
  cases++;
  const legacy = harness({ enabled: false, fixture: true });
  await legacy.initialize();
  assert.equal(legacy.client.snapshot().model_id, F);
  assert.equal(
    legacy.document.getElementById("host-model-select").children[0].disabled,
    true,
  );
  assert.equal(
    legacy.document.getElementById("host-model-select").children[2].disabled,
    false,
  );
  cases++;
  await h.client.release();
  assert.equal(h.client.snapshot(), null);
  assert(h.calls.every((call) => !call.url.includes(capability)));
  cases++;
  console.log(
    JSON.stringify({
      status: "PASS",
      cases,
      scope:
        "Fake DOM/transport only; static metadata gating, profile closure and context lifecycle",
    }),
  );
}
main().catch((error) => {
  process.stderr.write(error.stack + "\n");
  process.exitCode = 1;
});
