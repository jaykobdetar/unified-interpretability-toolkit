"use strict";
const assert = require("node:assert/strict");
const Profiles = require("../web/profile-client.js"),
  Hosts = require("../web/host-client.js");
const A = "m_" + "a".repeat(64),
  B = "m_" + "b".repeat(64),
  C = "m_" + "c".repeat(64);
const contexts = {
  [A]: { model_id: A, context_id: "1".repeat(32), capability: "2".repeat(64) },
  [B]: { model_id: B, context_id: "3".repeat(32), capability: "4".repeat(64) },
  [C]: { model_id: C, context_id: "5".repeat(32), capability: "6".repeat(64) },
};
const binding = {
  version: 2,
  source_identity: "e".repeat(64),
  model_identity: "f".repeat(64),
  tensor: 1,
  name: "matrix",
  dtype: "BF16",
  shape: [2, 3],
  rows: 2,
  cols: 3,
  slice: { leading_indices: [], display_axes: [0, 1] },
};
const defer = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => {
    resolve = a;
    reject = b;
  });
  return { promise, resolve, reject };
};
const flush = async () => {
  for (let i = 0; i < 30; i++) await Promise.resolve();
};
class Element {
  constructor(tag, doc) {
    this.tag = tag;
    this.ownerDocument = doc;
    this.children = [];
    this.events = {};
    this.textContent = "";
    this.value = "";
    this.disabled = false;
    this.hidden = false;
  }
  append(...items) {
    this.children.push(...items);
  }
  replaceChildren(...items) {
    this.children = items;
    this.textContent = "";
  }
  setAttribute() {}
  addEventListener(event, fn) {
    this.events[event] = fn;
  }
}
const response = (body, status = 200) => ({
  ok: status < 400,
  status,
  json: async () => body,
});
const cases = [];
const test = (name, fn) => cases.push({ name, fn });
async function fixture() {
  const calls = [],
    held = defer();
  let serverContext = null,
    ui,
    hold = null,
    refuse = false,
    holdRelease = false,
    heldBinding = null;
  const doc = { createElement: (t) => new Element(t, doc) },
    root = new Element("section", doc);
  const host = Hosts.create({
    schedule: () => 1,
    cancel: () => {},
    fetchImpl: async (url, options = {}) => {
      const body = options.body ? JSON.parse(options.body) : null;
      calls.push({ url, body });
      if (url === "/api/models")
        return response({
          api_version: 1,
          profiles_enabled: true,
          models: [A, B, C].map((model_id) => ({
            model_id,
            name: "fixture",
            fixture_eligible: true,
          })),
        });
      if (url === "/api/view-contexts") {
        if (body.model_id === hold) {
          if (!refuse) serverContext = contexts[body.model_id];
          return held.promise;
        }
        serverContext = contexts[body.model_id];
        return response({ api_version: 1, ...serverContext });
      }
      if (url.endsWith("/release"))
        return holdRelease ? held.promise : response({ api_version: 1 });
      if (url.includes("/binding?")) {
        const q = new URL(url, "http://local").searchParams;
        assert.equal(q.get("context"), serverContext.context_id);
        if (heldBinding) {
          const pending = heldBinding;
          heldBinding = null;
          return pending.promise;
        }
        return response({ api_version: 1, source_binding: binding });
      }
      if (url.startsWith("/api/profiles/")) {
        assert.equal(
          body.context_id,
          serverContext.context_id,
          "Never send obsolete profile ownership",
        );
        if (url.endsWith("/start"))
          return response(
            {
              version: 1,
              ...serverContext,
              job_id: "7".repeat(32),
              job_capability: "8".repeat(64),
              state: "complete",
              accepted: {
                revision: "9".repeat(64),
                visited_values: 6,
                total_values: 6,
                complete: true,
              },
              cleanup_pending: false,
              resume_available: false,
            },
            202,
          );
        if (url.endsWith("/reconcile"))
          return response({
            state: "cancelled",
            cleanup_pending: false,
            resume_available: false,
            no_owned_work: true,
          });
        return response({
          version: 1,
          ...serverContext,
          job_id: body.job_id,
          state: "cancelled",
          accepted: null,
          cleanup_pending: false,
          resume_available: false,
        });
      }
      throw new Error("Unexpected mock route " + url);
    },
  });
  host.mountProfiles(root, {
    mount: (...args) => {
      ui = Profiles.mount(...args);
      return ui;
    },
  });
  await host.initialize(
    async () => {
      await host.prepareProfileChange();
      await host.setProfileSelection(binding);
    },
    () => {},
  );
  const find = (n) =>
    n.textContent === "Start full slice"
      ? n
      : n.children.map(find).find(Boolean);
  return {
    host,
    ui,
    root,
    calls,
    held,
    start: () => find(root),
    hold(model, rejected = false) {
      hold = model;
      refuse = rejected;
    },
    holdRelease() {
      holdRelease = true;
    },
    holdBinding() {
      heldBinding = defer();
      return heldBinding;
    },
  };
}
test("deferred successful model acquisition blocks Start until new binding is installed", async () => {
  const f = await fixture();
  f.hold(B);
  const selecting = f.host.select(B);
  await flush();
  assert.equal(
    f.start().disabled,
    true,
    "Start disabled for entire acquire interval",
  );
  await assert.rejects(f.ui.client.start(binding, 0, 6));
  assert.equal(
    f.calls.filter((c) => c.url === "/api/profiles/start").length,
    0,
  );
  f.held.resolve(response({ api_version: 1, ...contexts[B] }));
  await selecting;
  assert.equal(f.host.snapshot().model_id, B);
  assert.equal(f.start().disabled, false);
  await f.ui.client.start(binding, 0, 6);
  assert.equal(
    f.calls.filter((c) => c.url === "/api/profiles/start")[0].body.context_id,
    contexts[B].context_id,
  );
  await f.host.select(C);
  assert.equal(f.host.snapshot().model_id, C);
  assert.equal(f.ui.client.snapshot().cleanup_pending, false);
});
test("ordinary acquire refusal revalidates and restores the previous binding", async () => {
  const f = await fixture();
  f.hold(B, true);
  const selecting = f.host.select(B);
  await flush();
  assert.equal(f.start().disabled, true);
  await assert.rejects(f.ui.client.start(binding, 0, 6));
  f.held.resolve(
    response(
      { api_version: 1, code: "reader_busy", error: "Reader busy" },
      409,
    ),
  );
  await assert.rejects(selecting, /Reader busy/);
  assert.equal(f.host.snapshot().model_id, A);
  assert.equal(f.start().disabled, false);
  await f.ui.client.start(binding, 0, 6);
  assert.equal(
    f.calls.filter((c) => c.url === "/api/profiles/start")[0].body.context_id,
    contexts[A].context_id,
  );
});
test("unknown acquire result keeps admission suspended and rejects stale binding installation", async () => {
  const f = await fixture();
  f.hold(B);
  const selecting = f.host.select(B);
  await flush();
  f.held.reject(new Error("Transport unavailable"));
  await assert.rejects(selecting, /Transport/);
  await f.host.setProfileSelection(binding);
  assert.equal(f.start().disabled, true);
  await assert.rejects(f.host.select(C), /ownership is unresolved/);
  await assert.rejects(f.ui.client.start(binding, 0, 6));
  assert.equal(
    f.calls.filter((c) => c.url === "/api/profiles/start").length,
    0,
  );
});
test("release holds the same admission barrier and clears binding after success", async () => {
  const f = await fixture();
  f.holdRelease();
  const releasing = f.host.release();
  await flush();
  assert.equal(f.start().disabled, true);
  await assert.rejects(f.ui.client.start(binding, 0, 6));
  await f.host.setProfileSelection(binding);
  assert.equal(f.start().disabled, true);
  f.held.resolve(response({ api_version: 1 }));
  await releasing;
  assert.equal(f.host.snapshot(), null);
  assert.equal(f.ui.client.snapshot().enabled, false);
});
test("refused release revalidates the previous selection", async () => {
  const f = await fixture();
  f.holdRelease();
  const releasing = f.host.release();
  await flush();
  f.held.resolve(response({ api_version: 1, error: "Reader busy" }, 409));
  await assert.rejects(releasing, /Reader busy/);
  assert.equal(f.host.snapshot().model_id, A);
  assert.equal(f.start().disabled, false);
});
test("source preparation invalidates late binding callbacks until a fresh binding verifies", async () => {
  const f = await fixture(),
    held = f.holdBinding();
  const old = f.host.setProfileSelection(binding);
  await flush();
  await f.host.prepareProfileChange();
  held.resolve(response({ api_version: 1, source_binding: binding }));
  await old;
  assert.equal(f.start().disabled, true);
  await assert.rejects(f.ui.client.start(binding, 0, 6));
  await f.host.setProfileSelection(binding);
  assert.equal(f.start().disabled, false);
});
test("conclusive acquire refusal cannot restore an unverifiable old binding", async () => {
  const f = await fixture();
  f.hold(B, true);
  const selecting = f.host.select(B);
  await flush();
  const held = f.holdBinding();
  f.held.resolve(response({ api_version: 1, error: "Reader busy" }, 409));
  await flush();
  held.resolve(
    response({ api_version: 1, source_binding: { ...binding, tensor: 99 } }),
  );
  await assert.rejects(selecting, /Reader busy/);
  assert.equal(f.start().disabled, true);
  await assert.rejects(f.ui.client.start(binding, 0, 6));
});
test("unknown release cannot enable an obsolete binding", async () => {
  const f = await fixture();
  f.holdRelease();
  const releasing = f.host.release();
  await flush();
  f.held.reject(new Error("Transport unavailable"));
  await assert.rejects(releasing, /Transport/);
  await f.host.setProfileSelection(binding);
  await assert.rejects(f.ui.client.start(binding, 0, 6));
  assert.equal(f.start().disabled, true);
  await assert.rejects(f.host.select(B), /ownership is unresolved/);
});
test("declined tab only clears uncertain admission on authoritative reconciliation", async () => {
  let starts = 0,
    confirmed = false;
  const calls = [];
  const client = Profiles.create({
    post: async (path) => {
      calls.push(path);
      if (path.endsWith("/start")) {
        starts++;
        throw new Error("409 owned by another tab");
      }
      if (!confirmed) throw new Error("Uncertain transport");
      return {
        state: "cancelled",
        cleanup_pending: false,
        resume_available: false,
        no_owned_work: true,
      };
    },
  });
  client.configure(
    { profiles_enabled: true },
    {
      model_id: A,
      context_id: contexts[A].context_id,
      tab_capability: "e".repeat(64),
    },
    binding,
  );
  await assert.rejects(client.start(binding, 0, 6));
  assert.equal(client.snapshot().cleanup_pending, true);
  await assert.rejects(client.poll());
  assert.equal(client.snapshot().cleanup_pending, true);
  await assert.rejects(client.start(binding, 0, 6));
  assert.equal(starts, 1);
  confirmed = true;
  await client.poll();
  assert.equal(client.snapshot().cleanup_pending, false);
  assert.equal(starts, 1);
  await assert.rejects(client.start(binding, 0, 6));
  assert.equal(starts, 2, "Only a later explicit Start retries admission");
});
(async () => {
  for (const { name, fn } of cases) {
    await fn();
    console.log("PASS " + name);
  }
  console.log(cases.length + " source transition checks passed");
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
