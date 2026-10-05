"use strict";
const assert = require("node:assert/strict");
const cases = [];
const test = (name, fn) => cases.push({ name, fn });
const { create } = require("../web/profile-client.js");
const context = {
  model_id: "m_" + "a".repeat(64),
  context_id: "context",
  tab_capability: "b".repeat(64),
};
const binding = { rows: 3, cols: 4 };
const admitted = {
  ...context,
  tab_capability: undefined,
  job_id: "c".repeat(32),
  job_capability: "d".repeat(64),
  state: "partial",
  accepted: { revision: "e".repeat(64), visited_values: 5, total_values: 12 },
  cleanup_pending: false,
  resume_available: false,
};
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => {
    resolve = a;
    reject = b;
  });
  return { promise, resolve, reject };
};
function setup(post) {
  const c = create({ post });
  c.configure({ profiles_enabled: true }, context, binding);
  return c;
}
test("disabled default never starts a request", async () => {
  let calls = 0;
  const c = create({
    post: () => {
      calls++;
    },
  });
  await assert.rejects(c.start(binding, 0, 1));
  assert.equal(calls, 0);
  assert.equal(c.snapshot().resume_available, false);
});
test("explicit start, accepted paired page, private state projection", async () => {
  const calls = [];
  const c = setup(async (path, data) => {
    calls.push(path);
    return path.endsWith("start")
      ? admitted
      : {
          binding,
          axis: "rows",
          start: 0,
          end: 2,
          revision: admitted.accepted.revision,
          original: [],
          control: [],
        };
  });
  await c.start(binding, 0, 5);
  assert.equal(c.snapshot().state, "partial");
  assert.ok(!JSON.stringify(c.snapshot()).includes(admitted.job_capability));
  await c.page("rows", 0, 2);
  await assert.rejects(c.start(binding, 0, 5));
  assert.equal(calls.length, 2);
});
test("late admission after context change cancels exact old owner", async () => {
  const d = deferred(),
    calls = [];
  const c = setup(async (p, b) => {
    calls.push({ p, b });
    return p.endsWith("start") ? d.promise : { state: "stopping" };
  });
  const starting = c.start(binding, 0, 5);
  c.configure(
    { profiles_enabled: true },
    { ...context, context_id: "next" },
    binding,
  );
  d.resolve(admitted);
  await assert.rejects(starting, { name: "AbortError" });
  assert.equal(calls[1].p, "/api/profiles/cancel");
  assert.equal(calls[1].b.context_id, "context");
  assert.equal(c.snapshot().cleanup_pending, true);
  await assert.rejects(c.start(binding, 0, 5));
});
test("lost admission remains uncertain and blocks automatic replacement", async () => {
  const d = deferred();
  const c = setup(() => d.promise);
  const p = c.start(binding, 0, 5);
  c.configure(
    { profiles_enabled: true },
    { ...context, context_id: "next" },
    binding,
  );
  d.reject(new Error("transport lost"));
  await assert.rejects(p);
  assert.equal(c.snapshot().cleanup_pending, true);
  await assert.rejects(c.start(binding, 0, 5));
});
test("cancel stays busy until matching status confirms cleanup", async () => {
  let cleaned = false;
  const c = setup(async (p) =>
    p.endsWith("start")
      ? admitted
      : {
          ...admitted,
          job_capability: undefined,
          state: cleaned ? "cancelled" : "stopping",
          cleanup_pending: !cleaned,
        },
  );
  await c.start(binding, 0, 5);
  await c.cancel();
  await c.poll();
  assert.equal(c.snapshot().cleanup_pending, true);
  cleaned = true;
  await c.poll();
  assert.equal(c.snapshot().cleanup_pending, false);
  assert.equal(c.snapshot().accepted, null);
});
test("stale page after selection change refuses", async () => {
  const d = deferred();
  const c = setup(async (p) =>
    p.endsWith("start") ? admitted : p.endsWith("page") ? d.promise : {},
  );
  await c.start(binding, 0, 5);
  const p = c.page("rows", 0, 2);
  c.configure({ profiles_enabled: true }, context, { rows: 4, cols: 3 });
  d.resolve({
    binding,
    axis: "rows",
    start: 0,
    end: 2,
    revision: admitted.accepted.revision,
  });
  await assert.rejects(p, { name: "AbortError" });
});
test("foreign cleanup acknowledgment cannot unlock client", async () => {
  const c = setup(async (p) =>
    p.endsWith("start")
      ? admitted
      : { ...admitted, job_id: "f".repeat(32), state: "cancelled" },
  );
  await c.start(binding, 0, 5);
  await c.cancel();
  await assert.rejects(c.poll());
  assert.equal(c.snapshot().cleanup_pending, true);
});

test("lost response reconciles existing tab ownership without replaying start", async () => {
  const calls = [];
  const c = setup(async (p) => {
    calls.push(p);
    if (p.endsWith("start")) throw new Error("lost");
    return {
      state: "cancelled",
      cleanup_pending: false,
      resume_available: false,
    };
  });
  await assert.rejects(c.start(binding, 0, 5));
  await c.poll();
  assert.equal(c.snapshot().cleanup_pending, false);
  assert.deepEqual(calls, ["/api/profiles/start", "/api/profiles/reconcile"]);
});
test("heartbeat only carries existing private ownership", async () => {
  const calls = [];
  const c = setup(async (p, b) => {
    calls.push({ p, b });
    return admitted;
  });
  await c.start(binding, 0, 5);
  await c.heartbeat();
  assert.equal(calls[1].p, "/api/profiles/heartbeat");
  assert.equal(calls[1].b.job_capability, admitted.job_capability);
  assert.ok(!("values" in calls[1].b));
});
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    for (const { name, fn } of cases) {
      await fn();
      console.log("PASS " + name);
    }
    assert.equal(cases.length, 9);
    console.log("9 client checks passed");
  })().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  }),
);
