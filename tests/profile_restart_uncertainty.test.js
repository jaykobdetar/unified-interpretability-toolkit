"use strict";
const assert = require("node:assert/strict");
const { create } = require(process.argv[2] || "../web/profile-client.js");
const context = {
  model_id: "m_" + "1".repeat(64),
  context_id: "original",
  tab_capability: "2".repeat(64),
};
const binding = { rows: 3, cols: 4 };
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => {
    resolve = a;
    reject = b;
  });
  return { promise, resolve, reject };
};
const response = (n) => ({
  ...context,
  job_id: String(n).repeat(32),
  job_capability: String(n).repeat(64),
  state: "complete",
  accepted: { revision: "9".repeat(64), visited_values: 12, total_values: 12 },
  cleanup_pending: false,
  resume_available: false,
});
function setup() {
  const replacement = deferred(),
    calls = [];
  let starts = 0,
    current = null;
  const client = create({
    post: async (path, data) => {
      const action = path.split("/").at(-1);
      calls.push({ action, data });
      if (action === "start") {
        current = {
          ...response(++starts),
          model_id: data.model_id,
          context_id: data.context_id,
        };
        return starts === 2 ? replacement.promise : current;
      }
      if (action === "reconcile") {
        assert.ok(
          data.context_id === context.context_id &&
            data.tab_capability === context.tab_capability,
          "Reconcile captured original owner",
        );
        assert.ok(
          !("job_capability" in data),
          "No obsolete job capability in reconciliation",
        );
        current = null;
        return {
          state: "cancelled",
          cleanup_pending: false,
          resume_available: false,
        };
      }
      if (action === "cancel") {
        assert.ok(
          current && data.job_capability === current.job_capability,
          "Cancel exact replacement owner",
        );
        current = { ...current, state: "cancelled", accepted: null };
        return current;
      }
      assert.ok(
        current && data.job_capability === current.job_capability,
        "Never use obsolete owner",
      );
      return current;
    },
  });
  client.configure({ profiles_enabled: true }, context, binding);
  return { client, replacement, calls };
}
const cases = [];
function test(name, fn) {
  cases.push({ name, fn });
}
async function lostRestart(mode) {
  const { client, replacement, calls } = setup();
  await client.start(binding, 0, 12);
  const pending = client.start(binding, 0, 12, { restart: true });
  const observed = assert.rejects(pending);
  if (mode === "selection")
    client.configure(
      { profiles_enabled: true },
      { ...context, context_id: "next" },
      binding,
    );
  if (mode === "cancel") await client.cancel();
  replacement.reject(new Error("Lost accepted replacement response"));
  await observed;
  assert.equal(
    client.snapshot().cleanup_pending,
    true,
    "Uncertain restart requires cleanup",
  );
  assert.equal(
    client.snapshot().accepted,
    null,
    "Old accepted revision must be invalidated",
  );
  await assert.rejects(client.page("rows", 0, 1));
  await assert.rejects(client.start(binding, 0, 12, { restart: true }));
  await client.heartbeat();
  assert.equal(
    calls.filter((c) => c.action === "start").length,
    2,
    "No third admission before reconciliation",
  );
  assert.equal(
    calls.filter((c) =>
      ["page", "status", "heartbeat", "cancel"].includes(c.action),
    ).length,
    0,
    "No obsolete capability requests",
  );
  await client.poll();
  assert.equal(calls.at(-1).action, "reconcile");
  assert.equal(client.snapshot().cleanup_pending, false);
  await client.start(binding, 0, 12); // No stale job forcing Restart after cleanup.
  assert.equal(calls.filter((c) => c.action === "start").length, 3);
}
test("lost explicit Restart response reconciles and clears obsolete ownership", () =>
  lostRestart("none"));
test("selection change during lost Restart reconciles captured original owner", () =>
  lostRestart("selection"));
test("cancel during lost Restart reconciles replacement rather than old job", () =>
  lostRestart("cancel"));
test("cancel during pending Restart with late success cancels exact replacement", async () => {
  const { client, replacement, calls } = setup();
  await client.start(binding, 0, 12);
  const pending = client.start(binding, 0, 12, { restart: true });
  const observed = assert.rejects(pending, { name: "AbortError" });
  await client.cancel();
  replacement.resolve(response(2));
  await observed;
  const cancelled = calls.filter((c) => c.action === "cancel");
  assert.equal(cancelled.length, 1);
  assert.ok(
    cancelled[0].data.job_capability === response(2).job_capability,
    "Only replacement capability cancelled",
  );
  await client.poll();
  assert.equal(client.snapshot().cleanup_pending, false);
  await client.start(binding, 0, 12);
});
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    let failed = 0;
    for (const { name, fn } of cases) {
      try {
        await fn();
        console.log("PASS " + name);
      } catch (e) {
        failed++;
        console.error("FAIL " + name + ": " + e.message);
      }
    }
    console.log(
      `${cases.length - failed}/${cases.length} restart uncertainty cases passed`,
    );
    process.exitCode = failed ? 1 : 0;
  })(),
);
