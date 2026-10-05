// Small transport/DOM doubles only: no browser, listener, model, or subprocess.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.value = "";
    this.textContent = "";
    this.style = {};
  }
  append(...items) {
    this.children.push(...items);
  }
  after(node) {
    this.afterNode = node;
  }
  setAttribute() {}
  replaceChildren(...items) {
    this.children = items;
  }
  querySelector() {
    return { value: "1" };
  }
}
const main = new Element("main"),
  events = {},
  timers = new Map(),
  sent = [],
  starts = [];
let timerId = 0,
  captured,
  selected = {
    id: 0,
    name: "fixture A",
    dtype: "BF16",
    shape: [4, 4],
    rows: 4,
    cols: 4,
  };
const report = { schema: "weight-atlas.analytics.v1", tensor: "fixture A" };
let cancelCount = 0,
  pollCount = 0;
const response = (payload) =>
  Promise.resolve({ ok: true, json: () => Promise.resolve(payload) });
const sandbox = {
  document: {
    createElement: (tag) => new Element(tag),
    querySelector: () => main,
    body: new Element("body"),
  },
  window: {
    atlasAnalyticsBridge: {
      selected: () => selected,
      jump: (data) => sent.push(data),
    },
    addEventListener: (name, fn) => (events[name] = fn),
  },
  mountAnalytics: (root, handlers) => {
    captured = handlers;
    return { setReport: () => {} };
  },
  renderModelOutliers: () => {},
  fetch: (path, options) => {
    if (!options) return response({ available: true });
    const data = options.body ? JSON.parse(options.body) : null;
    if (path.endsWith("/start"))
      return new Promise((resolve) => starts.push({ resolve, data }));
    if (path.endsWith("/cancel")) {
      cancelCount++;
      return response({ status: "cancelled" });
    }
    if (path.endsWith("/poll")) {
      pollCount++;
      return response({ status: "complete", result: report });
    }
    return response({ available: true });
  },
  AbortController,
  setTimeout: (fn, ms) => {
    timers.set(++timerId, { fn, ms });
    return timerId;
  },
  clearTimeout: (id) => timers.delete(id),
  console,
};
vm.createContext(sandbox);
let source = fs
  .readFileSync(
    new URL("../../web/analytics-mount.js", import.meta.url),
    "utf8",
  )
  .replace(
    /^import\s*\{[\s\S]*?\}\s*from\s*["']\.\/analytics-panel\.js["'];?\s*(?:\r?\n|$)/,
    "",
  );
vm.runInContext(source, sandbox);
const settle = async () => {
  for (let i = 0; i < 20; i++) await Promise.resolve();
};
await settle();
const first = captured.load({ seed: 7, svd: false });
const rejection = assert.rejects(first, /superseded|cancelled/);
await settle();
assert.equal(starts.length, 1);
selected = { ...selected, id: 1, name: "fixture B" };
events["atlas:tensor"]({ detail: selected });
starts.shift().resolve({
  ok: true,
  json: () => Promise.resolve({ job: "first-owner", status: "running" }),
});
await rejection;
assert.ok(cancelCount >= 1);
assert.equal(pollCount, 0);
const second = captured.load({ seed: 9, svd: false });
await settle();
assert.equal(starts[0].data.tensor, 1);
assert.equal(starts[0].data.seed, 9);
starts.shift().resolve({
  ok: true,
  json: () => Promise.resolve({ job: "second-owner", status: "running" }),
});
await settle();
for (const [id, t] of [...timers])
  if (t.ms === 200) {
    timers.delete(id);
    t.fn();
  }
assert.equal(await second, report);
assert.equal(pollCount, 1);
captured.jump({ axis: "row", index: 2 });
assert.equal(sent[0].index, 2);
const third = captured.load({ seed: 1, svd: false });
await settle();
const blocked = captured.load({ seed: 2, svd: false });
await assert.rejects(blocked, /active/);
starts.shift().resolve({
  ok: true,
  json: () => Promise.resolve({ job: "third-owner", status: "running" }),
});
await settle();
events.pagehide();
await settle();
const thirdRejected = assert.rejects(third, /superseded|cancelled/);
for (const [id, t] of [...timers])
  if (t.ms === 200) {
    timers.delete(id);
    t.fn();
  }
await thirdRejected;
selected = { ...selected, dtype: "F32" };
assert.throws(() => captured.load({ seed: 1, svd: false }), /BF16/);
assert.equal(starts.length, 0);
selected = {
  ...selected,
  dtype: "BF16",
  shape: [256, 256],
  rows: 256,
  cols: 256,
};
events["atlas:tensor"]({ detail: selected });
const summaryRequest = captured.load({ seed: 77, scope: "svd_summary" });
await settle();
assert.deepEqual(Object.keys(starts[0].data).sort(), [
  "region",
  "scope",
  "seed",
  "tensor",
]);
assert.equal(starts[0].data.scope, "svd_summary");
starts.shift().resolve({
  ok: true,
  json: () =>
    Promise.resolve({
      job: "summary-owner",
      status: "complete",
      result: report,
    }),
});
await summaryRequest;
const fieldControls = main.afterNode.children.find(
  (x) => x.className === "analytics-controls",
);
// This double stores className exactly as assigned by the production mount.
const rowsInput = fieldControls.children[2].children[0];
rowsInput.value = 129;
assert.throws(() => captured.load({ seed: 77, scope: "svd_summary" }), /128/);
assert.equal(starts.length, 0);
console.log(
  "PASS: mount selected-window request, late admission cancellation, one pending operation, native jump, pagehide ownership cleanup",
);
