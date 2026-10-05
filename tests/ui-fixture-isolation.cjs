"use strict";
const assert = require("node:assert/strict");
const app = require("./support/ui-fixture.cjs"),
  inference = require("./support/inference-fixture.cjs");
const first = app.createFixture(),
  second = app.createFixture();
first.get("row").value = "9";
first.catalog[0].max_abs = 99;
first.run("state.viewEpoch=27");
assert.equal(second.get("row").value, "");
assert.equal(second.catalog[0].max_abs, 0.5);
assert.equal(second.run("state.viewEpoch"), 0);
assert.equal(first.pending.length, 0);
assert.equal(second.pending.length, 0);
const a = inference.createFixture(),
  b = inference.createFixture();
a.get("infer-rate").value = "9";
a.take("/api/inference");
assert.equal(b.get("infer-rate").value, "2");
assert.equal(a.requests.length, 0);
assert.equal(b.requests.length, 1);
assert.notEqual(a.timers, b.timers);
assert.notEqual(a.context, b.context);
console.log(
  "PASS: explicit test factories allocate independent DOM, catalog, VM, transport, timer and ownership state; requiring a factory runs no scenario.",
);
