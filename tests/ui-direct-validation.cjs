"use strict";
// Direct validation assertions complete without awaiting a viewer or transport reply.
const { createFixture } = require("./support/ui-fixture.cjs");
const {
  assert,
  context,
  run,
  copy,
  get,
  pending,
  model,
  currentSettings,
  view,
} = createFixture();
context.fixture = model;
run("populateModel(fixture,{tensor:11})");
context.validView = view(currentSettings());
run("validateView(validView,settings())");
for (const side of ["left", "right"]) {
  context.wrongView = copy(context.validView);
  context.wrongView.legends[side].id = "unexpected";
  assert.throws(
    () => run("validateView(wrongView,settings())"),
    /legend rule does not match/,
  );
}
for (const [row, col] of [
  [0.1, 0],
  [0, 0.1],
  [-0.1, 0],
  [0, -0.1],
]) {
  context.row = row;
  context.col = col;
  const before = pending.length;
  run("inspectAt(row,col)");
  assert.equal(
    pending.length,
    before,
    "Fractional addresses must synchronously queue no source read",
  );
  assert(get("error").textContent.includes("whole-number row and column"));
}
assert.equal(run("state.current"), null);
assert.equal(run("state.pendingInspection"), null);
console.log(
  "PASS: both mismatched legend identities and fractional row/column addresses are rejected directly; no unresolved viewer/transport wait can mask these assertions.",
);
