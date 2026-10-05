"use strict";
// Selected-status races, with the actual app state machine and fake transport.
const { createFixture } = require("./support/ui-fixture.cjs");
const {
  fs,
  vm,
  assert,
  crypto,
  path,
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
} = createFixture();
(async () => {
  context.fixture = model;
  run("bind();populateModel(fixture,null)");
  const older = run("selectTensor(12)");
  await tick();
  const old = take("/api/tensor-status");
  const newest = run("selectTensor(13)");
  await tick();
  const next = take("/api/tensor-status");
  next.resolve(statusUpdate(catalog[2]));
  await tick();
  await complete(take("/api/view"));
  await newest;
  old.resolve(statusUpdate(catalog[1]));
  await older;
  assert.equal(run("state.current.tensor.id"), 13);
  assert.equal(run("state.tensor.max_abs"), 34);
  assert.equal(
    pending.length,
    0,
    "Late selected status must not start another view",
  );
  const first = run("selectTensor(13)");
  await tick();
  const firstRequest = take("/api/tensor-status");
  const last = run("selectTensor(13)");
  await tick();
  const lastRequest = take("/api/tensor-status");
  lastRequest.resolve(statusUpdate(catalog[2]));
  await tick();
  await complete(take("/api/view"));
  await last;
  const stale = statusUpdate(catalog[2]);
  stale.tensor_status.max_abs = 999;
  firstRequest.resolve(stale);
  await first;
  assert.equal(run("state.tensor.max_abs"), 34);
  assert.equal(pending.length, 0);
  console.log(
    "PASS: older tensor status and same-ID reselection status cannot replace the latest selection or reopen a stale view. Pure DOM/transport only.",
  );
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
