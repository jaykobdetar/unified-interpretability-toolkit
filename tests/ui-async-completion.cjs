"use strict";
const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const helper = require.resolve("./support/async-completion.cjs");
const pending = "new Promise(() => {})";
const faults = [
  ["lost direct settlement", pending],
  ["lost chained settlement", `Promise.resolve().then(() => ${pending})`],
  ["lost nested async settlement", `(async () => { await ${pending}; })()`],
  [
    "one unsettled parallel branch",
    `Promise.all([Promise.resolve(), ${pending}])`,
  ],
  ["empty race", "Promise.race([])"],
  ["unsettled final cleanup", `Promise.resolve().finally(() => ${pending})`],
  [
    "unsettled recovery",
    `Promise.reject(Error('held')).catch(() => ${pending})`,
  ],
  [
    "microtask drops resolver",
    "new Promise(resolve => queueMicrotask(() => {}))",
  ],
  ["callback drops resolver", "new Promise(resolve => setImmediate(() => {}))"],
  [
    "event callback drops resolver",
    "new Promise(resolve => { const e = new (require('node:events').EventEmitter)(); e.once('done', () => {}); e.emit('done'); })",
  ],
];
function child(expression, tracked) {
  const main = `(${expression}).then(() => console.log('completed')).catch(error => { console.error(error); process.exitCode = 1; })`;
  const script = tracked
    ? `require(${JSON.stringify(helper)}).requireCompletion(${main});`
    : `${main};`;
  const result = spawnSync(process.execPath, ["-e", script], {
    encoding: "utf8",
    timeout: 5000,
  });
  assert.ifError(result.error);
  assert.equal(result.signal, null, "a timeout is not a correctness catch");
  return {
    status: result.status,
    stdout: result.stdout,
    stderr: result.stderr,
  };
}
const rows = [];
for (const [name, expression] of faults) {
  const old = child(expression, false);
  const current = child(expression, true);
  assert.equal(old.status, 0, name);
  assert.equal(old.stdout, "", "legacy exit occurs before completion");
  assert.equal(current.status, 1, name);
  assert.match(current.stderr, /main promise completed/);
  rows.push({ name, old_caught: false, new_caught: true, old, current });
}
for (const expression of [
  "Promise.resolve()",
  "new Promise(resolve => setImmediate(resolve))",
]) {
  for (const tracked of [false, true]) {
    const result = child(expression, tracked);
    assert.equal(result.status, 0);
    assert.equal(result.stdout, "completed\n");
    assert.equal(result.stderr, "");
  }
}
const rejection = child(
  "Promise.reject(Error('guarded assertion failed'))",
  true,
);
assert.equal(rejection.status, 1);
assert.match(rejection.stderr, /guarded assertion failed/);
assert(!rejection.stderr.includes("main promise completed"));
console.log(
  JSON.stringify({
    faults: rows,
    completed_controls: 4,
    rejected_control: rejection,
  }),
);
console.log(
  "PASS: ten premature successful exits are caught; completed and rejected controls retain their results",
);
