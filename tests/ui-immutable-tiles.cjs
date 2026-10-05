"use strict";
const fs = require("fs"),
  vm = require("vm"),
  assert = require("assert");
const source = fs.readFileSync("web/app.js", "utf8");
const fn = source.slice(
  source.indexOf("function tileSource("),
  source.indexOf("function createViewer("),
);
const context = {
  URLSearchParams,
  apiURL: (url) => url,
  assert: (test, message) => assert.ok(test, message),
};
vm.createContext(context);
vm.runInContext(fn, context);
const tensor = { id: 4, rows: 128, cols: 256, max_level: 8, slice: [1, 2] };
const make = (rule, binding) =>
  context.tileSource(tensor, rule, binding).getTileUrl(7, 0, 0);
const a = "a".repeat(64),
  b = "b".repeat(64);
const original = make("tensor_linear", a);
assert.strictEqual(make("tensor_linear", a), original);
assert.notStrictEqual(make("tensor_linear", b), original);
assert.notStrictEqual(make("tensor_asinh", a), original);
assert.strictEqual(
  new URLSearchParams(original.split("?")[1]).get("binding"),
  a,
);
assert.strictEqual(
  new URLSearchParams(make("tensor_linear").split("?")[1]).get("binding"),
  null,
);
assert.throws(() => make("tensor_linear", "bad"));
console.log("PASS identity-bearing tile URLs and legacy fallback");
