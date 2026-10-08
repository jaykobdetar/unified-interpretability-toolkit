"use strict";

const assert = require("node:assert/strict"),
  vm = require("node:vm"),
  { enableModules, load } = require("./support/comparison-controller.cjs");
enableModules(__filename);
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    const document = {},
      context = vm.createContext({ document });
    await load(
      "const legacy = 7; function initialize() { throw Error('startup'); } initialize();",
      context,
    );
    assert.equal(vm.runInContext("legacy", context), 7);
    await load(
      "export const state = { value: 8 }; if (typeof document !== 'undefined') throw Error('startup');",
      context,
    );
    assert.equal(context.state.value, 8);
    assert.equal(context.document, document);
    await assert.rejects(
      load("export const broken = missing;", context),
      /missing/,
    );
    assert.equal(context.document, document);
    await assert.rejects(
      load("export { value } from './unexpected.js';", context),
      /Unexpected comparison dependency/,
    );
    assert.equal(context.document, document);
    console.log(
      "PASS: classic/module controller isolation, errors and document restoration",
    );
  })(),
);
