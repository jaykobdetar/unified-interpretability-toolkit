"use strict";
const assert = require("assert/strict");
const {
  experimentRecord,
  AtlasExperimentLog,
  utf8Size,
  redactExperiment,
} = require("../web/inference.js");
const request = {
  prompt: "private prompt fixture",
  source_model: {
    repo: "synthetic",
    revision: "fixture",
    weights_sha256: "fixture",
    path: "/private/model",
  },
  layer: 7,
  activation_site: "attention",
  max_new_tokens: 2,
  edits: [],
};
const snapshot = {
  session: "SECRET-CAPABILITY",
  worker_alive: false,
  status: "complete",
  source_directory: "/private/model",
  steps: [
    {
      index: 0,
      activation: Array(576).fill(0.25),
      input_token_id: 42,
      token_id: 8,
      token_piece: " result",
      generated_text: " result",
      activation_site: "attention",
      activation_kind: "after o_proj before residual",
      layer: 7,
      top_logits: [{ id: 8, value: 1.25 }],
    },
  ],
  details: {
    prompt_ids: [10, 42],
    runtime: {
      python: "3.12",
      torch: "2.8",
      source_directory: "/private/runtime",
      session: "SECRET-CAPABILITY",
    },
    generated_tokens: 1,
    load_ms: 5,
    reason: "token_limit",
  },
};
const plain = experimentRecord(request, snapshot);
assert(!JSON.stringify(plain).includes("private prompt fixture"));
assert(!JSON.stringify(plain).includes("SECRET-CAPABILITY"));
assert(!JSON.stringify(plain).includes("/private/"));
assert(!Object.hasOwn(plain.request, "prompt_ids"));
assert(!Object.hasOwn(plain.steps[0], "input_token_id"));
assert.equal(plain.privacy.request_replayable, false);
assert.equal(plain.worker_cleanup_confirmed, true);
const privateRecord = experimentRecord(request, snapshot, {
  includePrompt: true,
});
assert.equal(privateRecord.request.prompt, request.prompt);
assert.deepEqual(privateRecord.request.prompt_ids, [10, 42]);
assert.equal(privateRecord.steps[0].input_token_id, 42);
assert.equal(privateRecord.privacy.request_replayable, true);
const redacted = redactExperiment(privateRecord);
assert(!JSON.stringify(redacted).includes("private prompt fixture"));
assert(!Object.hasOwn(redacted.request, "prompt_ids"));
assert(!Object.hasOwn(redacted.steps[0], "input_token_id"));
assert.equal(privateRecord.request.prompt, "private prompt fixture");
for (const status of [
  "cancelled",
  "error",
  "resource_limit",
  "time_limit",
  "connection_lost",
]) {
  const record = experimentRecord(request, {
    ...snapshot,
    status,
    worker_alive: null,
  });
  assert.equal(record.complete, false);
  assert.equal(record.status, status);
  assert.equal(record.worker_cleanup_confirmed, false);
}
const log = new AtlasExperimentLog({ maxRuns: 2 });
log.append(plain);
log.append(privateRecord);
const prior = log.text();
assert.throws(() => log.append(plain), /full/);
assert.equal(log.text(), prior);
assert.equal(log.records.length, 2);
plain.request.layer = 29;
assert.equal(log.records[0].request.layer, 7);
log.clear();
assert.equal(log.records.length, 0);
const small = new AtlasExperimentLog({ maxBytes: 100 });
assert.throws(() => small.append(privateRecord), /full/);
assert.equal(small.records.length, 0);
assert.throws(() =>
  experimentRecord(request, {
    ...snapshot,
    steps: Array(33).fill(snapshot.steps[0]),
  }),
);
assert.throws(
  () =>
    experimentRecord(request, {
      ...snapshot,
      steps: [
        { ...snapshot.steps[0], activation: [NaN, ...Array(575).fill(0)] },
      ],
    }),
  /Nonfinite/,
);
assert.throws(
  () =>
    experimentRecord(
      { ...request, prompt: "a".repeat(1024 * 1024) },
      snapshot,
      { includePrompt: true },
    ),
  /1 MiB/,
);
assert.equal(utf8Size("A😀é"), 7);
console.log(
  JSON.stringify(
    {
      status: "PASS",
      checks: 23,
      scope:
        "Opt-in prompt data, explicit redaction, capability/path exclusion, exact capture provenance, incomplete statuses, detached snapshots, run/byte caps with no silent drop, nonfinite refusal and UTF-8 size",
    },
    null,
    2,
  ),
);
