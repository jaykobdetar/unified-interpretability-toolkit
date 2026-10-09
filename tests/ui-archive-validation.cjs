"use strict";
// Public pure codec witnesses: exact first errors and detached export bytes.
const assert = require("node:assert/strict"),
  codec = require("../web/inference.js"),
  importer = require(
    process.env.ATLAS_TEST_IMPORT_CODEC || "../web/inference-import.js",
  );
const record = codec.experimentRecord(
  {
    prompt: "Synthetic archive fixture",
    source_model: {},
    layer: 29,
    activation_site: "mlp",
    max_new_tokens: 1,
    edits: [
      {
        tensor: "fixture",
        shape: [2, 3],
        kind: "element",
        operation: "zero",
        row: 1,
        col: 2,
      },
    ],
  },
  {
    status: "complete",
    worker_alive: false,
    steps: [
      {
        index: 0,
        activation: [0, 0.5],
        token_id: 49151,
        top_logits: [{ id: 49151, value: 1 }],
        compute_ms: 0,
      },
    ],
    details: { prompt_ids: [1, 42], runtime: { python: "fixture" } },
  },
  { includePrompt: true, createdAt: "2026-01-01T00:00:00.000Z" },
);
const read = (value, options = {}) =>
  importer.read(JSON.stringify(value), { ...codec, ...options });
assert.equal(
  JSON.stringify(read(record, { includePrompt: true })[0]),
  JSON.stringify(record),
);
assert.deepEqual(read(record), [codec.redactExperiment(record)]);
const detached = read(record, { includePrompt: true })[0];
detached.steps[0].activation[0] = 9;
assert.equal(record.steps[0].activation[0], 0);

const cases = [
  [
    "envelope before runtime",
    (r) => {
      r.extra = true;
      r.runtime = null;
    },
    "Unknown or invalid experiment fields",
  ],
  [
    "runtime before timestamp",
    (r) => {
      r.runtime = null;
      r.created_at = "invalid";
    },
    "Invalid runtime or summary",
  ],
  [
    "timestamp before terminal state",
    (r) => {
      r.created_at = "invalid";
      r.status = "running";
    },
    "Invalid experiment schema or timestamp",
  ],
  [
    "terminal state before mode",
    (r) => {
      r.status = "running";
      r.request.mode = "other";
    },
    "Only terminal archived runs can be imported",
  ],
  [
    "mode before trace",
    (r) => {
      r.request.mode = "other";
      r.steps = null;
    },
    "Unsupported archived mode",
  ],
  [
    "trace budget before edit cap",
    (r) => {
      r.steps = Array(33).fill(r.steps[0]);
      r.request.edits = Array(9).fill({});
    },
    "Invalid archived trace cap",
  ],
  [
    "activation before edit fields",
    (r) => {
      r.steps[0].activation = [];
      r.request.edits[0].kind = "other";
    },
    "Invalid archived activation",
  ],
  [
    "edit fields before capture",
    (r) => {
      r.request.edits[0].kind = "other";
      r.request.layer = 30;
    },
    "Unknown or invalid experiment fields",
  ],
  [
    "capture before budget",
    (r) => {
      r.request.layer = 30;
      r.request.max_new_tokens = 0;
    },
    "Invalid capture request",
  ],
  [
    "budget before token IDs",
    (r) => {
      r.request.max_new_tokens = 0;
      r.request.prompt_ids = [-1];
    },
    "Invalid generation budget",
  ],
  [
    "token IDs before runtime leaves",
    (r) => {
      r.request.prompt_ids = [-1];
      r.runtime.python = 1;
    },
    "Invalid archived token IDs or budget",
  ],
  [
    "runtime leaves before trace relations",
    (r) => {
      r.runtime.python = 1;
      r.steps[0].layer = 28;
    },
    "Invalid archived python",
  ],
  [
    "trace relations before candidate IDs",
    (r) => {
      r.steps[0].layer = 28;
      r.steps[0].top_logits.push(r.steps[0].top_logits[0]);
    },
    "Trace layer differs",
  ],
  [
    "candidate IDs before consent",
    (r) => {
      r.steps[0].top_logits.push(r.steps[0].top_logits[0]);
      r.privacy.prompt_included = "yes";
    },
    "Invalid candidate IDs",
  ],
  [
    "consent before table shape",
    (r) => {
      r.privacy.prompt_included = "yes";
      r.steps[0].top_logits[0].extra = 1;
    },
    "Invalid consent or termination",
  ],
  [
    "table shape before rebuilt equality",
    (r) => {
      r.steps[0].top_logits[0].extra = 1;
      r.settings.seed = 9;
    },
    "Unknown or invalid experiment fields",
  ],
  [
    "rebuilt equality",
    (r) => {
      r.settings.seed = 9;
    },
    "Archive differs from the supported export contract",
  ],
];
for (const [label, mutate, message] of cases) {
  const bad = structuredClone(record);
  mutate(bad);
  const before = JSON.stringify(bad);
  assert.throws(() => read(bad), { name: "Error", message }, label);
  assert.equal(JSON.stringify(bad), before, label + " mutated its input");
}
console.log(
  `PASS: detached archive bytes, prompt redaction and ${cases.length} exact validation-order witnesses.`,
);
