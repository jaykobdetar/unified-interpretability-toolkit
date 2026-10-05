"use strict";
const assert = require("node:assert/strict"),
  codec = require("../web/inference.js"),
  I = codec.AtlasExperimentImport;
const request = {
    prompt: "private fixture",
    source_model: {
      repo: "fixture",
      revision: "fixture",
      weights_sha256: "a".repeat(64),
    },
    layer: 7,
    activation_site: "mlp",
    max_new_tokens: 1,
    edits: [],
  },
  snapshot = {
    status: "complete",
    worker_alive: false,
    steps: [
      {
        index: 0,
        activation: Array(576).fill(0.5),
        input_token_id: 42,
        token_id: 2,
        generated_text: "generated fixture",
        top_logits: [{ id: 2, value: 1 }],
      },
    ],
    details: { prompt_ids: [1, 42], runtime: { python: "fixture" } },
  };
const privateRecord = codec.experimentRecord(request, snapshot, {
    includePrompt: true,
  }),
  plain = codec.experimentRecord(request, snapshot),
  text = JSON.stringify(privateRecord),
  read = (s, opts = {}) => I.read(s, { ...codec, ...opts });
assert.deepEqual(read(text), [codec.redactExperiment(privateRecord)]);
assert.deepEqual(read(text, { includePrompt: true }), [privateRecord]);
assert.deepEqual(read(JSON.stringify(plain)), [plain]);
const log = new codec.AtlasExperimentLog();
log.append(plain);
const before = log.text();
const incoming = read(text);
assert.equal(I.append(log, incoming), 1);
incoming[0].steps[0].activation[0] = 9;
assert.equal(log.records[1].steps[0].activation[0], 0.5);
assert.deepEqual(read(log.text()), log.records);
for (const mutation of [
  (r) => (r.session = "secret"),
  (r) => (r.request.path = "/private"),
  (r) => (r.steps[0].source_directory = "/private"),
  (r) => (r.complete = false),
  (r) => (r.status = "running"),
  (r) => r.steps[0].activation.push(0.5),
  (r) => (r.steps[0].top_logits = Array(11).fill({ id: 2, value: 1 })),
  (r) => (r.steps[0].activation[0] = null),
  (r) => (r.settings.seed = 10),
  (r) => (r.steps[0].attention = { probabilities: [2], key_positions: [0] }),
  (r) => (r.request.unknown = true),
]) {
  const r = JSON.parse(text);
  mutation(r);
  assert.throws(() => read(JSON.stringify(r)));
}
assert.throws(
  () =>
    read(
      text.replace(
        '"status":"complete"',
        '"status":"error","status":"complete"',
      ),
    ),
  /Duplicate/,
);
assert.throws(() => read('{"__proto__":{}}'), /Unsafe/);
assert.throws(() => read("[".repeat(25) + "0" + "]".repeat(25)), /nesting/);
assert.throws(() => read(" ".repeat(I.MAX_BYTES + 1)), /1 MiB/);
const tooMany = new codec.AtlasExperimentLog({ maxRuns: 1 });
tooMany.append(plain);
const prior = tooMany.text();
assert.throws(() => I.append(tooMany, read(text)), /unchanged/);
assert.equal(tooMany.text(), prior);
const small = new codec.AtlasExperimentLog({ maxBytes: 100 });
assert.throws(() => I.append(small, [plain]));
assert.equal(small.records.length, 0);
for (const status of ["cancelled", "error", "connection_lost"]) {
  const r = codec.experimentRecord(request, {
    ...snapshot,
    status,
    worker_alive: null,
  });
  assert.deepEqual(read(JSON.stringify(r)), [r]);
  assert.equal(r.worker_cleanup_confirmed, false);
}
const pairReq = {
  mode: "prompt_pair",
  source_model: request.source_model,
  layer: 7,
  activation_site: "block",
  prompts: ["a", "b"],
  positions: [{ a: 0, b: 0 }],
  preview_digest: "b".repeat(64),
};
const pair = codec.experimentRecord(
  pairReq,
  {
    ...snapshot,
    steps: [
      {
        index: 0,
        activation: [0, 0],
        prompt_pair: {
          a: {
            position: 0,
            token_id: 10,
            token_piece: "a",
            activation: [1, 2],
          },
          b: {
            position: 0,
            token_id: 11,
            token_piece: "b",
            activation: [1, 2],
          },
          token_equal: false,
          prefix_equal: false,
          metrics: { a_l2: 1, b_l2: 1, delta_l2: 0, cosine: 1 },
        },
      },
    ],
  },
  { includePrompt: true },
);
assert.deepEqual(read(JSON.stringify(pair)), [codec.redactExperiment(pair)]);
// Semantic rejection is atomic: neither the existing log nor external sinks change.
const sink = { text: "unchanged" },
  reject = (mutate) => {
    const r = JSON.parse(text);
    mutate(r);
    const original = log.text();
    assert.throws(() => {
      const incoming = read(JSON.stringify(r));
      I.append(log, incoming, {
        beforeCommit: () => {
          sink.text = "must not commit";
        },
      });
    });
    assert.equal(log.text(), original);
    assert.equal(sink.text, "unchanged");
  };
for (const mutation of [
  (r) => (r.request.max_new_tokens = -1),
  (r) => (r.request.max_new_tokens = 0),
  (r) => (r.request.max_new_tokens = 33),
  (r) => (r.request.max_new_tokens = true),
  (r) => (r.runtime.python = 2),
  (r) => (r.runtime.deterministic_algorithms = "true"),
  (r) => (r.request.layer = 30),
  (r) => (r.request.activation_site = "unknown"),
  (r) => (r.request.prompt_ids = Array(129).fill(0)),
  (r) => (r.request.prompt_ids = [true]),
  (r) => (r.request.prompt = 7),
  (r) => (r.created_at = "2026-02-30T00:00:00.000Z"),
  (r) => (r.baseline.generated_ids = Array(576).fill(1)),
  (r) => (r.baseline.generated_ids = [49152]),
  (r) => (r.baseline.generated_ids = [false]),
  (r) => (r.summary.generated_tokens = 2),
  (r) => (r.summary.compute_total_ms = -1),
  (r) => (r.steps[0].token_id = -1),
  (r) => (r.steps[0].top_logits[0].id = "2"),
  (r) => (r.steps[0].top_logits[0].value = "1"),
  (r) => delete r.steps[0].top_logits[0].value,
  (r) => (r.steps[0].eos = 1),
  (r) => (r.steps[0].generated_text = 5),
  (r) => (r.steps[0].layer = 6),
  (r) => (r.steps[0].mode = "prompt_pair"),
  (r) =>
    (r.request.edits = [
      {
        tensor: "fixture",
        shape: [1, 1],
        kind: "element",
        operation: "zero",
        row: 100,
        col: 100,
      },
    ]),
  (r) =>
    (r.request.edits = [
      {
        tensor: "fixture",
        shape: [2, 2],
        kind: "rows",
        operation: "zero",
        start: 0,
        end: 3,
      },
    ]),
  (r) =>
    (r.request.edits = [
      {
        tensor: "fixture",
        shape: [2, 2],
        kind: "columns",
        operation: "zero",
        start: 1,
        end: 1,
      },
    ]),
  (r) =>
    (r.request.edits = [
      {
        tensor: "fixture",
        shape: [1, 1],
        kind: "element",
        operation: "zero",
        row: 0,
        col: 0,
        scale: 1,
      },
    ]),
])
  reject(mutation);
const session = JSON.parse(before);
session.records = [plain, plain];
for (const mutation of [
  (v) => (v.created_at = "invalid"),
  (v) => (v.created_at = "2026-02-30T00:00:00.000Z"),
  (v) => (v.persistence = "automatic"),
  (v) => (v.limits.bytes = 1),
  (v) => (v.limits.runs = 1),
  (v) => (v.limits.runs = true),
]) {
  const v = structuredClone(session);
  mutation(v);
  assert.throws(() => read(JSON.stringify(v)));
}
for (const mutate of [
  (r) => (r.request.positions = []),
  (r) => (r.request.positions[0].a = 128),
  (r) => (r.request.positions[0].b = true),
  (r) => (r.request.preview_digest = "invalid"),
  (r) => (r.steps[0].prompt_pair.a.position = 1),
  (r) => (r.steps[0].prompt_pair.token_equal = 1),
  (r) => (r.steps[0].prompt_pair.metrics.cosine = 2),
  (r) => (r.request.prompts = ["a"]),
  (r) => (r.request.max_new_tokens = 1),
]) {
  const r = structuredClone(pair);
  mutate(r);
  assert.throws(() => read(JSON.stringify(r)));
}
const legacy = structuredClone(plain);
legacy.request.source_model = {};
assert.deepEqual(read(JSON.stringify(legacy)), [legacy]);
console.log(
  "PASS: bounded detached archive import; separate prompt retention, export-contract/schema checks, terminal state, caps/atomic append, duplicate/deep/unknown/ownership rejection and paired-capture redaction; no execution APIs.",
);
