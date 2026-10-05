"use strict";
const architecture = require("./fixtures/inference-architecture.json");
const fs = require("fs"),
  vm = require("vm"),
  assert = require("assert/strict");
const {
    sweepRequest,
    experimentRecord,
    redactExperiment,
  } = require("../web/inference.js"),
  fixture = require("./fixtures/sweep-plan-v2.json");
const { request, plan } = fixture,
  source_model = request.source_model;
const values = {
  targets: "offset 0 0 0",
  prompts: ["public synthetic"],
  seed: "7",
  layer: "7",
  site: "block",
  operation: "zero",
  scale: ".5",
};
assert.deepEqual(sweepRequest(values, source_model, architecture), request);
for (const update of [
  { targets: "head 0 9" },
  { targets: "offset 0 0,0 3" },
  { targets: "head 0 0\nhead 0 0" },
  { targets: "head 0 0\nhead 0 1\nhead 0 2" },
  { seed: true },
  { seed: 4294967296 },
  { operation: "scale", scale: true },
  { operation: "scale", scale: Infinity },
])
  assert.throws(() =>
    sweepRequest({ ...values, ...update }, source_model, architecture),
  );
const ids = plan.cases.map((c) => c.id + "/prompt-1"),
  coverage = {
    planned_ids: ids,
    completed_ids: ids.slice(0, 2),
    unrun_ids: ids.slice(2),
    interrupted_id: null,
    complete: false,
  };
function step(index) {
  const empty = index === 0;
  return {
    index,
    mode: "sweep",
    phase: "fixed_context_probe",
    layer: 7,
    activation_site: "block",
    activation_kind: "fixture selected block",
    position: 2,
    input_token_id: 123 + index,
    activation: Array(576).fill(0.25),
    compute_ms: 1,
    compute_total_ms: index + 1,
    sweep: {
      record_id: ids[index],
      case_id: plan.cases[index].id,
      role: plan.cases[index].role,
      prompt_index: 0,
      selected_cells: empty ? 0 : 576,
      changed_cells: empty ? 0 : 500,
      parameter_delta_l2: empty ? 0 : 1,
      restoration_verified: true,
      metrics: {
        logit_delta_rms: empty ? 0 : 0.1,
        logit_delta_max_abs: empty ? 0 : 1,
        softmax_total_variation: empty ? 0 : 0.2,
        baseline_argmax_id: 1,
        baseline_argmax_logit_delta: empty ? 0 : -1,
        edited_argmax_id: 1,
        context: "matched fixed original prompt",
        semantics:
          "prompt-set sensitivity; no inferred causal purpose or general head importance",
      },
      candidates: [
        {
          id: 1,
          piece: "candidate",
          baseline_logit: 2,
          edited_logit: empty ? 2 : 1,
          delta: empty ? 0 : -1,
        },
      ],
    },
  };
}
const accepted = { ...request, plan_digest: plan.digest },
  snapshot = {
    status: "time_limit",
    worker_alive: false,
    steps: [step(0), step(1)],
    details: {
      sweep_plan: plan,
      sweep_coverage: coverage,
      reason: "remaining total budget below cleanup margin",
    },
  };
const omitted = experimentRecord(accepted, snapshot),
  included = experimentRecord(accepted, snapshot, { includePrompt: true });
assert(!Object.hasOwn(omitted.request, "prompts"));
assert(!Object.hasOwn(omitted.request, "plan_digest"));
assert(!Object.hasOwn(omitted.sweep_plan, "digest"));
assert(omitted.steps.every((s) => !Object.hasOwn(s, "input_token_id")));
assert.equal(omitted.summary.sweep_coverage.unrun_ids.length, 1);
assert(!omitted.complete);
assert.deepEqual(included.request.prompts, request.prompts);
assert(included.privacy.request_replayable);
assert.equal(included.steps[1].input_token_id, 124);
const redacted = redactExperiment(included);
assert(redacted.steps.every((s) => !Object.hasOwn(s, "input_token_id")));
assert(!Object.hasOwn(redacted.request, "plan_digest"));
class Element {
  constructor() {
    this.value = "";
    this.textContent = "";
    this.disabled = false;
    this.hidden = false;
    this.checked = false;
    this.listeners = {};
    this.children = [];
    this.width = 512;
    this.height = 288;
  }
  replaceChildren(...v) {
    this.children = v;
  }
  append(...v) {
    this.children.push(...v);
  }
  addEventListener(k, f) {
    this.listeners[k] = f;
  }
  getContext() {
    return { clearRect() {}, fillRect() {} };
  }
}
const elements = new Map(),
  requests = [],
  get = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  },
  el = (id) => get("infer-" + id);
for (const [id, value] of Object.entries({
  prompt: "public synthetic",
  layer: "7",
  site: "block",
  task: "generation",
  rate: "2",
  mode: "step",
  limit: "2",
  observation: "none",
  "sweep-targets": "offset 0 0 0",
  "sweep-operation": "zero",
  "sweep-scale": ".5",
  "sweep-seed": "7",
}))
  el(id).value = value;
const window = { addEventListener() {} };
const context = vm.createContext({
  console,
  document: { getElementById: get, createElement: () => new Element() },
  window,
  fetch: (url, options) =>
    new Promise((resolve) =>
      requests.push({
        url,
        options,
        resolve: (body, status = 200) =>
          resolve({ ok: status < 400, status, json: async () => body }),
      }),
    ),
  setTimeout: () => 1,
  clearTimeout() {},
});
vm.runInContext(fs.readFileSync("web/inference.js", "utf8"), context);
const tick = async () => {
    for (let i = 0; i < 15; i++) await Promise.resolve();
  },
  take = (part) => {
    const i = requests.findIndex((r) => r.url.endsWith(part));
    assert(i >= 0, part);
    return requests.splice(i, 1)[0];
  },
  click = (id) => el(id).listeners.click(),
  submit = () => el("form").listeners.submit({ preventDefault() {} });
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    take("/api/inference").resolve({
      architecture,
      model: "fixture",
      engine: "fixture",
      comparison: { source_model, max_edits: 8, tensors: [] },
      sweep: plan.limits,
    });
    await tick();
    el("task").value = "sweep";
    el("task").listeners.change();
    assert(el("start").disabled);
    assert(el("limit").disabled);
    assert(el("observation").disabled);
    const refusal = click("sweep-preview");
    take("/sweep-plan").resolve({ error: "Synthetic refusal" }, 503);
    await refusal;
    assert(el("error").textContent.includes("Synthetic refusal"));
    assert.equal(requests.length, 0);
    assert(el("start").disabled);
    const preview = click("sweep-preview"),
      planning = take("/sweep-plan");
    assert.deepEqual(JSON.parse(planning.options.body), request);
    planning.resolve({ plan });
    await preview;
    assert(!el("start").disabled);
    assert(
      el("sweep-plan-status").textContent.includes("not a complete head sweep"),
    );
    assert(el("sweep-plan-status").textContent.includes("90 CPU s total"));
    el("sweep-seed").value = "8";
    el("sweep-seed").listeners.input();
    assert(el("start").disabled);
    await submit();
    assert.equal(requests.length, 0);
    assert(el("error").textContent.includes("fresh expanded plan"));
    el("sweep-seed").value = "7";
    el("sweep-seed").listeners.input();
    const stalePlan = click("sweep-preview"),
      old = take("/sweep-plan");
    el("sweep-seed").value = "8";
    el("sweep-seed").listeners.input();
    old.resolve({ plan });
    await stalePlan;
    assert(el("start").disabled);
    el("sweep-seed").value = "7";
    el("sweep-seed").listeners.input();
    const review = click("sweep-preview");
    take("/sweep-plan").resolve({ plan });
    await review;
    const run = submit(),
      start = take("/start");
    assert.deepEqual(JSON.parse(start.options.body), accepted);
    start.resolve({
      session: "sweep-owner",
      status: "loading",
      steps: [],
      details: {
        sweep_plan: plan,
        sweep_coverage: {
          planned_ids: ids,
          completed_ids: [],
          unrun_ids: ids,
          interrupted_id: null,
          complete: false,
        },
      },
    });
    await run;
    assert(el("sweep-targets").disabled);
    assert(el("task").disabled);
    take("/poll").resolve({ session: "sweep-owner", ...snapshot });
    await tick();
    assert(el("sweep-progress").textContent.includes("2 / 3 completed"));
    assert(el("sweep-progress").textContent.includes("1 unrun"));
    assert.equal(el("sweep-results").children.length, 2);
    assert.equal(requests.length, 0);
    click("step");
    click("step");
    assert(el("alignment").textContent.includes("no continuation"));
    assert(
      el("score-context").textContent.includes("no inferred causal purpose"),
    );
    assert(el("edited-ids").textContent.includes("original bits restored"));
    assert.equal(el("scores").children[0].children[3].textContent, "-1.000000");
    // No automatic retry/chaining. Only this explicit new submit creates a request.
    const next = submit();
    take("/start").resolve({
      session: "sweep-next",
      status: "running",
      steps: [],
      details: { sweep_plan: plan },
    });
    await next;
    const stale = take("/poll"),
      reset = click("reset");
    take("/reset").resolve({
      session: null,
      status: "idle",
      steps: [],
      details: {},
    });
    await reset;
    stale.resolve({ session: "sweep-next", ...snapshot });
    await tick();
    assert(el("sweep-results-section").hidden);
    assert.equal(el("sweep-results").children.length, 0);
    assert(el("start").disabled);
    assert.equal(requests.length, 0);
    window.atlasInferenceSelectionChanged({
      source_model: { ...source_model, repo: "Qwen/Qwen3-8B" },
      tensor: "model.layers.0.self_attn.q_proj.weight",
      shape: [576, 576],
      row: 70,
      col: 0,
    });
    assert(el("sweep-use-inspected").disabled);
    console.log(
      JSON.stringify(
        {
          status: "PASS",
          scope:
            "Strict explicit subset input, reviewed deterministic plan binding, stale plan/seed invalidation, no automatic refusal retry or chaining, bounded partial coverage, owned reset/stale poll, fixed-context labels, Qwen inspector rejection, prompt/digest/all-prefill-token export redaction",
        },
        null,
        2,
      ),
    );
  })().catch((e) => {
    console.error(e);
    process.exitCode = 1;
  }),
);
// Full-layer requests stay in one reviewed job and cannot exceed the trace cap.
assert.deepEqual(
  sweepRequest({ ...values, targets: "layer 4" }, source_model, architecture)
    .targets,
  [{ kind: "layer_heads", layer: 4 }],
);
assert.throws(
  () =>
    sweepRequest(
      { ...values, targets: "layer 4", prompts: ["a", "b"] },
      source_model,
      architecture,
    ),
  /32-record/,
);
assert.throws(() =>
  sweepRequest(
    { ...values, targets: "layer 4", operation: "scale" },
    source_model,
    architecture,
  ),
);
assert.throws(() =>
  sweepRequest(
    { ...values, targets: "layer 4\nhead 0 0" },
    source_model,
    architecture,
  ),
);
assert.equal(
  sweepRequest(
    { ...values, targets: "query_head 4 8" },
    source_model,
    architecture,
  ).targets[0].kind,
  "query_head",
);
const alternate = {
  ...architecture,
  width: 24,
  layers: 2,
  query_heads: 4,
  head_dim: 8,
};
assert.equal(
  sweepRequest(
    { ...values, targets: "head 1 3", layer: "1" },
    source_model,
    alternate,
  ).targets[0].head,
  3,
);
assert.throws(() =>
  sweepRequest(
    { ...values, targets: "head 2 3", layer: "1" },
    source_model,
    alternate,
  ),
);
const { AtlasPlayback } = require("../web/inference.js"),
  dynamicPlayback = new AtlasPlayback(24);
dynamicPlayback.accept({
  status: "complete",
  steps: [{ index: 0, activation: Array(24).fill(0) }],
});
assert.throws(() =>
  dynamicPlayback.accept({
    steps: [{ index: 0, activation: Array(576).fill(0) }],
  }),
);
