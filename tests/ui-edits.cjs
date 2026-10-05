"use strict";
const architecture = require("./fixtures/inference-architecture.json");
const fs = require("fs"),
  vm = require("vm"),
  assert = require("assert/strict");
const { draftEdit, editRangeSummary } = require("../web/inference.js");
const source_model = {
  repo: "HuggingFaceTB/SmolLM2-135M",
  revision: "pinned",
  weights_sha256: "pinned",
};
const tensor = "model.layers.0.self_attn.q_proj.weight";
const contract = {
  source_model,
  max_edits: 8,
  tensors: [{ name: tensor, shape: [576, 576], aliases: [] }],
};
const selection = { source_model, tensor, shape: [576, 576], row: 70, col: 5 };
assert.deepEqual(draftEdit(selection, contract, "rows", "zero", 64, 128, 0), {
  tensor,
  shape: [576, 576],
  kind: "rows",
  operation: "zero",
  start: 64,
  end: 128,
});
for (const s of [
  { ...selection, source_model: null },
  { ...selection, shape: [4096, 4096] },
  { ...selection, tensor: "lm_head.weight" },
])
  assert.throws(() => draftEdit(s, contract, "element", "zero", 0, 1, 0));
for (const [start, end] of [
  [0, 0],
  [0, 577],
  [-1, 2],
  ["", 1],
  [0.1, 2],
])
  assert.throws(() =>
    draftEdit(selection, contract, "rows", "zero", start, end, 0),
  );
for (const scale of ["", Infinity, 101])
  assert.throws(() =>
    draftEdit(selection, contract, "element", "scale", 0, 1, scale),
  );
class Element {
  constructor() {
    this.value = "";
    this.textContent = "";
    this.disabled = false;
    this.hidden = false;
    this.listeners = {};
    this.children = [];
    this.width = 512;
    this.height = 288;
  }
  replaceChildren(...items) {
    this.children = items;
  }
  append(...items) {
    this.children.push(...items);
  }
  addEventListener(k, f) {
    this.listeners[k] = f;
  }
  getContext() {
    return { clearRect() {}, fillRect() {} };
  }
}
function setup() {
  const elements = new Map(),
    requests = [],
    get = (id) => {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    };
  for (const [key, value] of Object.entries({
    rate: "2",
    limit: "4",
    layer: "0",
    mode: "step",
    prompt: "synthetic",
    kind: "element",
    operation: "zero",
    factor: ".5",
  }))
    get("infer-" + key).value = value;
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
  const take = (suffix) => {
    const index = requests.findIndex((r) => r.url.endsWith(suffix));
    assert(index >= 0, suffix);
    return requests.splice(index, 1)[0];
  };
  return {
    get,
    take,
    window,
    click: (id) => get("infer-" + id).listeners.click(),
    submit: () => get("infer-form").listeners.submit({ preventDefault() {} }),
  };
}
const tick = async () => {
  for (let i = 0; i < 12; i++) await Promise.resolve();
};
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    const unavailable = setup();
    unavailable.take("/api/inference").resolve({ error: "missing" }, 404);
    await tick();
    unavailable.window.atlasInferenceSelectionChanged(selection);
    assert(
      unavailable
        .get("infer-status")
        .textContent.includes("run-atlas.sh serves the weight viewer only"),
    );
    assert(unavailable.get("infer-start").disabled);
    assert(unavailable.get("infer-add").disabled);
    assert(
      unavailable
        .get("infer-selection")
        .textContent.includes("no local inference coordinator"),
    );
    assert(!unavailable.get("infer-selection").textContent.includes("Qwen"));
    assert(unavailable.get("prepare-edit").disabled);
    const ui = setup();
    ui.take("/api/inference").resolve({
      architecture,
      model: "SmolLM2",
      engine: "CPU",
      comparison: contract,
    });
    await tick();
    ui.window.atlasInferenceSelectionChanged({
      ...selection,
      source_model: null,
    });
    assert(ui.get("infer-add").disabled);
    assert(
      ui
        .get("infer-selection")
        .textContent.includes("no verified edit mapping"),
    );
    ui.window.atlasInferenceSelectionChanged(selection);
    assert(!ui.get("infer-add").disabled);
    assert(!ui.get("prepare-edit").disabled);
    assert(
      ui.get("infer-target-summary").textContent.includes("Capture: layer 0"),
    );
    assert(ui.get("infer-target-summary").textContent.includes("independent"));
    assert(
      editRangeSummary(
        draftEdit(selection, contract, "rows", "zero", 64, 128, 0),
      ).includes("64–127 inclusive"),
    );
    assert(
      editRangeSummary(
        draftEdit(selection, contract, "rows", "zero", 64, 128, 0),
      ).includes("36,864 targeted"),
    );
    ui.click("head");
    assert(
      ui.get("infer-edits").children[0].textContent.includes("rows [64,128)"),
    );
    const start = ui.submit(),
      request = ui.take("/start"),
      body = JSON.parse(request.options.body);
    assert.deepEqual(body.source_model, source_model);
    assert.equal(body.edits[0].start, 64);
    assert.equal(body.edits[0].end, 128);
    ui.window.atlasInferenceSelectionChanged(null);
    assert(ui.get("infer-clear-edits").disabled);
    request.resolve({
      session: "comparison-owner",
      status: "running",
      steps: [],
      details: { comparison_phase: "baseline" },
    });
    await start;
    const base = {
      token_id: 1,
      token_piece: "a",
      generated_text: "a",
      generated_ids: [1],
      eos: false,
    };
    const step = {
      index: 0,
      activation: Array(576).fill(0.5),
      position: 1,
      input_token_id: 0,
      token_id: 2,
      token_piece: "b",
      generated_text: "b",
      compute_ms: 1,
      compute_total_ms: 1,
      layer: 0,
      phase: "prefill",
      baseline: base,
      edited: { ...base, token_id: 2, generated_text: "b", generated_ids: [2] },
      alignment: "matched_prefix",
      activation_branch: "edited",
      candidates: [
        { id: 1, piece: "a", baseline_logit: 2, edited_logit: 1, delta: -1 },
      ],
    };
    ui.take("/poll").resolve({
      session: "comparison-owner",
      status: "complete",
      steps: [step],
      details: {},
    });
    await tick();
    ui.click("step");
    assert.equal(ui.get("infer-baseline-output").textContent, "a");
    assert.equal(ui.get("infer-output").textContent, "b");
    assert(
      ui
        .get("infer-score-context")
        .textContent.includes("Matched consumed prefix"),
    );
    assert.equal(
      ui.get("infer-scores").children[0].children[3].textContent,
      "-1.000000",
    );
    assert(ui.get("infer-baseline-ids").textContent.includes("1"));
    assert(ui.get("infer-edited-ids").textContent.includes("2"));
    console.log(
      JSON.stringify(
        {
          status: "PASS",
          scope:
            "Pinned inspector identity and Qwen rejection, whole query-head draft, immutable comparison request, paired logits/text/IDs, persistent static-launch guidance",
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
