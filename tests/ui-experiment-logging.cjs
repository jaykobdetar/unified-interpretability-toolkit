"use strict";
const fs = require("fs"),
  vm = require("vm"),
  assert = require("assert/strict");
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
  click() {}
  remove() {}
}
const elements = new Map(),
  requests = [],
  downloads = [],
  get = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
for (const [key, value] of Object.entries({
  rate: "2",
  limit: "1",
  layer: "7",
  site: "mlp",
  mode: "step",
  prompt: "private synthetic input",
  kind: "element",
  operation: "zero",
  factor: ".5",
}))
  get("infer-" + key).value = value;
const context = vm.createContext({
  console,
  document: {
    getElementById: get,
    createElement: () => new Element(),
    body: { append() {} },
  },
  window: { addEventListener() {} },
  Blob: class {
    constructor(parts) {
      this.text = parts.join("");
    }
  },
  URL: {
    createObjectURL: (blob) => {
      downloads.push(blob.text);
      return "blob:fixture";
    },
    revokeObjectURL() {},
  },
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
  for (let i = 0; i < 12; i++) await Promise.resolve();
};
const take = (suffix) => {
  const i = requests.findIndex((r) => r.url.endsWith(suffix));
  assert(i >= 0, suffix);
  return requests.splice(i, 1)[0];
};
const click = (id) => get("infer-" + id).listeners.click();
const submit = () =>
  get("infer-form").listeners.submit({ preventDefault() {} });
const step = {
  index: 0,
  activation: Array(576).fill(0.5),
  activation_site: "mlp",
  activation_kind: "after down_proj before residual",
  position: 2,
  input_token_id: 42,
  token_id: 8,
  token_piece: " result",
  generated_text: " result",
  compute_ms: 1,
  compute_total_ms: 1,
  layer: 7,
  phase: "prefill",
  top_logits: [{ id: 8, value: 1 }],
};
const terminal = (session, status = "complete") => ({
  session,
  status,
  worker_alive: false,
  steps: status === "complete" ? [step] : [],
  details: {
    prompt_ids: [10, 20, 42],
    reason: status === "complete" ? "token_limit" : status,
    runtime: { torch: "fixture", session: "MUST-NOT-LEAK" },
  },
});
(async () => {
  take("/api/inference").resolve({ model: "fixture", engine: "CPU" });
  await tick();
  async function run(id, status = "complete") {
    const start = submit();
    const request = take("/start");
    assert.equal(JSON.parse(request.options.body).activation_site, "mlp");
    request.resolve({
      session: id,
      status: "loading",
      worker_alive: true,
      steps: [],
      details: {},
    });
    await start;
    take("/poll").resolve(terminal(id, status));
    await tick();
  }
  await run("without-consent");
  assert(get("infer-log-status").textContent.includes("0 / 8"));
  get("infer-logging").checked = true;
  get("infer-include-prompt").checked = true;
  await run("first-private");
  assert(get("infer-log-status").textContent.includes("1 / 8"));
  click("export-log");
  let file = JSON.parse(downloads.at(-1));
  assert.equal(file.records[0].request.prompt, "private synthetic input");
  assert.equal(file.records[0].request.activation_site, "mlp");
  assert(!downloads.at(-1).includes("first-private"));
  assert(!downloads.at(-1).includes("MUST-NOT-LEAK"));
  get("infer-include-prompt").checked = false;
  click("export-log");
  file = JSON.parse(downloads.at(-1));
  assert(!downloads.at(-1).includes("private synthetic input"));
  assert(!Object.hasOwn(file.records[0].steps[0], "input_token_id"));
  await run("failed-model", "error");
  click("export-log");
  file = JSON.parse(downloads.at(-1));
  assert.equal(file.records[1].status, "error");
  assert.equal(file.records[1].complete, false);
  // Cancel before the first token and retain an incomplete record; late poll ignored.
  const started = submit();
  take("/start").resolve({
    session: "cancel-owner",
    status: "loading",
    worker_alive: true,
    steps: [],
    details: {},
  });
  await started;
  const stale = take("/poll");
  const cancel = click("cancel");
  take("/cancel").resolve(terminal("cancel-owner", "cancelled"));
  await cancel;
  stale.resolve(terminal("cancel-owner"));
  await tick();
  assert(get("infer-log-status").textContent.includes("3 / 8"));
  for (let i = 3; i < 8; i++) await run("run-" + i);
  await run("overflow");
  assert(get("infer-log-status").textContent.includes("full"));
  assert(get("infer-start").disabled);
  click("export-log");
  assert.equal(JSON.parse(downloads.at(-1)).records.length, 8);
  click("export-run");
  file = JSON.parse(downloads.at(-1));
  assert.equal(file.status, "complete");
  assert(!downloads.at(-1).includes("overflow"));
  const reset = click("reset");
  take("/reset").resolve({
    session: null,
    status: "idle",
    worker_alive: false,
    steps: [],
    details: {},
  });
  await reset;
  assert(get("infer-start").disabled);
  assert(!get("infer-export-run").disabled);
  click("clear-log");
  assert(!get("infer-start").disabled);
  assert(get("infer-log-status").textContent.includes("0 / 8"));
  // Reset of an accepted active run records the last received trace as cancelled.
  const active = submit();
  take("/start").resolve({
    session: "reset-active",
    status: "loading",
    worker_alive: true,
    steps: [],
    details: {},
  });
  await active;
  const oldPoll = take("/poll");
  const clear = click("reset");
  take("/reset").resolve({
    session: null,
    status: "idle",
    worker_alive: false,
    steps: [],
    details: {},
  });
  await clear;
  oldPoll.resolve(terminal("reset-active"));
  await tick();
  click("export-log");
  file = JSON.parse(downloads.at(-1));
  assert.equal(file.records.length, 1);
  assert.equal(file.records[0].status, "cancelled");
  assert(file.records[0].termination.includes("last received tab snapshot"));
  assert.equal(requests.length, 0);
  console.log(
    JSON.stringify(
      {
        status: "PASS",
        checks: 20,
        scope:
          "Logging off by default; accepted run/component provenance; separate prompt consent and export redaction; failed/cancelled/reset records; stale response ignored; eight-run cap, pending overflow, capability exclusion, explicit clear and recovery",
      },
      null,
      2,
    ),
  );
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
