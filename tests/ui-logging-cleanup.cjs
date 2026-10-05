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
const timers = [],
  elements = new Map(),
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
    new Promise((resolve, reject) =>
      requests.push({
        url,
        options,
        reject,
        resolve: (body, status = 200) =>
          resolve({ ok: status < 400, status, json: async () => body }),
      }),
    ),
  setTimeout: (fn, ms) => {
    timers.push({ fn, ms });
    return timers.length;
  },
  clearTimeout: (id) => {
    timers[id - 1] = null;
  },
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

const fire = async () => {
  const i = timers.findIndex((t) => t?.ms === 250);
  assert(i >= 0, "cleanup poll scheduled");
  const t = timers[i];
  timers[i] = null;
  t.fn();
  await tick();
};
vm.runInContext(
  "globalThis.confirmations=0;const originalConfirm=AtlasExperimentLog.prototype.confirmCleanup;AtlasExperimentLog.prototype.confirmCleanup=function(index){confirmations++;return originalConfirm.call(this,index);};globalThis.appended=[];const originalAppend=AtlasExperimentLog.prototype.append;AtlasExperimentLog.prototype.append=function(record){appended.push(JSON.parse(JSON.stringify(record)));return originalAppend.call(this,record);};",
  context,
);
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    take("/api/inference").resolve({ model: "fixture", engine: "CPU" });
    await tick();
    get("infer-logging").checked = true;
    const first = submit();
    take("/start").resolve({
      session: "interrupted-owner",
      status: "running",
      worker_alive: true,
      steps: [step],
      details: { prompt_ids: [10, 20, 42] },
    });
    await first;
    take("/poll").reject(new Error("controlled poll loss"));
    await tick();
    take("/cancel").resolve({
      session: "interrupted-owner",
      status: "stopping",
      worker_alive: false,
      steps: [step],
      details: { cleanup_pending: true, prompt_ids: [10, 20, 42] },
    });
    await tick();
    assert(get("infer-status").textContent.includes("compute stopping"));
    assert(get("infer-start").disabled);
    assert(get("infer-log-status").textContent.includes("1 / 8"));
    const records = () => JSON.parse(JSON.stringify(context.appended));
    let saved = records();
    assert.equal(saved.length, 1);
    assert.equal(saved[0].status, "connection_lost");
    assert.equal(saved[0].worker_cleanup_confirmed, false);
    assert.equal(saved[0].termination, "transport_error");
    assert(!JSON.stringify(saved).includes("private synthetic input"));
    assert(!Object.hasOwn(saved[0].steps[0], "input_token_id"));
    await submit();
    assert.equal(requests.length, 0);
    await fire();
    let req = take("/poll");
    assert.equal(JSON.parse(req.options.body).session, "interrupted-owner");
    req.reject(new Error("second controlled status loss"));
    await tick();
    assert(get("infer-start").disabled);
    assert.equal(records().length, 1);
    await fire();
    take("/poll").resolve({
      session: "interrupted-owner",
      status: "stopping",
      worker_alive: false,
      steps: [step],
      details: { cleanup_pending: true },
    });
    await tick();
    assert(get("infer-start").disabled);
    assert.equal(records().length, 1);
    assert.equal(records()[0].worker_cleanup_confirmed, false);
    assert.equal(context.confirmations, 0);
    await fire();
    take("/poll").resolve({
      ...terminal("interrupted-owner", "cancelled"),
      steps: [step],
    });
    await tick();
    assert(!get("infer-start").disabled);
    click("export-log");
    let file = JSON.parse(downloads.at(-1));
    assert.equal(file.records.length, 1);
    assert.equal(file.records[0].status, "connection_lost");
    assert.equal(file.records[0].complete, false);
    assert.equal(file.records[0].worker_cleanup_confirmed, true);
    assert.equal(context.confirmations, 1);
    assert(!Object.hasOwn(file.records[0].steps[0], "input_token_id"));
    const second = submit();
    take("/start").resolve({
      session: "next-owner",
      status: "running",
      worker_alive: true,
      steps: [],
      details: {},
    });
    await second;
    take("/poll").resolve(terminal("next-owner"));
    await tick();
    click("export-log");
    file = JSON.parse(downloads.at(-1));
    assert.deepEqual(
      file.records.map((r) => r.status),
      ["connection_lost", "complete"],
    );
    assert(!downloads.at(-1).includes("interrupted-owner"));
    assert(!downloads.at(-1).includes("private synthetic input"));
    // A lost cancellation response also retains ownership. A later running status retries owned cancellation.
    const third = submit();
    take("/start").resolve({
      session: "unknown-cleanup",
      status: "running",
      worker_alive: true,
      steps: [],
      details: {},
    });
    await third;
    take("/poll").reject(new Error("poll lost"));
    await tick();
    take("/cancel").reject(new Error("cancel lost"));
    await tick();
    assert(get("infer-start").disabled);
    assert.equal(records().length, 3);
    assert.equal(records()[2].worker_cleanup_confirmed, false);
    await fire();
    take("/poll").resolve({
      session: "unknown-cleanup",
      status: "running",
      worker_alive: true,
      steps: [],
      details: {},
    });
    await tick();
    req = take("/cancel");
    assert.equal(JSON.parse(req.options.body).session, "unknown-cleanup");
    req.resolve({
      session: "unknown-cleanup",
      status: "stopping",
      worker_alive: true,
      steps: [],
      details: { cleanup_pending: true },
    });
    await tick();
    assert(get("infer-start").disabled);
    await fire();
    take("/poll").resolve(terminal("unknown-cleanup", "cancelled"));
    await tick();
    click("export-log");
    file = JSON.parse(downloads.at(-1));
    assert.equal(file.records.length, 3);
    assert.equal(file.records[2].worker_cleanup_confirmed, true);
    assert.equal(file.records[2].status, "connection_lost");
    assert.equal(requests.length, 0);
    console.log(
      "PASS OBS-2: owned cleanup survives poll/cancel transport loss; one incomplete consent-redacted outcome; cleanup confirmation updates in place; next run preserves prior outcome",
    );
  })().catch((e) => {
    console.error(e);
    process.exitCode = 1;
  }),
);
