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
  const elems = new Map(),
    requests = [],
    get = (id) => {
      if (!elems.has(id)) elems.set(id, new Element());
      return elems.get(id);
    };
  for (const [key, value] of Object.entries({
    rate: "2",
    limit: "3",
    layer: "0",
    mode: "step",
    prompt: "synthetic",
    kind: "element",
    operation: "zero",
  }))
    get("infer-" + key).value = value;
  const context = vm.createContext({
    console,
    document: { getElementById: get, createElement: () => new Element() },
    window: { addEventListener() {} },
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
  return {
    get,
    click: (id) => get("infer-" + id).listeners.click(),
    submit: () => get("infer-form").listeners.submit({ preventDefault() {} }),
    take: (suffix) => {
      const i = requests.findIndex((r) => r.url.endsWith(suffix));
      assert(i >= 0, suffix);
      return requests.splice(i, 1)[0];
    },
  };
}
const tick = async () => {
  for (let i = 0; i < 10; i++) await Promise.resolve();
};
const branch = (ids, text, eos = false) => ({
  generated_ids: ids,
  generated_text: text,
  token_id: ids.at(-1),
  token_piece: text,
  eos,
});
const paired = (index, left, right) => ({
  index,
  baseline: left,
  edited: right,
  activation: Array(576).fill(0.5),
  position: 4 + index,
  input_token_id: 42,
  token_id: (right || left).token_id,
  token_piece: (right || left).token_piece,
  generated_text: (right || left).generated_text,
  compute_ms: 1,
  compute_total_ms: index + 1,
  phase: index ? "decode" : "prefill",
  layer: 0,
  alignment: left && right ? "matched_prefix" : "branch_ended",
  activation_branch: right ? "edited" : "baseline",
  candidates: [
    {
      id: 1,
      piece: "synthetic",
      baseline_logit: left ? 2 : null,
      edited_logit: right ? 3 : null,
      delta: left && right ? 1 : null,
    },
  ],
});
(async () => {
  for (const ended of ["baseline", "edited"]) {
    const ui = setup();
    ui.take("/api/inference").resolve({ model: "fixture", engine: "CPU" });
    await tick();
    const short = branch(
        [1],
        ended === "baseline" ? "BASE EOS" : "EDIT EOS",
        true,
      ),
      long0 = branch([2], "LONG 0"),
      long1 = branch([2, 3], "LONG 0 1");
    const steps =
      ended === "baseline"
        ? [paired(0, short, long0), paired(1, null, long1)]
        : [paired(0, long0, short), paired(1, long1, null)];
    const start = ui.submit();
    ui.take("/start").resolve({
      session: "owner",
      status: "loading",
      steps: [],
      details: {},
    });
    await start;
    ui.take("/poll").resolve({
      session: "owner",
      status: "running",
      steps,
      details: {},
    });
    await tick();
    ui.click("step");
    ui.click("step");
    const output = ended === "baseline" ? "baseline-output" : "output",
      ids = ended === "baseline" ? "baseline-ids" : "edited-ids";
    assert.equal(ui.get("infer-" + output).textContent, short.generated_text);
    assert.equal(ui.get("infer-" + ids).textContent, "Token IDs: 1");
    assert(
      ui.get("infer-score-context").textContent.includes("One branch ended"),
    );
    assert.equal(
      ui.get("infer-scores").children[0].children[ended === "baseline" ? 1 : 2]
        .textContent,
      "—",
    );
    assert.equal(
      ui.get("infer-scores").children[0].children[3].textContent,
      "—",
    );
    const cancel = ui.click("cancel");
    ui.take("/cancel").resolve({
      session: "owner",
      status: "cancelled",
      steps,
      details: {},
    });
    await cancel;
    assert.equal(ui.get("infer-" + output).textContent, short.generated_text);
    assert.equal(ui.get("infer-" + ids).textContent, "Token IDs: 1");
    ui.click("replay");
    assert.equal(
      ui.get("infer-baseline-output").textContent,
      "Baseline text will appear here.",
    );
    ui.click("step");
    const longOutput = ended === "baseline" ? "output" : "baseline-output";
    assert.equal(ui.get("infer-" + longOutput).textContent, "LONG 0");
    const reset = ui.click("reset");
    ui.take("/reset").resolve({
      session: null,
      status: "idle",
      steps: [],
      details: {},
    });
    await reset;
    assert.equal(ui.get("infer-baseline-ids").textContent, "");
    assert.equal(ui.get("infer-edited-ids").textContent, "");
  }
  // A valid empty decoded branch is retained, including after a worker error.
  const ui = setup();
  ui.take("/api/inference").resolve({ model: "fixture", engine: "CPU" });
  await tick();
  const empty = branch([1], "", true),
    steps = [
      paired(0, empty, branch([2], "first")),
      paired(1, null, branch([2, 3], "first second")),
    ];
  const start = ui.submit();
  ui.take("/start").resolve({
    session: "empty",
    status: "loading",
    steps: [],
    details: {},
  });
  await start;
  ui.take("/poll").resolve({
    session: "empty",
    status: "error",
    steps,
    details: {
      error: "controlled fixture",
      baseline: { generated_text: "FUTURE SUMMARY", generated_ids: [99] },
      edited: { generated_text: "FUTURE SUMMARY", generated_ids: [99] },
    },
  });
  await tick();
  ui.click("step");
  ui.click("step");
  assert.equal(ui.get("infer-baseline-output").textContent, "");
  assert.equal(ui.get("infer-baseline-ids").textContent, "Token IDs: 1");
  ui.click("replay");
  ui.click("step");
  assert.equal(ui.get("infer-output").textContent, "first");
  console.log(
    JSON.stringify(
      {
        status: "PASS",
        scope:
          "Both asymmetric EOS/length directions preserve last branch text/IDs before done and after cancellation/error; scores stay absent; empty decoded text, rewind, reset, and no future-summary leakage",
      },
      null,
      2,
    ),
  );
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
