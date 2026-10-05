"use strict";
// Acceptance helpers only; importing this file starts no process/browser/request.
const assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  crypto = require("node:crypto");
const terminal = new Set([
  "idle",
  "complete",
  "cancelled",
  "error",
  "resource_limit",
  "time_limit",
  "client_timeout",
]);
const clean = (s) =>
  !!s &&
  s.worker_alive === false &&
  !s.details?.cleanup_pending &&
  terminal.has(s.status);
const ids = (plan) =>
  plan.cases.flatMap((c) =>
    Array.from(
      { length: plan.prompt_count },
      (_, i) => `${c.id}/prompt-${i + 1}`,
    ),
  );
function sweepCheck(plan, snapshot) {
  const planned = ids(plan),
    steps = snapshot.steps || [],
    coverage = snapshot.details?.sweep_coverage;
  assert.equal(planned.length, plan.records);
  assert(steps.length <= plan.records);
  assert(coverage, "coverage required");
  assert.deepEqual(coverage.planned_ids, planned);
  assert.deepEqual(coverage.completed_ids, planned.slice(0, steps.length));
  assert.deepEqual(coverage.unrun_ids, planned.slice(steps.length));
  assert(
    coverage.interrupted_id === null ||
      coverage.unrun_ids.includes(coverage.interrupted_id),
  );
  assert.equal(coverage.complete, steps.length === plan.records);
  if (snapshot.status === "complete")
    assert(coverage.complete, "partial coverage cannot pass as complete");
  for (const [i, step] of steps.entries()) {
    const value = step.sweep,
      c = plan.cases[Math.floor(i / plan.prompt_count)];
    assert.equal(step.index, i);
    assert.equal(value.record_id, planned[i]);
    assert.equal(value.case_id, c.id);
    assert.equal(value.role, c.role);
    assert.equal(value.prompt_index, i % plan.prompt_count);
    assert.equal(value.restoration_verified, true);
    assert.equal(value.selected_cells, c.selected_cells);
    assert(
      Number.isInteger(value.changed_cells) &&
        value.changed_cells >= 0 &&
        value.changed_cells <= value.selected_cells,
    );
    assert.equal(step.activation.length, 576);
    assert(step.activation.every(Number.isFinite));
    assert(
      Number.isFinite(value.parameter_delta_l2) &&
        value.parameter_delta_l2 >= 0,
    );
    const m = value.metrics;
    assert.equal(m.context, "matched fixed original prompt");
    for (const k of [
      "logit_delta_rms",
      "logit_delta_max_abs",
      "softmax_total_variation",
      "baseline_argmax_logit_delta",
    ])
      assert(Number.isFinite(m[k]));
    assert(
      m.logit_delta_rms >= 0 &&
        m.logit_delta_max_abs >= 0 &&
        m.softmax_total_variation >= 0 &&
        m.softmax_total_variation <= 1 + 1e-12,
    );
    assert(value.candidates.length >= 1 && value.candidates.length <= 10);
    assert.equal(
      new Set(value.candidates.map((x) => x.id)).size,
      value.candidates.length,
    );
    for (const candidate of value.candidates) {
      assert(
        Number.isInteger(candidate.id) &&
          candidate.id >= 0 &&
          candidate.id < 49152,
      );
      assert(
        [
          candidate.baseline_logit,
          candidate.edited_logit,
          candidate.delta,
        ].every(Number.isFinite),
      );
      assert.equal(
        candidate.delta,
        candidate.edited_logit - candidate.baseline_logit,
      );
    }
    if (c.role === "empty_control") {
      for (const key of [
        "logit_delta_rms",
        "logit_delta_max_abs",
        "softmax_total_variation",
        "baseline_argmax_logit_delta",
      ])
        assert.equal(m[key], 0);
      assert(value.candidates.every((x) => x.delta === 0));
    }
  }
  return {
    planned_ids: planned,
    completed_ids: coverage.completed_ids,
    unrun_ids: coverage.unrun_ids,
    interrupted_id: coverage.interrupted_id,
    complete: snapshot.status === "complete" && coverage.complete,
    observed_records: steps.length,
  };
}
function pairCheck(record) {
  assert.equal(record.request.mode, "prompt_pair");
  assert.equal(record.complete, true);
  assert.equal(record.steps.length, 2);
  let maximumMetricError = 0;
  for (const step of record.steps) {
    const p = step.prompt_pair,
      a = p.a.activation,
      b = p.b.activation;
    assert.equal(a.length, 576);
    assert.equal(b.length, 576);
    assert([...a, ...b, ...step.activation].every(Number.isFinite));
    const d = b.map((x, i) => x - a[i]);
    assert.deepEqual(step.activation, d);
    const na = Math.hypot(...a),
      nb = Math.hypot(...b),
      nd = Math.hypot(...d),
      cos =
        na && nb
          ? Math.max(
              -1,
              Math.min(1, a.reduce((sum, x, i) => sum + x * b[i], 0) / na / nb),
            )
          : null;
    for (const [key, value] of Object.entries({
      a_l2: na,
      b_l2: nb,
      delta_l2: nd,
      cosine: cos,
    })) {
      if (value === null) assert.equal(p.metrics[key], null);
      else {
        const error = Math.abs(value - p.metrics[key]);
        maximumMetricError = Math.max(maximumMetricError, error);
        assert(error < 1e-8, `${key} independent metric mismatch`);
      }
    }
  }
  return { pairs: 2, maximum_independent_metric_error: maximumMetricError };
}
function redacted(record) {
  assert.equal(record.privacy.prompt_included, false);
  for (const key of [
    "prompt",
    "prompts",
    "prompt_ids",
    "preview_digest",
    "plan_digest",
  ])
    assert(!Object.hasOwn(record.request, key));
  for (const step of record.steps) {
    assert(!Object.hasOwn(step, "input_token_id"));
    if (step.prompt_pair)
      for (const key of ["a", "b"]) {
        assert(!Object.hasOwn(step.prompt_pair[key], "token_id"));
        assert(!Object.hasOwn(step.prompt_pair[key], "token_piece"));
      }
  }
  if (record.sweep_plan) assert(!Object.hasOwn(record.sweep_plan, "digest"));
}
async function hashFile(file, buffer, open = fs.promises.open) {
  // Reuse one backing buffer: streaming allocations can retain tens of MiB until GC.
  const handle = await open(file, "r"),
    hash = crypto.createHash("sha256");
  try {
    while (true) {
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, null);
      if (!bytesRead) break;
      hash.update(buffer.subarray(0, bytesRead));
    }
    return hash.digest("hex");
  } finally {
    await handle.close();
  }
}
async function verifyPins(directory) {
  assert(directory, "Explicit ATLAS_MODEL_DIR required");
  const pins = JSON.parse(
      fs.readFileSync(
        path.join(__dirname, "../../docs/models/smollm2-135m.json"),
        "utf8",
      ),
    ).files,
    buffer = Buffer.allocUnsafe(1024 * 1024);
  for (const [name, expected] of Object.entries(pins))
    assert.equal(
      await hashFile(path.join(directory, name), buffer),
      expected,
      `Pinned file changed: ${name}`,
    );
  return pins;
}
function save(out, name, data) {
  const text = JSON.stringify(data, null, 2) + "\n";
  assert(Buffer.byteLength(text) <= 1024 * 1024);
  const target = path.join(out, name);
  fs.writeFileSync(target + ".tmp", text);
  fs.renameSync(target + ".tmp", target);
}
function watch(page, base, out) {
  const state = {
    latest: null,
    owner: null,
    owners: [],
    starts: [],
    errors: [],
    pending: new Set(),
    maximum_worker_rss_mib: 0,
  };
  page.on("response", (response) => {
    const route = new URL(response.url());
    if (
      route.origin !== base ||
      ![
        "/api/inference/start",
        "/api/inference/poll",
        "/api/inference/cancel",
      ].includes(route.pathname)
    )
      return;
    const task = (async () => {
      const raw = await response.body();
      assert(raw.length <= 2 * 1024 * 1024);
      const snapshot = JSON.parse(raw);
      if (!response.ok()) return;
      if (route.pathname.endsWith("/start")) {
        state.owner = snapshot.session;
        state.owners.push(snapshot.session);
        state.starts.push(
          response.request().postDataJSON().mode || "generation",
        );
      }
      if (snapshot.session !== state.owner) return;
      state.latest = snapshot;
      state.maximum_worker_rss_mib = Math.max(
        state.maximum_worker_rss_mib,
        snapshot.peak_worker_rss_mib || 0,
      );
      save(out, "progress.json", {
        status: snapshot.status,
        cleanup_confirmed: clean(snapshot),
        received_records: snapshot.steps?.length || 0,
        coverage: snapshot.details?.sweep_coverage || null,
        starts: state.starts,
        maximum_worker_rss_mib: state.maximum_worker_rss_mib,
        scope:
          "Last observed response only; guard interruption is not a complete result",
      });
    })().catch((e) => state.errors.push(e.message));
    state.pending.add(task);
    task.finally(() => state.pending.delete(task));
  });
  return state;
}
async function settled(w, expect) {
  await expect
    .poll(() => clean(w.latest), { timeout: 105000, intervals: [100, 250] })
    .toBe(true);
  await Promise.all([...w.pending]);
  assert.deepEqual(w.errors, []);
  return w.latest;
}
async function download(page, out, name) {
  const event = page.waitForEvent("download");
  await page.locator("#infer-export-run").click();
  const file = await event,
    target = path.join(out, name);
  await file.saveAs(target);
  assert(fs.statSync(target).size <= 1024 * 1024);
  return JSON.parse(fs.readFileSync(target, "utf8"));
}
async function cleanup(page, base, w) {
  if (!w.owner || clean(w.latest)) return;
  const action = async (name) => {
    const response = await page.request.post(base + "/api/inference/" + name, {
      headers: { Origin: base, "X-Atlas-Local": "1" },
      data: { session: w.owner },
      timeout: 2000,
    });
    if (response.ok()) w.latest = await response.json();
  };
  await action("cancel");
  const deadline = Date.now() + 4000;
  while (!clean(w.latest) && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 250));
    await action("poll");
  }
}
module.exports = {
  hashFile,
  verifyPins,
  clean,
  ids,
  sweepCheck,
  pairCheck,
  redacted,
  save,
  watch,
  settled,
  download,
  cleanup,
};
