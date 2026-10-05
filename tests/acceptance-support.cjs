"use strict";
// Tiny helper fixtures only. No browser, sockets, processes, timers or model reads.
const assert = require("node:assert/strict"),
  H = require("./acceptance/support.cjs"),
  { resolved_plan: plan } = require("./fixtures/sweep-maximum-acceptance.json");
function snapshot(count, status = count === 10 ? "complete" : "time_limit") {
  const planned = H.ids(plan),
    steps = Array.from({ length: count }, (_, i) => {
      const c = plan.cases[Math.floor(i / 2)],
        empty = c.role === "empty_control";
      return {
        index: i,
        activation: Array(576).fill(0),
        sweep: {
          record_id: planned[i],
          case_id: c.id,
          role: c.role,
          prompt_index: i % 2,
          restoration_verified: true,
          selected_cells: c.selected_cells,
          changed_cells: empty ? 0 : 1,
          parameter_delta_l2: empty ? 0 : 0.5,
          metrics: {
            context: "matched fixed original prompt",
            logit_delta_rms: empty ? 0 : 0.1,
            logit_delta_max_abs: empty ? 0 : 0.5,
            softmax_total_variation: empty ? 0 : 0.01,
            baseline_argmax_logit_delta: empty ? 0 : -0.5,
          },
          candidates: [
            {
              id: 1,
              baseline_logit: 1,
              edited_logit: empty ? 1 : 0.5,
              delta: empty ? 0 : -0.5,
            },
          ],
        },
      };
    });
  return {
    status,
    worker_alive: false,
    steps,
    details: {
      sweep_coverage: {
        planned_ids: planned,
        completed_ids: planned.slice(0, count),
        unrun_ids: planned.slice(count),
        interrupted_id: null,
        complete: count === 10,
      },
    },
  };
}
assert.equal(H.sweepCheck(plan, snapshot(10)).complete, true);
const partial = H.sweepCheck(plan, snapshot(3));
assert.equal(partial.complete, false);
assert.equal(partial.unrun_ids.length, 7);
assert.throws(() => H.sweepCheck(plan, snapshot(3, "complete")));
let bad = snapshot(3);
bad.steps[2].sweep.candidates[0].delta = 0;
assert.throws(() => H.sweepCheck(plan, bad));
bad = snapshot(3);
bad.steps[2].sweep.restoration_verified = false;
assert.throws(() => H.sweepCheck(plan, bad));
bad = snapshot(3);
bad.details.sweep_coverage.unrun_ids = [];
assert.throws(() => H.sweepCheck(plan, bad));
assert(
  !H.clean({
    status: "stopping",
    worker_alive: false,
    details: { cleanup_pending: true },
  }),
);
assert(!H.clean({ status: "stopping", worker_alive: false, details: {} }));
assert(
  !H.clean({
    status: "complete",
    worker_alive: false,
    details: { cleanup_pending: true },
  }),
);
assert(H.clean(snapshot(10)));
const a = Array(576).fill(0),
  b = [1, ...Array(575).fill(0)];
const record = {
  complete: true,
  request: { mode: "prompt_pair" },
  privacy: { prompt_included: false },
  steps: [0, 1].map((index) => ({
    index,
    activation: b,
    prompt_pair: {
      a: { activation: a },
      b: { activation: b },
      metrics: { a_l2: 0, b_l2: 1, delta_l2: 1, cosine: null },
    },
  })),
};
assert.equal(H.pairCheck(record).pairs, 2);
H.redacted(record);
record.steps[1].prompt_pair.a.token_id = 1;
assert.throws(() => H.redacted(record));
delete record.steps[1].prompt_pair.a.token_id;
record.steps[0].prompt_pair.metrics.delta_l2 = 0;
assert.throws(() => H.pairCheck(record));
console.log(
  "PASS: full versus partial coverage, missing-case refusal, independent candidate/vector metrics, restoration flag, authoritative cleanup, every-record privacy",
);
