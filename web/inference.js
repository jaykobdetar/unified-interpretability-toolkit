"use strict";
// Bounded session playback; draft edits apply only to a disposable model in RAM.
class AtlasPlayback {
  constructor(width = null) {
    this.width = width;
    this.reset();
  }
  reset() {
    this.steps = [];
    this.cursor = -1;
    this.paused = false;
    this.status = "idle";
    this.replaying = false;
  }
  accept(snapshot) {
    if (!Array.isArray(snapshot.steps) || snapshot.steps.length > 32)
      throw new Error("Invalid trace size");
    const width = this.width ?? snapshot.steps[0]?.activation?.length;
    snapshot.steps.forEach((s, i) => {
      if (
        s.index !== i ||
        !Number.isInteger(width) ||
        width < 1 ||
        width * snapshot.steps.length > 18432 ||
        s.activation?.length !== width ||
        !s.activation.every(Number.isFinite)
      )
        throw new Error("Invalid activation record");
    });
    this.steps = snapshot.steps;
    this.status = snapshot.status;
  }
  advance() {
    if (this.cursor + 1 < this.steps.length) {
      this.cursor++;
      return true;
    }
    return false;
  }
  rewind() {
    this.cursor = -1;
    this.paused = true;
    this.replaying = true;
  }
  get current() {
    return this.steps[this.cursor] || null;
  }
  branchAtCursor(name) {
    for (
      let index = Math.min(this.cursor, this.steps.length - 1);
      index >= 0;
      index--
    ) {
      const branch = this.steps[index][name];
      if (branch !== null && branch !== undefined) return branch;
    }
    return null;
  }
  get active() {
    return ["loading", "running", "stopping"].includes(this.status);
  }
}
function sameSource(left, right) {
  return (
    !!left &&
    !!right &&
    ["repo", "revision", "weights_sha256"].every((k) => left[k] === right[k])
  );
}
function draftEdit(selection, contract, kind, operation, start, end, factor) {
  const mapping = contract?.tensors.find((t) => t.name === selection?.tensor);
  if (
    !sameSource(selection?.source_model, contract?.source_model) ||
    !mapping ||
    JSON.stringify(mapping.shape) !== JSON.stringify(selection.shape)
  )
    throw new Error(
      "This selection has no verified edit mapping. Inspect a supported pinned SmolLM2 2-D tensor.",
    );
  const edit = {
    tensor: selection.tensor,
    shape: [...selection.shape],
    kind,
    operation,
  };
  if (kind === "element") {
    edit.row = selection.row;
    edit.col = selection.col;
    if (
      ![edit.row, edit.col].every(
        (v, i) => Number.isSafeInteger(v) && v >= 0 && v < edit.shape[i],
      )
    )
      throw new Error("Invalid inspected native coordinates.");
  } else if (["rows", "columns"].includes(kind)) {
    if (start === "" || end === "")
      throw new Error("Enter both native range endpoints.");
    edit.start = Number(start);
    edit.end = Number(end);
    if (
      !Number.isSafeInteger(edit.start) ||
      !Number.isSafeInteger(edit.end) ||
      edit.start < 0 ||
      edit.start >= edit.end ||
      edit.end > edit.shape[kind === "rows" ? 0 : 1]
    )
      throw new Error("Use a nonempty in-bounds [start,end) range.");
  } else throw new Error("Unknown edit target.");
  if (operation === "scale") {
    if (factor === "")
      throw new Error("Enter a finite scale between -100 and 100.");
    edit.scale = Number(factor);
    if (!Number.isFinite(edit.scale) || Math.abs(edit.scale) > 100)
      throw new Error("Scale must be finite, between -100 and 100.");
  } else if (operation !== "zero") throw new Error("Unknown operation.");
  return edit;
}
function editRangeSummary(edit) {
  if (edit.kind === "element")
    return `One original value at row ${edit.row}, column ${edit.col}.`;
  const count =
    (edit.end - edit.start) * edit.shape[edit.kind === "rows" ? 1 : 0];
  return `${edit.kind === "rows" ? "Rows" : "Columns"} ${edit.start}–${edit.end - 1} inclusive (viewer convention) = [${edit.start}, ${edit.end}) with end exclusive (edit convention) · ${count.toLocaleString("en-US")} targeted values. Other axis: entire extent.`;
}
function observationSelection(kind, head, site, architecture) {
  if (!kind || kind === "none") return null;
  if (kind === "attention") {
    if (
      !["string", "number"].includes(typeof head) ||
      head === "" ||
      !Number.isInteger(Number(head)) ||
      Number(head) < 0 ||
      !architecture ||
      Number(head) >= architecture.query_heads ||
      site !== "attention"
    )
      throw new Error(
        "Select attention output and a query head from the verified architecture.",
      );
    return { kind, head: Number(head) };
  }
  if (kind === "logit_lens" && site === "block") return { kind };
  throw new Error(
    "The selected observation requires its matching capture output.",
  );
}
function pairPositions(text, preview) {
  if (
    typeof text !== "string" ||
    text.length > 256 ||
    !preview?.tokens ||
    preview.tokens.length !== 2
  )
    throw new Error("Review a token preview and enter 1–8 position pairs.");
  const lines = text.trim().split(/\n/);
  if (lines.length < 1 || lines.length > 8)
    throw new Error("Use 1–8 position pairs.");
  const seen = new Set();
  return lines.map((line) => {
    const match = line.trim().match(/^(\d{1,3})\s*,\s*(\d{1,3})$/);
    if (!match) throw new Error("Use one A,B token position pair per line.");
    const a = Number(match[1]),
      b = Number(match[2]),
      key = a + "," + b;
    if (
      a >= preview.tokens[0].length ||
      b >= preview.tokens[1].length ||
      seen.has(key)
    )
      throw new Error("Each pair must be unique and inside both token lists.");
    seen.add(key);
    return { a, b };
  });
}
function sweepRequest(values, source_model, architecture) {
  if (!source_model || !architecture)
    throw new Error("Pinned model architecture metadata is required.");
  const lines = values.targets.trim().split(/\n/);
  if (lines.length < 1 || lines.length > 2)
    throw new Error(
      "Choose 1–2 explicit targets or one layer L for all output heads.",
    );
  const targets = lines.map((line) => {
    let m = line.trim().match(/^layer (\d+)$/);
    if (m) return { kind: "layer_heads", layer: Number(m[1]) };
    m = line.trim().match(/^(head|query_head) (\d+) (\d+)$/);
    if (m) return { kind: m[1], layer: Number(m[2]), head: Number(m[3]) };
    m = line.trim().match(/^offset (\d+) (\d+(?:,\d+){0,7}) (\d+)$/);
    if (m)
      return {
        kind: "offset",
        layer: Number(m[1]),
        heads: m[2].split(",").map(Number),
        offset: Number(m[3]),
      };
    throw new Error(
      "Use layer L, head L H, query_head L H or offset L H[,H...] O.",
    );
  });
  if (
    targets.some(
      (t) =>
        !Number.isSafeInteger(t.layer) ||
        t.layer >= architecture.layers ||
        (t.kind === "layer_heads"
          ? targets.length !== 1
          : t.kind === "offset"
            ? t.offset >= architecture.head_dim ||
              t.heads.some((h) => h >= architecture.query_heads) ||
              new Set(t.heads).size !== t.heads.length
            : t.head >= architecture.query_heads),
    )
  )
    throw new Error(
      "Use native layers, heads and offsets from the verified architecture; layer mode must stand alone.",
    );
  if (new Set(targets.map((t) => JSON.stringify(t))).size !== targets.length)
    throw new Error("Choose distinct targets.");
  const prompts = values.prompts;
  if (
    prompts.length < 1 ||
    prompts.length > 2 ||
    prompts.some(
      (p) => typeof p !== "string" || !p.trim() || utf8Size(p) > 2048,
    )
  )
    throw new Error("Choose 1–2 nonempty prompts of at most 2048 UTF-8 bytes.");
  if (
    targets[0].kind === "layer_heads" &&
    ((1 + 2 * architecture.query_heads) * prompts.length > 32 ||
      values.operation !== "zero")
  )
    throw new Error(
      "All-head ablation requires zero and must fit the 32-record cap; choose one prompt or an explicit subset.",
    );
  const seed = Number(values.seed),
    capture_layer = Number(values.layer);
  if (
    !["string", "number"].includes(typeof values.seed) ||
    !["string", "number"].includes(typeof values.layer) ||
    values.seed === "" ||
    values.layer === "" ||
    !Number.isSafeInteger(seed) ||
    seed < 0 ||
    seed > 4294967295 ||
    !Number.isInteger(capture_layer) ||
    capture_layer < 0 ||
    capture_layer >= architecture.layers
  )
    throw new Error("Use a seed 0–4294967295 and a native capture layer.");
  const request = {
    mode: "sweep",
    source_model,
    prompts,
    targets,
    seed,
    capture_layer,
    activation_site: values.site,
    operation: values.operation,
  };
  if (values.operation === "scale") {
    request.scale = Number(values.scale);
    if (
      !["string", "number"].includes(typeof values.scale) ||
      values.scale === "" ||
      !Number.isFinite(request.scale) ||
      Math.abs(request.scale) > 100
    )
      throw new Error("Use a finite scale between -100 and 100.");
  } else if (values.operation !== "zero") throw new Error("Use zero or scale.");
  return request;
}
const EXPERIMENT_BYTES = 1024 * 1024,
  EXPERIMENT_RUNS = 8;
const utf8Size = (text) => {
  let size = 0;
  for (const char of text) {
    const code = char.codePointAt(0);
    size += code < 128 ? 1 : code < 2048 ? 2 : code < 65536 ? 3 : 4;
  }
  return size;
};
function finiteJSON(value) {
  const visit = (value) => {
    if (typeof value === "number" && !Number.isFinite(value))
      throw new Error("Nonfinite experiment data cannot be saved.");
    if (value && typeof value === "object")
      for (const item of Object.values(value)) visit(item);
  };
  visit(value);
  return JSON.stringify(value, null, 2);
}
function picked(value, keys) {
  const out = {};
  for (const key of keys)
    if (value && Object.hasOwn(value, key)) out[key] = value[key];
  return out;
}
function workerCleanupConfirmed(snapshot) {
  return (
    snapshot.worker_alive === false &&
    !snapshot.details?.cleanup_pending &&
    [
      "idle",
      "complete",
      "cancelled",
      "error",
      "resource_limit",
      "time_limit",
      "client_timeout",
      "connection_lost",
    ].includes(snapshot.status)
  );
}
function recordRequestSettings(request, details, includePrompt) {
  const settings = picked(
    request,
    /* schema-fields: generation_settings */ [
      "mode",
      "max_new_tokens",
      "layer",
      "activation_site",
    ] /* end-schema-fields */,
  );
  settings.activation_site = settings.activation_site || "block";
  if (request.observation)
    settings.observation = picked(
      request.observation,
      /* schema-fields: observation_request */ [
        "kind",
        "head",
      ] /* end-schema-fields */,
    );
  settings.source_model = picked(
    request.source_model,
    /* schema-fields: source_model */ [
      "repo",
      "revision",
      "weights_sha256",
    ] /* end-schema-fields */,
  );
  settings.edits = (request.edits || []).map((edit) =>
    picked(
      edit,
      /* schema-fields: edit_projection */ [
        "tensor",
        "shape",
        "kind",
        "operation",
        "start",
        "end",
        "row",
        "col",
        "scale",
      ] /* end-schema-fields */,
    ),
  );
  if (settings.edits.length > 8)
    throw new Error("Experiment edit cap exceeded.");
  recordModeSettings(request, settings, details, includePrompt);
  return settings;
}
function recordModeSettings(request, settings, details, includePrompt) {
  if (request.mode === "sweep") {
    delete settings.edits;
    Object.assign(
      settings,
      picked(
        request,
        /* schema-fields: sweep_settings */ [
          "targets",
          "operation",
          "scale",
          "seed",
          "capture_layer",
        ] /* end-schema-fields */,
      ),
    );
    if (includePrompt) {
      settings.prompts = [...request.prompts];
      settings.plan_digest = request.plan_digest;
    }
  } else if (request.mode === "prompt_pair") {
    delete settings.edits;
    settings.positions = request.positions.map((p) =>
      picked(
        p,
        /* schema-fields: pair_positions */ ["a", "b"] /* end-schema-fields */,
      ),
    );
    if (includePrompt) {
      settings.prompts = [...request.prompts];
      settings.preview_digest = request.preview_digest;
    }
  } else if (includePrompt) {
    settings.prompt = request.prompt;
    if (Array.isArray(details.prompt_ids))
      settings.prompt_ids = [...details.prompt_ids];
  }
}
const recordBranch = (value) =>
  value === null
    ? null
    : picked(
        value,
        /* schema-fields: sequence */ [
          "token_id",
          "token_piece",
          "generated_text",
          "generated_ids",
          "eos",
          "compute_ms",
          "compute_total_ms",
        ] /* end-schema-fields */,
      );
function recordStepCore(snapshot, request, includePrompt, step, index) {
  if (
    step.index !== index ||
    !Array.isArray(step.activation) ||
    !step.activation.length ||
    step.activation.length * snapshot.steps.length > 18432
  )
    throw new Error("Invalid experiment trace.");
  const out = picked(
    step,
    /* schema-fields: step */ [
      "index",
      "phase",
      "position",
      "input_token_id",
      "token_id",
      "token_piece",
      "generated_text",
      "compute_ms",
      "compute_total_ms",
      "layer",
      "activation_site",
      "activation_kind",
      "activation_branch",
      "alignment",
      "score_kind",
      "eos",
    ] /* end-schema-fields */,
  );
  // The last prompt token is also prompt content; omit it at prefill when excluded.
  if (!includePrompt && (index === 0 || request.mode === "sweep"))
    delete out.input_token_id;
  out.activation = [...step.activation];
  return out;
}
function projectSweepStep(step, out) {
  if (step.sweep) {
    out.mode = "sweep";
    out.sweep = picked(
      step.sweep,
      /* schema-fields: sweep_record */ [
        "record_id",
        "case_id",
        "role",
        "prompt_index",
        "selected_cells",
        "changed_cells",
        "parameter_delta_l2",
        "restoration_verified",
        "metrics",
        "candidates",
      ] /* end-schema-fields */,
    );
  }
}
function projectPairStep(step, out, includePrompt) {
  if (step.prompt_pair) {
    out.mode = "prompt_pair";
    const p = step.prompt_pair;
    out.prompt_pair = {
      token_equal: p.token_equal,
      prefix_equal: p.prefix_equal,
      metrics: picked(
        p.metrics,
        /* schema-fields: pair_metrics */ [
          "a_l2",
          "b_l2",
          "delta_l2",
          "cosine",
        ] /* end-schema-fields */,
      ),
    };
    for (const key of ["a", "b"])
      out.prompt_pair[key] = picked(
        p[key],
        includePrompt
          ? ["position", "token_id", "token_piece", "activation"]
          : ["position", "activation"],
      );
  }
}
function projectAttentionStep(step, out, includePrompt) {
  if (step.attention) {
    out.attention = picked(
      step.attention,
      /* schema-fields: attention_projection */ [
        "layer",
        "query_head",
        "kv_head",
        "head_dim",
        "query_position",
        "key_positions",
        "probabilities",
        "semantics",
      ] /* end-schema-fields */,
    );
    if (includePrompt)
      out.attention.key_token_ids = step.attention.key_token_ids;
  }
}
function projectLensStep(step, out) {
  if (step.logit_lens) {
    out.logit_lens = picked(
      step.logit_lens,
      /* schema-fields: lens_projection */ [
        "layer",
        "position",
        "score_kind",
        "lens_argmax_id",
        "final_argmax_id",
        "semantics",
      ] /* end-schema-fields */,
    );
    out.logit_lens.candidates = step.logit_lens.candidates.map((entry) =>
      picked(
        entry,
        /* schema-fields: lens_candidate */ [
          "id",
          "piece",
          "lens_logit",
          "final_logit",
          "delta_lens_minus_final",
        ] /* end-schema-fields */,
      ),
    );
  }
}
function projectLogitStep(step, out) {
  if (step.top_logits)
    out.top_logits = step.top_logits.map((entry) =>
      picked(
        entry,
        /* schema-fields: logit */ ["id", "value"] /* end-schema-fields */,
      ),
    );
}
function projectComparisonStep(step, out, branch) {
  if (Object.hasOwn(step, "baseline")) {
    out.baseline = branch(step.baseline);
    out.edited = branch(step.edited);
    out.candidates = step.candidates.map((entry) =>
      picked(
        entry,
        /* schema-fields: candidate */ [
          "id",
          "piece",
          "baseline_logit",
          "edited_logit",
          "delta",
        ] /* end-schema-fields */,
      ),
    );
  }
}
function recordTraceSteps(snapshot, request, includePrompt) {
  const steps = snapshot.steps.map((step, index) => {
    const out = recordStepCore(snapshot, request, includePrompt, step, index);
    projectSweepStep(step, out);
    projectPairStep(step, out, includePrompt);
    projectAttentionStep(step, out, includePrompt);
    projectLensStep(step, out);
    projectLogitStep(step, out);
    projectComparisonStep(step, out, recordBranch);
    return out;
  });
  return steps;
}
function recordPrivacy(request, settings, includePrompt) {
  return {
    generated_outputs_included: request.mode !== "prompt_pair",
    prompt_included: includePrompt,
    request_replayable:
      includePrompt &&
      (typeof settings.prompt === "string" || Array.isArray(settings.prompts)),
    note:
      request.mode === "prompt_pair"
        ? includePrompt
          ? "Private prompts, source tokens and activation captures: keep this file private."
          : "Prompts and source tokens omitted; captures remain private and exact request replay is unavailable."
        : includePrompt
          ? "Private prompt and generated data: keep this file private."
          : "Prompt omitted; exact request replay unavailable. Generated outputs may still reveal prompt content.",
  };
}
function recordLimits(request, details) {
  return request.mode === "sweep"
    ? {
        interventions: details.sweep_plan?.cases?.length,
        prompts: request.prompts.length,
        probes_per_prompt: 1,
        records: details.sweep_plan?.records,
        prefills: details.sweep_plan?.prefills,
        trace_cap: 32,
        edits_per_intervention: 8,
        snapshot_cells: 65536,
        wall_seconds: 120,
        worker_cpu_seconds: 90,
      }
    : request.mode === "prompt_pair"
      ? {
          prompt_tokens_each: 128,
          prefills: 2,
          positions: 8,
          new_tokens: 0,
          trace_steps: 8,
          vector_equivalents: 24,
        }
      : { prompt_tokens: 128, new_tokens: 32, trace_steps: 32, edits: 8 };
}
function recordEnvelope(
  request,
  snapshot,
  settings,
  details,
  steps,
  includePrompt,
  createdAt,
) {
  return {
    schema: "weight-atlas-experiment-v1",
    created_at: createdAt,
    status: snapshot.status,
    complete: snapshot.status === "complete",
    worker_cleanup_confirmed: workerCleanupConfirmed(snapshot),
    termination: snapshot.termination || null,
    privacy: recordPrivacy(request, settings, includePrompt),
    request: settings,
    settings: {
      seed: 0,
      sampling:
        request.mode === "prompt_pair"
          ? "none (no generation)"
          : request.mode === "sweep"
            ? "none (fixed-context score probe)"
            : "greedy",
      dtype: "float32",
      deterministic_algorithms: true,
    },
    runtime: picked(
      details.runtime,
      /* schema-fields: runtime */ [
        "python",
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "platform",
        "machine",
        "device",
        "dtype",
        "sampling",
        "seed",
        "deterministic_algorithms",
        "numerical_threads",
        "attention_backend",
      ] /* end-schema-fields */,
    ),
    limits: recordLimits(request, details),
    summary: picked(
      details,
      /* schema-fields: summary */ [
        "reason",
        "comparison_phase",
        "generated_tokens",
        "record_count",
        "coverage",
        "sweep_coverage",
        "sweep_current",
        "compute_total_ms",
        "load_ms",
        "error",
      ] /* end-schema-fields */,
    ),
    baseline: picked(
      details.baseline,
      /* schema-fields: branch_summary */ [
        "generated_ids",
        "generated_text",
        "reason",
      ] /* end-schema-fields */,
    ),
    edited: picked(
      details.edited,
      /* schema-fields: branch_summary */ [
        "generated_ids",
        "generated_text",
        "reason",
      ] /* end-schema-fields */,
    ),
    steps,
  };
}
function experimentRecord(
  request,
  snapshot,
  { includePrompt = false, createdAt = new Date().toISOString() } = {},
) {
  if (!request || !Array.isArray(snapshot.steps) || snapshot.steps.length > 32)
    throw new Error("A bounded accepted run is required.");
  const details = snapshot.details || {},
    settings = recordRequestSettings(request, details, includePrompt);
  const steps = recordTraceSteps(snapshot, request, includePrompt);
  const result = recordEnvelope(
    request,
    snapshot,
    settings,
    details,
    steps,
    includePrompt,
    createdAt,
  );
  if (request.mode === "sweep")
    result.sweep_plan = picked(
      details.sweep_plan,
      /* schema-fields: sweep_plan */ [
        "version",
        "scope",
        "architecture",
        "intervention_semantics",
        "control_semantics",
        "seed",
        "control_version",
        "source_model",
        "targets",
        "cases",
        "prompt_count",
        "records",
        "prefills",
        "capture_layer",
        "activation_site",
        "coverage",
        "limits",
      ] /* end-schema-fields */,
    );
  const json = finiteJSON(result);
  if (utf8Size(json) > EXPERIMENT_BYTES)
    throw new Error(
      "This run exceeds the 1 MiB export cap. No file was saved.",
    );
  return JSON.parse(json); // Detached from mutable UI/snapshot objects.
}
function redactExperiment(record) {
  const clean = JSON.parse(finiteJSON(record));
  delete clean.request.prompt;
  delete clean.request.prompt_ids;
  delete clean.request.prompts;
  delete clean.request.preview_digest;
  delete clean.request.plan_digest;
  if (clean.steps[0]) delete clean.steps[0].input_token_id;
  for (const step of clean.steps) {
    if (clean.request.mode === "sweep") delete step.input_token_id;
    if (step.attention) delete step.attention.key_token_ids;
    if (step.prompt_pair)
      for (const key of ["a", "b"]) {
        delete step.prompt_pair[key].token_id;
        delete step.prompt_pair[key].token_piece;
      }
  }
  clean.privacy = {
    generated_outputs_included: clean.request.mode !== "prompt_pair",
    prompt_included: false,
    request_replayable: false,
    note:
      clean.request.mode === "prompt_pair"
        ? "Prompts and source tokens omitted; captures remain private and exact request replay is unavailable."
        : "Prompt omitted; exact request replay unavailable. Generated outputs may still reveal prompt content.",
  };
  return clean;
}
class AtlasExperimentLog {
  constructor({ maxBytes = EXPERIMENT_BYTES, maxRuns = EXPERIMENT_RUNS } = {}) {
    this.maxBytes = maxBytes;
    this.maxRuns = maxRuns;
    this.createdAt = new Date().toISOString();
    this.records = [];
  }
  envelope(records = this.records) {
    return {
      schema: "weight-atlas-session-log-v1",
      created_at: this.createdAt,
      persistence:
        "browser session memory; durable only after explicit file download",
      limits: { bytes: this.maxBytes, runs: this.maxRuns },
      records,
    };
  }
  text() {
    return finiteJSON(this.envelope());
  }
  get bytes() {
    return utf8Size(this.text());
  }
  append(record) {
    const detached = JSON.parse(finiteJSON(record)),
      next = [...this.records, detached];
    if (
      next.length > this.maxRuns ||
      utf8Size(finiteJSON(this.envelope(next))) > this.maxBytes
    )
      throw new Error(
        "Session log is full. This run is held pending; export it and the log, then explicitly clear/discard before starting another run.",
      );
    this.records = next;
  }
  confirmCleanup(index) {
    if (
      !Number.isSafeInteger(index) ||
      index < 0 ||
      index >= this.records.length
    )
      throw new Error("Missing interrupted log record.");
    // false -> true reduces serialized size; preserve the original outcome/trace.
    this.records[index] = {
      ...this.records[index],
      worker_cleanup_confirmed: true,
    };
  }
  clear() {
    this.records = [];
    this.createdAt = new Date().toISOString();
  }
}

// The closed import codec is requested only by an explicit archive action.
const AtlasExperimentImport =
  typeof module !== "undefined" ? require("./inference-import.js") : null;
const loadExperimentImport = (() => {
  let importCodecPromise = null;
  let importedExperimentCodec;
  return function loadExperimentImport() {
    const ready = () => {
      const c = importedExperimentCodec;
      if (!c || typeof c.read !== "function" || typeof c.append !== "function")
        throw new Error("Archive importer did not initialize");
      return c;
    };
    if (importCodecPromise) return importCodecPromise;
    importCodecPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.atlasRegisterImport = (codec) => {
        importedExperimentCodec = codec;
      };
      script.src = "/inference-import.js";
      script.async = true;
      const failed = () => {
        clearTimeout(timeout);
        script.remove();
        reject(
          new Error(
            "Archive importer unavailable; existing records unchanged. Retry explicitly.",
          ),
        );
      };
      const timeout = setTimeout(failed, 15000);
      script.onload = () => {
        clearTimeout(timeout);
        script.remove();
        try {
          resolve(ready());
        } catch (error) {
          reject(error);
        }
      };
      script.onerror = failed;
      document.head.append(script);
    }).catch((error) => {
      importCodecPromise = null;
      throw error;
    });
    return importCodecPromise;
  };
})();

class AtlasLogStorage {
  constructor(storage, tabStorage, makeOwner) {
    this.storage = storage;
    this.tabStorage = tabStorage;
    this.makeOwner = makeOwner;
    this.expected = undefined;
    this.owner = null;
  }
  key() {
    let owner =
      this.owner || this.tabStorage.getItem("weight-atlas.experiment-owner.v1");
    if (!owner) {
      owner = this.makeOwner();
      if (!/^[a-zA-Z0-9-]{1,80}$/.test(owner))
        throw new Error("Invalid local log owner");
      this.owner = owner;
    }
    if (!/^[a-zA-Z0-9-]{1,80}$/.test(owner))
      throw new Error("Invalid local log owner");
    return "weight-atlas.experiments.v1:" + owner;
  }
  async mutate(locks, operation) {
    if (!locks?.request)
      throw new Error("Browser archive writes require Web Locks");
    return locks.request("weight-atlas-experiment-storage-v1", () => {
      this.key();
      this.tabStorage.setItem(
        "weight-atlas.experiment-owner.v1",
        this.owner || this.key().split(":")[1],
      );
      return operation(this);
    });
  }
  list() {
    const keys = [];
    for (let i = 0; i < this.storage.length; i++) {
      const k = this.storage.key(i);
      if (/^weight-atlas\.experiments\.v1:[a-zA-Z0-9-]{1,80}$/.test(k))
        keys.push(k);
    }
    if (keys.length > 8)
      throw new Error(
        "More than 8 saved archives; export/remove old browser data before creating another",
      );
    return keys.sort();
  }
  readKey(key) {
    if (!/^weight-atlas\.experiments\.v1:[a-zA-Z0-9-]{1,80}$/.test(key))
      throw new Error("Invalid archive key");
    const text = this.storage.getItem(key);
    if (text !== null && utf8Size(text) > EXPERIMENT_BYTES)
      throw new Error("Saved log exceeds 1 MiB");
    return text;
  }
  save(text) {
    if (utf8Size(text) > EXPERIMENT_BYTES)
      throw new Error("Durable log exceeds 1 MiB");
    const key = this.key(),
      current = this.readKey(key);
    if (this.expected === undefined && current !== null)
      throw new Error(
        "A saved log exists: explicitly restore or delete it before saving",
      );
    if (this.expected !== undefined && current !== this.expected)
      throw new Error(
        "Saved archive changed in another tab; export your records and reload the archive before saving",
      );
    if (current === null && this.list().length >= 8)
      throw new Error("8 browser archive slots are full");
    this.storage.setItem(key, text);
    this.expected = text;
  }
  read() {
    return this.readKey(this.key());
  }
  adopt(text) {
    if (this.readKey(this.key()) !== text)
      throw new Error("Saved archive changed during restore");
    this.expected = text;
  }
  clear() {
    this.storage.removeItem(this.key());
    this.expected = null;
  }
}
class AtlasFileJournal {
  constructor(handle) {
    this.handle = handle;
    this.writing = false;
  }
  async save(text) {
    if (this.writing) throw new Error("File journal write still pending");
    if (utf8Size(text) > EXPERIMENT_BYTES)
      throw new Error("File journal exceeds 1 MiB");
    this.writing = true;
    let writable;
    try {
      writable = await this.handle.createWritable();
      await writable.write(text);
      await writable.close();
    } catch (error) {
      try {
        await writable?.abort();
      } catch {}
      throw error;
    } finally {
      this.writing = false;
    }
  }
}

if (typeof module !== "undefined")
  module.exports = {
    AtlasExperimentImport,
    loadExperimentImport,
    AtlasLogStorage,
    AtlasFileJournal,
    editRangeSummary,
    AtlasPlayback,
    draftEdit,
    sameSource,
    experimentRecord,
    AtlasExperimentLog,
    utf8Size,
    redactExperiment,
    observationSelection,
    pairPositions,
    sweepRequest,
  };
if (typeof document !== "undefined") {
  (() => {
    const el = (id) => document.getElementById("infer-" + id),
      play = new AtlasPlayback();
    let epoch = 0,
      playbackEpoch = 0,
      session = null,
      pending = false,
      timer = null,
      polling = null,
      enabled = false,
      details = {},
      architecture = null,
      contract = null,
      draft = [],
      selection = null,
      availability = "Checking local inference availability…",
      acceptedRequest = null,
      loggingRun = false,
      promptConsent = false,
      loggedRun = false,
      pendingLog = null,
      logError = "",
      lastWorkerAlive = null,
      observationModes = [],
      pairSupported = false,
      pairPreview = null,
      sweepSupported = false,
      sweepPlan = null,
      sweepKey = null,
      planEpoch = 0,
      planning = false,
      cleanupAfterTransport = false,
      incompleteLog = false,
      logRecordIndex = null;
    let busyElsewhere = false,
      acceptedAt = null,
      readinessEpoch = 0;
    const experimentLog = new AtlasExperimentLog();
    let durableStorage = null,
      fileJournal = null,
      saving = false,
      durableError = "",
      durableNote =
        "Persistence off. No automatic restore or worker session adoption.",
      importing = false,
      durableDirty = false,
      archivePending = false,
      archiveEpoch = 0;
    const storage = () =>
      durableStorage ||
      (durableStorage = new AtlasLogStorage(
        globalThis.localStorage,
        globalThis.sessionStorage,
        () => globalThis.crypto.randomUUID(),
      ));
    const archiveOptions = () => ({
      experimentRecord,
      redactExperiment,
      includePrompt: !!el("import-prompts").checked,
    });
    const durableText = () =>
      finiteJSON(
        experimentLog.envelope(
          el("durable-prompts").checked
            ? experimentLog.records
            : experimentLog.records.map(redactExperiment),
        ),
      );
    async function saveDurable() {
      if (saving) {
        durableDirty = true;
        return;
      }
      if (!el("persist").checked && !fileJournal) {
        durableError = "";
        render();
        return;
      }
      saving = true;
      durableError = "";
      render();
      try {
        durableDirty = false;
        const text = durableText();
        if (el("persist").checked) {
          await storage().mutate(globalThis.navigator?.locks, (s) =>
            s.save(text),
          );
        }
        if (fileJournal) await fileJournal.save(text);
        durableNote =
          "Saved " +
          experimentLog.records.length +
          " bounded archived records" +
          (el("persist").checked ? " in this tab’s browser slot" : "") +
          (fileJournal ? " and/or the selected file" : "") +
          ".";
      } catch (error) {
        durableError =
          "Private log persistence failed: " +
          error.message +
          ". Records remain in memory; export, retry or disable the failed sink before another run.";
      } finally {
        saving = false;
        render();
        if (durableDirty && !durableError) {
          durableDirty = false;
          saveDurable();
        }
      }
    }
    async function appendArchive(text, adopt = false, id = archiveEpoch) {
      const codec = await loadExperimentImport();
      if (id !== archiveEpoch) return;
      const records = codec.read(text, archiveOptions());
      codec.append(experimentLog, records, {
        beforeCommit: () => {
          if (adopt) storage().adopt(text);
        },
      });
      write(
        "import-status",
        records.length +
          " archived records appended locally. Historical provenance is unverified. Run inputs and worker ownership are unchanged.",
      );
      saveDurable();
      render();
    }

    function cancelArchive() {
      if (!archivePending) return;
      archiveEpoch++;
      archivePending = false;
      importing = false;
      write(
        "import-status",
        "Archive operation cancelled. Existing records and run inputs unchanged.",
      );
      render();
    }
    async function importArchive(getText, adopt = false) {
      if (pending || play.active || saving || importing) return;
      const id = ++archiveEpoch;
      archivePending = importing = true;
      render();
      try {
        const text = await getText();
        if (id === archiveEpoch) await appendArchive(text, adopt, id);
      } catch (error) {
        if (id === archiveEpoch)
          write(
            "import-status",
            error.message + " Existing records unchanged.",
          );
      } finally {
        if (id === archiveEpoch) {
          archivePending = importing = false;
          render();
        }
      }
    }

    const write = (id, value) => {
      el(id).textContent = value;
    };
    async function api(action, data) {
      const response = await fetch(
        "/api/inference" + (action ? "/" + action : ""),
        action
          ? {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-Atlas-Local": "1",
              },
              body: JSON.stringify(data),
            }
          : {},
      );
      const body = await response.json();
      if (!response.ok) {
        const error = new Error(body.error || "Inference request failed");
        error.code = body.code;
        error.status = response.status;
        throw error;
      }
      return body;
    }
    function message(value) {
      write("error", value);
      el("error").hidden = !value;
    }
    function updateSelection(value) {
      selection = value;
      if (value) {
        const coordinate =
          value[el("kind").value === "columns" ? "col" : "row"];
        el("range-start").value = String(coordinate);
        el("range-end").value = String(coordinate + 1);
      }
      render();
    }
    function renderDraft() {
      const active =
          pending ||
          play.active ||
          ["prompt_pair", "sweep"].includes(el("task").value),
        mapping = contract?.tensors.find((t) => t.name === selection?.tensor);
      const supported =
        sameSource(selection?.source_model, contract?.source_model) &&
        !!mapping &&
        JSON.stringify(mapping.shape) === JSON.stringify(selection.shape);
      write(
        "selection",
        !enabled
          ? "View only · no local inference coordinator is available. Inspecting a weight does not change it."
          : selection
            ? supported
              ? `${selection.tensor} · native [${selection.row}, ${selection.col}] · shape ${selection.shape.join(" × ")}${mapping.aliases.length ? " · tied: edits also change " + mapping.aliases.join(", ") : ""}`
              : "View only · this source or tensor has no verified edit mapping for the connected coordinator."
            : "Inspect a native 2-D weight above to select an edit target.",
      );
      document.getElementById("edit-readiness").textContent = !enabled
        ? "View only · local inference is unavailable."
        : supported
          ? "This selected weight supports an in-memory edit. Review the draft before running."
          : selection
            ? "View only · no verified edit mapping for this source/tensor."
            : "Inspect a supported native matrix to choose an edit target.";
      document.getElementById("prepare-edit").disabled =
        !enabled || !supported || pending || play.active;
      document.getElementById("prepare-edit").hidden = !enabled;
      document.getElementById("nav-experiment").hidden = !enabled;
      const inspected = selection
        ? `${selection.tensor} [${selection.row}, ${selection.col}]`
        : "none";
      const targets = [...new Set(draft.map((e) => e.tensor))];
      write(
        "target-summary",
        `Inspected: ${inspected}. Draft edit targets: ${targets.length ? targets.join(", ") : "none (empty comparison control)"}. Capture: layer ${el("layer").value}, ${el("site").value || "block"} output${el("observation").value === "attention" ? ", query head " + el("head-index").value : ""}. Inspection, edit targets and capture layer are independent; changing one does not silently change the others.`,
      );
      if (supported) {
        try {
          write(
            "range-preview",
            editRangeSummary(
              draftEdit(
                selection,
                contract,
                el("kind").value,
                el("operation").value,
                el("range-start").value,
                el("range-end").value,
                el("factor").value,
              ),
            ),
          );
        } catch (error) {
          write("range-preview", error.message);
        }
      } else
        write(
          "range-preview",
          "A supported selection is required before an edit range can be previewed.",
        );
      el("add").disabled = active || !supported || draft.length >= 8;
      write(
        "head",
        selection?.tensor?.endsWith(".o_proj.weight")
          ? "Ablate inspected output head (columns)"
          : "Zero inspected query rows (query intervention)",
      );
      el("head").disabled =
        active ||
        !supported ||
        draft.length >= 8 ||
        !architecture ||
        !/^model\.layers\.\d+\.self_attn\.(q_proj|o_proj)\.weight$/.test(
          selection.tensor,
        );
      el("clear-edits").disabled = active || !draft.length;
      for (const name of ["kind", "operation"]) el(name).disabled = active;
      for (const name of ["range-start", "range-end"])
        el(name).disabled = active || el("kind").value === "element";
      el("factor").disabled = active || el("operation").value !== "scale";
      const list = el("edits");
      list.replaceChildren();
      draft.forEach((edit, index) => {
        const item = document.createElement("li"),
          remove = document.createElement("button");
        item.textContent = `${edit.tensor} · ${edit.kind === "element" ? `[${edit.row},${edit.col}]` : `${edit.kind} [${edit.start},${edit.end})`} · ${edit.operation}${edit.operation === "scale" ? " × " + edit.scale : ""}${edit.tensor === "model.embed_tokens.weight" ? " (also tied output head)" : ""} `;
        const range = document.createElement("span");
        range.textContent = " " + editRangeSummary(edit) + " ";
        item.append(range);
        remove.type = "button";
        remove.textContent = "Remove";
        remove.disabled = active;
        remove.addEventListener("click", () => {
          draft.splice(index, 1);
          render();
        });
        item.append(remove);
        list.append(item);
      });
    }
    function renderComparison(step) {
      const paired = step && Object.hasOwn(step, "baseline");
      const baseline = paired ? play.branchAtCursor("baseline") : null,
        edited = paired ? play.branchAtCursor("edited") : null;
      write(
        "baseline-output",
        paired
          ? (baseline?.generated_text ?? "Baseline ended.")
          : "Baseline text will appear here.",
      );
      write(
        "baseline-ids",
        paired
          ? "Token IDs: " + (baseline?.generated_ids ?? []).join(", ")
          : "",
      );
      write(
        "edited-ids",
        paired ? "Token IDs: " + (edited?.generated_ids ?? []).join(", ") : "",
      );
      if (paired)
        write("output", edited?.generated_text ?? "Edited branch ended.");
      write(
        "score-context",
        !paired
          ? "Scores are raw logits, not probabilities."
          : step.alignment === "matched_prefix"
            ? "Matched consumed prefix · Δ = edited − baseline raw logit."
            : step.alignment === "different_prefix"
              ? "Different generated prefixes · free-running differences combine weight and context effects, not matched-context causal effects."
              : "One branch ended · unavailable scores and deltas are shown as —.",
      );
      const body = el("scores");
      body.replaceChildren();
      for (const candidate of step?.candidates || []) {
        const row = document.createElement("tr");
        for (const value of [
          JSON.stringify(candidate.piece) + " / " + candidate.id,
          ...["baseline_logit", "edited_logit", "delta"].map((k) =>
            candidate[k] === null ? "—" : candidate[k].toFixed(6),
          ),
        ]) {
          const cell = document.createElement("td");
          cell.textContent = value;
          row.append(cell);
        }
        body.append(row);
      }
    }
    function captureLog(snapshot) {
      lastWorkerAlive = snapshot.worker_alive ?? null;
      if (loggedRun && incompleteLog && workerCleanupConfirmed(snapshot)) {
        if (pendingLog) pendingLog.worker_cleanup_confirmed = true;
        else if (logRecordIndex !== null)
          experimentLog.confirmCleanup(logRecordIndex);
        incompleteLog = false;
        saveDurable();
      }
      if (
        !loggingRun ||
        loggedRun ||
        !acceptedRequest ||
        ["loading", "running", "stopping"].includes(snapshot.status)
      )
        return;
      loggedRun = true;
      incompleteLog =
        snapshot.status === "connection_lost" &&
        !workerCleanupConfirmed(snapshot);
      try {
        const record = experimentRecord(acceptedRequest, snapshot, {
          includePrompt: promptConsent,
        });
        pendingLog = record;
        const index = experimentLog.records.length;
        experimentLog.append(record);
        logRecordIndex = index;
        pendingLog = null;
        logError = "";
        saveDurable();
      } catch (error) {
        logError = error.message;
      }
    }
    function downloadExperiment(text, name) {
      if (utf8Size(text) > EXPERIMENT_BYTES)
        throw new Error(
          "The 1 MiB export cap was exceeded. No file was saved.",
        );
      const url = URL.createObjectURL(
          new Blob([text], { type: "application/json;charset=utf-8" }),
        ),
        link = document.createElement("a");
      link.href = url;
      link.download = name;
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    }
    function renderObservation(step) {
      const kind = el("observation").value,
        active =
          pending ||
          play.active ||
          ["prompt_pair", "sweep"].includes(el("task").value);
      el("observation").disabled = active || !observationModes.length;
      el("head-index").disabled = active || kind !== "attention";
      el("site").disabled =
        pending ||
        play.active ||
        (!["prompt_pair", "sweep"].includes(el("task").value) &&
          kind &&
          kind !== "none");
      const list = el("observation-values");
      list.replaceChildren();
      const a = step?.attention,
        lens = step?.logit_lens,
        branch = step?.activation_branch || "generated";
      el("observation-result").hidden = !a && !lens;
      if (!a && !lens) {
        write(
          "observation-context",
          "No observation at the current playback position.",
        );
        return;
      }
      const row = (values) => {
        const tr = document.createElement("tr");
        for (const value of values) {
          const td = document.createElement("td");
          td.textContent = String(value);
          tr.append(td);
        }
        list.append(tr);
      };
      if (a) {
        write(
          "observation-title",
          "Attention · selected last-query distribution",
        );
        write(
          "observation-context",
          `${branch} branch · layer ${a.layer}, query head ${a.query_head} (KV head ${a.kv_head}, width ${a.head_dim}) · query position ${a.query_position}. Genuine eager post-mask softmax probabilities; only this query/head, not all attention and not causal attribution.`,
        );
        write(
          "observation-columns",
          "Key position · token ID · probability (0–1)",
        );
        a.probabilities.forEach((probability, i) => {
          row([
            a.key_positions[i],
            a.key_token_ids[i],
            probability.toPrecision(7),
          ]);
          const meter = document.createElement("meter");
          meter.min = 0;
          meter.max = 1;
          meter.value = probability;
          meter.title = `Key ${i}: ${probability}`;
          list.children[list.children.length - 1].children[2].append(meter);
        });
      } else {
        write(
          "observation-title",
          "Logit lens · diagnostic intermediate readout",
        );
        write(
          "observation-context",
          `${branch} branch · layer ${lens.layer}, consumed position ${lens.position}. Post-block residual → actual final RMSNorm → tied output head. Raw logits, same consumed prefix; not an early-exit prediction or a causal edit effect.`,
        );
        write(
          "observation-columns",
          "Token / ID · lens logit · final logit · Δ lens − final",
        );
        lens.candidates.forEach((c) =>
          row([
            `${JSON.stringify(c.piece)} / ${c.id}`,
            c.lens_logit.toFixed(6),
            c.final_logit.toFixed(6),
            c.delta_lens_minus_final.toFixed(6),
          ]),
        );
      }
    }
    function acceptPreview(snapshot) {
      if (
        acceptedRequest?.mode === "prompt_pair_preview" &&
        snapshot.status === "complete" &&
        snapshot.details.preview
      ) {
        pairPreview = {
          ...snapshot.details.preview,
          prompts: [...acceptedRequest.prompts],
        };
        el("pairs").value =
          `${pairPreview.tokens[0].length - 1},${pairPreview.tokens[1].length - 1}`;
      }
    }
    function renderPair(step) {
      const selected = el("task").value === "prompt_pair",
        busy = pending || play.active;
      el("task").disabled = busy || (!pairSupported && !sweepSupported);
      el("pair-controls").hidden = !selected;
      el("prompt-b").disabled = busy;
      el("pairs").disabled = busy;
      el("pair-preview").disabled = busy || !pairSupported;
      el("limit").disabled = busy || selected;
      const valid =
        pairPreview &&
        JSON.stringify(pairPreview.prompts) ===
          JSON.stringify([el("prompt").value, el("prompt-b").value]);
      if (selected && !valid) el("start").disabled = true;
      write(
        "start",
        selected
          ? "Compare selected prompt positions"
          : "Run comparison locally",
      );
      write(
        "preview-status",
        valid
          ? "Review these exact tokenizer positions and choose up to 8 explicit pairs. No model was loaded for the preview."
          : "Preview the two prompts to review their exact tokens before comparison.",
      );
      for (const [i, key] of ["a", "b"].entries())
        write(
          "preview-" + key,
          valid
            ? pairPreview.tokens[i]
                .map((t) => `${t.position}: ${t.id} ${JSON.stringify(t.piece)}`)
                .join("\n")
            : "No current token preview.",
        );
      const p = step?.prompt_pair,
        pairRun = acceptedRequest?.mode === "prompt_pair";
      write("left-title", pairRun ? "Prompt A capture" : "Baseline");
      write("right-title", pairRun ? "Prompt B capture" : "Edited");
      write(
        "activation-title",
        pairRun
          ? "Activation difference · B − A"
          : "Actual activation · edited branch (baseline after edited EOS)",
      );
      if (pairRun && !p) {
        write("baseline-output", "No prompt A capture selected in playback.");
        write("output", "No prompt B capture selected in playback.");
      }
      if (acceptedRequest?.mode === "prompt_pair_preview") {
        write("left-title", "Prompt A preview");
        write("right-title", "Prompt B preview");
        write(
          "baseline-output",
          "Tokenizer preview only. Review tokens in the form.",
        );
        write("output", "No model activation capture or text generation.");
      }
      if (p) {
        for (const [key, out] of [
          ["a", "baseline-output"],
          ["b", "output"],
        ])
          write(
            out,
            `Position ${p[key].position} · token ${p[key].token_id} ${JSON.stringify(p[key].token_piece)} · ${p[key].activation.length} captured values`,
          );
        write("baseline-ids", `L2 norm A: ${p.metrics.a_l2}`);
        write("edited-ids", `L2 norm B: ${p.metrics.b_l2}`);
        write(
          "score-context",
          `Explicit positional alignment · token IDs ${p.token_equal ? "equal" : "different"} · consumed prefixes ${p.prefix_equal ? "equal" : "different"}. L2 distance ${p.metrics.delta_l2}; cosine ${p.metrics.cosine === null ? "undefined for a zero norm" : p.metrics.cosine}. Descriptive comparison, not semantic alignment or a causal intervention.`,
        );
      }
    }
    function currentSweepRequest() {
      return sweepRequest(
        {
          targets: el("sweep-targets").value,
          prompts: [
            el("prompt").value,
            ...(el("sweep-second").checked ? [el("sweep-prompt-b").value] : []),
          ],
          seed: el("sweep-seed").value,
          layer: el("layer").value,
          site: el("site").value || "block",
          operation: el("sweep-operation").value,
          scale: el("sweep-scale").value,
        },
        contract?.source_model,
        architecture,
      );
    }
    function invalidateSweep() {
      sweepPlan = null;
      sweepKey = null;
      planning = false;
      ++planEpoch;
    }
    async function previewSweep() {
      if (pending || play.active || planning || !sweepSupported) return;
      const id = ++planEpoch;
      planning = true;
      message("");
      render();
      try {
        const request = currentSweepRequest(),
          key = JSON.stringify(request);
        if (utf8Size(key) > 8120)
          throw new Error(
            "Plan leaves insufficient room for the 8 KiB accepted request.",
          );
        const result = await api("sweep-plan", request);
        if (id !== planEpoch || key !== JSON.stringify(currentSweepRequest()))
          return;
        sweepPlan = result.plan;
        sweepKey = key;
      } catch (error) {
        if (id === planEpoch) {
          sweepPlan = null;
          sweepKey = null;
          message(error.message);
        }
      } finally {
        if (id === planEpoch) {
          planning = false;
          render();
        }
      }
    }
    function renderSweep(step) {
      const selected = el("task").value === "sweep",
        busy = pending || play.active;
      el("sweep-controls").hidden = !selected;
      el("sweep-preview").disabled = busy || planning || !sweepSupported;
      for (const key of [
        "sweep-targets",
        "sweep-seed",
        "sweep-operation",
        "sweep-second",
      ])
        el(key).disabled = busy;
      el("sweep-prompt-b").disabled = busy || !el("sweep-second").checked;
      el("sweep-scale").disabled =
        busy || el("sweep-operation").value !== "scale";
      const supported =
        sameSource(selection?.source_model, contract?.source_model) &&
        !!architecture &&
        /^model\.layers\.\d+\.self_attn\.(q_proj|o_proj)\.weight$/.test(
          selection?.tensor || "",
        ) &&
        contract?.tensors.some(
          (t) =>
            t.name === selection.tensor &&
            JSON.stringify(t.shape) === JSON.stringify(selection.shape),
        );
      el("sweep-use-inspected").disabled = busy || !supported;
      let valid = false;
      if (selected) {
        try {
          valid =
            !!sweepPlan && sweepKey === JSON.stringify(currentSweepRequest());
        } catch {}
        el("limit").disabled = true;
        el("start").disabled = el("start").disabled || planning || !valid;
        write("start", "Run reviewed bounded sweep");
      }
      write(
        "sweep-plan",
        sweepPlan
          ? JSON.stringify(sweepPlan, null, 2)
          : "No reviewed plan. Preview the explicit targets, seed and matched controls before Run.",
      );
      write(
        "sweep-plan-status",
        planning
          ? "Preparing bounded plan…"
          : sweepPlan
            ? `${sweepPlan.targets.length} named targets + empty/matched controls · ${sweepPlan.records} case/prompt probes · ${sweepPlan.prefills} prefills maximum · seed ${sweepPlan.seed}. ${sweepPlan.coverage}. 120 s wall / 90 CPU s total, no retries or automatic continuation.`
            : "Up to 2 explicit targets with 2 prompts, or one layer of output heads with 1 prompt. Existing 32-record cap.",
      );
      const activePlan = details.sweep_plan,
        covered = details.sweep_coverage;
      el("sweep-results-section").hidden = !activePlan;
      write(
        "sweep-progress",
        activePlan
          ? `${covered?.completed_ids?.length ?? 0} / ${activePlan.records} completed · ${covered?.unrun_ids?.length ?? activePlan.records} unrun · current/interrupted ${covered?.interrupted_id || details.sweep_current || "none"}. Status ${play.status}. Unfinished output heads: ${covered?.unfinished_heads?.join(", ") || "none"}. Paired target/control completion is required; no unrun case is scored as zero.`
          : "",
      );
      const list = el("sweep-results");
      list.replaceChildren();
      for (const record of play.steps) {
        if (!record.sweep) continue;
        const value = record.sweep,
          tr = document.createElement("tr");
        for (const v of [
          value.record_id,
          value.role,
          value.metrics.logit_delta_rms,
          value.metrics.softmax_total_variation,
          value.metrics.baseline_argmax_logit_delta,
        ]) {
          const td = document.createElement("td");
          td.textContent = String(v);
          tr.append(td);
        }
        list.append(tr);
      }
      if (acceptedRequest?.mode === "sweep") {
        write("left-title", "Baseline next-token probe");
        write("right-title", "Edited next-token probe");
        write("activation-title", "Edited fixed-context probe activation");
        if (!step?.sweep) {
          write("baseline-output", "No sweep record selected in playback.");
          write("output", "No free-running text generation.");
        }
      }
      if (step?.sweep) {
        const v = step.sweep,
          m = v.metrics;
        write("baseline-output", `Argmax token ID: ${m.baseline_argmax_id}`);
        write("output", `Argmax token ID: ${m.edited_argmax_id}`);
        write("baseline-ids", `${v.record_id} · ${v.role}`);
        write(
          "edited-ids",
          `${v.selected_cells} selected cells, ${v.changed_cells} changed; parameter Δ L2 ${v.parameter_delta_l2}; original bits restored.`,
        );
        write(
          "score-context",
          `Matched fixed original prompt · all-vocabulary RMS logit Δ ${m.logit_delta_rms}, max |Δ| ${m.logit_delta_max_abs}, softmax TV ${m.softmax_total_variation}. Table: top-five union only, raw FP32 logits. Prompt-set sensitivity; no inferred causal purpose or general head importance.`,
        );
        const scores = el("scores");
        scores.replaceChildren();
        for (const c of v.candidates) {
          const tr = document.createElement("tr");
          for (const text of [
            `${JSON.stringify(c.piece)} / ${c.id}`,
            c.baseline_logit.toFixed(6),
            c.edited_logit.toFixed(6),
            c.delta.toFixed(6),
          ]) {
            const td = document.createElement("td");
            td.textContent = text;
            tr.append(td);
          }
          scores.append(tr);
        }
      }
    }
    function renderLogging() {
      const blocked = pending || play.active || saving || importing;
      el("cancel-import").disabled = !archivePending;
      el("logging").disabled = blocked;
      el("include-prompt").disabled = blocked;
      el("export-run").disabled =
        blocked ||
        (!pendingLog &&
          (!acceptedRequest || acceptedRequest.mode === "prompt_pair_preview"));
      el("export-log").disabled = blocked || !experimentLog.records.length;
      el("clear-log").disabled =
        blocked || (!experimentLog.records.length && !logError);
      write(
        "clear-log",
        pendingLog || logError
          ? "Clear log + discard pending record"
          : "Clear session log",
      );
      for (const id of [
        "persist",
        "durable-prompts",
        "restore-log",
        "delete-saved-log",
        "choose-journal",
        "detach-journal",
        "save-durable",
        "find-archives",
        "archive-select",
        "restore-archive",
        "delete-archive",
        "import-file",
        "import-prompts",
        "import-log",
      ])
        el(id).disabled = blocked;
      el("choose-journal").disabled =
        blocked || typeof globalThis.showSaveFilePicker !== "function";
      el("detach-journal").disabled = blocked || !fileJournal;
      write(
        "durable-status",
        durableError ||
          (saving ? "Saving bounded private log…" : durableNote) +
            (typeof globalThis.showSaveFilePicker !== "function"
              ? " Automatic filesystem journal unavailable in this browser; explicit downloads remain available."
              : ""),
      );
      write(
        "log-status",
        logError ||
          `${el("logging").checked ? "Logging enabled for new accepted runs" : "Logging off"} · ${experimentLog.records.length} / 8 records · ${experimentLog.bytes} / 1048576 UTF-8 bytes · browser-session memory only until downloaded.`,
      );
    }
    function render() {
      const step = play.current,
        active = play.active,
        computeMs =
          details.compute_total_ms ?? play.steps.at(-1)?.compute_total_ms;
      renderDraft();
      renderLogging();
      write(
        "run-inputs",
        acceptedRequest
          ? JSON.stringify(acceptedRequest, null, 2)
          : "No accepted run.",
      );
      el("start").disabled =
        !enabled ||
        busyElsewhere ||
        pending ||
        active ||
        !!logError ||
        saving ||
        importing ||
        !!durableError;
      el("check").disabled = pending || active;
      el("cancel").disabled = !session || pending || !active;
      el("reset").disabled = !session || pending;
      el("pause").disabled =
        !session ||
        pending ||
        (!active && play.cursor >= play.steps.length - 1);
      el("step").disabled = pending || play.cursor + 1 >= play.steps.length;
      el("replay").disabled = pending || active || !play.steps.length;
      for (const id of ["prompt", "limit", "layer", "site"])
        el(id).disabled = pending || active;
      write("pause", play.paused ? "Resume playback" : "Pause playback");
      const mode = play.replaying
        ? "Recorded replay"
        : active
          ? acceptedRequest?.mode === "prompt_pair_preview"
            ? "Token preview"
            : acceptedRequest?.mode === "prompt_pair"
              ? "Two prompt prefills"
              : acceptedRequest?.mode === "sweep"
                ? "Bounded sweep"
                : "Live generation"
          : "Recorded session";
      write(
        "status",
        session
          ? `${mode} · compute ${play.status} · playback ${play.paused ? "paused" : !active && play.cursor >= play.steps.length - 1 ? "finished" : "playing"} · shown ${play.cursor + 1} / ${play.steps.length} computed`
          : availability,
      );
      write(
        "timing",
        `${session && acceptedAt !== null && active ? "Elapsed " + Math.max(0, (Date.now() - acceptedAt) / 1000).toFixed(1) + " s since acceptance · " : ""}${details.comparison_phase ? "Comparison: " + details.comparison_phase + " · " : ""}Model load: ${Number.isFinite(details.load_ms) ? details.load_ms.toFixed(0) + " ms" : "not measured yet"} · inference compute: ${Number.isFinite(computeMs) ? computeMs.toFixed(1) + " ms" : "not measured yet"} · playback: ${el("rate").value} steps/s`,
      );
      write(
        "output",
        step?.generated_text || "Generated text will appear here.",
      );
      renderComparison(step);
      renderObservation(step);
      renderPair(step);
      renderSweep(step);
      const canvas = el("activation"),
        ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!step) {
        write(
          "alignment",
          `No activation selected. Capture width: ${architecture?.width ?? "awaiting model metadata"}.`,
        );
        write("scale", "No activation scale yet");
        write("value", "");
        return;
      }
      const max = Math.max(...step.activation.map(Math.abs));
      for (let i = 0; i < step.activation.length; i++) {
        const value = max ? step.activation[i] / max : 0,
          t = Math.abs(value),
          neutral = [242, 242, 242],
          edge = value < 0 ? [37, 99, 242] : [242, 38, 38];
        ctx.fillStyle = `rgb(${neutral.map((n, j) => Math.round(n + (edge[j] - n) * t)).join(",")})`;
        ctx.fillRect(
          (i % 32) * (canvas.width / 32),
          Math.floor(i / 32) *
            (canvas.height / Math.ceil(step.activation.length / 32)),
          canvas.width / 32,
          canvas.height / Math.ceil(step.activation.length / 32),
        );
      }
      write(
        "alignment",
        step.sweep
          ? `${step.sweep.record_id} · layer ${step.layer} · consumed position ${step.position}. Same original prompt on both sides; one next-token probe, no continuation. ${step.activation_kind}`
          : step.prompt_pair
            ? `Pair ${step.index + 1} · layer ${step.layer} · ${step.activation_kind}. Captures at the two declared consumed positions; no generated or predicted tokens.`
            : `${step.activation_branch ? "Activation: " + step.activation_branch + " · " : ""}Step ${step.index + 1} · ${step.phase} consumed position ${step.position} (token ${step.input_token_id}) → predicts token ${step.token_id} ${JSON.stringify(step.token_piece)}${step.eos ? " · EOS" : ""}. Layer ${step.layer} · ${step.activation_kind || "post-block residual"}. This activation belongs to the consumed position, before the predicted token is consumed.`,
      );
      write(
        "scale",
        `Activation units · linear scale ±${max.toPrecision(6)} per step · blue negative / white zero / red positive. Row-major layout of vector indices 0–${step.activation.length - 1}; no matrix geometry implied.`,
      );
      const index = Math.min(
        step.activation.length - 1,
        Math.max(0, Number(el("index").value) || 0),
      );
      write(
        "value",
        `activation[${index}] = ${step.activation[index]} · forward compute ${step.compute_ms.toFixed(2)} ms`,
      );
    }
    function clearPlayback() {
      clearTimeout(timer);
      timer = null;
      ++playbackEpoch;
    }
    function schedule() {
      clearPlayback();
      if (!session || play.paused) return;
      const id = epoch,
        generation = playbackEpoch,
        target = session;
      timer = setTimeout(
        () => {
          if (
            id !== epoch ||
            generation !== playbackEpoch ||
            target !== session ||
            play.paused
          )
            return;
          timer = null;
          if (play.advance()) render();
          if (play.active || play.cursor + 1 < play.steps.length) schedule();
        },
        1000 / Number(el("rate").value),
      );
    }
    async function poll(id) {
      if (id !== epoch || !session) return;
      const target = session;
      try {
        let snapshot = await api("poll", { session: target });
        if (id !== epoch || session !== target) return;
        if (
          cleanupAfterTransport &&
          !workerCleanupConfirmed(snapshot) &&
          snapshot.status !== "stopping" &&
          !snapshot.details?.cleanup_pending
        ) {
          snapshot = await api("cancel", { session: target });
          if (id !== epoch || session !== target) return;
        }
        if (cleanupAfterTransport && !workerCleanupConfirmed(snapshot))
          snapshot = { ...snapshot, status: "stopping" };
        if (workerCleanupConfirmed(snapshot)) cleanupAfterTransport = false;
        const wasStopping = play.status === "stopping";
        play.accept(snapshot);
        details = snapshot.details;
        acceptPreview(snapshot);
        captureLog(snapshot);
        if (wasStopping && !play.active) message("");
        render();
        if (snapshot.details.error) message(snapshot.details.error);
        if (play.active) polling = setTimeout(() => poll(id), 250);
      } catch (error) {
        if (id !== epoch) return;
        if (play.status === "stopping") {
          message(
            error.message + "; cleanup remains pending. Retrying status.",
          );
          render();
          polling = setTimeout(() => poll(id), 250);
          return;
        }
        // Retain this owner and one incomplete outcome while cancellation/reaping is uncertain.
        cleanupAfterTransport = true;
        let stopped = {
          status: "stopping",
          worker_alive: null,
          steps: play.steps,
          details,
        };
        try {
          stopped = await api("cancel", { session: target });
        } catch {
          /* The owned status path retries cleanup; no replacement run. */
        }
        if (id !== epoch || session !== target) return;
        captureLog({
          ...stopped,
          status: "connection_lost",
          worker_alive: workerCleanupConfirmed(stopped) ? false : null,
          termination: "transport_error",
        });
        play.accept(stopped);
        details = stopped.details;
        play.paused = true;
        cleanupAfterTransport = !workerCleanupConfirmed(stopped);
        if (cleanupAfterTransport) {
          play.status = "stopping";
          polling = setTimeout(() => poll(id), 250);
        }
        message(
          error.message +
            (cleanupAfterTransport
              ? "; cleanup remains pending. Retrying status."
              : ""),
        );
        render();
      }
    }
    async function start(event, previewOnly = false) {
      event.preventDefault();
      if (
        pending ||
        play.active ||
        !enabled ||
        busyElsewhere ||
        logError ||
        saving ||
        importing ||
        durableError
      )
        return;
      const id = ++epoch;
      clearTimeout(polling);
      clearPlayback();
      pending = true;
      play.paused = true;
      if (!session) availability = "Starting local generation…";
      message("");
      render();
      try {
        let request;
        if (el("task").value === "sweep") {
          if (!sweepSupported)
            throw new Error("This backend does not offer bounded sweeps.");
          request = currentSweepRequest();
          if (!sweepPlan || sweepKey !== JSON.stringify(request))
            throw new Error(
              "Review a fresh expanded plan for these exact targets, prompts and seed.",
            );
          request.plan_digest = sweepPlan.digest;
        } else if (el("task").value === "prompt_pair") {
          if (!pairSupported || !contract)
            throw new Error(
              "This backend does not offer pinned prompt-pair captures.",
            );
          const prompts = [el("prompt").value, el("prompt-b").value];
          if (prompts.some((p) => !p.trim() || utf8Size(p) > 2048))
            throw new Error(
              "Use two nonempty prompts of at most 2048 UTF-8 bytes each.",
            );
          request = {
            mode: previewOnly ? "prompt_pair_preview" : "prompt_pair",
            prompts,
            source_model: contract.source_model,
          };
          if (!previewOnly) {
            if (
              !pairPreview ||
              JSON.stringify(pairPreview.prompts) !== JSON.stringify(prompts)
            )
              throw new Error(
                "Review a fresh preview for these exact prompts.",
              );
            Object.assign(request, {
              layer: Number(el("layer").value),
              activation_site: el("site").value || "block",
              positions: pairPositions(el("pairs").value, pairPreview),
              preview_digest: pairPreview.digest,
            });
          }
        } else {
          request = {
            prompt: el("prompt").value,
            max_new_tokens: Number(el("limit").value),
            layer: Number(el("layer").value),
            activation_site: el("site").value || "block",
            ...(contract
              ? {
                  source_model: contract.source_model,
                  edits: draft.map((edit) => ({
                    ...edit,
                    shape: [...edit.shape],
                  })),
                }
              : {}),
          };
          const observation = observationSelection(
            el("observation").value,
            el("head-index").value,
            request.activation_site,
            architecture,
          );
          if (observation) {
            if (!observationModes.includes(observation.kind))
              throw new Error(
                "This backend does not offer the selected observation.",
              );
            request.observation = observation;
          }
        }
        if (utf8Size(JSON.stringify(request)) > 8192)
          throw new Error(
            "Complete request exceeds the 8 KiB limit; shorten the prompts.",
          );
        const snapshot = await api("start", request);
        if (id !== epoch) return;
        session = snapshot.session;
        acceptedAt = Date.now();
        availability =
          "Ready · local CPU · " +
          (contract ? "baseline then edited comparison" : "fixed weights");
        acceptedRequest = request;
        loggingRun =
          request.mode !== "prompt_pair_preview" && !!el("logging").checked;
        promptConsent = !!el("include-prompt").checked;
        loggedRun = false;
        incompleteLog = false;
        logRecordIndex = null;
        cleanupAfterTransport = false;
        play.reset();
        play.accept(snapshot);
        details = snapshot.details;
        acceptPreview(snapshot);
        captureLog(snapshot);
        play.paused = el("mode").value === "step";
        poll(id);
        schedule();
      } catch (error) {
        if (id === epoch) {
          if (!session) {
            play.status = "error";
            availability =
              "Generation unavailable · retry manually after checking the error.";
          }
          message(error.message);
        }
      } finally {
        if (id === epoch) {
          pending = false;
          render();
        }
      }
    }
    async function stop(reset) {
      if (reset) cancelArchive();
      if (pending || !session) return;
      const id = ++epoch,
        target = session;
      pending = true;
      play.paused = true;
      clearPlayback();
      clearTimeout(polling);
      render();
      try {
        const snapshot = await api(reset ? "reset" : "cancel", {
          session: target,
        });
        if (id !== epoch) return;
        if (reset) {
          captureLog({
            status: play.active ? "cancelled" : play.status,
            worker_alive: false,
            steps: play.steps,
            details,
            termination: "reset; last received tab snapshot",
          });
          session = null;
          acceptedRequest = null;
          pairPreview = null;
          invalidateSweep();
          play.reset();
          details = {};
          message("");
        } else {
          play.accept(snapshot);
          details = snapshot.details;
          captureLog(snapshot);
          if (play.active) poll(id);
        }
      } catch (error) {
        if (id === epoch) {
          if (error.code === "cleanup_pending" || play.status === "stopping") {
            play.status = "stopping";
            message(error.message);
            poll(id);
          } else {
            message(
              error.message +
                "; worker lease expires within 15 seconds without polling.",
            );
            play.status = "error";
          }
        }
      } finally {
        if (id === epoch) {
          pending = false;
          render();
        }
      }
    }
    el("task").addEventListener("change", () => {
      if (!["prompt_pair", "sweep"].includes(el("task").value)) {
        if (el("observation").value === "attention")
          el("site").value = "attention";
        if (el("observation").value === "logit_lens")
          el("site").value = "block";
      }
      render();
    });
    for (const key of ["prompt", "prompt-b"])
      el(key).addEventListener("input", () => {
        pairPreview = null;
        invalidateSweep();
        render();
      });
    for (const key of [
      "sweep-targets",
      "sweep-seed",
      "sweep-operation",
      "sweep-scale",
      "sweep-second",
      "sweep-prompt-b",
      "layer",
      "site",
    ]) {
      el(key).addEventListener("input", () => {
        invalidateSweep();
        render();
      });
      el(key).addEventListener("change", () => {
        invalidateSweep();
        render();
      });
    }
    el("sweep-preview").addEventListener("click", previewSweep);
    el("sweep-use-inspected").addEventListener("click", () => {
      if (pending || play.active || el("sweep-use-inspected").disabled) return;
      const layer = selection.tensor.match(/layers\.(\d+)\./)[1],
        head = Math.floor(
          (selection.tensor.endsWith(".o_proj.weight")
            ? selection.col
            : selection.row) / architecture.head_dim,
        );
      el("sweep-targets").value =
        `${selection.tensor.endsWith(".o_proj.weight") ? "head" : "query_head"} ${layer} ${head}`;
      invalidateSweep();
      render();
    });
    el("pair-preview").addEventListener("click", () =>
      start({ preventDefault() {} }, true),
    );
    el("observation").addEventListener("change", () => {
      const kind = el("observation").value;
      if (kind === "attention") el("site").value = "attention";
      if (kind === "logit_lens") el("site").value = "block";
      render();
    });
    el("logging").addEventListener("change", render);
    el("include-prompt").addEventListener("change", render);
    el("export-run").addEventListener("click", () => {
      if (pending || play.active) return;
      try {
        const record =
          pendingLog ||
          experimentRecord(
            acceptedRequest,
            {
              status: play.status,
              worker_alive: lastWorkerAlive,
              steps: play.steps,
              details,
            },
            { includePrompt: !!el("include-prompt").checked },
          );
        downloadExperiment(
          finiteJSON(
            el("include-prompt").checked ? record : redactExperiment(record),
          ),
          "weight-atlas-experiment.json",
        );
        message("");
      } catch (error) {
        message(error.message);
      }
    });
    el("export-log").addEventListener("click", () => {
      if (!pending && !play.active) {
        try {
          downloadExperiment(
            finiteJSON(
              experimentLog.envelope(
                el("include-prompt").checked
                  ? experimentLog.records
                  : experimentLog.records.map(redactExperiment),
              ),
            ),
            "weight-atlas-session-log.json",
          );
          message("");
        } catch (error) {
          message(error.message);
        }
      }
    });
    el("clear-log").addEventListener("click", () => {
      if (!pending && !play.active && !saving && !importing) {
        experimentLog.clear();
        pendingLog = null;
        logError = "";
        saveDurable();
        render();
      }
    });
    el("persist").addEventListener("change", () => {
      if (!pending && !play.active && !saving) {
        durableNote = el("persist").checked
          ? "Browser persistence enabled for this tab."
          : "Browser persistence off; existing saved data remains until explicitly deleted.";
        saveDurable();
      }
    });
    el("durable-prompts").addEventListener("change", () => {
      if (!pending && !play.active && !saving) saveDurable();
    });
    el("save-durable").addEventListener("click", () => {
      if (!pending && !play.active && !saving && !importing) saveDurable();
    });
    el("choose-journal").addEventListener("click", async () => {
      if (
        pending ||
        play.active ||
        saving ||
        importing ||
        typeof globalThis.showSaveFilePicker !== "function"
      )
        return;
      importing = true;
      render();
      try {
        const handle = await globalThis.showSaveFilePicker({
          suggestedName: "weight-atlas-session-log.json",
          types: [
            {
              description: "Private experiment log JSON",
              accept: { "application/json": [".json"] },
            },
          ],
        });
        fileJournal = new AtlasFileJournal(handle);
        durableNote =
          "File journal selected; every terminal logged run updates this bounded file.";
      } catch (error) {
        if (error.name !== "AbortError") message(error.message);
      } finally {
        importing = false;
        await saveDurable();
        render();
      }
    });
    el("detach-journal").addEventListener("click", () => {
      if (!pending && !play.active && !saving && !importing) {
        fileJournal = null;
        durableNote = "File journal updates stopped; the file remains on disk.";
        saveDurable();
      }
    });
    el("import-log").addEventListener("click", async () => {
      if (pending || play.active || saving || importing) return;
      const file = el("import-file").files?.[0];
      if (!file) {
        write("import-status", "Select one archived JSON file first.");
        return;
      }
      if (file.size > EXPERIMENT_BYTES) {
        write(
          "import-status",
          "Import exceeds 1 MiB; existing records unchanged.",
        );
        return;
      }
      await importArchive(() => file.text());
    });
    el("cancel-import").addEventListener("click", cancelArchive);
    el("restore-log").addEventListener("click", () =>
      importArchive(() => {
        const text = storage().read();
        if (!text) throw new Error("No saved log for this tab");
        return text;
      }, true),
    );
    el("find-archives").addEventListener("click", () => {
      if (!pending && !play.active && !saving && !importing) {
        try {
          const select = el("archive-select");
          select.replaceChildren();
          for (const [i, key] of storage().list().entries()) {
            const option = document.createElement("option");
            option.value = key;
            option.textContent =
              "Private archive " + (i + 1) + " · " + key.slice(-8);
            select.append(option);
          }
          write(
            "import-status",
            "Saved archive names listed locally. Choose one and explicitly restore; no worker is adopted.",
          );
        } catch (error) {
          write("import-status", error.message);
        }
      }
    });
    el("restore-archive").addEventListener("click", () =>
      importArchive(() => {
        const text = storage().readKey(el("archive-select").value);
        if (!text) throw new Error("Archive no longer exists");
        return text;
      }),
    );
    el("delete-archive").addEventListener("click", async () => {
      if (pending || play.active || saving || importing) return;
      importing = true;
      render();
      try {
        const key = el("archive-select").value;
        storage().readKey(key);
        await storage().mutate(globalThis.navigator?.locks, (s) => {
          if (key === s.key()) s.clear();
          else s.storage.removeItem(key);
        });
        write(
          "import-status",
          "Selected browser archive deleted. Other tabs must reconcile a changed archive before saving; worker sessions and files are unchanged.",
        );
      } catch (error) {
        write("import-status", error.message);
      } finally {
        importing = false;
        render();
      }
    });
    el("delete-saved-log").addEventListener("click", async () => {
      if (pending || play.active || saving || importing) return;
      importing = true;
      render();
      try {
        await storage().mutate(globalThis.navigator?.locks, (s) => s.clear());
        el("persist").checked = false;
        durableError = "";
        durableNote =
          "This tab’s browser log deleted. In-memory records and files remain.";
      } catch (error) {
        durableError = error.message;
      } finally {
        importing = false;
        render();
      }
    });

    el("form").addEventListener("submit", start);
    window.atlasInferenceSelectionChanged = updateSelection;
    el("kind").addEventListener("change", () => updateSelection(selection));
    el("operation").addEventListener("change", render);
    for (const id of ["range-start", "range-end", "factor", "head-index"])
      el(id).addEventListener("input", render);
    document.getElementById("prepare-edit").addEventListener("click", () => {
      if (document.getElementById("prepare-edit").disabled) return;
      const panel = document.getElementById("inference-panel");
      panel.open = true;
      panel.scrollIntoView({ block: "start" });
      el("kind").focus();
    });
    function addEdit(head = false) {
      if (pending || play.active || draft.length >= 8) return;
      try {
        const outputHead =
          head && selection.tensor.endsWith(".self_attn.o_proj.weight");
        const first = head
          ? Math.floor(
              (outputHead ? selection.col : selection.row) /
                architecture.head_dim,
            ) * architecture.head_dim
          : el("range-start").value;
        const edit = draftEdit(
          selection,
          contract,
          head ? (outputHead ? "columns" : "rows") : el("kind").value,
          head ? "zero" : el("operation").value,
          first,
          head ? first + architecture.head_dim : el("range-end").value,
          el("factor").value,
        );
        if (
          head &&
          !/^model\.layers\.\d+\.self_attn\.(q_proj|o_proj)\.weight$/.test(
            edit.tensor,
          )
        )
          throw new Error("Select a verified query or output projection.");
        draft.push(edit);
        message("");
        render();
      } catch (error) {
        message(error.message);
      }
    }
    el("add").addEventListener("click", () => addEdit());
    el("head").addEventListener("click", () => addEdit(true));
    el("clear-edits").addEventListener("click", () => {
      if (!pending && !play.active) {
        draft = [];
        render();
      }
    });
    el("pause").addEventListener("click", () => {
      play.paused = !play.paused;
      render();
      schedule();
    });
    el("step").addEventListener("click", () => {
      play.paused = true;
      clearPlayback();
      play.advance();
      render();
    });
    el("replay").addEventListener("click", () => {
      clearPlayback();
      play.rewind();
      render();
    });
    el("cancel").addEventListener("click", () => stop(false));
    el("reset").addEventListener("click", () => stop(true));
    el("rate").addEventListener("change", () => {
      render();
      schedule();
    });
    el("index").addEventListener("input", render);
    el("activation").addEventListener("click", (event) => {
      const r = el("activation").getBoundingClientRect();
      el("index").value = Math.min(
        (play.current?.activation.length || architecture?.width || 1) - 1,
        Math.floor(
          ((event.clientY - r.top) / r.height) *
            Math.ceil(
              (play.current?.activation.length || architecture?.width || 1) /
                32,
            ),
        ) *
          32 +
          Math.floor(((event.clientX - r.left) / r.width) * 32),
      );
      render();
    });
    // Page closure uses an authenticated same-origin keepalive, with lease as fallback.
    window.addEventListener("pagehide", () => {
      if (session && play.active)
        fetch("/api/inference/cancel", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Atlas-Local": "1" },
          body: JSON.stringify({ session }),
          keepalive: true,
        }).catch(() => {});
    });
    selection = window.atlasInferenceSelection || null;
    render();
    async function checkReadiness() {
      if (pending || play.active) return;
      const checkedEpoch = epoch,
        checkedReadiness = ++readinessEpoch;
      el("check").disabled = true;
      const current = () =>
        checkedReadiness === readinessEpoch &&
        checkedEpoch === epoch &&
        !pending &&
        !play.active;
      return api("")
        .then((info) => {
          if (!current()) return;
          enabled = true;
          busyElsewhere = !!info.busy;
          document.getElementById("inference-panel").hidden = false;
          document.getElementById("mode-status").textContent = busyElsewhere
            ? "Local worker busy"
            : "Weight viewer + local experiments";
          sweepSupported = !!info.sweep;
          pairSupported = !!info.prompt_pair;
          observationModes = info.observations?.kinds || [];
          contract = info.comparison || null;
          availability =
            "Ready · local CPU · " +
            (contract ? "baseline then edited comparison" : "fixed weights");
          write(
            "model",
            `${info.model} · ${info.engine} · greedy FP32 · seed 0`,
          );
          architecture = info.architecture || contract?.architecture || null;
          if (architecture) {
            play.width = architecture.width;
            const options = (id, values) => {
              const prior = el(id).value;
              el(id).replaceChildren();
              for (const [value, label] of values) {
                const option = document.createElement("option");
                option.value = String(value);
                option.textContent = label;
                el(id).append(option);
              }
              el(id).value = values.some(([v]) => String(v) === prior)
                ? prior
                : String(values[0][0]);
            };
            options(
              "layer",
              Array.from({ length: architecture.layers }, (_, i) => [
                i,
                String(i),
              ]),
            );
            options("site", Object.entries(architecture.capture_sites));
            el("head-index").max = architecture.query_heads - 1;
            el("index").max = architecture.width - 1;
            el("activation").height = 16 * Math.ceil(architecture.width / 32);
            write(
              "sweep-help",
              `layer L: all output heads; head L H: output-projection columns; query_head L H or offset L H[,H...] O: query-row interventions. Layers 0–${architecture.layers - 1}, heads 0–${architecture.query_heads - 1}, offsets 0–${architecture.head_dim - 1}. One layer plan, one total budget.`,
            );
          }
          // Do not adopt another tab's session or show its prompt/trace. No ownership takeover.
          if (info.busy) {
            const owner =
              info.busy_owner === "analytics job"
                ? "An analytics job"
                : "Another inference session";
            availability = `Busy · ${owner.toLowerCase()} owns the local worker. No queue; check readiness after it finishes.`;
            message(
              `${owner} owns the local worker. Your work has not been queued.`,
            );
          } else message("");
          render();
        })
        .catch(() => {
          if (!current()) return;
          enabled = false;
          busyElsewhere = false;
          document.getElementById("inference-panel").hidden = true;
          document.getElementById("mode-status").textContent =
            "Weight viewer · inference unavailable";
          availability =
            "Inference unavailable · run-atlas.sh serves the weight viewer only. Qwen3-8B is viewable, not an inference target. Launch python3 tools/live_inference.py --model /absolute/path/to/complete/smollm2-135m --python /absolute/path/to/cpu-environment/bin/python and open http://127.0.0.1:8796 (renderer: 8797).";
          render();
        })
        .finally(() => {
          if (checkedReadiness === readinessEpoch && checkedEpoch === epoch)
            el("check").disabled = pending || play.active;
        });
    }
    window.atlasRefreshAvailability = checkReadiness;
    el("check").addEventListener("click", checkReadiness);
    checkReadiness();
  })();
}
