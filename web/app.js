"use strict";
// Derived from validated reference frontend. Progressive extension; frozen API SHA256: 665f2e600fd22676f757f4ee59f3603d92bd3811a4444551bab062ffe057ffd7
const $ = (id) => document.getElementById(id);
const SIDES = ["left", "right"];
const RULE_IDS = [
  "global_linear",
  "global_asinh",
  "tensor_linear",
  "tensor_asinh",
  "tensor_magnitude",
  "tensor_magnitude_asinh",
  "tensor_robust99",
  "tensor_signed_percentile",
];
const state = {
  model: null,
  tensor: null,
  current: null,
  viewEpoch: 0,
  modelEpoch: 0,
  inspectEpoch: 0,
  openStarted: performance.now(),
  polling: false,
  pollController: null,
  viewFailed: false,
  modelController: null,
  viewController: null,
  inspectController: null,
  pendingInspection: null,
  viewers: {},
  selected: null,
  inspectionData: null,
  recolor: null,
  loading: true,
  syncing: false,
  frame: false,
};
const number = (value) => Number(value).toLocaleString("en-US");
const exact = (value) =>
  value === null || value === undefined ? "—" : String(value);
function node(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}
function text(id, value) {
  $(id).textContent = value;
}
function showError(message) {
  text("error", message);
  $("error").hidden = !message;
}
function status(message) {
  text("status", message);
}
function option(value, label) {
  const n = node("option", "", label);
  n.value = String(value);
  return n;
}
function assert(condition, message) {
  if (!condition) throw new Error("API contract: " + message);
}
async function json(url, signal) {
  if (globalThis.AtlasTools)
    return AtlasTools.readJSON(url, {
      signal,
      onState: (event) => globalThis.atlasWorkspace?.retry(event, url, signal),
    });
  const bound = globalThis.AtlasHost?.bindRead(url);
  const response = await fetch(bound?.url || url, {
    signal,
    cache: "no-store",
  });
  bound?.assertCurrent();
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`HTTP ${response.status}: expected API JSON.`);
  }
  if (!response.ok) {
    const e = new Error(data.error || `HTTP ${response.status}`);
    e.status = response.status;
    e.code = data.code;
    throw e;
  }
  assert(data.api_version === 1, "expected api_version 1.");
  bound?.check(data);
  return data;
}
const apiURL = (url) => (globalThis.AtlasHost ? AtlasHost.url(url) : url);
function settings() {
  return {
    tensor: state.tensor?.id,
    ...(state.tensor?.slice ? { slice: state.tensor.slice.join(",") } : {}),
    left: $("left-rule").value,
    right: $("right-rule").value,
  };
}
function sameSettings(a, b) {
  return (
    a.tensor === b.tensor &&
    (a.slice || "") === (b.slice || "") &&
    a.left === b.left &&
    a.right === b.right
  );
}
function validateTensor(t) {
  if (t?.available === false) {
    assert(
      typeof t.unavailable_reason === "string" && Number.isSafeInteger(t.id),
      "unavailable tensor reason required.",
    );
    return;
  }
  if (t?.dtype !== undefined)
    assert(
      ["BF16", "F16", "F32"].includes(t.dtype) &&
        t.element_bytes === (t.dtype === "F32" ? 4 : 2),
      "supported source dtype and byte width required.",
    );
  assert(
    t && Number.isSafeInteger(t.id) && t.id >= 0,
    "tensor ID must be a nonnegative integer.",
  );
  assert(
    Number.isSafeInteger(t.rows) &&
      t.rows > 0 &&
      Number.isSafeInteger(t.cols) &&
      t.cols > 0,
    "positive native rows/columns required.",
  );
  assert(
    Array.isArray(t.shape) &&
      t.shape.every((n) => Number.isSafeInteger(n) && n > 0),
    "native shape required.",
  );
  assert(
    t.count === t.shape.reduce((a, b) => a * b, 1) &&
      t.cols === t.shape.at(-1) &&
      t.rows === (t.shape.length === 1 ? 1 : t.shape.at(-2)),
    "native dimensions must contain exactly the stored values.",
  );
  assert(
    t.max_level === Math.ceil(Math.log2(Math.max(t.rows, t.cols))),
    "unexpected Deep Zoom maximum level.",
  );
  assert(t.min_level === 0, "numeric pyramid must start at level 0.");
  if (t.shape.length === 1)
    assert(
      t.rows === 1 && t.cols === t.shape[0],
      "vectors must stay one unpadded row.",
    );
}
function validateView(data, s) {
  validateTensor(data.tensor);
  if (globalThis.AtlasTools)
    AtlasTools.requireBinding(
      data.source_binding,
      AtlasTools.sourceBinding(state.model, state.tensor),
    );
  assert(data.tensor.id === s.tensor, "view returned a different tensor.");
  assert(
    data.tensor.rows === state.tensor.rows &&
      data.tensor.cols === state.tensor.cols,
    "view dimensions disagree with the catalog.",
  );
  assert(
    data.tile_size === 256 && data.overlap === 0,
    "expected 256px tiles with no overlap.",
  );
  assert(
    data.source_values_unchanged === true,
    "view must preserve source values.",
  );
  for (const side of SIDES) {
    const l = data.legends?.[side];
    assert(l && l.id === s[side], "legend rule does not match " + side + ".");
    assert(
      Number.isFinite(l.min) &&
        Number.isFinite(l.max) &&
        l.min <= 0 &&
        l.max >= 0 &&
        l.zero === 0,
      "finite raw bounds and exact zero required.",
    );
    assert(
      typeof l.formula === "string" &&
        typeof l.scope === "string" &&
        typeof l.units === "string",
      "formula, scope and units required.",
    );
    if (["tensor_magnitude", "tensor_magnitude_asinh"].includes(s[side]))
      assert(
        l.min === 0 && l.palette === "sequential-purple-v1",
        "unsigned magnitude bounds and palette required.",
      );
    if (s[side].endsWith("asinh"))
      assert(Number.isFinite(l.s) && l.s > 0, "positive asinh scale required.");
  }
}
function setInteraction(enabled) {
  for (const id of [
    "fit",
    "zoom-in",
    "zoom-out",
    "scalar-zoom",
    "row",
    "col",
    "inspect-submit",
  ])
    $(id).disabled = !enabled;
}
function publishInferenceSelection(selection) {
  window.atlasInferenceSelection = selection;
  window.atlasInferenceSelectionChanged?.(selection);
}
function clearInspection(keep = false) {
  publishInferenceSelection(null);
  state.inspectEpoch++;
  state.inspectController?.abort();
  state.inspectController = null;
  state.pendingInspection = null;
  if (keep && state.inspectionData && state.selected) {
    showInspection({
      ...state.inspectionData,
      transformed: { left: null, right: null },
      transform_errors: { left: "Updating colors…", right: "Updating colors…" },
    });
  } else {
    state.selected = null;
    state.inspectionData = null;
    $("inspection").replaceChildren(
      node("p", "empty-state", "Tap either image or enter an address."),
    );
    text("pinned-readout", "No pinned value yet.");
  }
  for (const side of SIDES) state.viewers[side]?.clearOverlays();
}
function stopViews() {
  for (const side of SIDES) {
    const v = state.viewers[side];
    if (v) v.destroy();
    delete state.viewers[side];
    $("" + side + "-canvas").replaceChildren(
      node("div", "canvas-message", "Loading the requested tensor…"),
    );
    text(side + "-resolution", "Awaiting numeric tiles");
    text(side + "-magnification", "—");
  }
}
function deactivate(keep = false) {
  if (!keep) state.recolor = null;
  cancelHover();
  clearOverview();
  state.pollController?.abort();
  state.viewFailed = false;
  globalThis.atlasWorkspace?.changed("deactivate");
  state.viewEpoch++;
  state.viewController?.abort();
  state.viewController = null;
  clearInspection(keep);
  state.current = null;
  state.loading = true;
  $("comparison").setAttribute("aria-busy", "true");
  setInteraction(false);
  stopViews();
}
function calibrationScope(model) {
  const supported = (model.catalog || []).filter((t) => t.available !== false),
    unavailable = (model.catalog?.length || 0) - supported.length;
  const completed = model.coverage?.calibrated_tensors;
  const pending = Number.isSafeInteger(completed)
    ? Math.max(0, supported.length - completed)
    : supported.filter((t) => !t.calibration_complete).length;
  return {
    supported,
    unavailable,
    pending,
    globalSupported:
      model.global_calibration_supported !== false && unavailable === 0,
  };
}
function drawCoverage(model) {
  const c = model.coverage || {},
    scope = calibrationScope(model);
  text(
    "readiness-summary",
    c.statistics_complete
      ? "Color scales ready"
      : c.active_tensor !== null && c.active_tensor !== undefined
        ? "Calibrating supported weights…"
        : !scope.globalSupported
          ? scope.supported.length === 0
            ? "No supported color calibration"
            : scope.pending
              ? "Local calibration pending · global unavailable"
              : "Supported local scales ready · global unavailable"
          : "Local scales on demand",
  );
  const done =
    c.calibrated_tensors ??
    model.catalog?.filter((t) => t.calibration_complete).length ??
    0;
  text(
    "calibration-progress",
    `${done} / ${scope.supported.length} supported tensors calibrated${scope.unavailable ? " · " + scope.unavailable + " unavailable tensors excluded; global calibration unavailable" : ""} · ${number(c.values_streamed || 0)} / ${number(model.parameter_count)} values scanned${c.active_tensor !== null && c.active_tensor !== undefined ? " · scanning tensor " + c.active_tensor : ""}. Updates arrive after completed tensors; no time estimate is available.`,
  );
  text(
    "source-coverage",
    c.source_complete === true
      ? `Headers/index complete · ${number(c.sha_hashed_shards || 0)} shards freshly hashed · ${number(c.sha_expected_matched_shards || 0)} matched saved expectations · ${number(c.sha_missing_expected_shards || 0)} hashed without expectations`
      : "Source validation incomplete",
  );
  text(
    "statistics-coverage",
    `${number(c.values_streamed || 0)} / ${number(model.parameter_count)} values · ${c.statistics_complete ? "complete" : !scope.globalSupported ? "global unavailable; supported local scope only" : c.active_tensor !== null ? "calibrating tensor " + c.active_tensor : "partial / not started"}`,
  );
  text("identity-note", model.identity_validation);
  $("calibrate-all").disabled = scope.pending === 0 || c.all_requested;
  text(
    "calibrate-all",
    scope.globalSupported
      ? "Calibrate full model"
      : "Calibrate supported tensors",
  );
  $("calibration-note").hidden = model.calibration_complete;
  text(
    "calibration-note",
    !scope.globalSupported
      ? "Global calibration is unavailable because the catalog includes unsupported tensors. Supported tensors can still be calibrated locally; unavailable tensors are excluded. " +
          (scope.pending
            ? "Use Calibrate supported tensors for the remaining supported work."
            : scope.supported.length
              ? "All supported tensors are calibrated; local views remain available."
              : "There are no supported tensors to calibrate.")
      : "Global calibration is incomplete. Local views become available after the selected tensor is fully scanned. Raw source inspection is available immediately. Use Calibrate full model to enable global rules.",
  );
  if (c.calibration_error) showError(c.calibration_error);
  text(
    "render-coverage",
    `${number(c.materialized_tiles || 0)} cached tiles · ${number(c.materialized_bytes || 0)} bytes`,
  );
  text(
    "render-coverage-note",
    (c.all_pixels_materialized === true
      ? "All scalar pixels materialized. "
      : "Complete scalar-pixel coverage not materialized. ") +
      (c.rendering_policy || "Fine tiles generated on demand.") +
      " Counts at last status refresh.",
  );
}
function layerKey(t) {
  const match = t.name.match(/(?:^|\.)layers\.(\d+)\./);
  return match ? match[1] : "shared";
}
function tensorDescription(t) {
  const layer = t.name.match(/(?:^|\.)layers\.(\d+)\./)?.[1];
  const definitions = [
    [
      "self_attn.q_proj.weight",
      "Attention queries",
      "Query and key projections are paired to calculate attention scores. These stored weights are not attention probabilities.",
    ],
    [
      "self_attn.k_proj.weight",
      "Attention keys",
      "Keys are compared with queries when attention scores are calculated. These stored weights are not attention probabilities.",
    ],
    [
      "self_attn.v_proj.weight",
      "Attention values",
      "Projects the features that attention mixes together. A weight is distinct from the activation it helps calculate.",
    ],
    [
      "self_attn.o_proj.weight",
      "Attention output",
      "Projects combined attention outputs back into the hidden features. Head groups, when verified, run along input columns.",
    ],
    [
      "mlp.gate_proj.weight",
      "Feed-forward gate",
      "Weights of the gating projection in the feed-forward block.",
    ],
    [
      "mlp.up_proj.weight",
      "Feed-forward expansion",
      "Weights that project into the wider feed-forward features.",
    ],
    [
      "mlp.down_proj.weight",
      "Feed-forward output",
      "Weights that project feed-forward features back to the hidden width.",
    ],
    [
      "input_layernorm.weight",
      "Input normalization",
      "One stored scale per feature, displayed as a native vector.",
    ],
    [
      "post_attention_layernorm.weight",
      "Feed-forward normalization",
      "One stored normalization scale per feature, displayed as a native vector.",
    ],
    [
      "self_attn.q_norm.weight",
      "Query normalization",
      "A native vector of query-normalization scales.",
    ],
    [
      "self_attn.k_norm.weight",
      "Key normalization",
      "A native vector of key-normalization scales.",
    ],
    [
      "embed_tokens.weight",
      "Token embeddings",
      "One stored feature vector per token row; a row number is not a decoded token label.",
    ],
    [
      "lm_head.weight",
      "Vocabulary output",
      "Projects hidden features to vocabulary scores.",
    ],
    [
      "model.norm.weight",
      "Final normalization",
      "One final normalization scale per feature.",
    ],
  ];
  const match = definitions.find(([suffix]) => t.name.endsWith(suffix));
  return {
    title:
      (layer !== undefined ? `Layer ${layer} · ` : "") +
      (match?.[1] || "Stored tensor"),
    description:
      match?.[2] || "Original values in their stored native dimensions.",
  };
}
function describeTensor() {
  const info = tensorDescription(state.tensor);
  $("tensor-description").replaceChildren(
    node("strong", "", info.title),
    node("span", "", info.description),
  );
}
function drawCatalog() {
  if (!state.model) return;
  const search = $("tensor-search").value.trim().toLowerCase(),
    layer = $("layer-filter").value;
  const items = state.model.catalog.filter(
    (t) =>
      (layer === "all" || layerKey(t) === layer) &&
      (t.name + " " + tensorDescription(t).title)
        .toLowerCase()
        .includes(search),
  );
  const fragment = document.createDocumentFragment();
  for (const t of items) {
    const b = node("button", "tensor-option", tensorDescription(t).title);
    b.type = "button";
    b.setAttribute("aria-pressed", String(t.id === state.tensor?.id));
    b.setAttribute("aria-label", `Select ${t.name}`);
    b.dataset.tensorId = String(t.id);
    b.append(
      node("code", "tensor-native-name", t.name),
      node(
        "small",
        "",
        `${t.shape.join(" × ")} · ${number(t.count)} values${t.available === false ? " · unavailable: " + t.unavailable_reason : t.shape.length > 2 ? " · slice required" : ""}`,
      ),
    );
    b.addEventListener("click", () => selectTensor(t.id));
    fragment.append(b);
  }
  if (!items.length)
    fragment.append(node("p", "small muted", "No matching tensors."));
  $("tensor-list").replaceChildren(fragment);
  text(
    "filtered-count",
    `${items.length} of ${state.model.catalog.length} tensors shown`,
  );
}
function updateRuleAvailability() {
  if (!state.model) return;
  for (const side of SIDES)
    for (const o of $(side + "-rule").options) {
      const unavailable =
        o.value === "tensor_signed_percentile" && state.tensor?.dtype === "F32";
      o.disabled =
        unavailable ||
        (o.value.startsWith("global_") && !state.model.calibration_complete);
      o.textContent =
        (state.model.rules.find((r) => r.id === o.value)?.title || o.value) +
        (unavailable ? " · unavailable for F32" : "");
      o.title = unavailable
        ? "Exact F32 absolute-value rank index pending; no approximate ranks."
        : "";
    }
}
function refuseUnsupportedView() {
  if (
    state.tensor?.dtype !== "F32" ||
    !SIDES.some(
      (side) => $(side + "-rule").value === "tensor_signed_percentile",
    )
  )
    return false;
  deactivate();
  allowRawInspection();
  globalThis.atlasWorkspace?.rawReady();
  describeTensor();
  text("tensor-name", state.tensor.name);
  text("tensor-shape", `${state.tensor.shape.join(" × ")} native shape · F32`);
  const reason =
    "Tensor signed percentile is unavailable for F32: its exact absolute-value rank index is pending. Choose another rule; raw inspection remains available.";
  showError(reason);
  status("Requested rule unsupported · no replacement field.");
  for (const side of SIDES) {
    text(side + "-formula", "Choose a supported rule");
    text(side + "-scope", "No active view");
    text(side + "-min", "—");
    text(side + "-max", "—");
    text(side + "-parameter", "");
    text(side + "-units", "");
  }
  return true;
}
function validateModelCatalog(model) {
  assert(
    Array.isArray(model.catalog) && model.catalog.length > 0,
    "nonempty full tensor catalog required.",
  );
  for (const t of model.catalog) validateTensor(t);
  assert(
    new Set(model.catalog.map((t) => t.id)).size === model.catalog.length,
    "duplicate tensor IDs.",
  );
  assert(
    model.catalog.reduce((sum, t) => sum + t.count, 0) ===
      model.parameter_count,
    "catalog total differs from checkpoint parameter count.",
  );
  const rules = model.rules.filter((r) => RULE_IDS.includes(r.id));
  assert(
    rules.length >= 7 && new Set(rules.map((r) => r.id)).size === rules.length,
    "all supported pointwise rules required.",
  );
  return rules;
}
function drawModelSummary(model) {
  text("model-name", model.name);
  document.title = `Unified Interpretability Toolkit · ${model.name}`;
  text(
    "model-scope",
    `${number(model.parameter_count)} original values · ${number(model.catalog.length)} stored tensors · native tensor coordinates`,
  );
  text("representation", model.representation);
  text("revision", model.revision);
  text(
    "source-directory",
    model.content_digest || model.source_directory || model.source_identity,
  );
  text("source-bytes", number(model.source_bytes));
  text("tensor-count", String(model.catalog.length));
  text("render-semantics", model.render_semantics);
  drawCoverage(model);
  const peak = model.catalog.find((t) => t.max_abs === model.global_max),
    matrices = model.catalog.filter(
      (t) => t.rows > 1 && t.cols > 1 && t.calibration_complete,
    );
  const matrixMax = matrices.length
    ? Math.max(...matrices.map((t) => t.max_abs))
    : null;
  text(
    "global-scale-note",
    !calibrationScope(model).globalSupported
      ? "Global scale unavailable: this catalog includes unsupported tensors. Local rules use each complete supported tensor; calibrating the supported subset cannot enable global rules."
      : !model.calibration_complete
        ? "Global scale unavailable until every source value has been calibrated. Current local rules use the complete selected tensor."
        : `Global bounds are ±${exact(model.global_max)}${peak ? " (reached in " + peak.name + ")" : ""}.` +
          (matrixMax !== null
            ? ` Largest matrix magnitude: ${exact(matrixMax)}.`
            : "") +
          " This can make global matrix colors pale. Tensor asinh uses separately labeled calibration to show local detail.",
  );
}
function populateModelFilters(model, rules, prior) {
  const layers = [
    ...new Set(model.catalog.map(layerKey).filter((v) => v !== "shared")),
  ].sort((a, b) => Number(a) - Number(b));
  $("layer-filter").replaceChildren(
    option("all", "All layers and shared tensors"),
    option("shared", "Shared / embedding / final tensors"),
    ...layers.map((v) => option(v, "Layer " + v)),
  );
  $("layer-filter").value = "all";
  for (const side of SIDES) {
    $("" + side + "-rule").replaceChildren(
      ...rules.map((r) => option(r.id, r.title)),
    );
    $("" + side + "-rule").value =
      prior && RULE_IDS.includes(prior[side])
        ? prior[side]
        : side === "left"
          ? model.calibration_complete
            ? "global_linear"
            : "tensor_linear"
          : "tensor_asinh";
    $("" + side + "-rule").disabled = false;
    for (const o of $("" + side + "-rule").options)
      o.disabled = o.value.startsWith("global_") && !model.calibration_complete;
    if (
      !model.calibration_complete &&
      $("" + side + "-rule").value.startsWith("global_")
    )
      $("" + side + "-rule").value =
        side === "left" ? "tensor_linear" : "tensor_asinh";
  }
}
function selectModelTensor(model, prior) {
  const candidates = model.catalog
    .filter((t) => t.available !== false && t.shape.length <= 2)
    .sort(
      (a, b) =>
        Number(b.calibration_complete) - Number(a.calibration_complete) ||
        (a.calibration_complete ? 0 : a.count - b.count),
    );
  state.tensor =
    model.catalog.find((t) => t.id === prior?.tensor) ||
    candidates[0] ||
    model.catalog[0];
  if (prior?.slice !== undefined)
    state.tensor = {
      ...state.tensor,
      slice: prior.slice ? prior.slice.split(",").map(Number) : [],
    };
}
function populateModel(model, prior) {
  const rules = validateModelCatalog(model);
  state.model = model;
  drawExamples();
  drawModelSummary(model);
  populateModelFilters(model, rules, prior);
  selectModelTensor(model, prior);
  updateRuleAvailability();
  drawCatalog();
  globalThis.atlasWorkspace?.model(model);
}
async function refreshModel() {
  try {
    await globalThis.AtlasHost?.prepareProfileChange?.();
  } catch (e) {
    showError(e.message);
    return;
  }
  globalThis.AtlasHost?.setProfileSelection?.(null);
  if (typeof window.CustomEvent === "function")
    window.dispatchEvent(new CustomEvent("atlas:tensor", { detail: null }));
  const prior = state.tensor ? settings() : null,
    id = ++state.modelEpoch;
  state.modelController?.abort();
  state.modelController = new AbortController();
  state.model = null;
  for (const side of SIDES) $(side + "-rule").disabled = true;
  deactivate();
  showError("");
  status("Loading checkpoint catalog and coverage…");
  $("refresh-model").disabled = true;
  try {
    const model = await json("/api/model", state.modelController.signal);
    if (id !== state.modelEpoch) return;
    populateModel(model, prior);
    await prepareTensor();
  } catch (e) {
    if (e.name !== "AbortError" && id === state.modelEpoch) {
      if (e.status === 503 && !e.code) {
        text(
          "calibration-note",
          "Initializing: full source statistics are not ready. No color calibration is assumed. Refresh status to retry.",
        );
        $("calibration-note").hidden = false;
        status("Initializing · full statistics unavailable.");
      } else {
        showError(e.message);
        status("Catalog unavailable. Refresh status to retry.");
      }
      for (const side of SIDES) $(side + "-rule").disabled = true;
    }
  } finally {
    if (id === state.modelEpoch) $("refresh-model").disabled = false;
  }
}
async function selectTensor(id) {
  if (!state.model) return;
  const tensor = state.model.catalog.find((t) => t.id === id);
  if (!tensor) return;
  const selectedModel = state.model;
  try {
    await globalThis.AtlasHost?.prepareProfileChange?.();
  } catch (e) {
    showError(e.message);
    return;
  }
  if (state.model !== selectedModel) return;
  globalThis.AtlasHost?.setProfileSelection?.(null);
  state.tensor = tensor;
  globalThis.atlasWorkspace?.changed("tensor");
  deactivate();
  drawCatalog();
  const epoch = state.modelEpoch,
    view = state.viewEpoch;
  try {
    const update = await json("/api/tensor-status?tensor=" + id);
    if (
      epoch !== state.modelEpoch ||
      view !== state.viewEpoch ||
      state.tensor?.id !== id
    )
      return;
    applyStatus(update);
    await prepareTensor();
  } catch (e) {
    if (
      e.name !== "AbortError" &&
      epoch === state.modelEpoch &&
      view === state.viewEpoch &&
      state.tensor?.id === id
    )
      showError(e.message);
  }
  if (window.innerWidth <= 760) $("tensor-browser").open = false;
}
function tileSource(t, rule, binding) {
  if (binding !== undefined)
    assert(
      typeof binding === "string" && /^[a-f0-9]{64}$/.test(binding),
      "Invalid tile binding.",
    );
  return {
    width: t.cols,
    height: t.rows,
    tileSize: 256,
    tileOverlap: 0,
    minLevel: 0,
    maxLevel: t.max_level,
    getTileUrl: (level, x, y) =>
      apiURL(
        "/tile?" +
          new URLSearchParams({
            tensor: t.id,
            ...(t.slice ? { slice: t.slice.join(",") } : {}),
            rule,
            level,
            x,
            y,
            ...(binding ? { binding } : {}),
          }),
      ),
  };
}
function createViewer(side, view, s, id, signal) {
  const host = $(side + "-canvas"),
    stage = node("div", "canvas-stage");
  stage.id = `${side}-stage-${id}`;
  host.replaceChildren(stage);
  const viewer = createNativeViewer(stage);
  state.viewers[side] = viewer;
  viewer.setMouseNavEnabled(false);
  const valid = () => id === state.viewEpoch && state.viewers[side] === viewer;
  const opened = waitForViewerOpen(viewer, signal, valid);
  bindViewerDrawing(viewer, side, valid);
  bindViewerInspection(viewer, host, view, signal, valid);
  viewer.open(tileSource(view.tensor, s[side], view.tile_bindings?.[side]));
  return { viewer, stage, opened };
}

function createNativeViewer(stage) {
  return OpenSeadragon({
    element: stage,
    drawer: "canvas",
    showNavigationControl: false,
    showNavigator: false,
    tabIndex: -1,
    maxZoomPixelRatio: 24,
    minPixelRatio: 1,
    visibilityRatio: 0.25,
    constrainDuringPan: true,
    preserveViewport: false,
    imageSmoothingEnabled: false,
    blendTime: 0,
    animationTime: 0,
    immediateRender: true,
    maxImageCacheCount: 32,
    imageLoaderLimit: 1,
    timeout: 60000,
    gestureSettingsMouse: {
      clickToZoom: false,
      dblClickToZoom: true,
      scrollToZoom: true,
    },
    gestureSettingsTouch: {
      clickToZoom: false,
      dblClickToZoom: true,
      pinchToZoom: true,
    },
  });
}

function waitForViewerOpen(viewer, signal, valid) {
  return new Promise((resolve, reject) => {
    const abort = () => reject(new DOMException("Cancelled", "AbortError"));
    signal.addEventListener("abort", abort, { once: true });
    viewer.addHandler("open", () => {
      signal.removeEventListener("abort", abort);
      if (!valid() || signal.aborted) {
        abort();
        return;
      }
      resolve(viewer);
    });
    viewer.addHandler("open-failed", (event) => {
      signal.removeEventListener("abort", abort);
      reject(new Error(event.message || "Tensor image could not open."));
    });
  });
}

function bindViewerDrawing(viewer, side, valid) {
  viewer.addHandler("tile-drawn", () => {
    if (valid() && !document.body.dataset.firstTileMs) {
      document.body.dataset.firstTileMs = String(
        performance.now() - state.openStarted,
      );
      text(
        "first-paint",
        `First numeric tile drawn ${(performance.now() - state.openStarted).toFixed(0)} ms after app script start`,
      );
    }
  });
  viewer.addHandler("viewport-change", () => {
    if (valid()) {
      cancelHover();
      synchronize(side);
      updateOverview();
      globalThis.atlasWorkspace?.changed("viewport");
    }
    scheduleResolution();
  });
  for (const event of [
    "animation",
    "update-viewport",
    "tile-drawn",
    "tile-loaded",
    "resize",
  ])
    viewer.addHandler(event, () => {
      if (valid()) scheduleResolution();
    });
  viewer.addHandler("tile-load-failed", (event) => {
    if (!valid()) return;
    state.viewFailed = true;
    showError(
      "A numeric tile could not load. Use Refresh status to retry; no replacement values are fabricated.",
    );
    status("Tile request failed · requested view may be incomplete.");
  });
}

function bindViewerInspection(viewer, host, view, signal, valid) {
  viewer.addHandler("canvas-click", (event) => {
    if (
      !valid() ||
      !event.quick ||
      state.loading ||
      !state.current ||
      !viewer.world.getItemCount()
    )
      return;
    const item = viewer.world.getItemAt(0),
      p = item.viewportToImageCoordinates(
        viewer.viewport.pointFromPixel(event.position),
      );
    const row = Math.floor(p.y),
      col = Math.floor(p.x);
    if (
      row >= 0 &&
      col >= 0 &&
      row < view.tensor.rows &&
      col < view.tensor.cols
    )
      inspectAt(row, col);
  });
  host.addEventListener(
    "pointermove",
    (event) => {
      if (
        !valid() ||
        state.loading ||
        event.pointerType === "touch" ||
        event.buttons ||
        !viewer.world.getItemCount()
      )
        return;
      const box = host.getBoundingClientRect(),
        p = viewer.world
          .getItemAt(0)
          .viewportToImageCoordinates(
            viewer.viewport.pointFromPixel(
              new OpenSeadragon.Point(
                event.clientX - box.left,
                event.clientY - box.top,
              ),
              true,
            ),
          );
      queueHover(Math.floor(p.y), Math.floor(p.x));
    },
    { signal },
  );
  host.addEventListener("pointerleave", () => cancelHover(), { signal });
}

function drawLegend(side, l) {
  text(side + "-min", exact(l.min));
  text(side + "-max", exact(l.max));
  text(side + "-scope", `${l.title} · ${l.scope}`);
  const magnitude = ["tensor_magnitude", "tensor_magnitude_asinh"].includes(
    l.id,
  );
  $(side + "-gradient").classList.toggle("magnitude", magnitude);
  text(side + "-mid", magnitude ? "" : "0");
  text(
    side + "-parameter",
    ["tensor_robust99", "tensor_magnitude_asinh"].includes(l.id)
      ? `${l.id === "tensor_magnitude_asinh" ? "Median nonzero |x| s = " + exact(l.s) + "; " : ""}Q99 = ${exact(l.q99)}; effective divisor D = ${exact(l.effective_divisor)}${l.zero_quantile_fallback ? " (Q99=0 fallback)" : ""}. ${number(l.clipped_count)} original values clipped (${(100 * l.clipped_fraction).toFixed(3)}%).`
      : l.id === "tensor_signed_percentile"
        ? "Signed absolute-magnitude mid-CDF; ties share ranks; zeros map to zero."
        : magnitude
          ? `Unsigned magnitude; exact tensor M = ${exact(l.max)}`
          : l.s === null || l.s === undefined
            ? "Linear mapping; raw bounds shown."
            : `Exact asinh scale s = ${exact(l.s)}`,
  );
  text(side + "-formula", l.formula);
  text(
    side + "-units",
    l.units +
      (magnitude
        ? " · Light → dark purple; sign is retained in the raw inspector."
        : " · pooled pixels average the transformed coordinate.") +
      (l.id.startsWith("tensor_")
        ? " Detail within this tensor; independent scale. Colors cannot compare magnitudes across tensors."
        : " Global checkpoint reference scale."),
  );
  $(side + "-gradient").classList.toggle(
    "all-zero",
    l.min === 0 && l.max === 0,
  );
}
function tensorContextKey() {
  return JSON.stringify([
    state.model?.source_identity,
    state.model?.model_identity,
    state.model?.revision,
    state.modelEpoch,
    state.model?.host_context?.context_id,
    state.tensor?.id,
    state.tensor?.name,
    state.tensor?.shape,
    state.tensor?.dtype,
    state.tensor?.slice || [],
  ]);
}
async function loadView() {
  if (refuseUnavailableTensor()) return;
  if (refuseUnsupportedView()) return;
  if (!state.model || !state.tensor?.calibration_complete) return;
  rememberViewContext();
  // Carry the last active viewport through consecutive rule changes, even while
  // an earlier recoloring request is still opening and state.current is null.
  const remembered = state.recolor,
    prior = remembered?.bounds;
  deactivate(!!remembered);
  const id = state.viewEpoch,
    restoreInspectionEpoch = state.inspectEpoch;
  state.viewController = new AbortController();
  const signal = state.viewController.signal,
    s = settings();
  const started = performance.now();
  drawViewLoading();
  try {
    assert(
      RULE_IDS.includes(s.left) && RULE_IDS.includes(s.right),
      "unsupported color rule.",
    );
    const view = await json("/api/view?" + new URLSearchParams(s), signal);
    if (id !== state.viewEpoch) return;
    validateView(view, s);
    const pair = SIDES.map((side) => createViewer(side, view, s, id, signal));
    await Promise.all(pair.map((p) => p.opened));
    if (
      id !== state.viewEpoch ||
      signal.aborted ||
      !sameSettings(s, settings())
    )
      return;
    activateLoadedView(view, s, pair, prior);
    // An explicit inspection begun after recolor invalidation owns the pin,
    // whether its source read has completed or is still pending.
    if (remembered?.selected && state.inspectEpoch === restoreInspectionEpoch)
      inspectAt(...remembered.selected, true);
    if (state.model.coverage?.all_requested) pollStatus();
    if (!state.viewFailed)
      status(
        `Same tensor active · metadata and viewer setup ${(performance.now() - started).toFixed(0)} ms. Numeric tiles load separately.`,
      );
  } catch (e) {
    refuseViewLoad(e, id);
  }
}

function rememberViewContext() {
  const contextKey = tensorContextKey();
  const pendingRaw =
    state.pendingInspection?.key === contextKey
      ? state.pendingInspection.selected
      : null;
  if (
    state.current?.tensor.id === state.tensor.id &&
    JSON.stringify(state.current.tensor.slice || []) ===
      JSON.stringify(state.tensor.slice || []) &&
    state.viewers.left?.world.getItemCount()
  ) {
    const b = state.viewers.left.viewport.getBounds(true);
    state.recolor = {
      key: contextKey,
      bounds: b,
      selected: state.selected ? [...state.selected] : null,
    };
  } else if (
    !state.current &&
    !state.recolor &&
    (pendingRaw ||
      (state.selected && state.inspectionData?.tensor === state.tensor.id))
  ) {
    // First calibration can finish before a valid raw read returns. Remember
    // the newest same-context address, never its unvalidated response. Opening
    // re-reads it once through the normal binding and inspection-epoch checks.
    state.recolor = {
      key: contextKey,
      bounds: null,
      selected: [...(pendingRaw || state.selected)],
    };
  } else if (state.recolor?.key !== contextKey) state.recolor = null;
}

function drawViewLoading() {
  showError("");
  status("Loading both color rules for the same tensor…");
  describeTensor();
  text("tensor-name", state.tensor.name);
  text(
    "tensor-shape",
    `${state.tensor.shape.join(" × ")} native shape${state.tensor.shape.length > 2 ? " · fixed [" + state.tensor.slice.join(", ") + "]" : ""} · ${number(state.tensor.count)} values · rows ↓ / columns →${state.tensor.rows === 1 ? " · one unwrapped vector row" : ""}`,
  );
  for (const side of SIDES) {
    text(side + "-min", "—");
    text(side + "-max", "—");
    text(side + "-formula", "Waiting for exact legend…");
    text(side + "-scope", "Requested view loading");
    text(side + "-parameter", "");
    text(side + "-units", "");
  }
}

function activateLoadedView(view, s, pair, prior) {
  state.current = { ...view, settings: s };
  state.loading = false;
  state.syncing = true;
  try {
    for (const p of pair) {
      if (prior) p.viewer.viewport.fitBounds(prior, true);
      else p.viewer.viewport.goHome(true);
      p.stage.classList.add("active");
      p.viewer.setMouseNavEnabled(true);
    }
  } finally {
    state.syncing = false;
  }
  for (const side of SIDES) drawLegend(side, view.legends[side]);
  $("row").max = String(view.tensor.rows - 1);
  $("col").max = String(view.tensor.cols - 1);
  for (const key of ["row", "col"])
    $(key).value = String(
      Math.min(Number($(key).max), Math.max(0, Number($(key).value) || 0)),
    );
  $("comparison").setAttribute("aria-busy", "false");
  setInteraction(true);
  drawCatalog();
  scheduleResolution();
  globalThis.atlasWorkspace?.loaded();
  loadOverview();
  if (state.selected) markers();
}

function refuseViewLoad(e, id) {
  if (e.name !== "AbortError" && id === state.viewEpoch) {
    state.current = null;
    state.loading = false;
    setInteraction(false);
    stopViews();
    if (e.status === 503 && !e.code) {
      text(
        "calibration-note",
        "Initializing: completed numeric statistics are unavailable. Refresh status to retry.",
      );
      $("calibration-note").hidden = false;
      status("Initializing · no active color view.");
    } else {
      showError(e.message);
      status("View unavailable · previous images are inactive.");
    }
  }
}

function synchronize(side) {
  if (state.syncing || state.loading || !state.current) return;
  const a = state.viewers[side],
    b = state.viewers[side === "left" ? "right" : "left"];
  if (!a || !b || !a.world.getItemCount() || !b.world.getItemCount()) return;
  const ar = a.viewport.getBounds(true),
    br = b.viewport.getBounds(true);
  if (
    ["x", "y", "width", "height"].every((k) => Math.abs(ar[k] - br[k]) < 1e-9)
  )
    return;
  state.syncing = true;
  try {
    b.viewport.fitBounds(ar, true);
  } finally {
    state.syncing = false;
  }
}
function scheduleResolution() {
  if (state.frame) return;
  state.frame = true;
  requestAnimationFrame(() => {
    state.frame = false;
    updateResolution();
  });
}
function updateResolution() {
  if (!state.current || state.loading) return;
  const t = state.current.tensor;
  updateViewportBounds();
  for (const side of SIDES) {
    const viewer = state.viewers[side];
    if (!viewer?.world.getItemCount()) continue;
    const item = viewer.world.getItemAt(0);
    const levels = [
      ...new Set((item.lastDrawn || []).map((d) => d.tile.level)),
    ].sort((a, b) => a - b);
    const factors = levels.map((level) => 2 ** (t.max_level - level));
    const ready = item.getFullyLoaded();
    const scalar = levels.length === 1 && levels[0] === t.max_level;
    if (
      ready &&
      SIDES.every((s) =>
        state.viewers[s]?.world.getItemAt(0)?.getFullyLoaded(),
      ) &&
      !document.body.dataset.bothViewsReadyMs
    )
      document.body.dataset.bothViewsReadyMs = String(
        performance.now() - state.openStarted,
      );
    let label = levels.length
      ? scalar
        ? "Scalar-level cells · one source address"
        : `Pooled cells · ${Math.min(...factors) === Math.max(...factors) ? factors[0] : Math.min(...factors) + "–" + Math.max(...factors)} per-side source blocks`
      : "Loading numeric tiles…";
    if (levels.length && !ready) label += " · loading";
    text(side + "-resolution", label);
    $(side + "-resolution").classList.toggle("scalar", scalar && ready);
    const ratio = viewer.viewport.viewportToImageZoom(
      viewer.viewport.getZoom(true),
    );
    text(side + "-magnification", `${ratio.toFixed(2)} CSS px / scalar`);
    $(side + "-canvas").dataset.loadedLevels = levels.join(",");
  }
}
function markers() {
  for (const side of SIDES) {
    const v = state.viewers[side];
    if (!v?.world.getItemCount()) continue;
    v.clearOverlays();
    if (!state.selected) continue;
    const [row, col] = state.selected,
      marker = node("div", "scalar-marker");
    marker.setAttribute("aria-hidden", "true");
    v.addOverlay({
      element: marker,
      location: v.world.getItemAt(0).imageToViewportRectangle(col, row, 1, 1),
      checkResize: false,
    });
  }
}
function fact(dl, key, value) {
  dl.append(node("dt", "", key), node("dd", "", value));
}
function showInspection(data) {
  text(
    "pinned-readout",
    `Pinned [${data.native_indices.join(", ")}] · raw ${data.raw_exact}`,
  );
  const box = $("inspection");
  box.replaceChildren(
    node("div", "inspection-label", `Original ${data.dtype || "BF16"} value`),
    node("code", "raw-value", data.raw_exact),
    node(
      "p",
      "native-index",
      `Native index [${data.native_indices.join(", ")}]`,
    ),
  );
  const dl = node("dl", "facts");
  fact(
    dl,
    `${data.dtype || "BF16"} bytes · little endian`,
    data.raw_hex_le || data.bf16_hex_le,
  );
  if (data.classification && data.classification !== "finite")
    fact(dl, "Nonfinite source value", data.classification);
  fact(dl, "Original shard", data.shard);
  fact(dl, "Byte offset", exact(data.byte_offset));
  fact(dl, "Display row / column", `${data.row} / ${data.col}`);
  box.append(dl);
  const transforms = node("div", "transforms");
  for (const side of SIDES) {
    const cell = node("div");
    cell.append(
      node("strong", "", `${side === "left" ? "A" : "B"} · transformed`),
      node(
        "code",
        "",
        data.transformed[side] === null
          ? data.transform_errors?.[side] ||
              (data.classification && data.classification !== "finite"
                ? "Unavailable: nonfinite source"
                : "Awaiting calibration")
          : exact(data.transformed[side]),
      ),
    );
    transforms.append(cell);
  }
  box.append(transforms);
  box.append(
    node(
      "p",
      "small muted",
      "One exact source address, shared by both views. This value is not the mean of the displayed pooled cell.",
    ),
  );
}
function validateInspection(data, t, row, col) {
  if (globalThis.AtlasTools)
    AtlasTools.requireBinding(
      data.source_binding,
      AtlasTools.sourceBinding(state.model, t),
    );
  assert(
    data.tensor === t.id && data.row === row && data.col === col,
    "inspection address does not match the request.",
  );
  const dtype = data.dtype || "BF16",
    width = { BF16: 2, F16: 2, F32: 4 }[dtype],
    raw = data.raw_hex_le || data.bf16_hex_le;
  assert(
    width &&
      (!t.dtype || t.dtype === dtype) &&
      (data.element_bytes === undefined || data.element_bytes === width),
    "source dtype and byte width must match the selected tensor.",
  );
  assert(
    typeof data.raw_exact === "string" &&
      typeof raw === "string" &&
      new RegExp("^[0-9a-fA-F]{" + 2 * width + "}$").test(raw),
    "exact decimal string and dtype-sized original bytes required.",
  );
  assert(
    Array.isArray(data.native_indices) &&
      data.native_indices.every(Number.isSafeInteger),
    "native source indices required.",
  );
  assert(
    typeof data.shard === "string" &&
      Number.isSafeInteger(data.byte_offset) &&
      data.byte_offset >= 0,
    "original shard and safe integer byte offset required.",
  );
  assert(
    (data.transforms_ready === false &&
      [data.transformed?.left, data.transformed?.right].every(
        (v) => v === null || Number.isFinite(v),
      )) ||
      (Number.isFinite(data.transformed?.left) &&
        Number.isFinite(data.transformed?.right)),
    "transforms must be finite or explicitly uncalibrated.",
  );
  if (t.shape.length === 1)
    assert(
      data.native_indices.length === 1 && data.native_indices[0] === col,
      "vector source index must equal the column.",
    );
  else
    assert(
      JSON.stringify(data.native_indices) ===
        JSON.stringify([...(t.slice || []), row, col]),
      "slice native index mismatch.",
    );
}
async function inspectAt(row, col, keepVisible = false) {
  if (!state.tensor) return;
  const t = state.tensor;
  if (
    !Number.isSafeInteger(row) ||
    !Number.isSafeInteger(col) ||
    row < 0 ||
    col < 0 ||
    row >= t.rows ||
    col >= t.cols
  ) {
    showError("Enter a whole-number row and column inside this tensor.");
    return;
  }
  const id = ++state.inspectEpoch,
    view = state.viewEpoch,
    s = { ...settings(), row, col };
  state.inspectController?.abort();
  state.inspectController = new AbortController();
  state.pendingInspection = {
    epoch: id,
    key: tensorContextKey(),
    selected: [row, col],
  };
  if (!keepVisible)
    $("inspection").replaceChildren(
      node("p", "empty-state", "Reading the original source bytes…"),
    );
  try {
    const data = await json(
      "/api/inspect?" + new URLSearchParams(s),
      state.inspectController.signal,
    );
    if (
      id !== state.inspectEpoch ||
      view !== state.viewEpoch ||
      t.id !== state.tensor?.id
    )
      return;
    validateInspection(data, t, row, col);
    state.inspectionData = data;
    state.selected = [row, col];
    $("row").value = String(row);
    $("col").value = String(col);
    showInspection(data);
    markers();
    if (!state.viewFailed) showError("");
    publishInferenceSelection(
      t.shape.length <= 2
        ? {
            source_model: state.model.inference_source_model || null,
            tensor: t.name,
            shape: [...t.shape],
            row,
            col,
          }
        : null,
    );
  } catch (e) {
    if (
      e.name !== "AbortError" &&
      id === state.inspectEpoch &&
      view === state.viewEpoch
    ) {
      $("inspection").replaceChildren(
        node("p", "empty-state", "This address could not be read."),
      );
      showError(e.message);
    }
  } finally {
    if (state.pendingInspection?.epoch === id) state.pendingInspection = null;
  }
}
function zoomBy(factor) {
  if (!state.current) return;
  state.viewers.left.viewport.zoomBy(factor);
  state.viewers.left.viewport.applyConstraints(true);
  synchronize("left");
  scheduleResolution();
}
function fitView() {
  if (!state.current) return;
  state.viewers.left.viewport.goHome(true);
  synchronize("left");
  scheduleResolution();
}
function scalarZoom() {
  if (!state.current) return;
  const v = state.viewers.left,
    item = v.world.getItemAt(0),
    t = state.current.tensor;
  const [row, col] = state.selected || [
    Math.floor(t.rows / 2),
    Math.floor(t.cols / 2),
  ];
  const center = item.imageToViewportCoordinates(col + 0.5, row + 0.5);
  v.viewport.zoomTo(v.viewport.imageToViewportZoom(1), null, true);
  v.viewport.panTo(center, true);
  v.viewport.applyConstraints(true);
  synchronize("left");
  scheduleResolution();
}
function focusView() {
  const side =
    window.innerWidth <= 760
      ? $("comparison").dataset.mobileSide || "left"
      : "left";
  const host = $(side + "-canvas");
  host.scrollIntoView?.({ block: "center" });
  host.focus?.({ preventScroll: true });
}
function bind() {
  bindWelcomeAndTheme();
  window.atlasFocusView = focusView;
  for (const [id, target] of [
    ["skip-to-view", "workspace"],
    ["nav-inspect", "scalar-inspector"],
    ["nav-experiment", "inference-panel"],
  ])
    $(id).addEventListener("click", (event) => {
      event.preventDefault();
      const region = $(target);
      if (target === "inference-panel") region.open = true;
      region.scrollIntoView?.({ block: "start" });
      region.focus?.({ preventScroll: true });
    });
  $("nav-view").addEventListener("click", (event) => {
    event.preventDefault();
    focusView();
  });
  for (const side of SIDES)
    $("show-" + side).addEventListener("click", () => {
      $("comparison").dataset.mobileSide = side;
      for (const s of SIDES)
        $("show-" + s).setAttribute("aria-pressed", String(s === side));
      cancelHover();
    });
  $("calibrate-all").addEventListener("click", async () => {
    if (!state.model || calibrationScope(state.model).pending === 0) return;
    try {
      await calibrationAction("all=1");
      pollStatus();
    } catch (e) {
      showError(e.message);
    }
  });
  $("refresh-model").addEventListener("click", () => {
    refreshModel();
    window.atlasRefreshAvailability?.();
  });
  $("tensor-search").addEventListener("input", drawCatalog);
  $("layer-filter").addEventListener("change", drawCatalog);
  for (const side of SIDES)
    $(side + "-rule").addEventListener("change", loadView);
  $("fit").addEventListener("click", fitView);
  $("zoom-in").addEventListener("click", () => zoomBy(2));
  $("zoom-out").addEventListener("click", () => zoomBy(0.5));
  $("scalar-zoom").addEventListener("click", scalarZoom);
  $("inspect-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if ($("row").value.trim() === "" || $("col").value.trim() === "") {
      showError("Enter both a row and a column.");
      return;
    }
    inspectAt(Number($("row").value), Number($("col").value));
  });
  for (const side of SIDES)
    $(side + "-canvas").addEventListener("keydown", (event) => {
      handleViewKey(event);
    });
  if (window.innerWidth <= 760) $("tensor-browser").open = false;
}

function panViewKey(event) {
  event.preventDefault();
  const v = state.viewers.left,
    b = v.viewport.getBounds(true);
  v.viewport.panTo(
    new OpenSeadragon.Point(
      b.x +
        b.width *
          (0.5 +
            (event.key === "ArrowRight"
              ? 0.2
              : event.key === "ArrowLeft"
                ? -0.2
                : 0)),
      b.y +
        b.height *
          (0.5 +
            (event.key === "ArrowDown"
              ? 0.2
              : event.key === "ArrowUp"
                ? -0.2
                : 0)),
    ),
    true,
  );
  v.viewport.applyConstraints(true);
  synchronize("left");
  scheduleResolution();
}

function inspectViewKey(event) {
  event.preventDefault();
  const t = state.tensor,
    [r, c] = state.selected || [Math.floor(t.rows / 2), Math.floor(t.cols / 2)];
  inspectAt(
    Math.max(
      0,
      Math.min(
        t.rows - 1,
        r + (event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0),
      ),
    ),
    Math.max(
      0,
      Math.min(
        t.cols - 1,
        c +
          (event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0),
      ),
    ),
  );
}

function handleViewKey(event) {
  if (event.key === "Home") {
    event.preventDefault();
    fitView();
  }
  if (
    event.shiftKey &&
    ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key) &&
    state.current
  ) {
    panViewKey(event);
    return;
  }
  if (
    ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key) &&
    state.tensor &&
    !state.loading
  ) {
    inspectViewKey(event);
  }
  if (event.key === "+" || event.key === "=") {
    event.preventDefault();
    zoomBy(2);
  }
  if (event.key === "-") {
    event.preventDefault();
    zoomBy(0.5);
  }
}

function refuseUnavailableTensor() {
  const t = state.tensor;
  if (t.available !== false && (t.shape.length <= 2 || Array.isArray(t.slice)))
    return false;
  deactivate();
  describeTensor();
  text("tensor-name", t.name);
  text("tensor-shape", `${t.shape.join(" × ")} native shape`);
  const reason =
    t.available === false
      ? t.unavailable_reason
      : "Choose every leading index and apply the 2D slice. Rows and columns are the final two axes.";
  status(reason);
  showError("");
  for (const side of SIDES) {
    text(side + "-scope", "No active view");
    text(side + "-formula", reason);
    text(side + "-min", "—");
    text(side + "-max", "—");
    text(side + "-units", "");
  }
  return true;
}
function drawSlicePicker() {
  const host = $("slice-picker");
  if (!host) return;
  host.replaceChildren();
  const t = state.tensor;
  host.hidden = !t || t.available === false || t.shape.length <= 2;
  if (host.hidden) return;
  host.append(
    node(
      "p",
      "small",
      `2D slice: rows = axis ${t.shape.length - 2}, columns = axis ${t.shape.length - 1}. Fix the leading indices explicitly.`,
    ),
  );
  const inputs = t.shape.slice(0, -2).map((size, i) => {
    const label = node("label", "", `Axis ${i} (0–${size - 1}) `),
      input = node("input");
    input.type = "number";
    input.min = "0";
    input.max = String(size - 1);
    input.step = "1";
    input.required = true;
    input.value = t.slice ? String(t.slice[i]) : "";
    label.append(input);
    host.append(label);
    return input;
  });
  const button = node("button", "", "Apply 2D slice");
  button.type = "button";
  button.addEventListener("click", async () => {
    try {
      const leading = inputs.map((input) =>
        /^\d+$/.test(input.value) ? Number(input.value) : NaN,
      );
      await globalThis.AtlasHost?.prepareProfileChange?.();
      if (state.tensor !== t) return;
      globalThis.AtlasHost?.setProfileSelection?.(null);
      state.tensor = AtlasTools.withSlice(t, leading);
      deactivate();
      globalThis.atlasWorkspace?.changed("tensor");
      prepareTensor();
    } catch (e) {
      showError(e.message);
    }
  });
  host.append(button);
}
async function calibrationAction(query) {
  if (globalThis.AtlasHost)
    throw new Error("Fixture preparation is an owner CLI action.");
  const r = await fetch("/api/calibrate?" + query, {
    method: "POST",
    headers: { "X-Atlas-Local": "1" },
  });
  const v = await r.json();
  if (!r.ok) throw new Error(v.error || "Calibration could not start.");
  return v;
}
function allowRawInspection() {
  if (
    !state.tensor ||
    state.tensor.available === false ||
    (state.tensor.shape.length > 2 && !state.tensor.slice)
  )
    return;
  for (const k of ["row", "col"]) {
    $(k).disabled = false;
    $(k).max = String(
      (k === "row" ? state.tensor.rows : state.tensor.cols) - 1,
    );
    $(k).value = String(
      Math.min(Number($(k).max), Math.max(0, Number($(k).value) || 0)),
    );
  }
  $("inspect-submit").disabled = false;
}
async function prepareTensor() {
  if (globalThis.AtlasHost?.setProfileSelection) {
    let selected = null;
    try {
      selected = AtlasTools.sourceBinding(state.model, state.tensor);
    } catch {}
    await AtlasHost.setProfileSelection(selected);
  }
  if (typeof window.CustomEvent === "function")
    window.dispatchEvent(
      new CustomEvent("atlas:tensor", { detail: state.tensor }),
    );
  drawSlicePicker();
  if (refuseUnavailableTensor()) return;
  updateRuleAvailability();
  if (refuseUnsupportedView()) return;
  if (state.tensor.calibration_complete) {
    await loadView();
    return;
  }
  deactivate();
  const id = state.viewEpoch,
    t = state.tensor;
  allowRawInspection();
  globalThis.atlasWorkspace?.rawReady();
  describeTensor();
  text("tensor-name", t.name);
  text(
    "tensor-shape",
    `${t.shape.join(" × ")} native shape · ${number(t.count)} values`,
  );
  for (const side of SIDES) {
    text(side + "-min", "—");
    text(side + "-max", "—");
    text(side + "-scope", "Complete selected-tensor calibration pending");
    text(side + "-formula", "No partial bound is used");
    text(side + "-parameter", "");
    text(side + "-units", "");
  }
  if (globalThis.AtlasHost) {
    status(
      "Raw inspection ready. Color scales require separately reviewed owner preparation before reopening this model.",
    );
    text(
      "calibration-note",
      "Color scales are not prepared. Model selection never starts a full scan.",
    );
    $("calibration-note").hidden = false;
    return;
  }
  status(
    "Reading the selected tensor for exact calibration. Raw addresses can already be inspected.",
  );
  try {
    await calibrationAction("tensor=" + t.id);
    if (id === state.viewEpoch) pollStatus();
  } catch (e) {
    if (id === state.viewEpoch) showError(e.message);
  }
}
function applyStatus(update) {
  assert(
    update.model_status_version === 1 &&
      state.model &&
      update.source_identity === state.model.source_identity &&
      update.model_identity === state.model.model_identity,
    "status source identity changed.",
  );
  assert(
    typeof update.calibration_complete === "boolean" &&
      (update.global_max === null || Number.isFinite(update.global_max)),
    "bounded global status required.",
  );
  const coverage = update.coverage;
  assert(
    coverage &&
      Number.isSafeInteger(coverage.calibrated_tensors) &&
      coverage.calibrated_tensors >= 0 &&
      coverage.calibrated_tensors <=
        state.model.catalog.filter((t) => t.available !== false).length &&
      Number.isSafeInteger(coverage.values_streamed) &&
      coverage.values_streamed >= 0 &&
      coverage.values_streamed <= state.model.parameter_count,
    "invalid readiness coverage.",
  );
  const patch = update.tensor_status;
  if (patch !== null) {
    assert(
      patch &&
        patch.id === state.tensor?.id &&
        typeof patch.calibration_complete === "boolean",
      "status returned another tensor.",
    );
    const allowed = [
      "id",
      "calibration_complete",
      "max_abs",
      "median_nonzero_abs",
      "q99",
      "robust_clipped_count",
      "exact_zero_count",
      "unique_bit_patterns",
      "calibration_method",
      "rule_status",
    ];
    assert(
      Object.keys(patch).every((key) => allowed.includes(key)),
      "unexpected tensor status field.",
    );
    state.tensor = { ...state.tensor, ...patch };
    state.model.catalog = state.model.catalog.map((t) =>
      t.id === patch.id ? { ...t, ...patch } : t,
    );
  }
  state.model = {
    ...state.model,
    global_max: update.global_max,
    calibration_complete: update.calibration_complete,
    coverage: { ...state.model.coverage, ...coverage },
  };
  if (typeof update.supported_calibration_complete === "boolean")
    state.model.supported_calibration_complete =
      update.supported_calibration_complete;
  drawCoverage(state.model);
  updateRuleAvailability();
}
async function pollStatus() {
  if (state.pollController && !state.pollController.signal.aborted) return;
  const controller = new AbortController(),
    signal = controller.signal,
    epoch = state.modelEpoch,
    view = state.viewEpoch;
  state.pollController = controller;
  state.polling = true;
  try {
    while (state.model && !signal.aborted) {
      const ms = state.tensor?.calibration_complete ? 1000 : 250;
      if (globalThis.AtlasTools) await AtlasTools.delay(ms, signal);
      else await new Promise((resolve) => setTimeout(resolve, ms));
      if (
        signal.aborted ||
        epoch !== state.modelEpoch ||
        view !== state.viewEpoch
      )
        return;
      const update = await json(
        "/api/progress?tensor=" + state.tensor.id,
        signal,
      );
      if (
        signal.aborted ||
        epoch !== state.modelEpoch ||
        view !== state.viewEpoch
      )
        return;
      const was = state.tensor?.calibration_complete;
      applyStatus(update);
      const model = state.model;
      if (model.calibration_complete) {
        $("calibration-note").hidden = true;
        text(
          "global-scale-note",
          `Complete checkpoint global bounds: ±${exact(model.global_max)}. Local scales remain separately labeled.`,
        );
      }
      if (!was && state.tensor?.calibration_complete) {
        await loadView();
        return;
      }
      if (
        !model.coverage.all_requested &&
        model.coverage.active_tensor === null &&
        state.tensor?.calibration_complete
      )
        break;
      if (model.coverage.calibration_error) break;
    }
  } catch (e) {
    if (
      e.name !== "AbortError" &&
      !signal.aborted &&
      epoch === state.modelEpoch &&
      view === state.viewEpoch
    ) {
      showError(e.message);
      status("Status unavailable. Refresh status to retry.");
    }
  } finally {
    if (state.pollController === controller) {
      state.pollController = null;
      state.polling = false;
    }
  }
}
// Pointer reads have their own cancellation epoch and never change the pinned selection.
const hover = { epoch: 0, timer: null, controller: null, key: null };
function cancelHover() {
  hover.epoch++;
  clearTimeout(hover.timer);
  hover.timer = null;
  hover.controller?.abort();
  hover.controller = null;
  hover.key = null;
  text("hover-readout", "Hover over either image to read one original value.");
}
function queueHover(row, col) {
  const t = state.tensor;
  if (!t || row < 0 || col < 0 || row >= t.rows || col >= t.cols) {
    cancelHover();
    return;
  }
  const key = `${state.viewEpoch}:${row}:${col}`;
  if (hover.key === key) return;
  cancelHover();
  hover.key = key;
  const id = hover.epoch,
    view = state.viewEpoch,
    s = { ...settings(), row, col };
  text("hover-readout", `Row ${row} · column ${col} · reading original value…`);
  hover.timer = setTimeout(async () => {
    const controller = new AbortController();
    hover.controller = controller;
    const timeout = setTimeout(() => controller.abort(), 2000);
    try {
      // No transport retry: at most one pending scalar read after 140 ms of rest.
      const response = await fetch(
        apiURL("/api/inspect?" + new URLSearchParams(s)),
        { signal: controller.signal, cache: "no-store" },
      );
      if (!response.ok)
        throw new Error("Original value unavailable; move to retry.");
      const data = await response.json();
      if (
        controller.signal.aborted ||
        id !== hover.epoch ||
        view !== state.viewEpoch ||
        t !== state.tensor
      )
        return;
      assert(data.api_version === 1, "expected api_version 1.");
      validateInspection(data, t, row, col);
      const head =
        globalThis.AtlasTools?.hoverHead(state.model, t, row, col) ||
        "Head unavailable · no verified layout for this source";
      text(
        "hover-readout",
        `Row ${row} · column ${col} · native [${data.native_indices.join(", ")}] · ${head} · raw ${data.dtype || "BF16"} ${data.raw_exact}`,
      );
    } catch (e) {
      if (id === hover.epoch && view === state.viewEpoch)
        text(
          "hover-readout",
          `Row ${row} · column ${col} · ${e.name === "AbortError" ? "Read timed out; move to retry." : e.message}`,
        );
    } finally {
      clearTimeout(timeout);
      if (id === hover.epoch) hover.controller = null;
    }
  }, 140);
}
function clearOverview() {
  const img = $("overview-image");
  img.hidden = true;
  img.onload = null;
  img.onerror = null;
  img.removeAttribute?.("src");
  $("overview-viewport").hidden = true;
  text("overview-status", "Waiting for numeric tiles");
  text(
    "viewport-bounds",
    "Native viewport bounds appear when the view is ready.",
  );
}
function loadOverview() {
  const t = state.tensor,
    img = $("overview-image"),
    box = $("tensor-overview"),
    id = state.viewEpoch;
  if (!state.current || !box.style) return;
  const level = Math.min(8, t.max_level),
    factor = 2 ** (t.max_level - level),
    w = Math.ceil(t.cols / factor),
    h = Math.ceil(t.rows / factor);
  const scale = Math.min(144 / w, 96 / h);
  box.style.width = `${Math.max(1, Math.round(scale * w))}px`;
  box.style.height = `${Math.max(4, Math.round(scale * h))}px`;
  img.onload = () => {
    if (id !== state.viewEpoch) return;
    img.hidden = false;
    text(
      "overview-status",
      `Overview · ${factor === 1 ? "native cells" : factor + " × " + factor + " pooled blocks"}`,
    );
    updateOverview();
  };
  img.onerror = () => {
    if (id === state.viewEpoch) {
      img.hidden = true;
      $("overview-viewport").hidden = true;
      text("overview-status", "Overview unavailable · refresh to retry");
    }
  };
  img.src = tileSource(t, settings().left).getTileUrl(level, 0, 0);
}
function updateViewportBounds() {
  const v = state.viewers.left,
    t = state.tensor;
  if (!state.current || !v?.world.getItemCount()) return;
  const b = v.viewport.getBounds(true),
    item = v.world.getItemAt(0),
    p = item.viewportToImageCoordinates(b.x, b.y),
    q = item.viewportToImageCoordinates(b.x + b.width, b.y + b.height);
  const r0 = Math.max(0, Math.floor(p.y)),
    c0 = Math.max(0, Math.floor(p.x)),
    r1 = Math.min(t.rows - 1, Math.ceil(q.y) - 1),
    c1 = Math.min(t.cols - 1, Math.ceil(q.x) - 1);
  text(
    "viewport-bounds",
    r1 < r0 || c1 < c0
      ? "Viewport is outside the tensor · use Fit tensor"
      : `Visible rows ${number(r0)}–${number(r1)} of ${number(t.rows)} ↓ · columns ${number(c0)}–${number(c1)} of ${number(t.cols)} → · zero-based, inclusive`,
  );
}
function updateOverview() {
  const v = state.viewers.left,
    t = state.tensor,
    box = $("overview-viewport");
  if (
    !state.current ||
    !v?.world.getItemCount() ||
    !box.style ||
    $("overview-image").hidden
  )
    return;
  const b = v.viewport.getBounds(true),
    item = v.world.getItemAt(0),
    p = item.viewportToImageCoordinates(b.x, b.y),
    q = item.viewportToImageCoordinates(b.x + b.width, b.y + b.height);
  const clip = (x, max) => Math.max(0, Math.min(max, x)),
    x = clip(p.x, t.cols),
    y = clip(p.y, t.rows),
    right = clip(q.x, t.cols),
    bottom = clip(q.y, t.rows);
  box.hidden = false;
  box.style.left = `${(100 * x) / t.cols}%`;
  box.style.top = `${(100 * y) / t.rows}%`;
  box.style.width = `${(100 * Math.max(0, right - x)) / t.cols}%`;
  box.style.height = `${(100 * Math.max(0, bottom - y)) / t.rows}%`;
  const scope =
    t.shape?.length > 2
      ? `Selected 2D slice [${(t.slice || []).join(", ")}]`
      : "Whole tensor";
  $("tensor-overview").setAttribute(
    "aria-label",
    `${scope} overview. Viewport rows ${Math.floor(y)} to ${Math.max(0, Math.ceil(bottom) - 1)}, columns ${Math.floor(x)} to ${Math.max(0, Math.ceil(right) - 1)}. Use Fit tensor to return to this displayed plane.`,
  );
}
function bindWelcomeAndTheme() {
  const welcome = $("welcome"),
    root = document.documentElement;
  try {
    welcome.hidden = localStorage.getItem("atlas-welcome-dismissed") === "1";
  } catch {
    /* Help stays available without storage. */
  }
  $("dismiss-help").addEventListener("click", () => {
    welcome.hidden = true;
    try {
      localStorage.setItem("atlas-welcome-dismissed", "1");
    } catch {}
    $("show-help").focus();
  });
  $("show-help").addEventListener("click", () => {
    welcome.hidden = false;
    try {
      localStorage.removeItem("atlas-welcome-dismissed");
    } catch {}
    $("dismiss-help").focus();
  });
  if (!root) return;
  let dark =
    window.matchMedia?.("(prefers-color-scheme: dark)").matches || false;
  try {
    const saved = localStorage.getItem("atlas-theme");
    if (["dark", "light"].includes(saved)) dark = saved === "dark";
  } catch {}
  const apply = () => {
    root.dataset.theme = dark ? "dark" : "light";
    $("theme-toggle").setAttribute("aria-pressed", String(dark));
    text("theme-toggle", dark ? "Light mode" : "Dark mode");
  };
  apply();
  $("theme-toggle").addEventListener("click", () => {
    dark = !dark;
    apply();
    try {
      localStorage.setItem("atlas-theme", dark ? "dark" : "light");
    } catch {}
  });
}
function drawExamples() {
  const box = $("guided-examples");
  if (!globalThis.AtlasTools?.guidedExamples) return;
  const examples = AtlasTools.guidedExamples(state.model);
  box.replaceChildren();
  for (const example of examples) {
    const card = node("a", "example-card");
    card.href = example.href;
    card.append(
      node("strong", "", example.title),
      node("span", "", example.description),
      node("small", "", example.location),
    );
    box.append(card);
  }
  if (!examples.length)
    box.append(
      node(
        "p",
        "small muted",
        "No verified starting findings are saved for this source. Choose a tensor in the browser; the inspector always reads original values.",
      ),
    );
}

async function initialize() {
  bind();
  if (globalThis.AtlasHost) {
    if (globalThis.AtlasProfiles && AtlasHost.mountProfiles)
      AtlasHost.mountProfiles($("profile-panel"), AtlasProfiles);
    $("calibrate-all").hidden = true;
    try {
      await AtlasHost.initialize(
        async () => {
          state.tensor = null;
          await refreshModel();
        },
        (message) => {
          state.modelEpoch++;
          state.modelController?.abort();
          deactivate();
          state.model = null;
          state.tensor = null;
          text("model-name", "Reader unavailable");
          showError(message);
        },
      );
    } catch (e) {
      showError(e.message);
      status("Choose an available installed model.");
    }
  } else await refreshModel();
}

// Small additive bridge for optional analytics UI; original state stays private.
window.atlasAnalyticsBridge = {
  selected: () => (state.model ? state.tensor : null),
  jump: ({ axis, index }) => {
    const t = state.tensor;
    if (
      !t ||
      !Number.isSafeInteger(index) ||
      index < 0 ||
      !["row", "column"].includes(axis) ||
      index >= (axis === "row" ? t.rows : t.cols)
    )
      return;
    const row =
      axis === "row"
        ? index
        : Math.max(
            0,
            Math.min(t.rows - 1, Math.trunc(Number($("row").value) || 0)),
          );
    const col =
      axis === "column"
        ? index
        : Math.max(
            0,
            Math.min(t.cols - 1, Math.trunc(Number($("col").value) || 0)),
          );
    for (const side of SIDES) {
      const v = state.viewers[side];
      if (
        !state.loading &&
        state.current?.tensor.id === t.id &&
        v?.world.getItemCount()
      )
        v.viewport.panTo(
          v.world.getItemAt(0).imageToViewportCoordinates(col + 0.5, row + 0.5),
          true,
        );
    }
    inspectAt(row, col);
  },
};

initialize();
