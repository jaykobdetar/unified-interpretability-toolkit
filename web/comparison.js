"use strict";
// Dedicated comparison state and routes: never emits an inference/edit target.
const $ = (id) => document.getElementById(id),
  SIDES = ["left", "right"],
  SPACE = "checkpoint-comparison-v1";
const state = {
  model: null,
  pair: null,
  view: null,
  epoch: 0,
  inspectEpoch: 0,
  modelEpoch: 0,
  controller: null,
  inspectController: null,
  modelController: null,
  viewers: {},
  syncing: false,
  poll: null,
};
const text = (id, value) => {
  $(id).textContent = String(value);
};
const assert = (ok, message) => {
  if (!ok) throw Error(message);
};
const error = (message) => {
  text("error", message);
  $("error").hidden = !message;
};
function envelope(v) {
  assert(
    v.coordinate_space === SPACE && v.inference_editable === false,
    "Invalid comparison coordinate boundary",
  );
  assert(
    v.comparison_identity &&
      v.sources?.a?.source_identity &&
      v.sources?.b?.source_identity,
    "Missing ordered comparison identities",
  );
  if (state.model) {
    assert(
      v.comparison_identity === state.model.comparison_identity &&
        v.sources.a.source_identity === state.model.sources.a.source_identity &&
        v.sources.b.source_identity === state.model.sources.b.source_identity,
      "Comparison source identity changed; refresh required",
    );
  }
}
async function request(path, params = {}, options = {}) {
  const url =
    "/api/comparison/" +
    path +
    "?" +
    new URLSearchParams({
      ...params,
      ...(state.model
        ? { comparison_identity: state.model.comparison_identity }
        : {}),
    });
  const r = await fetch(url, options),
    v = await r.json();
  if (!r.ok) throw Error(v.error || `Comparison request failed (${r.status})`);
  return v;
}
function settings() {
  return {
    tensor: state.pair.id,
    left: $("left").value,
    right: $("right").value,
    mapping: $("mapping").value,
  };
}
function clearView() {
  state.epoch++;
  state.inspectEpoch++;
  state.controller?.abort();
  state.inspectController?.abort();
  state.view = null;
  for (const v of Object.values(state.viewers)) v.destroy();
  state.viewers = {};
  for (const side of SIDES) {
    text(side + "-canvas", "");
    text(side + "-resolution", "No active view");
    for (const part of ["bounds", "scope", "formula", "units"])
      text(side + "-" + part, "");
  }
  for (const id of ["fit", "zoom-in", "zoom-out"]) $(id).disabled = true;
  text("inspection", "Choose a native coordinate.");
}
function rawControls() {
  const p = state.pair;
  for (const id of ["row", "col", "inspect"]) $(id).disabled = !p;
  if (p) {
    $("row").max = p.rows - 1;
    $("col").max = p.cols - 1;
    for (const id of ["row", "col"])
      $(id).value = String(
        Math.min(Number($(id).max), Math.max(0, Number($(id).value) || 0)),
      );
  }
}
function catalog() {
  const query = $("search").value.toLowerCase();
  const rows = state.model.catalog.filter((t) =>
    t.name.toLowerCase().includes(query),
  );
  const shown = rows.slice(0, 200);
  if (state.pair && !shown.some((t) => t.id === state.pair.id))
    shown.unshift(state.pair);
  $("tensor").replaceChildren(
    ...shown.map((t) => {
      const o = document.createElement("option");
      o.value = String(t.id);
      o.textContent = `${t.name} · ${t.shape.join("×")} · ${t.dtype_a}/${t.dtype_b}`;
      return o;
    }),
  );
  $("tensor").value = String(state.pair.id);
  text(
    "catalog-note",
    `${rows.length} matching tensors; showing up to 200 plus selection. Refine the search for another tensor.`,
  );
}
function drawPair() {
  const p = state.pair;
  text("name", p.name);
  text(
    "shape",
    `${p.shape.join(" × ")} native shape · ${p.dtype_a} A / ${p.dtype_b} B${p.shape.length === 1 ? " · one unwrapped vector row" : ""}`,
  );
  $("calibrate").disabled = !!p.calibration_complete;
  rawControls();
  text(
    "calibration",
    p.calibration_complete
      ? `Shared raw max ${p.scales.shared_raw_max}; separate B−A max ${p.scales.difference_max}.`
      : "Selected pair has no complete calibration. Raw originals remain inspectable.",
  );
}
async function refresh() {
  const ticket = ++state.modelEpoch;
  state.modelController?.abort();
  state.modelController = new AbortController();
  clearView();
  if (state.poll) clearTimeout(state.poll);
  state.poll = null;
  const old = state.pair?.id;
  state.model = null;
  state.pair = null;
  rawControls();
  $("tensor").disabled = true;
  $("calibrate").disabled = true;
  error("");
  try {
    const model = await request(
      "model",
      {},
      { signal: state.modelController.signal },
    );
    if (ticket !== state.modelEpoch) return;
    envelope(model);
    assert(
      model.compatibility?.complete && model.catalog?.length,
      "Complete named-shape compatibility required",
    );
    state.model = model;
    state.pair = model.catalog.find((p) => p.id === old) || model.catalog[0];
    text(
      "provenance",
      `A: ${model.sources.a.source_directory}\nIdentity A: ${model.sources.a.source_identity}\nB: ${model.sources.b.source_directory}\nIdentity B: ${model.sources.b.source_identity}\nPair identity: ${model.comparison_identity}\n${model.identity_validation}\nComplete named-shape match: ${model.compatibility.tensor_count} tensors. Mixed supported dtypes are explicitly allowed.`,
    );
    $("tensor").disabled = false;
    catalog();
    drawPair();
    if (state.pair.calibration_complete) await loadView();
    else
      text(
        "status",
        "Raw inspection ready. Calibrate the selected pair to view colors.",
      );
  } catch (e) {
    if (ticket === state.modelEpoch && e.name !== "AbortError")
      error(e.message);
  }
}
function drawLegend(side, l) {
  text(side + "-bounds", `${l.min} → ${l.max}`);
  text(side + "-scope", `${l.value_definition} · ${l.scope}`);
  text(side + "-formula", l.formula);
  text(side + "-units", `${l.units}. ${l.rounding}.`);
  $(side + "-gradient").classList.toggle(
    "unsigned",
    l.palette === "sequential-purple-v1",
  );
  $(side + "-gradient").classList.toggle("zero", l.bound === 0);
}
function tileSource(pair, quantity, mapping, identity) {
  return {
    width: pair.cols,
    height: pair.rows,
    tileSize: 256,
    tileOverlap: 0,
    minLevel: 0,
    maxLevel: pair.max_level,
    getTileUrl: (level, x, y) =>
      "/api/comparison/tile?" +
      new URLSearchParams({
        comparison_identity: identity,
        tensor: pair.id,
        quantity,
        mapping,
        level,
        x,
        y,
      }),
  };
}
function resolution(side, v, p) {
  if (!state.view || state.viewers[side] !== v) return;
  const item = v.world.getItemAt(0),
    levels = [
      ...new Set(
        (item?.lastDrawn || [])
          .map((t) => (t.tile || t).level)
          .filter(Number.isInteger),
      ),
    ];
  const factors = levels.map((l) => 2 ** (p.max_level - l));
  text(
    side + "-resolution",
    factors.length
      ? `Actual drawn pooling: ${factors.map((f) => (f === 1 ? "native scalars" : `${f} × ${f} blocks`)).join(", ")}; partial edges use actual counts.`
      : "Waiting for numeric tiles…",
  );
}
async function loadView() {
  if (!state.pair) return;
  clearView();
  drawPair();
  if (!state.pair.calibration_complete) {
    text(
      "status",
      "Calibrate the selected pair; raw inspection remains ready.",
    );
    return;
  }
  const epoch = state.epoch,
    s = settings(),
    identity = state.model.comparison_identity;
  state.controller = new AbortController();
  error("");
  try {
    const view = await request("view", s, { signal: state.controller.signal });
    if (epoch !== state.epoch) return;
    envelope(view);
    assert(
      view.pair?.id === s.tensor &&
        view.pair.name === state.pair.name &&
        JSON.stringify(view.pair.shape) === JSON.stringify(state.pair.shape),
      "Comparison pair mismatch",
    );
    for (const side of SIDES)
      assert(
        view.legends[side]?.quantity === s[side] &&
          view.legends[side]?.mapping === s.mapping,
        "Comparison legend mismatch",
      );
    state.view = view;
    for (const side of SIDES) {
      const v = OpenSeadragon({
        id: side + "-canvas",
        prefixUrl: "",
        drawer: "canvas",
        showNavigationControl: false,
        showNavigator: false,
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
      state.viewers[side] = v;
      drawLegend(side, view.legends[side]);
      v.addHandler("viewport-change", () => {
        if (epoch !== state.epoch || state.syncing) return;
        const other = state.viewers[side === "left" ? "right" : "left"];
        if (!v.world.getItemCount() || !other?.world.getItemCount()) return;
        state.syncing = true;
        try {
          other.viewport.fitBounds(v.viewport.getBounds(true), true);
        } finally {
          state.syncing = false;
        }
      });
      v.addHandler("canvas-click", (e) => {
        if (epoch !== state.epoch || !e.quick || !v.world.getItemCount())
          return;
        const point = v.world
          .getItemAt(0)
          .viewportToImageCoordinates(v.viewport.pointFromPixel(e.position));
        inspectAt(Math.floor(point.y), Math.floor(point.x));
      });
      v.addHandler("tile-drawn", () => {
        if (epoch === state.epoch) resolution(side, v, view.pair);
      });
      v.addHandler("tile-load-failed", () => {
        if (epoch === state.epoch)
          error(
            "Comparison tile failed. Refresh status or retry; no source values are substituted.",
          );
      });
      v.addHandler("open", () => {
        if (epoch !== state.epoch) return;
        if (SIDES.every((k) => state.viewers[k]?.world.getItemCount())) {
          for (const id of ["fit", "zoom-in", "zoom-out"])
            $(id).disabled = false;
          text(
            "status",
            "Native coordinates synchronized. A/B raw scale and B−A scale are separately labeled.",
          );
        }
      });
      v.open(tileSource(view.pair, s[side], s.mapping, identity));
    }
  } catch (e) {
    if (epoch === state.epoch && e.name !== "AbortError") {
      clearView();
      rawControls();
      error(e.message);
      text(
        "status",
        "Comparison view unavailable; raw inspection remains ready.",
      );
    }
  }
}
async function inspectAt(row, col) {
  const p = state.pair;
  if (!p) return;
  if (
    !Number.isInteger(row) ||
    !Number.isInteger(col) ||
    row < 0 ||
    col < 0 ||
    row >= p.rows ||
    col >= p.cols
  ) {
    error("Enter valid integer native row and column coordinates.");
    return;
  }
  const ticket = ++state.inspectEpoch,
    epoch = state.epoch;
  state.inspectController?.abort();
  state.inspectController = new AbortController();
  $("row").value = String(row);
  $("col").value = String(col);
  text("inspection", "Reading both originals…");
  try {
    const v = await request(
      "inspect",
      { tensor: p.id, row, col },
      { signal: state.inspectController.signal },
    );
    if (ticket !== state.inspectEpoch || epoch !== state.epoch) return;
    envelope(v);
    assert(
      v.pair_id === p.id && v.name === p.name && v.row === row && v.col === col,
      "Comparison address mismatch",
    );
    const box = $("inspection");
    box.replaceChildren();
    for (const side of ["a", "b"]) {
      const raw = v.originals?.[side];
      const bytes =
        raw?.dtype === "F32" ? 4 : ["BF16", "F16"].includes(raw?.dtype) ? 2 : 0;
      assert(
        raw?.original_source_value === true &&
          typeof raw.raw_exact === "string" &&
          bytes &&
          raw.element_bytes === bytes &&
          new RegExp(`^[0-9a-f]{${bytes * 2}}$`).test(raw.raw_hex_le),
        "Missing or inconsistent original scalar",
      );
      const title = document.createElement("h3"),
        body = document.createElement("pre");
      title.textContent = `Original ${side.toUpperCase()} · ${raw.dtype}`;
      body.textContent = `${raw.raw_exact}\nBytes (little endian): ${raw.raw_hex_le} · ${raw.classification}\n${raw.shard} @ ${raw.byte_offset}`;
      box.append(title, body);
    }
    assert(
      v.difference?.derived === true &&
        v.difference.original_source_value === false &&
        v.difference.direction === "B-A",
      "Invalid derived quantity boundary",
    );
    const title = document.createElement("h3"),
      body = document.createElement("pre");
    title.textContent = "Derived B − A · F64 arithmetic";
    body.textContent = `${v.difference.decimal_f64 ?? v.difference.unavailable_reason}\n${v.difference.arithmetic}\nNative index [${v.native_indices.join(", ")}] · comparison only; no inference edit target`;
    box.append(title, body);
  } catch (e) {
    if (
      ticket === state.inspectEpoch &&
      epoch === state.epoch &&
      e.name !== "AbortError"
    )
      error(e.message);
  }
}
async function calibrate() {
  if (!state.pair) return;
  const id = state.pair.id,
    identity = state.model.comparison_identity,
    epoch = state.epoch;
  $("calibrate").disabled = true;
  error("");
  try {
    await request(
      "calibrate",
      { tensor: id },
      { method: "POST", headers: { "X-Atlas-Local": "1" } },
    );
    if (epoch !== state.epoch) return;
    text("status", "Calibrating selected pair with bounded reads…");
    const poll = async () => {
      if (
        epoch !== state.epoch ||
        identity !== state.model?.comparison_identity
      )
        return;
      try {
        const m = await request("model");
        if (epoch !== state.epoch) return;
        envelope(m);
        const p = m.catalog.find((p) => p.id === id);
        if (m.progress.error) throw Error(m.progress.error);
        if (p.calibration_complete) {
          state.model = m;
          state.pair = p;
          drawPair();
          await loadView();
          return;
        }
        state.poll = setTimeout(poll, 2000);
      } catch (e) {
        if (epoch === state.epoch) {
          error(e.message);
          $("calibrate").disabled = false;
        }
      }
    };
    state.poll = setTimeout(poll, 2000);
  } catch (e) {
    if (epoch === state.epoch) {
      error(e.message);
      $("calibrate").disabled = false;
    }
  }
}
function bind() {
  $("refresh").addEventListener("click", refresh);
  $("search").addEventListener("input", () => {
    if (state.model) catalog();
  });
  $("tensor").addEventListener("change", () => {
    state.pair = state.model.catalog.find(
      (p) => p.id === Number($("tensor").value),
    );
    loadView();
  });
  for (const id of ["left", "right", "mapping"])
    $(id).addEventListener("change", loadView);
  $("calibrate").addEventListener("click", calibrate);
  $("inspect-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if ($("row").value === "" || $("col").value === "") {
      error("Enter both native coordinates.");
      return;
    }
    inspectAt(Number($("row").value), Number($("col").value));
  });
  $("fit").addEventListener("click", () =>
    state.viewers.left?.viewport.goHome(),
  );
  for (const [id, factor] of [
    ["zoom-in", 2],
    ["zoom-out", 0.5],
  ])
    $(id).addEventListener("click", () => {
      state.viewers.left?.viewport.zoomBy(factor);
      state.viewers.left?.viewport.applyConstraints();
    });
}
function initialize() {
  bind();
  refresh();
}
initialize();
