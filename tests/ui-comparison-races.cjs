"use strict";
// Actual comparison controller with tiny deterministic DOM/OSD/fetch doubles.
// No browser, socket, payload, host headroom check or production gate override.
const fs = require("fs"),
  vm = require("vm"),
  assert = require("assert/strict"),
  crypto = require("crypto"),
  path = require("path");
const source = fs.readFileSync(
  path.join(__dirname, "../web/comparison.js"),
  "utf8",
);
class Element {
  constructor(tag = "", id = "") {
    Object.assign(this, {
      tag,
      id,
      value: "",
      textContent: "",
      hidden: false,
      disabled: false,
      children: [],
      listeners: {},
      classes: new Set(),
    });
    this.classList = {
      toggle: (k, v) => (v ? this.classes.add(k) : this.classes.delete(k)),
    };
  }
  replaceChildren(...children) {
    this.children = children;
    this.textContent = "";
  }
  append(...children) {
    this.children.push(...children);
  }
  addEventListener(k, f) {
    this.listeners[k] = f;
  }
}
const elements = new Map(),
  get = (id) => {
    if (!elements.has(id)) elements.set(id, new Element("", id));
    return elements.get(id);
  };
get("left").value = "a";
get("right").value = "b";
get("mapping").value = "linear";
get("row").value = get("col").value = "0";
const pending = [],
  viewers = [],
  timers = [];
// Exact established scalar-viewer controls; asserted for every constructor, in both panels.
const expectedViewerControls = {
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
};
function OSD(options) {
  const captured = JSON.parse(JSON.stringify(options)),
    { id, prefixUrl, ...controls } = captured;
  assert(["left-canvas", "right-canvas"].includes(id));
  assert.equal(prefixUrl, "");
  assert.deepEqual(controls, expectedViewerControls);
  const v = {
    options: captured,
    handlers: {},
    destroyed: false,
    opened: false,
    bounds: { x: 0, y: 0, width: 1, height: 1 },
    fitCalls: 0,
    addHandler(k, f) {
      (this.handlers[k] ??= []).push(f);
    },
    emit(k, e = {}) {
      for (const f of this.handlers[k] || []) f(e);
    },
    open(s) {
      this.source = s;
    },
    destroy() {
      this.destroyed = true;
    },
  };
  v.item = { lastDrawn: [], viewportToImageCoordinates: (p) => p };
  v.world = {
    getItemCount: () => (!v.destroyed && v.opened ? 1 : 0),
    getItemAt: () => v.item,
  };
  v.viewport = {
    getBounds: () => ({ ...v.bounds }),
    fitBounds: (b) => {
      assert(++v.fitCalls < 100);
      v.bounds = { ...b };
      v.emit("viewport-change");
    },
    pointFromPixel: (p) => p,
    goHome: () => {
      v.bounds = { x: 0, y: 0, width: 1, height: 1 };
      v.emit("viewport-change");
    },
    zoomBy: (f) => {
      v.bounds.width /= f;
      v.bounds.height /= f;
      v.emit("viewport-change");
    },
    applyConstraints: () => {},
  };
  viewers.push(v);
  return v;
}
const context = vm.createContext({
  console,
  AbortController,
  URLSearchParams,
  OpenSeadragon: OSD,
  setTimeout: (f) => {
    timers.push(f);
    return timers.length;
  },
  clearTimeout: (i) => {
    timers[i - 1] = null;
  },
  document: { getElementById: get, createElement: (t) => new Element(t) },
  fetch: (url, options) =>
    new Promise((resolve, reject) =>
      pending.push({
        url,
        options,
        reject,
        resolve: (body, status = 200) =>
          resolve({
            ok: status >= 200 && status < 300,
            status,
            json: async () => body,
          }),
      }),
    ),
});
vm.runInContext(source.replace(/initialize\(\);\s*$/, ""), context);
const run = (s) => vm.runInContext(s, context),
  copy = (x) => JSON.parse(JSON.stringify(x)),
  tick = async () => {
    for (let i = 0; i < 12; i++) await Promise.resolve();
  };
const take = (part) => {
  const i = pending.findIndex((p) => p.url.includes(part));
  assert(i >= 0, "missing " + part);
  return pending.splice(i, 1)[0];
};
const boundary = {
  api_version: 1,
  coordinate_space: "checkpoint-comparison-v1",
  comparison_identity: "pair-identity",
  inference_editable: false,
  sources: {
    a: { source_identity: "source-a", source_directory: "/fixture-a" },
    b: { source_identity: "source-b", source_directory: "/fixture-b" },
  },
};
const catalog = [
  {
    id: 0,
    name: "weights",
    shape: [3, 5],
    rows: 3,
    cols: 5,
    count: 15,
    max_level: 3,
    dtype_a: "BF16",
    dtype_b: "F32",
    calibration_complete: true,
    scales: { shared_raw_max: 4, difference_max: 2 },
  },
  {
    id: 1,
    name: "norm",
    shape: [5],
    rows: 1,
    cols: 5,
    count: 5,
    max_level: 3,
    dtype_a: "F16",
    dtype_b: "BF16",
    calibration_complete: true,
    scales: { shared_raw_max: 1, difference_max: 0 },
  },
];
const model = {
  ...boundary,
  catalog,
  compatibility: { complete: true, tensor_count: 2 },
  identity_validation: "fingerprints, not fresh hashes",
  progress: { active: null, error: null },
};
const settings = () => copy(run("settings()"));
function view(s = settings()) {
  const p = catalog[s.tensor],
    legends = {};
  for (const side of ["left", "right"]) {
    const q = s[side],
      derived = ["delta", "abs_delta"].includes(q),
      bound = derived ? p.scales.difference_max : p.scales.shared_raw_max,
      unsigned = q === "abs_delta" || s.mapping === "magnitude";
    legends[side] = {
      quantity: q,
      mapping: s.mapping,
      bound,
      min: unsigned ? 0 : -bound,
      max: bound,
      scope: derived ? "shared derived B-A scale" : "shared original A+B scale",
      calibration_domain: derived ? "difference" : "shared_raw",
      value_definition: derived
        ? "derived B-A; not original"
        : "original source " + q,
      palette: unsigned ? "sequential-purple-v1" : "signed-blue-red",
      formula: "fixture formula",
      units: "mean of pointwise transformed quantity",
      rounding: "F64 derived arithmetic may round",
    };
  }
  return { ...boundary, pair: p, legends, tile_size: 256 };
}
function inspection(pair = 0, row = 0, col = 0) {
  return {
    ...boundary,
    pair_id: pair,
    name: catalog[pair].name,
    row,
    col,
    native_indices: catalog[pair].shape.length === 1 ? [col] : [row, col],
    originals: {
      a: {
        raw_exact: "-0.0",
        raw_hex_le: "0080",
        dtype: "BF16",
        element_bytes: 2,
        classification: "finite",
        original_source_value: true,
        shard: "a.safetensors",
        byte_offset: 100,
      },
      b: {
        raw_exact: "0.5",
        raw_hex_le: "0000003f",
        dtype: "F32",
        element_bytes: 4,
        classification: "finite",
        original_source_value: true,
        shard: "b.safetensors",
        byte_offset: 200,
      },
    },
    difference: {
      derived: true,
      original_source_value: false,
      direction: "B-A",
      decimal_f64: "0.5",
      value: 0.5,
      arithmetic: "F64 subtraction; not exact symbolic",
    },
  };
}
async function finishView(req, s = settings()) {
  const start = viewers.length;
  req.resolve(view(s));
  await tick();
  const vs = viewers.slice(start);
  assert.equal(vs.length, 2);
  for (const v of vs) {
    v.opened = true;
    v.emit("open");
  }
  return vs;
}
async function activate() {
  const task = run("loadView()");
  const vs = await finishView(take("/view"));
  await task;
  return vs;
}
const passed = [];
(async () => {
  run("bind()");
  const init = run("refresh()");
  take("/model").resolve(model);
  await tick();
  await finishView(take("/view"));
  await init;
  passed.push(
    "Both panels preserve exact established 32-tile cache, one-loader limit, canvas/no-smoothing, render/zoom and gesture controls on every construction",
  );
  assert.deepEqual(settings(), {
    tensor: 0,
    left: "a",
    right: "b",
    mapping: "linear",
  });
  assert.equal(get("left-bounds").textContent, get("right-bounds").textContent);
  assert(
    get("provenance").textContent.includes("source-a") &&
      get("provenance").textContent.includes("source-b"),
  );
  assert.equal(get("inspect").disabled, false);
  passed.push(
    "Ordered source identities and shared raw A/B scale appear with comparison-only coordinate boundary",
  );
  for (const side of ["left", "right"])
    for (const q of ["a", "b", "delta", "abs_delta"]) {
      get(side).value = q;
      await activate();
      const url = run(`state.viewers.${side}.source.getTileUrl(0,0,0)`);
      assert(url.startsWith("/api/comparison/tile?"));
      assert(
        url.includes("quantity=" + q) &&
          url.includes("comparison_identity=pair-identity"),
      );
      assert.equal(
        get(side + "-gradient").classes.has("unsigned"),
        q === "abs_delta",
      );
      if (q === "delta")
        assert(get(side + "-scope").textContent.includes("derived"));
    }
  passed.push(
    "Both panels select every quantity with ordered-identity-bound comparison tile URLs and distinct raw/difference legends",
  );
  get("mapping").value = "magnitude";
  await activate();
  assert(get("left-gradient").classes.has("unsigned"));
  get("mapping").value = "asinh";
  await activate();
  assert(
    run("state.viewers.left.source.getTileUrl(0,0,0)").includes(
      "mapping=asinh",
    ),
  );
  passed.push(
    "Common magnitude/asinh mapping survives API and tile selection without substituting a rule",
  );
  const p1 = run("inspectAt(0,0)"),
    r1 = take("/inspect"),
    p2 = run("inspectAt(1,1)"),
    r2 = take("/inspect");
  r2.resolve(inspection(0, 1, 1));
  await p2;
  const nodes = get("inspection").children;
  assert.equal(nodes[0].textContent, "Original A · BF16");
  assert(nodes[1].textContent.startsWith("-0.0"));
  assert.equal(nodes[4].textContent, "Derived B − A · F64 arithmetic");
  assert(nodes[5].textContent.includes("no inference edit target"));
  r1.resolve({
    ...inspection(),
    difference: { ...inspection().difference, decimal_f64: "stale" },
  });
  await p1;
  assert(!get("inspection").children.at(-1).textContent.includes("stale"));
  passed.push(
    "Inspector preserves signed original, dtype-sized bytes and labeled arithmetic difference; late address response ignored",
  );
  const before = pending.length;
  await run("inspectAt(-1,0)");
  await run("inspectAt(3,0)");
  await run("inspectAt(0,0.5)");
  assert.equal(pending.length, before);
  get("row").value = "";
  get("inspect-form").listeners.submit({ preventDefault() {} });
  assert.equal(pending.length, before);
  passed.push(
    "Invalid or empty native coordinates do not request source reads",
  );
  const old = run("loadView()"),
    oldReq = take("/view"),
    oldSettings = settings();
  get("left").value = "a";
  const current = run("loadView()");
  await finishView(take("/view"));
  oldReq.resolve(view(oldSettings));
  await Promise.all([old, current]);
  assert.equal(run("state.view.legends.left.quantity"), "a");
  passed.push(
    "Out-of-order view response cannot replace newer quantity selection",
  );
  const stale = run("state.viewers.left"),
    oldRead = run("inspectAt(0,0)"),
    readReq = take("/inspect");
  get("right").value = "b";
  await activate();
  readReq.resolve(inspection());
  await oldRead;
  assert.equal(get("inspection").textContent, "Choose a native coordinate.");
  get("error").textContent = "";
  stale.emit("tile-load-failed");
  assert.equal(get("error").textContent, "");
  run("state.viewers.left.emit('tile-load-failed')");
  assert(get("error").textContent.includes("tile failed"));
  passed.push(
    "Rule changes invalidate inspections and obsolete viewer errors while preserving active failures",
  );
  const left = run("state.viewers.left"),
    right = run("state.viewers.right");
  left.bounds = { x: 0.1, y: 0.2, width: 0.3, height: 0.4 };
  left.emit("viewport-change");
  assert.deepEqual(left.bounds, right.bounds);
  right.bounds = { x: 0.2, y: 0.3, width: 0.4, height: 0.5 };
  right.emit("viewport-change");
  assert.deepEqual(left.bounds, right.bounds);
  left.item.lastDrawn = [{ tile: { level: 2 } }];
  left.emit("tile-drawn");
  assert(get("left-resolution").textContent.includes("2 × 2"));
  passed.push(
    "Native geometry synchronizes in both directions; actual numeric pooling and partial-edge policy are visible",
  );
  get("tensor").value = "1";
  get("tensor").listeners.change();
  await finishView(take("/view"));
  await tick();
  assert.equal(run("state.viewers.left.source.height"), 1);
  get("right").value = "delta";
  await activate();
  assert(get("right-gradient").classes.has("zero"));
  assert(get("shape").textContent.includes("unwrapped vector"));
  passed.push(
    "Vectors remain one native row and zero-difference scale displays a uniform zero legend",
  );
  const bad = run("loadView()");
  take("/view").resolve({ ...view(), comparison_identity: "wrong" });
  await bad;
  assert.equal(run("state.view"), null);
  assert(get("error").textContent.includes("identity changed"));
  assert.equal(get("inspect").disabled, false);
  passed.push(
    "Mismatched comparison identity refuses colors while leaving original inspection available",
  );
  const wrongLegend = run("loadView()");
  const wl = view();
  wl.legends.left.quantity = "wrong";
  take("/view").resolve(wl);
  await wrongLegend;
  assert.equal(run("state.view"), null);
  assert(get("error").textContent.includes("legend mismatch"));
  passed.push("Wrong quantity legend cannot activate either panel");
  const rawBad = run("inspectAt(0,0)");
  const badRaw = inspection(1);
  badRaw.originals.a.raw_hex_le = "00";
  take("/inspect").resolve(badRaw);
  await rawBad;
  assert(get("error").textContent.includes("inconsistent original"));
  passed.push("Raw inspector refuses dtype/byte-width inconsistency");
  const ref = run("refresh()");
  const unready = {
    ...model,
    catalog: model.catalog.map((t) => ({
      ...t,
      calibration_complete: false,
      scales: null,
    })),
  };
  take("/model").resolve(unready);
  await ref;
  assert.equal(run("state.view"), null);
  assert.equal(get("inspect").disabled, false);
  assert.equal(pending.length, 0);
  const ir = run("inspectAt(0,0)");
  const special = inspection(1);
  special.originals.b = {
    ...special.originals.b,
    raw_exact: "Infinity",
    classification: "infinity",
  };
  special.difference = {
    ...special.difference,
    decimal_f64: null,
    value: null,
    unavailable_reason: "Nonfinite original; difference unavailable",
  };
  take("/inspect").resolve(special);
  await ir;
  assert(get("inspection").children.at(-1).textContent.includes("Nonfinite"));
  passed.push(
    "Uncalibrated pairs make no automatic scan or tile request; nonfinite originals remain inspectable with unavailable difference",
  );
  const cal = run("calibrate()"),
    calReq = take("/calibrate");
  assert.equal(calReq.options.method, "POST");
  assert.equal(calReq.options.headers["X-Atlas-Local"], "1");
  assert(calReq.url.includes("tensor=1") && !calReq.url.includes("all="));
  calReq.resolve({ ...boundary, queued: 1 }, 202);
  await cal;
  const poll = timers.filter(Boolean).at(-1);
  poll();
  take("/model").resolve(model);
  await tick();
  await finishView(take("/view"));
  await tick();
  assert.equal(run("state.view.pair.id"), 1);
  passed.push(
    "Explicit calibration queues only the selected pair; completed metadata activates its views",
  );
  const untrusted = run("loadView()");
  take("/view").resolve({ ...view(), inference_editable: true });
  await untrusted;
  assert.equal(run("state.view"), null);
  assert(get("error").textContent.includes("coordinate boundary"));
  passed.push(
    "Any inference-editable response is rejected at the comparison boundary",
  );
  const html = fs.readFileSync(
    path.join(__dirname, "../web/comparison.html"),
    "utf8",
  );
  assert(!html.includes("inference.js"));
  assert(!/["']\/api\/inference/.test(source));
  assert(
    /@media\s*\(\s*max-width\s*:\s*720px\s*\)/.test(
      fs.readFileSync(path.join(__dirname, "../web/comparison.css"), "utf8"),
    ),
  );
  passed.push(
    "Dedicated page has no inference controller or route; mobile panels stack with wrapping source metadata (static check)",
  );
  console.log(
    JSON.stringify(
      {
        status: "PASS",
        checks: passed.length,
        source_sha256: crypto.createHash("sha256").update(source).digest("hex"),
        viewer_controls: expectedViewerControls,
        constructors_checked: viewers.length,
        scope:
          "Actual comparison frontend plus deterministic DOM/transport/OSD doubles; no browser rendering or backend qualification",
        passed,
      },
      null,
      2,
    ),
  );
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
