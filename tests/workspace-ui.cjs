"use strict";
require("./support/comparison-controller.cjs").enableModules(__filename);
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    // Real workspace adapter + app with deterministic DOM/OSD/transport doubles.
    // Aborted responses are deliberately delivered; this is not browser rendering QA.
    const fs = require("fs"),
      vm = require("vm"),
      assert = require("assert/strict"),
      crypto = require("crypto"),
      path = require("path");
    const source = fs.readFileSync(
      path.join(__dirname, "../web/app.js"),
      "utf8",
    );
    class Element {
      constructor(tag = "", id = "") {
        this.tag = tag;
        this.id = id;
        this.value = "";
        this.textContent = "";
        this.hidden = false;
        this.disabled = false;
        this.children = [];
        this.dataset = {};
        this.attrs = {};
        this.listeners = {};
        this.classes = new Set();
        this.classList = {
          add: (x) => this.classes.add(x),
          remove: (x) => this.classes.delete(x),
          toggle: (x, v) => (v ? this.classes.add(x) : this.classes.delete(x)),
          contains: (x) => this.classes.has(x),
        };
      }
      get options() {
        return this.children;
      }
      append(...children) {
        this.children.push(...children);
      }
      replaceChildren(...children) {
        this.children = children;
        this.textContent = "";
        if (this.tag === "select") this.value = children[0]?.value || "";
      }
      setAttribute(k, v) {
        this.attrs[k] = v;
      }
      addEventListener(k, v) {
        this.listeners[k] = v;
      }
      click() {
        return this.listeners.click?.();
      }
      focus() {}
      remove() {}
    }
    const elements = new Map(),
      get = (id) => {
        if (!elements.has(id))
          elements.set(
            id,
            new Element(
              ["left-rule", "right-rule", "layer-filter"].includes(id)
                ? "select"
                : "div",
              id,
            ),
          );
        return elements.get(id);
      };
    const pending = [],
      viewers = [],
      frames = [],
      timers = [];
    function OSD() {
      const v = {
        handlers: {},
        destroyed: false,
        overlays: [],
        nav: false,
        bounds: { x: 0, y: 0, width: 1, height: 1 },
        zoom: 1,
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
        setMouseNavEnabled(v) {
          this.nav = v;
        },
        clearOverlays() {
          this.overlays = [];
        },
        addOverlay(v) {
          this.overlays.push(v);
        },
      };
      v.item = {
        lastDrawn: [],
        getFullyLoaded: () => true,
        viewportToImageCoordinates: (x, y) =>
          typeof x === "number" ? { x, y } : x,
        imageToViewportRectangle: (x, y, width, height) => ({
          x,
          y,
          width,
          height,
        }),
        imageToViewportCoordinates: (x, y) => ({ x, y }),
      };
      v.world = {
        getItemCount: () => (v.destroyed ? 0 : 1),
        getItemAt: () => v.item,
      };
      v.viewport = {
        getBounds: () => ({ ...v.bounds }),
        fitBounds: (b) => {
          v.fitCalls++;
          assert(v.fitCalls < 100, "viewport sync feedback loop");
          v.bounds = { ...b };
          v.emit("viewport-change");
        },
        goHome: () => {
          v.bounds = { x: 0, y: 0, width: 1, height: 1 };
          v.emit("viewport-change");
        },
        pointFromPixel: (p) => p,
        getZoom: () => v.zoom,
        viewportToImageZoom: (z) => z,
        imageToViewportZoom: (z) => z,
        zoomBy: (f) => {
          v.zoom *= f;
          v.bounds.width /= f;
          v.bounds.height /= f;
          v.emit("viewport-change");
        },
        zoomTo: (z) => {
          v.zoom = z;
        },
        panTo: (p) => {
          v.bounds.x = p.x;
          v.bounds.y = p.y;
          v.emit("viewport-change");
        },
        applyConstraints: () => {},
      };
      viewers.push(v);
      return v;
    }
    const location = { pathname: "/", search: "", hash: "" },
      historyEntries = ["/"],
      events = {},
      storage = new Map(),
      downloads = [];
    const updateLocation = (url) => {
      location.hash = url.includes("#") ? url.slice(url.indexOf("#")) : "";
      location.search = "";
    };
    const history = {
      pushState: (_, __, url) => {
        historyEntries.push(url);
        updateLocation(url);
      },
      replaceState: (_, __, url) => {
        historyEntries[historyEntries.length - 1] = url;
        updateLocation(url);
      },
    };
    const localStorage = {
      getItem: (k) => storage.get(k) || null,
      setItem: (k, v) => storage.set(k, v),
    };
    const tensorEvents = [];
    class CustomEvent {
      constructor(type, options) {
        this.type = type;
        this.detail = options.detail;
      }
    }
    class DownloadURL extends URL {
      static createObjectURL(blob) {
        downloads.push(blob);
        return "blob:fixture";
      }
      static revokeObjectURL() {}
    }
    const context = vm.createContext({
      console,
      CustomEvent,
      location,
      history,
      localStorage,
      Blob,
      TextEncoder,
      URL: DownloadURL,
      crypto,
      clearTimeout: (id) => {
        timers[id - 1] = null;
      },
      AbortController,
      DOMException,
      URLSearchParams,
      performance,
      setTimeout: (f, ms) => timers.push({ f, ms }),
      OpenSeadragon: OSD,
      requestAnimationFrame: (f) => frames.push(f),
      window: {
        innerWidth: 390,
        CustomEvent,
        dispatchEvent: (e) => {
          tensorEvents.push(e.detail);
          events[e.type]?.(e);
        },
        addEventListener: (k, f) => {
          events[k] = f;
        },
      },
      document: {
        body: new Element("body"),
        title: "",
        getElementById: get,
        createElement: (t) => new Element(t),
        createDocumentFragment: () => new Element("fragment"),
      },
      fetch: (url, options) =>
        new Promise((resolve, reject) =>
          pending.push({
            url,
            options,
            resolve: (body, status = 200) =>
              resolve({
                ok: status >= 200 && status < 300,
                status,
                json: async () => body,
              }),
            reject,
          }),
        ),
    });
    await require("./support/comparison-controller.cjs").load(
      fs.readFileSync(path.join(__dirname, "../web/atlas-tools.js"), "utf8"),
      context,
      path.join(__dirname, "../web/atlas-tools.js"),
    );
    await require("./support/comparison-controller.cjs").load(
      source.replace(/initialize\(\);\s*$/, ""),
      context,
      path.join(__dirname, "../web/app.js"),
    );
    await require("./support/comparison-controller.cjs").load(
      fs.readFileSync(
        path.join(__dirname, "../web/workspace-tools.js"),
        "utf8",
      ),
      context,
      path.join(__dirname, "../web/workspace-tools.js"),
    );
    for (const id of ["region-r0", "region-r1", "region-c0", "region-c1"])
      get(id).value = "0";
    get("note-kind").value = "region";
    const run = (s) => vm.runInContext(s, context),
      copy = (x) => JSON.parse(JSON.stringify(x)),
      tick = async () => {
        for (let i = 0; i < 30; i++) await Promise.resolve();
      };
    const take = (part) => {
      const i = pending.findIndex((p) => p.url.includes(part));
      assert(i >= 0, "missing " + part);
      return pending.splice(i, 1)[0];
    };
    const tensor = (id, name, shape, max_abs) => ({
      id,
      name,
      dtype: "BF16",
      element_bytes: 2,
      shape,
      calibration_complete: true,
      rows: shape.length === 1 ? 1 : shape[0],
      cols: shape.at(-1),
      count: shape.reduce((a, b) => a * b, 1),
      max_abs,
      max_level: Math.ceil(Math.log2(Math.max(...shape))),
      min_level: 0,
    });
    const catalog = [
      tensor(11, "model.layers.0.self_attn.q_proj.weight", [4, 4], 0.5),
      tensor(12, "model.layers.1.self_attn.q_proj.weight", [2, 2], 1),
      tensor(13, "model.layers.0.self_attn.k_norm.weight", [128], 34),
    ];
    const model = {
      model_identity: "d".repeat(64),
      source_identity: "a".repeat(64),
      api_version: 1,
      name: "Qwen3-8B test double",
      revision: "fixture",
      representation: "BF16",
      source_directory: "/fixture",
      source_bytes: 296,
      parameter_count: 148,
      global_max: 34,
      calibration_complete: true,
      catalog,
      rules: [
        "global_linear",
        "global_asinh",
        "tensor_linear",
        "tensor_asinh",
        "tensor_magnitude",
        "tensor_robust99",
        "tensor_signed_percentile",
      ].map((id) => ({ id, title: id, formula: "fixture formula" })),
      identity_validation: "Fixture identity",
      coverage: {
        active_tensor: null,
        all_requested: false,
        source_complete: true,
        sha_verified_shards: 3,
        sha_hashed_shards: 5,
        sha_expected_matched_shards: 3,
        sha_missing_expected_shards: 2,
        statistics_complete: true,
        values_streamed: 148,
        materialized_tiles: 2,
        materialized_bytes: 100,
        all_pixels_materialized: false,
        rendering_policy: "Fixture only",
      },
      render_semantics: "Fixture only",
    };
    function statusUpdate(t) {
      return {
        api_version: 1,
        model_status_version: 1,
        source_identity: model.source_identity,
        model_identity: model.model_identity,
        global_max: 34,
        calibration_complete: true,
        coverage: {
          calibrated_tensors: 3,
          values_streamed: 148,
          all_requested: false,
          active_tensor: null,
        },
        tensor_status: {
          id: t.id,
          calibration_complete: true,
          max_abs: t.max_abs,
        },
      };
    }
    const currentSettings = () => copy(run("settings()"));
    function view(s) {
      const t = catalog.find((t) => t.id === s.tensor),
        legends = {};
      for (const side of ["left", "right"]) {
        const max = s[side].startsWith("global") ? 34 : t.max_abs;
        legends[side] = {
          id: s[side],
          title: s[side],
          min: -max,
          max,
          zero: 0,
          s: s[side].endsWith("asinh") ? 0.01 * max : null,
          formula: "fixture formula",
          scope: s[side].split("_")[0],
          units: "raw weight",
        };
      }
      return {
        source_binding: copy(
          run("AtlasTools.sourceBinding(state.model,state.tensor)"),
        ),
        api_version: 1,
        tensor: t,
        legends,
        tile_size: 256,
        overlap: 0,
        source_values_unchanged: true,
      };
    }
    function inspect(t, row, col, raw) {
      return {
        source_binding: copy(
          run("AtlasTools.sourceBinding(state.model,state.tensor)"),
        ),
        api_version: 1,
        tensor: t.id,
        row,
        col,
        raw_exact: raw,
        bf16_hex_le: "803f",
        shard: "fixture.safetensors",
        byte_offset: 8 + 2 * (row * t.cols + col),
        native_indices: t.shape.length === 1 ? [col] : [row, col],
        transformed: { left: 0, right: 0 },
      };
    }
    async function complete(req, s = currentSettings()) {
      const before = viewers.length;
      req.resolve(view(s));
      await tick();
      const pair = viewers.slice(before);
      assert.equal(pair.length, 2);
      for (const v of pair) v.emit("open");
      await tick();
      const restore = pending.findIndex(
        (p) =>
          p.url.includes("/api/inspect") &&
          p.options.signal === run("state.inspectController?.signal"),
      );
      if (restore >= 0) {
        const req = pending.splice(restore, 1)[0],
          data = copy(run("state.inspectionData"));
        req.resolve({ ...data, transformed: { left: 0, right: 0 } });
        await tick();
      }
      return pair;
    }
    async function activate() {
      const p = run("loadView()");
      await complete(take("/api/view"));
      await p;
    }
    const checks = [];
    const click = (id) => get(id).listeners.click();
    const fire = async (ms, after = 0) => {
      const i = timers.findIndex((t, index) => index >= after && t?.ms === ms);
      assert(i >= 0, "timer " + ms);
      const t = timers[i];
      timers[i] = null;
      t.f();
      await tick();
    };
    const loaded = async () => {
      await tick();
      take("/api/model").resolve(model);
      await tick();
      await complete(take("/api/view"));
      await tick();
    };
    require("./support/async-completion.cjs").requireCompletion(
      (async () => {
        run("bind()");
        const init = run("refreshModel()");
        await tick();
        assert.equal(tensorEvents.at(-1), null);
        await loaded();
        await init;
        assert.equal(tensorEvents.at(-1).id, 11);
        assert.equal(run("window.atlasAnalyticsBridge.selected().id"), 11);
        checks.push(
          "analytics source invalidation and selected-tensor bridge coexist with QoL initialization",
        );
        assert.equal(get("tensor-browser").open, false);
        assert.equal(get("note-controls").disabled, false);
        assert.equal(get("region-controls").disabled, false);
        checks.push(
          "390px mobile initial state and region/note controls initialized (DOM double only)",
        );
        get("note-text").value = "local-only secret";
        click("note-save");
        click("note-save");
        assert.equal(storage.size, 1);
        assert.equal(JSON.parse([...storage.values()][0]).length, 1);
        assert.equal(get("note-list").children.length, 1);
        assert(!location.hash.includes("secret"));
        assert.equal(pending.length, 0);
        checks.push(
          "repeated Save updates one private note without a request or URL leak",
        );
        const refresh = run("refreshModel()");
        await tick();
        assert.equal(tensorEvents.at(-1), null);
        assert.equal(run("window.atlasAnalyticsBridge.selected()"), null);
        await loaded();
        await refresh;
        assert.equal(get("note-list").children.length, 1);
        assert(
          get("note-list").children[0].children[0].textContent.includes(
            "local-only",
          ),
        );
        checks.push("same-identity harmless model reload preserves notes");
        click("bookmark-save");
        assert.equal(historyEntries.length, 2);
        click("bookmark-save");
        assert.equal(historyEntries.length, 2);
        get("region-r0").value = "1";
        get("region-r1").value = "2";
        get("region-c0").value = "1";
        get("region-c1").value = "3";
        get("region-r0").listeners.input();
        await fire(350);
        assert.equal(historyEntries.length, 2);
        const savedHash = location.hash;
        assert(run("AtlasTools.parseBookmark(location.hash).region[0]") === 1);
        checks.push(
          "bookmark button makes one entry; repeated clicks and debounced edits do not spam history",
        );
        run("selectTensor(13)");
        await tick();
        take("/api/tensor-status").resolve(statusUpdate(catalog[2]));
        await tick();
        await complete(take("/api/view"));
        await fire(350);
        assert.equal(get("note-list").children.length, 0);
        const vectorHash = location.hash;
        assert.notEqual(vectorHash, savedHash);
        const anchorRequests = pending.length;
        updateLocation("/#workspace");
        events.hashchange();
        assert.equal(pending.length, anchorRequests);
        assert(!get("bookmark-status").textContent.includes("Invalid"));
        checks.push(
          "Ordinary section anchors never parse as source bookmarks or reload the view",
        );
        updateLocation("/" + savedHash);
        events.popstate();
        await loaded();
        assert.equal(run("state.tensor.id"), 11);
        assert.equal(get("region-r0").value, "1");
        assert.equal(get("note-list").children.length, 1);
        const requestsBefore = pending.length;
        events.hashchange();
        assert.equal(pending.length, requestsBefore);
        checks.push(
          "back navigation restores tensor/region/notes; duplicate hashchange is ignored",
        );
        updateLocation("/" + vectorHash);
        events.popstate();
        await loaded();
        assert.equal(run("state.tensor.id"), 13);
        assert.equal(get("note-list").children.length, 0);
        checks.push("forward navigation restores the vector scope");
        click("bookmark-reset");
        await loaded();
        assert.equal(location.hash, "");
        assert.equal(run("state.tensor.id"), 11);
        assert.equal(get("region-r0").value, "0");
        const resetEntries = historyEntries.length;
        click("bookmark-reset");
        await loaded();
        assert.equal(historyEntries.length, resetEntries);
        checks.push(
          "reset clears link, cancels prior context and restores default tensor/region without repeated-click history spam",
        );
        const first = run("refreshModel()");
        await tick();
        take("/api/model").resolve(
          { code: "backend_unavailable", error: "offline" },
          503,
        );
        await tick();
        assert(get("read-retry").textContent.includes("Retry 1/2"));
        const replacement = run("refreshModel()");
        await tick();
        await loaded();
        await replacement;
        await first;
        assert.equal(pending.length, 0);
        assert(!get("read-retry").textContent.includes("Retry"));
        checks.push(
          "refresh aborts old retry wait; only newest model/view can activate",
        );
        const tile = run("loadView()");
        const req = take("/api/view"),
          before = viewers.length;
        req.resolve(view(currentSettings()));
        await tick();
        const pair = viewers.slice(before);
        pair[0].emit("tile-load-failed");
        pair.forEach((v) => v.emit("open"));
        await tile;
        assert(get("status").textContent.includes("Tile request failed"));
        checks.push(
          "tile failure during open cannot be overwritten by active-view status",
        );
        const exp = click("region-export");
        const metadata = take("/api/model");
        await click("region-export");
        assert.equal(pending.length, 0);
        get("left-rule").value = "tensor_linear";
        const nav = run("loadView()");
        metadata.resolve(model);
        await exp;
        assert.equal(downloads.length, 0);
        assert.equal(
          pending.filter((p) => p.url.includes("/api/inspect")).length,
          0,
        );
        await complete(take("/api/view"));
        await nav;
        checks.push(
          "repeated export click launches once; rule change cancels before stale metadata can read cells or download",
        );
        const success = click("region-export");
        take("/api/model").resolve(model);
        await tick();
        take("/api/inspect").resolve(inspect(catalog[0], 0, 0, "1"));
        await tick();
        take("/api/model").resolve(model);
        await success;
        assert.equal(downloads.length, 1);
        assert((await downloads[0].text()).includes("raw_exact"));
        checks.push(
          "bounded original cell export completes through Blob download",
        );
        for (const action of [
          () => {
            run("state.selected=[0,0]");
            click("region-use-cell");
          },
          () => get("note-list").children[0].children[1].children[0].click(),
          () => get("note-list").children[0].children[1].children[1].click(),
        ]) {
          const work = click("region-export"),
            oldMetadata = take("/api/model"),
            downloadCount = downloads.length;
          action();
          oldMetadata.resolve(model);
          await work;
          assert.equal(downloads.length, downloadCount);
          assert.equal(pending.length, 0);
          assert(
            get("region-status").textContent.includes("cancelled") ||
              get("region-status").textContent.includes("selected"),
          );
        }
        checks.push(
          "Use inspected cell, Show pin and note Edit all cancel a pending export before stale callbacks can download",
        );
        const readOne = run(
          "AtlasTools.readJSON('/api/model',{onState:e=>atlasWorkspace.retry(e,'/api/model',undefined,'parallel')})",
        );
        take("/api/model").resolve(
          { code: "backend_unavailable", error: "offline" },
          503,
        );
        await tick();
        const readTwo = run(
          "AtlasTools.readJSON('/api/model',{onState:e=>atlasWorkspace.retry(e,'/api/model',undefined,'parallel')})",
        );
        take("/api/model").resolve({ api_version: 1 });
        await readTwo;
        assert(get("read-retry").textContent.includes("Retry 1/2"));
        await fire(250);
        take("/api/model").resolve({ api_version: 1 });
        await readOne;
        assert.equal(get("read-retry").textContent, "");
        checks.push(
          "concurrent success for the same URL cannot clear another request retry status",
        );
        const manualFail = run(
          "AtlasTools.readJSON('/api/model',{onState:e=>atlasWorkspace.retry(e,'/api/model',undefined,'export')})",
        );
        const failedPromise = assert.rejects(manualFail);
        take("/api/model").reject(new Error("offline"));
        await failedPromise;
        assert(
          get("read-retry").textContent.includes("Automatic retry stopped"),
        );
        const manualRead = run(
          "AtlasTools.readJSON('/api/model',{onState:e=>atlasWorkspace.retry(e,'/api/model',undefined,'export')})",
        );
        take("/api/model").resolve({ api_version: 1 });
        await manualRead;
        assert.equal(get("read-retry").textContent, "");
        checks.push(
          "manual retry retires only terminal failures owned by that operation",
        );
        get("note-list").children[0].children[1].children[1].click();
        get("note-text").value = "edited";
        click("note-save");
        assert(
          get("note-list").children[0].children[0].textContent.includes(
            "edited",
          ),
        );
        get("note-list").children[0].children[1].children[2].click();
        assert.equal(get("note-list").children.length, 0);
        checks.push(
          "visible edit and delete controls update only scoped local notes",
        );
        const pollTimerStart = timers.length;
        const poll = run("pollStatus()");
        await fire(1000, pollTimerStart);
        const stalePoll = take("/api/progress");
        run("selectTensor(13)");
        await tick();
        take("/api/tensor-status").resolve(statusUpdate(catalog[2]));
        await tick();
        stalePoll.resolve({ ...statusUpdate(catalog[0]), global_max: 999 });
        await poll;
        await complete(take("/api/view"));
        assert.equal(run("state.model.name"), model.name);
        assert.equal(run("state.model.global_max"), 34);
        assert.equal(run("state.tensor.id"), 13);
        checks.push(
          "navigation aborts polling and ignores stale model responses",
        );
        const unavailable = run("refreshModel()");
        await tick();
        take("/api/model").resolve(
          { code: "backend_unavailable", error: "still offline" },
          503,
        );
        await tick();
        await fire(250);
        take("/api/model").resolve(
          { code: "backend_unavailable", error: "still offline" },
          503,
        );
        await tick();
        await fire(750);
        take("/api/model").resolve(
          { code: "backend_unavailable", error: "still offline" },
          503,
        );
        await unavailable;
        assert(
          get("read-retry").textContent.includes("Automatic retry stopped"),
        );
        assert(get("status").textContent.includes("unavailable"));
        assert.equal(pending.length, 0);
        const manual = run("refreshModel()");
        await tick();
        await loaded();
        await manual;
        assert.equal(get("read-retry").textContent, "");
        checks.push(
          "retry exhaustion stays unavailable and explicit refresh recovers",
        );
        click("note-new");
        get("note-text").value = "only revision A";
        click("note-save");
        assert.equal(get("note-list").children.length, 1);
        const oldNotes = [...storage.entries()];
        const oldExport = click("region-export"),
          oldExportMetadata = take("/api/model"),
          downloadCount = downloads.length;
        const revisionPoll = run("refreshModel()");
        await tick();
        assert.equal(run("state.model"), null);
        assert.equal(get("note-controls").disabled, true);
        oldExportMetadata.resolve(model);
        await oldExport;
        assert.equal(downloads.length, downloadCount);
        take("/api/model").resolve({
          ...model,
          revision: "fixture-B",
          model_identity: "e".repeat(64),
        });
        await tick();
        await complete(take("/api/view"));
        await revisionPoll;
        assert.equal(run("state.model.revision"), "fixture-B");
        assert.equal(get("note-list").children.length, 0);
        assert(get("note-scope").textContent.includes("fixture-B"));
        assert.deepEqual([...storage.entries()], oldNotes);
        click("note-new");
        get("note-text").value = "only revision B";
        click("note-save");
        assert.equal(get("note-list").children.length, 1);
        assert.equal(storage.size, oldNotes.length + 1);
        const oldRevision = run("refreshModel()");
        await tick();
        await loaded();
        await oldRevision;
        assert(
          get("note-list").children[0].children[0].textContent.includes(
            "only revision A",
          ),
        );
        checks.push(
          "explicit revision refresh invalidates old loads/exports and notes before adopting equal-geometry metadata; old notes survive separately",
        );
        const snapshot = [...storage.entries()];
        run("state.model={...state.model,revision:'unexpected-stale-context'}");
        get("note-text").value = "must not attach";
        click("note-save");
        click("note-export");
        assert.deepEqual([...storage.entries()], snapshot);
        assert.equal(downloads.length, downloadCount);
        assert(get("note-status").textContent.includes("Model/tensor changed"));
        run("state.model={...state.model,revision:'fixture'}");
        checks.push(
          "note writes and exports independently refuse stale revision context",
        );
        updateLocation("/#wa=1&prompt=must-not-apply");
        events.popstate();
        await loaded();
        assert(get("bookmark-status").textContent.includes("Invalid"));
        assert.equal(pending.length, 0);
        checks.push(
          "invalid URL state ignored without unsafe coordinate reads",
        );
        run("state.tensor={...state.tensor,dtype:'F32',element_bytes:4}");
        get("right-rule").value = "tensor_signed_percentile";
        assert.equal(run("refuseUnsupportedView()"), true);
        assert.equal(get("inspect-submit").disabled, false);
        assert.equal(get("note-controls").disabled, false);
        assert.equal(get("region-controls").disabled, false);
        click("note-new");
        get("note-text").value = "F32 native note";
        click("note-save");
        assert(get("note-status").textContent.includes("Saved"));
        const f32export = click("region-export");
        take("/api/model").resolve(model);
        await f32export;
        assert.equal(pending.length, 0);
        assert(get("region-status").textContent.includes("Source changed"));
        checks.push(
          "F32 native tools remain available; a dtype switch not yet bound to current server metadata refuses before reading cells",
        );
        assert.equal(pending.length, 0);
        console.log(
          JSON.stringify(
            { status: "PASS", checks: checks.length, passed: checks },
            null,
            2,
          ),
        );
      })().catch((e) => {
        console.error(e);
        process.exitCode = 1;
      }),
    );
  })(),
);
