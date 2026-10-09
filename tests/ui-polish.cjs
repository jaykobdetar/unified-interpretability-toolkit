"use strict";
require("./support/comparison-controller.cjs").enableModules(__filename);
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    // Deterministic DOM/transport tests. No browser, server, model execution or sockets.
    const assert = require("node:assert/strict"),
      fs = require("node:fs"),
      vm = require("node:vm");
    const A = require("../web/atlas-tools.js"),
      evidence = require("./fixtures/ui-polish-observations.json");
    const catalog = evidence.records.map((r, id) => ({
      id,
      name: r.name,
      dtype: r.dtype,
      element_bytes: 2,
      shape: r.shape,
      rows: r.shape.length === 1 ? 1 : r.shape[0],
      cols: r.shape.at(-1),
      count: r.shape.reduce((a, b) => a * b, 1),
      max_level: Math.ceil(Math.log2(Math.max(...r.shape))),
      min_level: 0,
    }));
    const model = {
      source_identity: evidence.source_identity,
      revision: evidence.revision,
      model_identity: "a".repeat(64),
      catalog,
    };
    model.head_layout = require("./fixtures/smollm2-head-layout-v1.json");
    model.head_layout_binding = {
      source_identity: model.source_identity,
      model_identity: model.model_identity,
      weights_sha256: model.head_layout.source_model.weights_sha256,
      config_sha256: model.head_layout.source_model.config_sha256,
    };
    const examples = A.guidedExamples(model);
    assert.equal(examples.length, 3);
    for (const [i, e] of examples.entries()) {
      const b = A.resolveBookmark(A.parseBookmark(e.href), model);
      assert.deepEqual(b.region, evidence.records[i].region);
      const r = evidence.records[i];
      assert.equal(Math.min(...r.values), r.min);
      assert.equal(Math.max(...r.values), r.max);
      assert.equal(r.values.filter((v) => v > 0).length, r.positive);
      assert.equal(r.values.filter((v) => v < 0).length, r.negative);
    }
    assert.equal(
      A.guidedExamples({ ...model, source_identity: "b".repeat(64) }).length,
      0,
    );
    assert.equal(A.guidedExamples({ ...model, revision: "other" }).length, 0);
    assert.match(
      A.hoverHead(model, catalog[0], 64, 5),
      /Query head 1 · row offset 0/,
    );
    assert.match(
      A.hoverHead(model, catalog[1], 191, 5),
      /KV K head 2 · row offset 63/,
    );
    assert.match(
      A.hoverHead(
        model,
        { ...catalog[0], name: "model.layers.0.self_attn.o_proj.weight" },
        70,
        129,
      ),
      /input-column Q-head group 2 · column offset 1/,
    );
    assert.match(A.hoverHead(model, catalog[2], 0, 4), /not applicable/);
    assert.match(
      A.hoverHead(
        { ...model, source_identity: "b".repeat(64) },
        catalog[0],
        64,
        5,
      ),
      /unavailable/,
    );
    assert.match(
      A.hoverHead(model, { ...catalog[0], shape: [575, 576] }, 64, 5),
      /mismatch/,
    );
    const qe = require("./fixtures/ui-polish-qwen-observations.json");
    const qm = {
      source_identity: qe.source_identity,
      revision: qe.revision,
      model_identity: "c".repeat(64),
      catalog: qe.records.map((r, id) => ({
        ...r,
        id,
        rows: r.shape.length === 1 ? 1 : r.shape[0],
        cols: r.shape.at(-1),
      })),
    };
    assert.equal(A.guidedExamples(qm).length, 3);
    assert.equal(
      A.guidedExamples({ ...qm, source_identity: "b".repeat(64) }).length,
      0,
    );
    assert.match(A.hoverHead(qm, qm.catalog[0], 128, 5), /unavailable/);
    assert.match(A.hoverHead(qm, qm.catalog[1], 1023, 5), /unavailable/);
    for (const [i, e] of A.guidedExamples(qm).entries()) {
      assert.deepEqual(
        A.resolveBookmark(A.parseBookmark(e.href), qm).region,
        qe.records[i].region,
      );
      const r = qe.records[i];
      assert.equal(Math.min(...r.values), r.min);
      assert.equal(Math.max(...r.values), r.max);
      assert.equal(r.values.filter((v) => v > 0).length, r.positive);
      assert.equal(r.values.filter((v) => v < 0).length, r.negative);
    }
    class Element {
      constructor() {
        this.children = [];
        this.listeners = {};
        this.hidden = false;
        this.disabled = false;
        this.dataset = {};
        this.style = {};
        this.attrs = {};
        this.value = "";
        this.textContent = "";
        this.classList = { toggle() {}, add() {}, remove() {} };
      }
      append(...a) {
        this.children.push(...a);
      }
      replaceChildren(...a) {
        this.children = a;
      }
      setAttribute(k, v) {
        this.attrs[k] = v;
      }
      removeAttribute(k) {
        delete this[k];
      }
      addEventListener(k, f) {
        this.listeners[k] = f;
      }
      focus() {
        this.focused = true;
      }
    }
    const elems = new Map(),
      get = (id) => {
        if (!elems.has(id)) elems.set(id, new Element());
        return elems.get(id);
      };
    get("left-rule").value = "tensor_linear";
    get("right-rule").value = "tensor_asinh";
    let tid = 0;
    const timers = new Map(),
      requests = [],
      storage = new Map();
    const ctx = vm.createContext({
      console,
      AtlasTools: A,
      AbortController,
      DOMException,
      URLSearchParams,
      performance,
      localStorage: {
        getItem: (k) => storage.get(k),
        setItem: (k, v) => storage.set(k, v),
        removeItem: (k) => storage.delete(k),
      },
      setTimeout: (f, ms) => {
        timers.set(++tid, { f, ms });
        return tid;
      },
      clearTimeout: (id) => timers.delete(id),
      window: { matchMedia: () => ({ matches: true }) },
      document: {
        documentElement: new Element(),
        getElementById: get,
        createElement: () => new Element(),
      },
      fetch: (url, options) =>
        new Promise((resolve) =>
          requests.push({
            url,
            options,
            resolve: (body) => resolve({ ok: true, json: async () => body }),
          }),
        ),
      fixture: model,
    });
    await require("./support/comparison-controller.cjs").load(
      fs.readFileSync("web/app.js", "utf8").replace(/initialize\(\);\s*$/, ""),
      ctx,
      "web/app.js",
    );
    const run = (s) => vm.runInContext(s, ctx),
      tick = async () => {
        for (let i = 0; i < 12; i++) await Promise.resolve();
      };
    function fire(ms) {
      const entry = [...timers].find(([, t]) => t.ms === ms);
      assert(entry, "missing timer " + ms);
      timers.delete(entry[0]);
      entry[1].f();
    }
    function scalar(row, col, t = catalog[0]) {
      return {
        api_version: 1,
        source_binding: A.sourceBinding(model, t),
        tensor: t.id,
        row,
        col,
        dtype: "BF16",
        element_bytes: 2,
        raw_hex_le: "803f",
        raw_exact: "1",
        native_indices: t.shape.length === 1 ? [col] : [row, col],
        shard: "model.safetensors",
        byte_offset: 8,
        transformed: { left: 0.5, right: 0.6 },
      };
    }
    require("./support/async-completion.cjs").requireCompletion(
      (async () => {
        run(
          "state.model=fixture;state.tensor=fixture.catalog[0];state.loading=false;state.current={tensor:state.tensor};state.selected=[1,1]",
        );
        assert.match(
          run("tensorDescription(state.tensor).title"),
          /Layer 0.*Attention queries/,
        );
        run("queueHover(64,1);queueHover(64,2);queueHover(64,3)");
        assert.equal(requests.length, 0);
        assert.equal(timers.size, 1);
        fire(140);
        assert.equal(requests.length, 1);
        const old = requests.shift();
        run("queueHover(65,3)");
        assert(old.options.signal.aborted);
        fire(140);
        const fresh = requests.shift();
        fresh.resolve(scalar(65, 3));
        await tick();
        assert.match(
          get("hover-readout").textContent,
          /Row 65 · column 3.*native \[65, 3\].*Query head 1.*raw BF16 1/,
        );
        old.resolve(scalar(64, 3));
        await tick();
        assert.match(get("hover-readout").textContent, /Row 65/);
        assert.equal(run('state.selected.join(",")'), "1,1");
        run("queueHover(65,3)");
        assert.equal(timers.size, 0);
        assert.equal(requests.length, 0);
        run("queueHover(66,4)");
        fire(140);
        const wrong = requests.shift();
        wrong.resolve(scalar(66, 5));
        await tick();
        assert.match(get("hover-readout").textContent, /does not match/);
        run("queueHover(67,4)");
        fire(140);
        const late = requests.shift();
        run("cancelHover();state.viewEpoch++");
        late.resolve(scalar(67, 4));
        await tick();
        assert.match(get("hover-readout").textContent, /Hover over/);
        run("queueHover(-1,0);queueHover(576,0)");
        assert.equal(requests.length, 0);
        assert.equal(timers.size, 0);
        run("state.tensor=fixture.catalog[2];queueHover(0,17)");
        fire(140);
        requests.shift().resolve(scalar(0, 17, catalog[2]));
        await tick();
        assert.match(
          get("hover-readout").textContent,
          /native \[17\].*not applicable/,
        );
        run("queueHover(0,18)");
        fire(140);
        const expired = requests.shift();
        fire(2000);
        assert(expired.options.signal.aborted);
        expired.resolve(scalar(0, 18, catalog[2]));
        await tick();
        assert(!get("hover-readout").textContent.includes("raw BF16 1"));
        run(
          `state.tensor={id:9,rows:100000,cols:200000,max_level:18};state.current={tensor:state.tensor};state.viewers.left={world:{getItemCount:()=>1,getItemAt:()=>({viewportToImageCoordinates:(x,y)=>({x,y})})},viewport:{getBounds:()=>({x:50000,y:25000,width:50000,height:25000})}};loadOverview()`,
        );
        const image = get("overview-image");
        assert.match(image.src, /level=8&x=0&y=0/);
        assert.equal(get("tensor-overview").style.width, "144px");
        image.onload();
        assert.equal(get("overview-viewport").style.left, "25%");
        assert.equal(get("overview-viewport").style.width, "25%");
        assert.equal(get("overview-viewport").style.top, "25%");
        assert.equal(get("overview-viewport").style.height, "25%");
        run("updateViewportBounds()");
        assert.match(
          get("viewport-bounds").textContent,
          /rows 25,000–49,999.*columns 50,000–99,999/,
        );
        assert.equal(requests.length, 0);
        run("clearOverview()");
        assert(image.hidden);
        assert(get("overview-viewport").hidden);
        assert.equal(image.src, undefined);
        run("bindWelcomeAndTheme()");
        assert.equal(ctx.document.documentElement.dataset.theme, "dark");
        get("theme-toggle").listeners.click();
        assert.equal(storage.get("atlas-theme"), "light");
        get("dismiss-help").listeners.click();
        assert(get("welcome").hidden);
        assert.equal(storage.get("atlas-welcome-dismissed"), "1");
        get("show-help").listeners.click();
        assert(!get("welcome").hidden);
        assert(!storage.has("atlas-welcome-dismissed"));
        console.log(
          "PASS: source-bound observed examples; verified row/column head mappings; hover debounce, abort, late response, address validation, vector indices, timeout, no selection mutation; bounded overview and viewport geometry; reversible help/theme. Pure doubles only.",
        );
      })().catch((e) => {
        console.error(e);
        process.exitCode = 1;
      }),
    );
  })(),
);
