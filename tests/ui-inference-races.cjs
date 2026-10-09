"use strict";
require("./support/comparison-controller.cjs").enableModules(__filename);
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    const fs = require("fs"),
      vm = require("vm"),
      assert = require("assert/strict");
    const source = fs.readFileSync("web/inference.js", "utf8");
    const { AtlasPlayback } = require("../web/inference.js");
    const step = (i) => ({
      index: i,
      activation: Array(576).fill(i + 0.5),
      position: 4 + i,
      input_token_id: 10 + i,
      token_id: 11 + i,
      token_piece: " token",
      generated_text: " token".repeat(i + 1),
      compute_ms: 20,
      compute_total_ms: 20 * (i + 1),
      layer: 0,
      phase: i ? "decode" : "prefill",
    });
    const p = new AtlasPlayback();
    p.accept({ status: "running", steps: [step(0)] });
    assert.equal(p.advance(), true);
    assert.equal(p.current.index, 0);
    assert.equal(p.advance(), false);
    p.accept({ status: "complete", steps: [step(0), step(1)] });
    assert.equal(p.advance(), true);
    p.rewind();
    assert.equal(p.current, null);
    assert.equal(p.replaying, true);
    assert.equal(p.paused, true);
    p.reset();
    assert.equal(p.steps.length, 0);
    assert.throws(() =>
      p.accept({ steps: Array.from({ length: 33 }, (_, i) => step(i)) }),
    );
    assert.throws(() =>
      p.accept({ steps: [{ ...step(0), activation: [NaN] }] }),
    );
    class Element {
      constructor() {
        this.value = "";
        this.textContent = "";
        this.disabled = false;
        this.hidden = false;
        this.listeners = {};
        this.width = 512;
        this.height = 288;
      }
      replaceChildren(...items) {
        this.children = items;
      }
      append(...items) {
        this.children = (this.children || []).concat(items);
      }
      addEventListener(k, f) {
        this.listeners[k] = f;
      }
      getContext() {
        return { clearRect() {}, fillRect() {} };
      }
      getBoundingClientRect() {
        return { top: 0, left: 0, width: 512, height: 288 };
      }
    }
    const elems = new Map(),
      get = (id) => {
        if (!elems.has(id)) elems.set(id, new Element());
        return elems.get(id);
      };
    get("infer-rate").value = "2";
    get("infer-limit").value = "4";
    get("infer-layer").value = "0";
    get("infer-mode").value = "step";
    get("infer-prompt").value = "Hello";
    let timerId = 0;
    const timers = new Map(),
      requests = [];
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
      setTimeout: (f, ms) => {
        timers.set(++timerId, { f, ms });
        return timerId;
      },
      clearTimeout: (id) => timers.delete(id),
    });
    await require("./support/comparison-controller.cjs").load(
      source,
      context,
      require("node:path").join(__dirname, "../web/inference.js"),
    );
    const tick = async () => {
      for (let i = 0; i < 10; i++) await Promise.resolve();
    };
    const take = (part) => {
      const i = requests.findIndex((r) => r.url.endsWith(part));
      assert(i >= 0, part);
      return requests.splice(i, 1)[0];
    };
    const click = (id) => get("infer-" + id).listeners.click();
    const submit = () =>
      get("infer-form").listeners.submit({ preventDefault() {} });
    require("./support/async-completion.cjs").requireCompletion(
      (async () => {
        take("/api/inference").resolve({ model: "fixture", engine: "fixture" });
        await tick();
        assert.equal(get("infer-start").disabled, false);
        const start = submit();
        assert.equal(get("infer-start").disabled, true);
        await submit();
        assert.equal(requests.length, 1);
        take("/start").resolve({
          session: "one",
          status: "loading",
          steps: [],
          details: {},
        });
        await start;
        await tick();
        const stale = take("/poll");
        const reset = click("reset");
        take("/reset").resolve({
          session: null,
          status: "idle",
          steps: [],
          details: {},
        });
        await reset;
        stale.resolve({
          session: "one",
          status: "running",
          steps: [step(0)],
          details: {},
        });
        await tick();
        assert.equal(
          get("infer-output").textContent,
          "Generated text will appear here.",
        );
        assert(get("infer-status").textContent.startsWith("Ready"));
        const next = submit();
        take("/start").resolve({
          session: "two",
          status: "running",
          steps: [],
          details: {},
        });
        await next;
        take("/poll").resolve({
          session: "two",
          status: "complete",
          steps: [step(0), step(1)],
          details: {},
        });
        await tick();
        assert(get("infer-status").textContent.includes("shown 0 / 2"));
        click("step");
        assert(
          get("infer-alignment").textContent.includes("consumed position 4"),
        );
        assert(get("infer-status").textContent.includes("shown 1 / 2"));
        assert.equal(timers.size, 0);
        click("pause");
        assert([...timers.values()].some((t) => t.ms === 500));
        get("infer-rate").value = "8";
        get("infer-rate").listeners.change();
        assert([...timers.values()].some((t) => t.ms === 125));
        click("pause");
        assert.equal(timers.size, 0);
        click("replay");
        assert(get("infer-status").textContent.includes("Recorded replay"));
        assert(get("infer-status").textContent.includes("shown 0 / 2"));
        // A callback may already be queued when clearTimeout runs. Deliver it anyway.
        click("pause");
        const rewindTimer = [...timers.values()][0].f;
        click("replay");
        assert.equal(timers.size, 0);
        rewindTimer();
        assert(
          get("infer-status").textContent.includes("paused · shown 0 / 2"),
        );
        assert.equal(
          get("infer-output").textContent,
          "Generated text will appear here.",
        );
        assert.equal(timers.size, 0);
        click("pause");
        rewindTimer();
        assert(get("infer-status").textContent.includes("shown 0 / 2"));
        const oldRateTimer = [...timers.values()][0].f;
        get("infer-rate").value = "4";
        get("infer-rate").listeners.change();
        oldRateTimer();
        assert(get("infer-status").textContent.includes("shown 0 / 2"));
        const [timerKey, currentTimer] = [...timers.entries()][0];
        assert.equal(currentTimer.ms, 250);
        timers.delete(timerKey);
        currentTimer.f();
        assert(get("infer-status").textContent.includes("shown 1 / 2"));
        const resetTimer = [...timers.values()][0].f;
        const r = click("reset");
        take("/reset").resolve({
          session: null,
          status: "idle",
          steps: [],
          details: {},
        });
        await r;
        resetTimer();
        assert(get("infer-status").textContent.startsWith("Ready"));
        const bad = submit();
        take("/start").resolve({ error: "Prompt too long" }, 400);
        await bad;
        assert.equal(get("infer-error").textContent, "Prompt too long");
        assert.equal(get("infer-start").disabled, false);
        const again = submit();
        take("/start").resolve({
          session: "three",
          status: "loading",
          steps: [],
          details: {},
        });
        await again;
        resetTimer();
        assert(get("infer-status").textContent.includes("shown 0 / 0"));
        const late = take("/poll");
        const cancel = click("cancel");
        take("/cancel").resolve({
          session: "three",
          status: "cancelled",
          steps: [],
          details: {},
        });
        await cancel;
        late.resolve({
          session: "three",
          status: "complete",
          steps: [step(0)],
          details: {},
        });
        await tick();
        assert(get("infer-status").textContent.includes("cancelled"));
        assert.equal(
          get("infer-output").textContent,
          "Generated text will appear here.",
        );
        const cleanupRun = submit();
        take("/start").resolve({
          session: "cleanup",
          status: "running",
          steps: [],
          details: {},
        });
        await cleanupRun;
        take("/poll").resolve({
          session: "cleanup",
          status: "running",
          steps: [],
          details: {},
        });
        await tick();
        const cleanupCancel = click("cancel");
        take("/cancel").resolve({
          session: "cleanup",
          status: "stopping",
          steps: [],
          details: { cleanup_pending: true },
        });
        await cleanupCancel;
        assert.equal(get("infer-start").disabled, true);
        take("/poll").resolve({
          session: "cleanup",
          status: "cancelled",
          steps: [],
          details: {},
        });
        await tick();
        assert.equal(get("infer-start").disabled, false);
        assert(get("infer-status").textContent.includes("cancelled"));
        // A reset still awaiting reap retains the trace/capability and stays active.
        const resetRun = submit();
        take("/start").resolve({
          session: "reset-owner",
          status: "running",
          steps: [step(0)],
          details: {},
        });
        await resetRun;
        take("/poll").resolve({
          session: "reset-owner",
          status: "running",
          steps: [step(0)],
          details: {},
        });
        await tick();
        click("step");
        const savedOutput = get("infer-output").textContent;
        const pendingReset = click("reset");
        const resetRequest = take("/reset");
        assert.equal(
          JSON.parse(resetRequest.options.body).session,
          "reset-owner",
        );
        resetRequest.resolve(
          {
            error: "Worker cleanup pending; retry reset",
            code: "cleanup_pending",
          },
          503,
        );
        await pendingReset;
        assert.equal(get("infer-start").disabled, true);
        assert.equal(get("infer-reset").disabled, false);
        assert.equal(get("infer-output").textContent, savedOutput);
        const cleanupPoll = take("/poll");
        assert.equal(
          JSON.parse(cleanupPoll.options.body).session,
          "reset-owner",
        );
        await submit();
        assert.equal(requests.length, 0);
        // A transient status failure while stopping must not release local ownership.
        cleanupPoll.resolve({ error: "Temporary transport failure" }, 503);
        await tick();
        assert.equal(get("infer-start").disabled, true);
        assert(get("infer-status").textContent.includes("stopping"));
        const retryTimer = [...timers.entries()].find(([, t]) => t.ms === 250);
        assert(retryTimer);
        timers.delete(retryTimer[0]);
        retryTimer[1].f();
        const reapPoll = take("/poll");
        assert.equal(JSON.parse(reapPoll.options.body).session, "reset-owner");
        reapPoll.resolve({
          session: "reset-owner",
          status: "cancelled",
          steps: [step(0)],
          details: {},
        });
        await tick();
        assert.equal(get("infer-start").disabled, false);
        assert.equal(get("infer-output").textContent, savedOutput);
        // Even a rejected replacement must preserve the previously accepted owner/trace.
        const rejectedReplacement = submit();
        take("/start").resolve(
          { error: "A session is active; cancel or reset it first" },
          400,
        );
        await rejectedReplacement;
        assert.equal(get("infer-reset").disabled, false);
        assert.equal(get("infer-output").textContent, savedOutput);
        const retryReset = click("reset");
        const retryRequest = take("/reset");
        assert.equal(
          JSON.parse(retryRequest.options.body).session,
          "reset-owner",
        );
        retryRequest.resolve({
          session: null,
          status: "idle",
          steps: [],
          details: {},
        });
        await retryReset;
        assert.equal(get("infer-reset").disabled, true);
        assert.equal(
          get("infer-output").textContent,
          "Generated text will appear here.",
        );
        assert.equal(requests.length, 0);
        console.log(
          JSON.stringify(
            {
              status: "PASS",
              checks: 20,
              scope:
                "Playback and DOM/transport doubles: bounded trace, step, end, replay, reset, stale poll, cancelled poll, duplicate start, rates, pause, error, restart; rewind cancels timer, queued callbacks honor pause, stale timer generations/rates/sessions ignored; pending cleanup remains active and is polled until reaped; delayed reset/transport retries preserve ownership and trace; rejected replacement preserves Reset retry",
            },
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
