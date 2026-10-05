"use strict";
// Ordinary UI acceptance only. External guards own all limits and processes.
const { chromium, expect } = require("playwright/test");
const assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  crypto = require("node:crypto");
const H = require("./acceptance/support.cjs");
const base = process.env.ATLAS_TEST_URL,
  out = process.env.ATLAS_EVIDENCE_DIR;
const phase = process.env.ATLAS_ACCEPTANCE_PHASE || "combined",
  phaseB = phase === "full-sweep-only";
assert(
  ["combined", "full-sweep-only"].includes(phase),
  "Unknown acceptance phase",
);
assert(/^http:\/\/127\.0\.0\.1:\d+$/.test(base));
assert(out && process.env.ATLAS_CHROMIUM);
fs.mkdirSync(out, { recursive: true });
const save = (name, data) => H.save(out, name, data);
const sha = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
(async () => {
  const started = Date.now(),
    checks = [],
    errors = [],
    pending = new Set(),
    runs = [],
    owners = [],
    inspectionStates = [];
  let page,
    browser,
    active = null,
    cleanup = null;
  const scrub = (snapshot) => {
    const value = JSON.parse(JSON.stringify(snapshot));
    delete value.session;
    return value;
  };
  try {
    browser = await chromium.launch({
      headless: true,
      chromiumSandbox: true,
      executablePath: process.env.ATLAS_CHROMIUM,
      args: ["--renderer-process-limit=1", "--disable-gpu"],
    });
    const context = await browser.newContext({
      viewport: { width: 1360, height: 900 },
      acceptDownloads: true,
    });
    await context.route("**/*", (r) =>
      r
        .request()
        .url()
        .startsWith(base + "/")
        ? r.continue()
        : r.abort(),
    );
    page = await context.newPage();
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("response", (response) => {
      const route = new URL(response.url());
      if (
        route.origin !== base ||
        ![
          "/api/inference/start",
          "/api/inference/poll",
          "/api/inference/cancel",
          "/api/inference/reset",
        ].includes(route.pathname)
      )
        return;
      const promise = (async () => {
        const raw = await response.body();
        assert(raw.length <= 1024 * 1024);
        const body = JSON.parse(raw);
        if (!response.ok()) {
          save("action-error.json", {
            route: route.pathname,
            status: response.status(),
            body: scrub(body),
          });
          throw new Error(
            `Normal UI action ${route.pathname} returned ${response.status()}: ${body.error || ""}`,
          );
        }
        if (route.pathname.endsWith("/start")) {
          assert(
            runs.length < (phaseB ? 1 : 2),
            "No unplanned extra inference job",
          );
          active = {
            index: runs.length + 1,
            owner: body.session,
            request: response.request().postDataJSON(),
            latest: body,
            progress: [],
          };
          runs.push(active);
          owners.push(body.session);
          save(`run-${active.index}-accepted-request.json`, active.request);
        }
        if (!active) return;
        if (body.session !== active.owner && !route.pathname.endsWith("/reset"))
          return;
        if (route.pathname.endsWith("/reset")) {
          active.reset = scrub(body);
          save(`run-${active.index}-reset.json`, active.reset);
          return;
        }
        active.latest = body;
        if (active.progress.length < 128)
          active.progress.push({
            status: body.status,
            records: body.steps?.length || 0,
            worker_alive: body.worker_alive,
            coverage: body.details?.sweep_coverage || null,
          });
        save(`run-${active.index}-runtime.json`, scrub(body));
        save(`run-${active.index}-progress.json`, active.progress);
      })().catch((e) => errors.push(e.message));
      pending.add(promise);
      promise.finally(() => pending.delete(promise));
    });
    await page.goto(base);
    const el = (id) => page.locator("#infer-" + id);
    await page.locator("#inference-panel > summary").click();
    await expect(el("start")).toBeEnabled({ timeout: 15000 });
    const model = await (await context.request.get(base + "/api/model")).json();
    const metadata = await (
      await context.request.get(base + "/api/inference")
    ).json();
    assert.equal(metadata.head_layout.runtime_verified, false);
    assert.equal(metadata.head_layout.evidence, "pinned_configuration");
    assert.equal(
      await el("layer").locator("option").count(),
      metadata.architecture.layers,
    );
    assert.equal(
      await el("site").locator("option").count(),
      Object.keys(metadata.architecture.capture_sites).length,
    );
    save("model-binding.json", {
      source_identity: model.source_identity,
      model_identity: model.model_identity,
      revision: model.revision,
      inference_source_model: model.inference_source_model,
      head_layout: metadata.head_layout,
      viewer_receipt_bound: !!model.head_layout_binding,
      architecture: metadata.architecture,
    });
    // Bind actual served assets before treating the UI as candidate evidence.
    const bundle = await (
      await context.request.get(base + "/viewer.js")
    ).body();
    const files = [
      "web/vendor/openseadragon.min.js",
      "web/atlas-tools.js",
      "web/app.js",
      "web/workspace-tools.js",
      "web/inference.js",
    ];
    const expected = Buffer.concat(
      files.flatMap((file, i) => [
        ...(i ? [Buffer.from("\n;\n")] : []),
        fs.readFileSync(file),
      ]),
    );
    assert.equal(
      sha(bundle),
      sha(expected),
      "Browser bundle must contain current inference source",
    );
    save("served-assets.json", {
      viewer_js_sha256: sha(bundle),
      source_files: files.map((file) => ({
        file,
        sha256: sha(fs.readFileSync(file)),
      })),
    });
    const shot = async (name) => {
      await page.evaluate(
        () =>
          new Promise((r) =>
            requestAnimationFrame(() => requestAnimationFrame(r)),
          ),
      );
      await page.screenshot({ path: path.join(out, name + ".png") });
    };
    if (phaseB) {
      const priorRoot = process.env.ATLAS_PHASE_A_EVIDENCE;
      assert(
        priorRoot,
        "Phase B requires an explicit ATLAS_PHASE_A_EVIDENCE archive",
      );
      const prior = JSON.parse(
        fs.readFileSync(path.join(priorRoot, "manifest.json"), "utf8"),
      );
      assert.equal(
        prior.attempt_commit,
        "96d7251033b929546c0e0e05e9b22b50198b01fe",
      );
      const reused = [
        "acceptance/browser/served-assets.json",
        "acceptance/browser/inspection-states.json",
        "acceptance/browser/output-head-columns.png",
        "acceptance/browser/query-row-intervention.png",
        "acceptance/browser/reviewed-plan.json",
      ];
      for (const file of reused)
        assert.equal(
          sha(fs.readFileSync(path.join(priorRoot, file))),
          prior.files[file],
          "Retained phase A evidence hash mismatch",
        );
      assert.equal(
        sha(bundle),
        JSON.parse(fs.readFileSync(path.join(priorRoot, reused[0]), "utf8"))
          .viewer_js_sha256,
        "Phase A proof requires unchanged served assets",
      );
      save("phase-scope.json", {
        phase,
        prior_attempt_commit: prior.attempt_commit,
        reused_phase_a_files: reused,
        maximum_new_jobs: 1,
        original_browser_wall_seconds: 120,
        model_total_wall_seconds: 120,
        model_total_cpu_seconds: 90,
        scope:
          "One fresh full sweep; previous geometry assertions retained with matching source; no cancellation or continuation job",
      });
      checks.push(
        "Phase A O/Q geometry and loading-selection proof retained with exact file hashes and identical served assets",
      );
    } else {
      save("phase-scope.json", {
        phase: "combined-under-user-approved-1gib-browser-policy",
        maximum_new_jobs: 2,
        phases: [
          {
            id: "A",
            scope:
              "ordinary native selection and O/Q draft geometry; no model job",
          },
          {
            id: "B",
            scope:
              "one complete layer0 sweep;19records38prefills; original total model budget",
          },
          {
            id: "C",
            scope:
              "separate normal cancellation lifecycle; one new job only if B passes and elapsed is below80s; cancel/reap wait at most10s",
          },
        ],
        browser_total_wall_seconds: 120,
        scope:
          "One browser lifetime, no extensions, no continuation of unfinished heads; new cap must be enforced by bound outer guard",
      });
      const inspect = async (tensor) => {
        await page.locator("#tensor-search").fill(tensor);
        await page
          .getByRole("button", { name: "Select " + tensor, exact: true })
          .click();
        await expect(page.locator("#inspect-submit")).toBeEnabled();
        inspectionStates.push({
          tensor,
          elapsed_ms: Date.now() - started,
          view_busy: await page
            .locator("#comparison")
            .getAttribute("aria-busy"),
          readiness: await page.locator("#readiness-summary").textContent(),
          phase: "before_normal_inspect",
        });
        save("inspection-states.json", inspectionStates);
        await page.locator("#row").fill("70");
        await page.locator("#col").fill("575");
        await page.locator("#inspect-submit").click();
        await expect(el("selection")).toContainText(tensor);
        await expect(el("head")).toBeEnabled();
        inspectionStates.push({
          tensor,
          elapsed_ms: Date.now() - started,
          view_busy: await page
            .locator("#comparison")
            .getAttribute("aria-busy"),
          selection: await el("selection").textContent(),
          label: await el("head").textContent(),
          phase: "verified_action_ready",
        });
        save("inspection-states.json", inspectionStates);
      };
      await inspect("model.layers.0.self_attn.o_proj.weight");
      await expect(el("head")).toContainText("output head (columns)");
      await el("head").click();
      await expect(el("edits")).toContainText("columns [512,576)");
      await el("edits").scrollIntoViewIfNeeded();
      await shot("output-head-columns");
      await el("clear-edits").click();
      await inspect("model.layers.0.self_attn.q_proj.weight");
      await expect(el("head")).toContainText("query rows (query intervention)");
      await el("head").click();
      await expect(el("edits")).toContainText("rows [64,128)");
      await el("edits").scrollIntoViewIfNeeded();
      await shot("query-row-intervention");
      await el("clear-edits").click();
      checks.push(
        "Native row70/column575: O shortcut chooses columns512:576; Q shortcut chooses rows64:128 with distinct intervention labels",
      );
    }
    await el("task").selectOption("sweep");
    await el("prompt").fill("The capital of France is");
    await el("sweep-targets").fill("layer 0");
    await el("sweep-operation").selectOption("zero");
    await el("sweep-seed").fill("7");
    await el("layer").selectOption("0");
    await el("site").selectOption("attention");
    await el("mode").selectOption("step");
    await expect(el("sweep-second")).not.toBeChecked();
    await expect(el("include-prompt")).not.toBeChecked();
    await el("sweep-preview").click();
    await expect(el("start")).toBeEnabled();
    const plan = JSON.parse(await el("sweep-plan").textContent());
    assert.equal(plan.records, 19);
    assert.equal(plan.prefills, 38);
    assert.equal(plan.scope, "layer_heads");
    assert.equal(plan.targets.length, 9);
    assert(plan.control_semantics.includes("not a null/no-effect"));
    assert(
      plan.cases
        .slice(1)
        .every(
          (c) =>
            c.edits[0].kind === "columns" &&
            c.edits[0].tensor.endsWith(".o_proj.weight"),
        ),
    );
    save("reviewed-plan.json", plan);
    if (phaseB)
      assert.equal(
        plan.digest,
        JSON.parse(
          fs.readFileSync(
            path.join(
              process.env.ATLAS_PHASE_A_EVIDENCE,
              "acceptance/browser/reviewed-plan.json",
            ),
            "utf8",
          ),
        ).digest,
        "Phase B plan must match accepted Phase A plan",
      );
    await el("start").click();
    await expect.poll(() => runs.length, { timeout: 15000 }).toBe(1);
    await expect
      .poll(() => H.clean(active.latest), {
        timeout: 65000,
        intervals: [100, 250],
      })
      .toBe(true);
    await Promise.all([...pending]);
    assert.deepEqual(errors, []);
    const complete = active.latest;
    assert.equal(complete.status, "complete");
    H.sweepCheck(plan, complete);
    assert.equal(complete.steps.length, 19);
    assert.deepEqual(complete.details.sweep_coverage.unfinished_heads, []);
    assert.equal(complete.worker_alive, false);
    await expect(el("sweep-results").locator("tr")).toHaveCount(19);
    await expect(el("sweep-progress")).toContainText("19 / 19 completed");
    await expect(el("sweep-progress")).toContainText("0 unrun");
    await el("step").click();
    await expect(el("score-context")).toContainText(
      "Matched fixed original prompt",
    );
    if (phaseB) {
      const optional = {
        desktop: "unrun",
        export: "unrun",
        mobile: "unrun",
        pin_readback: "unrun",
      };
      checks.push(
        "One complete full-layer job: all19records, original paired controls, runtime metrics/restoration flags, no unfinished heads and worker reaped",
      );
      if (Date.now() - started < 70000) {
        await el("sweep-progress").scrollIntoViewIfNeeded();
        await shot("all-heads-complete-desktop");
        optional.desktop = "passed";
      }
      if (Date.now() - started < 75000) {
        const exported = await H.download(
          page,
          out,
          "full-sweep-export-redacted.json",
        );
        H.redacted(exported);
        assert.equal(exported.steps.length, 19);
        assert(
          owners.every((owner) => !JSON.stringify(exported).includes(owner)),
        );
        optional.export = "passed";
      }
      if (Date.now() - started < 80000) {
        await page.setViewportSize({ width: 390, height: 844 });
        await el("sweep-progress").scrollIntoViewIfNeeded();
        await shot("all-heads-complete-mobile");
        const mobileOverflow = await page.evaluate(
          () => document.documentElement.scrollWidth > innerWidth,
        );
        save("mobile-layout.json", {
          viewport: { width: 390, height: 844 },
          horizontal_overflow: mobileOverflow,
        });
        assert(!mobileOverflow, "Mobile page overflows");
        await page.setViewportSize({ width: 1360, height: 900 });
        optional.mobile = "passed";
      }
      await el("reset").click();
      await expect(el("status")).toContainText("Ready");
      await Promise.all([...pending]);
      assert.equal(runs.length, 1);
      assert.deepEqual(errors, []);
      let pins = null;
      if (Date.now() - started < 90000) {
        pins = await H.verifyPins(process.env.ATLAS_MODEL_DIR);
        optional.pin_readback = "passed";
      }
      save("browser-result.json", {
        status: "PASS",
        phase,
        checks,
        optional,
        unrun_optional: Object.entries(optional)
          .filter(([, value]) => value === "unrun")
          .map(([name]) => name),
        optional_skip_reason:
          "Original browser lifetime and cleanup margin; no time extension",
        full_sweep_records: 19,
        run_count: runs.length,
        elapsed_ms: Date.now() - started,
        browser: browser.version(),
        page_errors: errors,
        pinned_disk_hashes_checked_after: pins,
        cancellation_exercised: false,
        full_combined_workflow_qualified: false,
        scope:
          "Phase B only: one complete layer0 public-fixture sweep; phase A separately retained; phase C cancellation unrun; not proof of uninterrupted full-workflow memory fit",
      });
      return;
    }
    await el("sweep-progress").scrollIntoViewIfNeeded();
    await shot("all-heads-complete-desktop");
    const exported = await H.download(
      page,
      out,
      "full-sweep-export-redacted.json",
    );
    H.redacted(exported);
    assert.equal(exported.steps.length, 19);
    assert(owners.every((owner) => !JSON.stringify(exported).includes(owner)));
    checks.push(
      "One full-layer UI job completed19records with runtime per-case metrics, bitwise restoration flags and no unfinished heads; export redacts prompt fields and capability",
    );
    await page.setViewportSize({ width: 390, height: 844 });
    await el("sweep-progress").scrollIntoViewIfNeeded();
    await shot("all-heads-complete-mobile");
    const mobileOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    );
    save("mobile-layout.json", {
      viewport: { width: 390, height: 844 },
      horizontal_overflow: mobileOverflow,
    });
    assert(!mobileOverflow, "Mobile page overflows");
    await page.setViewportSize({ width: 1360, height: 900 });
    await el("reset").click();
    await expect(el("status")).toContainText("Ready");
    // Separately authorized lifecycle case; never continues incomplete heads.
    if (Date.now() - started >= 80000) {
      let pins = null;
      if (Date.now() - started < 90000)
        pins = await H.verifyPins(process.env.ATLAS_MODEL_DIR);
      save("browser-result.json", {
        status: "PASS_FULL_SWEEP_CANCELLATION_UNRUN",
        checks,
        cancellation_exercised: false,
        cancellation_unrun_reason:
          "Declared phase C requires elapsed below80s; original browser lifetime not extended",
        full_sweep_records: 19,
        run_count: runs.length,
        elapsed_ms: Date.now() - started,
        browser: browser.version(),
        page_errors: errors,
        pinned_disk_hashes_checked_after: pins,
        scope:
          "Complete layer0 public-fixture sweep and export/mobile checks; optional short cancellation phase skipped",
      });
      return;
    }
    await el("sweep-preview").click();
    await expect(el("start")).toBeEnabled();
    await el("start").click();
    await expect.poll(() => runs.length, { timeout: 15000 }).toBe(2);
    if (!H.clean(active.latest)) {
      await expect(el("cancel")).toBeEnabled();
      await el("cancel").click();
      await expect
        .poll(() => H.clean(active.latest), {
          timeout: 10000,
          intervals: [100, 250],
        })
        .toBe(true);
      await Promise.all([...pending]);
    }
    const cancelled = active.latest;
    H.sweepCheck(plan, cancelled);
    const cancellationExercised =
      cancelled.status === "cancelled" && cancelled.steps.length < 19;
    if (cancellationExercised) {
      assert(cancelled.details.sweep_coverage.unrun_ids.length > 0);
      assert(cancelled.details.sweep_coverage.unfinished_heads.length > 0);
      await expect(el("sweep-progress")).toContainText(
        `${cancelled.steps.length} / 19 completed`,
      );
      checks.push(
        "Normal cancellation produced honest partial coverage; owned worker reaped without continuation",
      );
    }
    await el("sweep-progress").scrollIntoViewIfNeeded();
    await shot("cancelled-or-raced-completion");
    await el("reset").click();
    await expect(el("status")).toContainText("Ready");
    await Promise.all([...pending]);
    assert.equal(runs.length, 2);
    assert.deepEqual(errors, []);
    const pins = await H.verifyPins(process.env.ATLAS_MODEL_DIR);
    save("browser-result.json", {
      status: cancellationExercised
        ? "PASS"
        : "PARTIAL_CANCELLATION_UNEXERCISED",
      checks,
      cancellation_exercised: cancellationExercised,
      cancellation_received_records: cancelled.steps.length,
      full_sweep_records: 19,
      run_count: runs.length,
      elapsed_ms: Date.now() - started,
      browser: browser.version(),
      page_errors: errors,
      pinned_disk_hashes_checked_after: pins,
      scope:
        "One complete layer0 short-public-prompt sweep plus distinct ordinary cancellation; no cross-session residency or general head-importance claim",
    });
    if (!cancellationExercised) process.exitCode = 3;
  } catch (e) {
    if (page)
      await page
        .screenshot({ path: path.join(out, "failure.png") })
        .catch(() => {});
    save("browser-failure.json", {
      status: "FAIL_OR_INTERRUPTED",
      error: e.stack,
      checks,
      page_errors: errors,
      runs: runs.map((r) => ({
        index: r.index,
        status: r.latest?.status,
        records: r.latest?.steps?.length || 0,
        coverage: r.latest?.details?.sweep_coverage || null,
      })),
    });
    throw e;
  } finally {
    try {
      if (page && active && !H.clean(active.latest)) {
        const w = { owner: active.owner, latest: active.latest };
        await H.cleanup(page, base, w);
        active.latest = w.latest;
      }
      cleanup = {
        worker_cleanup_confirmed: active ? H.clean(active.latest) : null,
        last_status: active?.latest?.status || null,
        runs: runs.length,
      };
    } catch (e) {
      cleanup = { worker_cleanup_confirmed: false, error: e.message };
    }
    save("browser-cleanup.json", cleanup);
    if (browser) await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
