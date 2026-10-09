"use strict";
// Benign feature acceptance only. Run each phase in a fresh, externally guarded
// browser lifetime. No response fault injection, model downloads or sandbox bypass.
const { chromium, expect } = require("playwright/test"),
  assert = require("node:assert/strict"),
  fs = require("node:fs");
const base = process.env.ATLAS_TEST_URL,
  out = process.env.ATLAS_EVIDENCE_DIR,
  phase = process.env.ATLAS_POLISH_PHASE;
assert(/^http:\/\/127\.0\.0\.1:\d+$/.test(base));
assert(out);
assert(["desktop", "mobile", "inference", "qwen"].includes(phase));
fs.mkdirSync(out, { recursive: true });
(async () => {
  const browser = await chromium.launch({
    headless: true,
    chromiumSandbox: true,
    executablePath: process.env.ATLAS_CHROMIUM,
    args: ["--renderer-process-limit=1", "--disable-gpu"],
  });
  const context = await browser.newContext({
    viewport:
      phase === "mobile"
        ? { width: 390, height: 844 }
        : { width: 1360, height: 900 },
  });
  const page = await context.newPage(),
    errors = [],
    checks = [];
  let owner = null;
  page.on("pageerror", (e) => errors.push(e.message));
  await context.route("**/*", (r) =>
    r
      .request()
      .url()
      .startsWith(base + "/")
      ? r.continue()
      : r.abort(),
  );
  const ready = async () => {
    await expect(page.locator("#comparison")).toHaveAttribute(
      "aria-busy",
      "false",
      { timeout: 20000 },
    );
    await page.waitForFunction(
      async () => {
        const __atlasController =
          typeof state === "undefined" ? await import("/app.js") : { state };
        return ["left", "right"].every((s) =>
          __atlasController.state.viewers[s]?.world
            .getItemAt(0)
            ?.getFullyLoaded(),
        );
      },
      {},
      { timeout: 20000 },
    );
  };
  const shot = async (name) => {
    await page.evaluate(
      () =>
        new Promise((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)),
        ),
    );
    await page.screenshot({ path: out + "/" + name + ".png" });
  };
  try {
    await page.goto(base);
    await ready();
    await expect(page.locator("#welcome")).toBeVisible();
    const model = await (await context.request.get(base + "/api/model")).json();
    assert.equal(
      await page.title(),
      "Unified Interpretability Toolkit · " + model.name,
    );
    if (phase !== "inference") {
      await expect(page.locator("#inference-panel")).toBeHidden();
      await expect(page.locator("#mode-status")).toContainText(
        "inference unavailable",
      );
      checks.push(
        "Viewer-only server hides unavailable experiments and loads the actual model name",
      );
    }
    if (phase === "desktop") {
      const section = process.env.ATLAS_DESKTOP_SECTION || "all";
      assert(
        ["all", "pointer", "navigation", "light", "dark"].includes(section),
      );
      if (["all", "pointer"].includes(section)) {
        await shot("desktop-first-visit");
        await page.locator("#dismiss-help").click();
        await expect(page.locator("#welcome")).toBeHidden();
        await page.locator("#show-help").click();
        await expect(page.locator("#welcome")).toBeVisible();
        await page.locator("#dismiss-help").click();
        // A fresh native catalog starts on embeddings. Exercise the actual
        // projection whose native coordinates the held pointer assertion uses.
        if (!(await page.locator("#tensor-browser").evaluate((e) => e.open)))
          await page.locator("#tensor-browser > summary").click();
        await page
          .locator("#tensor-search")
          .fill("model.layers.0.self_attn.q_proj.weight");
        await page.locator(".tensor-option").first().click();
        await ready();
        await page.locator("#row").fill("70");
        await page.locator("#col").fill("5");
        await page.locator("#inspect-submit").click();
        await expect(page.locator(".native-index")).toContainText("[70, 5]");
        await page.locator("#left-canvas").scrollIntoViewIfNeeded();
        const point = await page.evaluate(async () => {
          const __atlasController =
            typeof state === "undefined" ? await import("/app.js") : { state };

          const v = __atlasController.state.viewers.left,
            item = v.world.getItemAt(0),
            p = v.viewport.pixelFromPoint(
              item.imageToViewportCoordinates(6.5, 71.5),
              true,
            ),
            r = document.getElementById("left-canvas").getBoundingClientRect();
          return { x: r.left + p.x, y: r.top + p.y };
        });
        await page.mouse.move(point.x, point.y);
        await expect(page.locator("#hover-readout")).toContainText(
          "Row 71 · column 6",
          { timeout: 5000 },
        );
        await expect(page.locator("#hover-readout")).toContainText("raw BF16");
        await expect(page.locator("#hover-readout")).toContainText(
          "Query head 1",
        );
        const scalar = await (
          await context.request.get(
            base +
              "/api/inspect?" +
              new URLSearchParams({
                tensor: model.catalog.find(
                  (t) => t.name === "model.layers.0.self_attn.q_proj.weight",
                ).id,
                row: 71,
                col: 6,
                left: "tensor_linear",
                right: "tensor_asinh",
              }),
          )
        ).json();
        await expect(page.locator("#hover-readout")).toContainText(
          scalar.raw_exact,
        );
        await expect(page.locator(".native-index")).toContainText("[70, 5]");
        checks.push(
          "Actual pointer coordinates show exact source scalar and verified query head without changing the pinned inspector",
        );
        await page.locator("#zoom-in").click();
        await page.locator("#zoom-in").click();
        await ready();
        const recolorBounds = await page.evaluate(async () => {
          const __atlasController =
            typeof state === "undefined" ? await import("/app.js") : { state };

          const b =
            __atlasController.state.viewers.left.viewport.getBounds(true);
          return [b.x, b.y, b.width, b.height];
        });
        await page.locator("#left-rule").selectOption("tensor_asinh");
        await page.locator("#left-rule").selectOption("tensor_magnitude");
        await ready();
        const afterRecolor = await page.evaluate(async () => {
          const __atlasController =
            typeof state === "undefined" ? await import("/app.js") : { state };

          const b =
            __atlasController.state.viewers.left.viewport.getBounds(true);
          return [b.x, b.y, b.width, b.height];
        });
        assert(
          afterRecolor.every((n, i) => Math.abs(n - recolorBounds[i]) < 1e-7),
        );
        await expect(page.locator(".native-index")).toContainText("[70, 5]");
        checks.push(
          "Rapid consecutive recolorings preserve actual viewport and pinned scalar",
        );
      }
      if (["all", "navigation"].includes(section)) {
        if (section !== "all") {
          await page.locator("#dismiss-help").click();
          await page.locator("#row").fill("70");
          await page.locator("#col").fill("5");
          await page.locator("#inspect-submit").click();
          await expect(page.locator(".native-index")).toContainText("[70, 5]");
        }
        await expect(page.locator("#overview-image")).toBeVisible();
        const before = await page
          .locator("#overview-viewport")
          .getAttribute("style");
        await page.locator("#zoom-in").click();
        await ready();
        await expect
          .poll(() => page.locator("#overview-viewport").getAttribute("style"))
          .not.toBe(before);
        await page.locator("#fit").click();
        await ready();
        checks.push("Bounded overview paints and viewport box follows zoom");
        await page.locator("#left-canvas").focus();
        await page.keyboard.press("ArrowDown");
        await expect(page.locator(".native-index")).toContainText("[71, 5]");
        checks.push(
          "Keyboard arrow inspection preserves exact native coordinates",
        );
        await page.locator(".starting-points summary").click();
        await expect(page.locator(".example-card")).toHaveCount(3);
        await page.locator(".example-card").first().click();
        await ready();
        assert(page.url().includes("#wa=2"));
        await expect(page.locator("#region-r0")).toHaveValue("64");
        checks.push("Verified finding opens an actual source-bound bookmark");
        await page.locator(".starting-points summary").click();
      }
      if (["all", "light", "dark"].includes(section)) {
        if (section !== "all") {
          await page.locator("#dismiss-help").click();
          await page.locator(".starting-points summary").click();
          await page.locator(".example-card").first().click();
          await ready();
          await page.locator(".starting-points summary").click();
        }
        await page.locator("#workspace").scrollIntoViewIfNeeded();
        if (section !== "dark") await shot("desktop-exploration-light");
        const gradient = await page
          .locator("#left-gradient")
          .evaluate((e) => getComputedStyle(e).backgroundImage);
        await page.locator("#theme-toggle").click();
        await expect(page.locator("html")).toHaveAttribute(
          "data-theme",
          "dark",
        );
        assert.equal(
          await page
            .locator("#left-gradient")
            .evaluate((e) => getComputedStyle(e).backgroundImage),
          gradient,
        );
        await page.locator("#workspace").scrollIntoViewIfNeeded();
        if (section !== "light") await shot("desktop-exploration-dark");
        if (section === "dark") {
          fs.writeFileSync(
            out + "/legend-render.json",
            JSON.stringify(
              await page.locator("#left-gradient").evaluate((e) => ({
                rect: e.getBoundingClientRect().toJSON(),
                background: getComputedStyle(e).backgroundImage,
                height: getComputedStyle(e).height,
                visibility: getComputedStyle(e).visibility,
                opacity: getComputedStyle(e).opacity,
              })),
              null,
              2,
            ),
          );
          await page
            .locator("#left-gradient")
            .screenshot({ path: out + "/legend-dark.png" });
        }
        checks.push("Dark theme changes chrome, preserves numerical palette");
      }
    }
    if (phase === "mobile") {
      assert(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
      await shot("mobile-first-visit");
      await page.locator("#dismiss-help").click();
      await page.locator("#nav-view").click();
      await expect(page.locator("#left-canvas")).toBeFocused();
      assert(!page.url().includes("#workspace"));
      await expect(page.locator("#error")).toBeEmpty();
      await page.locator("#show-right").click();
      await expect(page.locator("#right-canvas")).toBeVisible();
      await expect(page.locator("#left-canvas")).toBeHidden();
      await page.locator("#show-left").click();
      assert.equal(
        await page.locator('.openseadragon-canvas[tabindex="0"]').count(),
        0,
      );
      checks.push(
        "Mobile jump navigation preserves bookmarks; compact A/B switch and single canvas keyboard stop work",
      );
      await expect(page.locator("#tensor-browser")).not.toHaveAttribute(
        "open",
        "",
      );
      await page.locator("#tensor-browser > summary").click();
      await page
        .locator("#tensor-search")
        .fill("model.layers.0.input_layernorm.weight");
      await page.locator(".tensor-option").first().click();
      await ready();
      await page.locator("#row").fill("0");
      await page.locator("#col").fill("17");
      await page.locator("#inspect-submit").click();
      await expect(page.locator(".native-index")).toContainText("[17]");
      checks.push(
        "Mobile tensor picker and vector inspection use one native row with a one-dimensional source index",
      );
      await page.locator("#theme-toggle").click();
      await page.locator("#workspace").scrollIntoViewIfNeeded();
      assert(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      );
      await shot("mobile-vector-dark");
      await page.locator(".workspace-tools > summary").click();
      await page.locator("#region-use-cell").click();
      await expect(page.locator("#region-c0")).toHaveValue("17");
      await page.locator(".workspace-tools").scrollIntoViewIfNeeded();
      await shot("mobile-tools-dark");
      checks.push(
        "Mobile region tools remain reachable with no horizontal page overflow",
      );
    }
    if (phase === "qwen") {
      const observed = require("./fixtures/ui-polish-qwen-observations.json");
      assert.equal(model.source_identity, observed.source_identity);
      assert.equal(model.revision, observed.revision);
      await page.locator("#dismiss-help").click();
      await page.locator(".starting-points summary").click();
      await expect(page.locator(".example-card")).toHaveCount(3);
      await page.locator(".example-card").first().click();
      await ready();
      await page.locator(".starting-points summary").click();
      await expect(page.locator("#region-r0")).toHaveValue("128");
      await page.locator("#row").fill("128");
      await page.locator("#col").fill("1");
      await page.locator("#inspect-submit").click();
      await expect(page.locator(".native-index")).toContainText("[128, 1]");
      await page.locator("#left-canvas").scrollIntoViewIfNeeded();
      const point = await page.evaluate(async () => {
        const __atlasController =
          typeof state === "undefined" ? await import("/app.js") : { state };

        const v = __atlasController.state.viewers.left,
          item = v.world.getItemAt(0),
          p = v.viewport.pixelFromPoint(
            item.imageToViewportCoordinates(6.5, 129.5),
            true,
          ),
          r = document.getElementById("left-canvas").getBoundingClientRect();
        return { x: r.left + p.x, y: r.top + p.y };
      });
      await page.mouse.move(point.x, point.y);
      await expect(page.locator("#hover-readout")).toContainText(
        "Row 129 · column 6",
      );
      await expect(page.locator("#hover-readout")).toContainText(
        "Query head 1",
      );
      const scalar = await (
        await context.request.get(
          base +
            "/api/inspect?" +
            new URLSearchParams({
              tensor: model.catalog.find(
                (t) => t.name === observed.records[0].name,
              ).id,
              row: 129,
              col: 6,
              left: "tensor_linear",
              right: "tensor_magnitude",
            }),
        )
      ).json();
      assert.equal(Number(scalar.raw_exact), observed.records[0].values[14]);
      await expect(page.locator("#hover-readout")).toContainText(
        scalar.raw_exact,
      );
      await expect(page.locator(".native-index")).toContainText("[128, 1]");
      await expect(page.locator("#overview-image")).toBeVisible();
      assert(
        await page
          .locator("#overview-image")
          .evaluate((e) => e.naturalWidth <= 256 && e.naturalHeight <= 256),
      );
      await page.locator("#workspace").scrollIntoViewIfNeeded();
      await shot("qwen-native-region");
      checks.push(
        "Current Qwen source has three observed bookmarks; native query-head hover matches retained BF16 evidence and independent source lookup; bounded overview paints; inference stays hidden",
      );
    }
    if (phase === "inference") {
      await expect(page.locator("#inference-panel")).toBeVisible();
      await page.locator("#inference-panel > summary").click();
      await expect(page.locator("#infer-status")).toContainText("Ready");
      await page.locator("#infer-check").click();
      await expect(page.locator("#infer-start")).toBeEnabled();
      await page.locator("#infer-limit").fill("1");
      await page.locator("#infer-mode").selectOption("step");
      const accepted = page.waitForResponse(
        (r) =>
          r.url() === base + "/api/inference/start" &&
          r.request().method() === "POST",
      );
      await page.locator("#infer-start").click();
      const response = await accepted;
      assert.equal(response.status(), 202);
      owner = (await response.json()).session;
      await expect(page.locator("#infer-timing")).toContainText("Elapsed", {
        timeout: 5000,
      });
      await page.locator("#infer-status").scrollIntoViewIfNeeded();
      await shot("inference-waiting");
      await expect(page.locator("#infer-status")).toContainText(
        "compute complete",
        { timeout: 100000 },
      );
      const result = await (
        await context.request.post(base + "/api/inference/poll", {
          data: { session: owner },
          headers: { "X-Atlas-Local": "1" },
        })
      ).json();
      assert.equal(result.worker_alive, false);
      assert.deepEqual(result.details.baseline, result.details.edited);
      assert(
        result.steps.every((s) => s.candidates.every((c) => c.delta === 0)),
      );
      await page.locator("#infer-step").click();
      await shot("inference-completed");
      checks.push(
        "One real CPU empty-edit comparison completes with baseline/edited parity and observed stage/elapsed labels; owned worker reaped",
      );
      fs.writeFileSync(
        out + "/inference-summary.json",
        JSON.stringify(
          {
            status: result.status,
            steps: result.steps.length,
            worker_alive: result.worker_alive,
            peak_worker_rss_mib: result.peak_worker_rss_mib,
            load_ms: result.details.load_ms,
          },
          null,
          2,
        ),
      );
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(
      out + "/ui-polish-browser.json",
      JSON.stringify(
        {
          status: "PASS",
          phase,
          section: process.env.ATLAS_DESKTOP_SECTION || null,
          checks,
          page_errors: errors,
          source_identity: model.source_identity,
          model_identity: model.model_identity,
          browser: browser.version(),
          screenshot_scope:
            "Viewport-sized desktop/mobile emulation; physical phone untested",
        },
        null,
        2,
      ),
    );
    console.log(JSON.stringify({ status: "PASS", phase, checks }));
  } catch (e) {
    fs.writeFileSync(
      out + "/failure.json",
      JSON.stringify(
        { status: "FAIL", phase, error: e.stack, checks, page_errors: errors },
        null,
        2,
      ),
    );
    throw e;
  } finally {
    if (owner)
      await context.request
        .post(base + "/api/inference/reset", {
          data: { session: owner },
          headers: { "X-Atlas-Local": "1" },
        })
        .catch(() => {});
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
