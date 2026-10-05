"use strict";
// Benign opt-in comparison controls, one real inference session at a time.
const { chromium, expect } = require("playwright/test");
const fs = require("fs"),
  assert = require("assert/strict");
const base = process.env.ATLAS_TEST_URL || "http://127.0.0.1:8796";
assert(/^http:\/\/127\.0\.0\.1:\d+$/.test(base));
const out = process.env.ATLAS_EVIDENCE_DIR || "results/edits-browser";
fs.mkdirSync(out, { recursive: true });
(async () => {
  const browser = await chromium.launch({
    headless: true,
    chromiumSandbox: true,
    executablePath: process.env.ATLAS_CHROMIUM,
    args: ["--renderer-process-limit=1", "--disable-gpu"],
  });
  const context = await browser.newContext({
      viewport: { width: 1440, height: 1100 },
    }),
    errors = [],
    checks = [];
  await context.route("**/*", (route) =>
    route
      .request()
      .url()
      .startsWith(base + "/")
      ? route.continue()
      : route.abort(),
  );
  const page = await context.newPage();
  page.on("pageerror", (error) => errors.push(error.message));
  const el = (id) => page.locator("#infer-" + id);
  const action = async (name, data) =>
    (
      await context.request.post(base + "/api/inference/" + name, {
        data,
        headers: { "X-Atlas-Local": "1" },
      })
    ).json();
  const views = [];
  const sample = async (label) =>
    views.push({
      label,
      ...(await page.evaluate(() => ({
        left: state.viewers.left.viewport.getBounds(true),
        right: state.viewers.right.viewport.getBounds(true),
        leftZoom: state.viewers.left.viewport.getZoom(true),
        rightZoom: state.viewers.right.viewport.getZoom(true),
        magnification:
          document.getElementById("left-magnification").textContent,
      }))),
    });
  const start = async () => {
    const response = page.waitForResponse(
      (r) =>
        r.url() === base + "/api/inference/start" &&
        r.request().method() === "POST",
    );
    await el("start").click();
    return (await response).json();
  };
  try {
    await page.goto(base);
    await page.locator("#inference-panel > summary").click();
    await expect(el("start")).toBeEnabled();
    await page.waitForFunction(
      () => document.body.dataset.bothViewsReadyMs,
      {},
      { timeout: 10000 },
    );
    await sample("startup");
    await el("mode").selectOption("step");
    await el("limit").fill("4");
    await page.locator("#row").fill("70");
    await page.locator("#col").fill("5");
    await page.locator("#inspect-submit").click();
    await expect(el("selection")).toContainText("native [70, 5]");
    await expect(el("head")).toBeEnabled();
    await sample("inspected");
    await el("head").click();
    await expect(el("edits")).toContainText("rows [64,128)");
    await sample("draft");
    const editedOwner = await start();
    await expect(el("status")).toContainText("compute complete", {
      timeout: 30000,
    });
    await el("step").click();
    await expect(el("score-context")).toContainText("Matched consumed prefix");
    assert((await el("scores").locator("tr").count()) >= 5);
    await expect(el("baseline-ids")).toContainText("Token IDs:");
    await expect(el("edited-ids")).toContainText("Token IDs:");
    const edited = await action("poll", { session: editedOwner.session });
    assert.equal(edited.worker_alive, false);
    assert.equal(edited.details.edits[0].start, 64);
    assert(
      edited.steps[0].candidates.every(
        (c) =>
          Number.isFinite(c.baseline_logit) &&
          Number.isFinite(c.edited_logit) &&
          c.delta === c.edited_logit - c.baseline_logit,
      ),
    );
    checks.push(
      "Inspected native row maps to complete query-head rows [64,128); paired scores/IDs and accepted inputs shown",
    );
    await sample("generated");
    const effect = Math.max(
      ...edited.steps[0].candidates.map((c) => Math.abs(c.delta)),
    );
    await el("clear-edits").click();
    await expect(el("run-inputs")).toContainText('"start": 64');
    await sample("cleared");
    await page.locator(".inference").scrollIntoViewIfNeeded();
    await page.screenshot({
      path: out + "/comparison-desktop.jpg",
      type: "jpeg",
      quality: 85,
    });
    await sample("after-desktop-screenshot");
    assert.equal(views.at(-1).leftZoom, views.at(-2).leftZoom);
    await page.locator("#workspace").scrollIntoViewIfNeeded();
    await page.screenshot({
      path: out + "/source-viewer-desktop.jpg",
      type: "jpeg",
      quality: 85,
    });
    fs.writeFileSync(out + "/viewports.json", JSON.stringify(views, null, 2));
    await el("reset").click();
    await expect(el("status")).toContainText("Ready");
    const emptyOwner = await start();
    await expect(el("status")).toContainText("compute complete", {
      timeout: 30000,
    });
    const empty = await action("poll", { session: emptyOwner.session });
    assert.deepEqual(empty.details.baseline, empty.details.edited);
    assert(
      empty.steps.every(
        (s) =>
          s.alignment === "matched_prefix" &&
          s.candidates.every((c) => c.delta === 0),
      ),
    );
    await el("step").click();
    checks.push(
      "Empty edit comparison has exact token/text/top-union parity after previous edited worker was reaped",
    );
    await page.setViewportSize({ width: 390, height: 844 });
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    await el("status").scrollIntoViewIfNeeded();
    await page.screenshot({
      path: out + "/comparison-mobile.jpg",
      type: "jpeg",
      quality: 85,
    });
    await page.locator(".edit-draft").scrollIntoViewIfNeeded();
    await page.screenshot({
      path: out + "/edit-draft-mobile.jpg",
      type: "jpeg",
      quality: 85,
    });
    checks.push(
      "390×844 viewport fits native editor and paired outputs without page overflow",
    );
    await el("reset").click();
    assert.deepEqual(errors, []);
    const result = {
      status: "PASS",
      checks,
      page_errors: errors,
      first_step_top_union_max_abs_delta: effect,
      fixture: "capital-france-public-synthetic",
      query_head: 1,
      empty_edit_parity: true,
      peak_worker_rss_mib: Math.max(
        edited.peak_worker_rss_mib,
        empty.peak_worker_rss_mib,
      ),
      physical_phone_tested: false,
    };
    fs.writeFileSync(
      out + "/comparison-browser.json",
      JSON.stringify(result, null, 2) + "\n",
    );
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
