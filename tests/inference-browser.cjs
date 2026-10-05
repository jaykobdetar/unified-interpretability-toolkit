"use strict";
// Explicit opt-in: installed Playwright/Chromium, fresh context, loopback only.
const { chromium, expect } = require("playwright/test");
const fs = require("fs"),
  assert = require("assert/strict");
const base = process.env.ATLAS_TEST_URL || "http://127.0.0.1:8796";
assert(/^http:\/\/127\.0\.0\.1:\d+$/.test(base));
const out = process.env.ATLAS_EVIDENCE_DIR || "results";
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
  });
  await context.route("**/*", (route) =>
    route
      .request()
      .url()
      .startsWith(base + "/")
      ? route.continue()
      : route.abort(),
  );
  await context.tracing.start({ screenshots: false, snapshots: false });
  const page = await context.newPage(),
    errors = [],
    requests = [],
    checks = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("request", (r) => requests.push(r.url()));
  const ui = (id) => page.locator("#infer-" + id),
    text = (id) => ui(id).innerText();
  const generate = async () => {
    const response = page.waitForResponse(
      (r) =>
        r.url() === base + "/api/inference/start" &&
        r.request().method() === "POST",
    );
    await ui("start").click();
    return (await response).json();
  };
  const api = async (action, data, headers = {}) =>
    context.request.post(base + "/api/inference/" + action, {
      data,
      headers: { "X-Atlas-Local": "1", ...headers },
    });
  try {
    await page.goto(base);
    await page.locator("#inference-panel > summary").click();
    await expect(ui("start")).toBeEnabled();
    await expect(page.locator("#left-resolution")).toContainText("cells");
    await ui("mode").selectOption("step");
    await ui("limit").fill("8");
    await ui("prompt").fill("The capital of France is");
    await generate();
    await expect(ui("status")).toContainText("compute complete", {
      timeout: 20000,
    });
    await expect(ui("status")).toContainText("shown 0 / 8");
    await ui("step").click();
    await expect(ui("alignment")).toContainText(
      "prefill consumed position 4 (token 314) → predicts token 260",
    );
    assert.equal(await text("output"), " the");
    await ui("step").click();
    await expect(ui("alignment")).toContainText(
      "decode consumed position 5 (token 260) → predicts token 3575",
    );
    await ui("index").fill("575");
    await expect(ui("value")).toContainText("activation[575] =");
    checks.push(
      "Actual prompt generation, manual steps, prefill/decode alignment, exact activation inspector",
    );
    await page.screenshot({
      path: out + "/live-inference-desktop.jpg",
      type: "jpeg",
      quality: 85,
      fullPage: false,
    });
    await ui("rate").selectOption("0.5");
    await ui("pause").click();
    await page.waitForTimeout(350);
    await ui("pause").click();
    await expect(ui("status")).toContainText("shown 2 / 8");
    await ui("rate").selectOption("8");
    await ui("pause").click();
    await expect(ui("status")).toContainText("shown 8 / 8", { timeout: 3000 });
    await expect(ui("status")).toContainText("finished");
    await expect(ui("step")).toBeDisabled();
    checks.push("Playback timing, pause/resume, speed change, terminal step");
    const beforeReplay = requests.filter((u) => u.endsWith("/start")).length;
    await ui("replay").click();
    await expect(ui("status")).toContainText("Recorded replay");
    await ui("step").click();
    assert.equal(await text("output"), " the");
    assert.equal(
      requests.filter((u) => u.endsWith("/start")).length,
      beforeReplay,
    );
    checks.push("Recorded replay performs no new generation");
    await ui("reset").click();
    await expect(ui("status")).toContainText("Ready");
    await expect(ui("output")).toContainText("will appear");
    await ui("limit").fill("32");
    const cancelOwner = await generate();
    await expect(ui("cancel")).toBeEnabled();
    await ui("cancel").click();
    await expect(ui("status")).toContainText("cancelled");
    const cancelled = await (
      await api("poll", { session: cancelOwner.session })
    ).json();
    assert.equal(cancelled.worker_alive, false);
    assert.equal(cancelled.status, "cancelled");
    checks.push(
      "Reset clears trace; cancel during model loading stops actual worker",
    );
    await ui("reset").click();
    await ui("prompt").fill("word ".repeat(200));
    await ui("start").click();
    await expect(ui("error")).toContainText("Prompt must encode", {
      timeout: 20000,
    });
    await expect(ui("start")).toBeEnabled();
    checks.push(
      "Token bound is enforced by real tokenizer and displayed as an error",
    );
    await ui("reset").click();
    await ui("prompt").fill("Hello world");
    await ui("layer").selectOption("29");
    await ui("limit").fill("4");
    const completedOwner = await generate();
    await expect(ui("status")).toContainText("compute complete", {
      timeout: 20000,
    });
    await ui("step").click();
    await expect(ui("alignment")).toContainText("Layer 29");
    checks.push("Recovery after invalid prompt; final layer capture");
    const metadata = await (
      await context.request.get(base + "/api/inference")
    ).json();
    assert.equal(metadata.queue_capacity, 0);
    assert.equal(typeof metadata.busy, "boolean");
    assert(!("session" in metadata));
    assert(!("steps" in metadata));
    assert(!("prompt" in metadata));
    const completed = await (
      await api("poll", { session: completedOwner.session })
    ).json();
    assert.equal(completed.worker_alive, false);
    assert.equal(completed.steps.length, 4);
    assert(completed.peak_worker_rss_mib < 1536);
    assert(completed.minimum_available_gib >= 3);
    await page.setViewportSize({ width: 390, height: 844 });
    await ui("step").click();
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    await page.screenshot({
      path: out + "/live-inference-mobile.jpg",
      type: "jpeg",
      quality: 85,
      fullPage: true,
    });
    checks.push(
      "390×844 mobile layout, controls, and activation display without horizontal overflow",
    );
    // Read-only weight inspection remains usable after inference.
    await page.locator("#inspect-submit").click();
    await expect(page.locator(".raw-value")).not.toBeEmpty();
    checks.push("Original BF16 inspector remains functional");
    const oldId = completed.session;
    await ui("reset").click();
    const fresh = await api("start", {
      prompt: "Hello world",
      max_new_tokens: 32,
      layer: 0,
    });
    assert.equal(fresh.status(), 202);
    const current = await fresh.json();
    const duplicate = await api("start", {
      prompt: "Hello world",
      max_new_tokens: 1,
      layer: 0,
    });
    assert.equal(duplicate.status(), 400);
    const stale = await api("cancel", { session: oldId });
    assert.equal(stale.status(), 409);
    const stopped = await (
      await api("cancel", { session: current.session })
    ).json();
    assert.equal(stopped.worker_alive, false);
    checks.push(
      "One-session exclusion and stale cancel cannot stop a newer session",
    );
    const origin = await api(
      "start",
      { prompt: "x" },
      { Origin: "https://example.invalid" },
    );
    assert.equal(origin.status(), 400);
    const invalid = await api("start", { prompt: "x", max_new_tokens: 33 });
    assert.equal(invalid.status(), 400);
    checks.push("Local-origin contract and bounded token request");
    assert.deepEqual(errors, []);
    assert(requests.every((u) => u.startsWith(base + "/")));
    const evidence = {
      status: "PASS",
      browser: browser.version(),
      checks,
      page_errors: errors,
      external_page_requests: 0,
      resource_sample: {
        peak_worker_rss_mib: completed.peak_worker_rss_mib,
        minimum_available_gib: completed.minimum_available_gib,
      },
      generated_ids: completed.steps.map((s) => s.token_id),
      screenshot_scope: "Desktop and mobile emulation; physical phone untested",
    };
    fs.writeFileSync(
      out + "/inference-browser.json",
      JSON.stringify(evidence, null, 2) + "\n",
    );
    console.log(JSON.stringify(evidence, null, 2));
  } finally {
    try {
      await context.tracing.stop({
        path: out + "/inference-browser-trace.zip",
      });
    } finally {
      await browser.close();
    }
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
