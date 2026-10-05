"use strict";
const { chromium, expect } = require("playwright/test"),
  assert = require("node:assert/strict"),
  fs = require("node:fs"),
  path = require("node:path"),
  crypto = require("node:crypto");
const base = process.env.ATLAS_TEST_URL,
  out = process.env.ATLAS_EVIDENCE_DIR;
assert(/^http:\/\/127\.0\.0\.1:\d+$/.test(base));
assert(out && process.env.ATLAS_CHROMIUM);
fs.mkdirSync(out, { recursive: true });
const sha = (b) => crypto.createHash("sha256").update(b).digest("hex");
let browser;
const checks = [],
  requests = [],
  errors = [];
(async () => {
  try {
    browser = await chromium.launch({
      headless: true,
      chromiumSandbox: true,
      executablePath: process.env.ATLAS_CHROMIUM,
      args: ["--renderer-process-limit=1", "--disable-gpu"],
    });
    const context = await browser.newContext({
      viewport: { width: 1280, height: 900 },
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
    const page = await context.newPage();
    page.setDefaultTimeout(8000);
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("request", (r) =>
      requests.push({ method: r.method(), path: new URL(r.url()).pathname }),
    );
    const el = (id) => page.locator("#infer-" + id);
    const open = async () => {
      await expect(page.locator("#inference-panel")).toBeVisible();
      await page.locator("#inference-panel > summary").click();
      await el("import-file").scrollIntoViewIfNeeded();
    };
    await page.goto(base);
    await open();
    const initial = await (
      await context.request.get(base + "/api/inference")
    ).json();
    assert.equal(initial.model, "HuggingFaceTB/SmolLM2-135M");
    assert.equal(initial.busy, false);
    assert(!requests.some((r) => r.path === "/inference-import.js"));
    await el("prompt").fill("Current browser inputs must remain here.");
    await el("layer").selectOption("3");
    await el("limit").fill("2");
    const snapshot = async () => ({
      prompt: await el("prompt").inputValue(),
      layer: await el("layer").inputValue(),
      limit: await el("limit").inputValue(),
      accepted: await el("run-inputs").textContent(),
    });
    const before = await snapshot();
    const codec = require("../web/inference.js"),
      req = {
        layer: 7,
        activation_site: "mlp",
        max_new_tokens: 1,
        prompt: "Historical synthetic archive prompt.",
        edits: [],
        source_model: {},
      },
      record = codec.experimentRecord(
        req,
        {
          status: "complete",
          worker_alive: false,
          steps: [
            {
              index: 0,
              activation: [0.5],
              token_id: 2,
              top_logits: [{ id: 2, value: 1 }],
            },
          ],
        },
        { includePrompt: true },
      );
    const archive = path.join(out, "historical-archive.json");
    fs.writeFileSync(archive, JSON.stringify(record));
    await expect(el("import-prompts")).not.toBeChecked();
    const chooser = page.waitForEvent("filechooser");
    await el("import-file").click();
    await (await chooser).setFiles(archive);
    const codecResponse = page.waitForResponse(
      (r) => r.url() === base + "/inference-import.js",
    );
    await el("import-log").click();
    const response = await codecResponse;
    assert.equal(response.status(), 200);
    assert.equal(
      sha(await response.body()),
      sha(fs.readFileSync("web/inference-import.js")),
    );
    await expect(el("log-status")).toContainText("1 / 8");
    assert.deepEqual(await snapshot(), before);
    await expect(el("import-status")).toContainText(
      "Historical provenance is unverified",
    );
    checks.push(
      "Actual file chooser imports one historical record through the trusted lazy codec; current inputs and worker ownership stay unchanged",
    );
    const downloaded = page.waitForEvent("download");
    await el("export-log").click();
    const download = await downloaded,
      target = path.join(out, "redacted-log.json");
    await download.saveAs(target);
    const raw = fs.readFileSync(target);
    assert(raw.length <= 1048576);
    const exported = JSON.parse(raw);
    assert.equal(exported.records[0].privacy.prompt_included, false);
    assert(!JSON.stringify(exported).includes(req.prompt));
    checks.push(
      "Default prompt redaction is preserved in the actual browser download",
    );
    await el("persist").check();
    await expect(el("durable-status")).toContainText("Saved");
    await page.reload();
    await open();
    await expect(el("log-status")).toContainText("0 / 8");
    await expect(el("persist")).not.toBeChecked();
    const reloadInputs = await snapshot();
    await el("restore-log").click();
    await expect(el("log-status")).toContainText("1 / 8");
    assert.deepEqual(await snapshot(), reloadInputs);
    checks.push(
      "Browser save survives reload, automatic restoration stays off, and explicit restore appends without adopting run inputs",
    );
    const final = await (
      await context.request.get(base + "/api/inference")
    ).json();
    assert.equal(final.busy, false);
    assert.deepEqual(errors, []);
    assert(requests.every((r) => r.method === "GET"));
    await page.screenshot({ path: path.join(out, "archive.png") });
    fs.writeFileSync(
      path.join(out, "archive-result.json"),
      JSON.stringify(
        {
          status: "PASS",
          checks,
          requests,
          page_errors: errors,
          download_bytes: raw.length,
          download_sha256: sha(raw),
          actual_coordinator: true,
          model_execution: false,
          head_binding_claim: false,
          native_file_dialog_permission_qualified: false,
        },
        null,
        2,
      ) + "\n",
    );
  } catch (e) {
    fs.writeFileSync(
      path.join(out, "failure.json"),
      JSON.stringify(
        { error: e.stack, checks, requests, page_errors: errors },
        null,
        2,
      ),
    );
    throw e;
  } finally {
    if (browser) await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
