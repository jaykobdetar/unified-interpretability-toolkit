"use strict";
const { chromium, expect } = require("playwright/test"),
  { spawn } = require("node:child_process"),
  net = require("node:net"),
  fs = require("node:fs"),
  path = require("node:path"),
  assert = require("node:assert/strict"),
  crypto = require("node:crypto");
const root = path.resolve(__dirname, ".."),
  out = process.env.ATLAS_EVIDENCE_DIR;
assert(out && process.env.ATLAS_CHROMIUM);
const held = require("./fixtures/held-bookmark-v2.json"),
  binary = path.join(root, "target/release/weight-atlas-rust"),
  fixture = path.join(root, "fixtures/tiny-bf16");
const sha = (value) => crypto.createHash("sha256").update(value).digest("hex");
assert.equal(
  sha(fs.readFileSync(path.join(fixture, "tiny.safetensors"))),
  held.fixture_sha256,
);
fs.mkdirSync(out, { recursive: true });
let server, browser, port;
const errors = [],
  requests = [],
  checks = [];
const wait = async (p, ms) => {
  let id;
  try {
    return await Promise.race([
      p,
      new Promise((_, reject) => {
        id = setTimeout(() => reject(Error("Owned fixture deadline")), ms);
      }),
    ]);
  } finally {
    clearTimeout(id);
  }
};
const command = (args) =>
  new Promise((resolve, reject) => {
    const child = spawn(binary, args, { cwd: root });
    const data = [];
    child.stdout.on("data", (x) => data.push(x));
    child.stderr.on("data", (x) => data.push(x));
    child.on("error", reject);
    child.on("close", (code) => {
      fs.writeFileSync(path.join(out, "calibrate.log"), Buffer.concat(data));
      code === 0 ? resolve() : reject(Error("Native calibration failed"));
    });
  });
(async () => {
  try {
    port = await new Promise((resolve, reject) => {
      const s = net.createServer();
      s.on("error", reject);
      s.listen(0, "127.0.0.1", () => {
        const p = s.address().port;
        s.close(() => resolve(p));
      });
    });
    const base = "http://127.0.0.1:" + port,
      cache = path.join(out, "cache");
    await command([
      "calibrate",
      "--model",
      fixture,
      "--cache",
      cache,
      "--revision",
      held.revision,
    ]);
    server = spawn(
      binary,
      [
        "serve",
        "--model",
        fixture,
        "--cache",
        cache,
        "--revision",
        held.revision,
        "--name",
        "26-value fixture",
        "--port",
        String(port),
      ],
      { cwd: root },
    );
    const log = fs.createWriteStream(path.join(out, "server.log"));
    server.stdout.pipe(log, { end: false });
    server.stderr.pipe(log, { end: false });
    server.once("exit", () => log.end());
    await wait(
      new Promise((resolve, reject) => {
        let raw = "";
        server.stdout.on("data", (b) => {
          raw += b;
          for (const line of raw.split("\n")) {
            try {
              if (JSON.parse(line).listening === base) resolve();
            } catch {}
          }
        });
        server.on("error", reject);
        server.once("exit", () =>
          reject(Error("Native server exited before readiness")),
        );
      }),
      5000,
    );
    browser = await chromium.launch({
      headless: true,
      chromiumSandbox: true,
      executablePath: process.env.ATLAS_CHROMIUM,
      args: ["--renderer-process-limit=1", "--disable-gpu"],
    });
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 },
    });
    await context.route("**/*", (r) =>
      r
        .request()
        .url()
        .startsWith(base + "/")
        ? r.continue()
        : r.abort(),
    );
    if (process.env.ATLAS_BENIGN_MUTANT_FILE) {
      const mutant = fs.readFileSync(process.env.ATLAS_BENIGN_MUTANT_FILE);
      assert.equal(sha(mutant), process.env.ATLAS_BENIGN_MUTANT_SHA256);
      await context.route(base + "/viewer.js", async (route) => {
        const response = await route.fetch();
        await route.fulfill({ response, body: mutant });
      });
    }
    const page = await context.newPage();
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("request", (r) =>
      requests.push({ method: r.method(), path: new URL(r.url()).pathname }),
    );
    await page.goto(base + held.fragment);
    await expect(page.locator("#comparison")).toHaveAttribute(
      "aria-busy",
      "false",
    );
    const model = await (await context.request.get(base + "/api/model")).json();
    assert.equal(model.source_identity, held.source_identity);
    assert.equal(model.model_identity, held.model_identity);
    assert.equal(model.revision, held.revision);
    assert.equal(await page.evaluate(() => state.tensor.name), "matrix");
    await expect(page.locator("#left-rule")).toHaveValue("global_linear");
    await expect(page.locator("#right-rule")).toHaveValue("tensor_asinh");
    for (const [name, value] of [
      ["r0", "0"],
      ["c0", "0"],
      ["r1", "1"],
      ["c1", "1"],
    ])
      await expect(page.locator("#region-" + name)).toHaveValue(value);
    const parsed = await page.evaluate(
        (fragment) => AtlasTools.parseBookmark(fragment),
        held.fragment,
      ),
      actual = await page.evaluate(() => {
        const viewer = state.viewers.left,
          b = viewer.viewport.getBounds(true),
          item = viewer.world.getItemAt(0),
          a = item.viewportToImageCoordinates(b.x, b.y, true),
          z = item.viewportToImageCoordinates(
            b.x + b.width,
            b.y + b.height,
            true,
          );
        return [a.x, a.y, z.x - a.x, z.y - a.y];
      });
    assert(actual.every((v, i) => Math.abs(v - parsed.viewport[i]) < 1e-7));
    checks.push(
      "Cold navigation accepts the held pre-format v2 source/model link and restores exact tensor, both rules, region and viewport",
    );
    await page.locator(".workspace-tools > summary").click();
    await page.locator("#bookmark-save").click();
    assert.equal(new URL(page.url()).hash, held.expected_resaved_fragment);
    await page.reload();
    await expect(page.locator("#comparison")).toHaveAttribute(
      "aria-busy",
      "false",
    );
    assert.equal(new URL(page.url()).hash, held.expected_resaved_fragment);
    checks.push(
      "Bookmark save and fresh page reload preserve the exact starting resaved fragment, including its signed zero",
    );
    await page.locator("#row").fill("1");
    await page.locator("#col").fill("1");
    await page.locator("#inspect-submit").click();
    await expect(page.locator(".native-index")).toContainText("[1, 1]");
    assert.deepEqual(errors, []);
    assert(requests.every((r) => r.method === "GET"));
    await page.screenshot({ path: path.join(out, "held-link.png") });
    fs.writeFileSync(
      path.join(out, "held-link-result.json"),
      JSON.stringify(
        {
          status: "PASS",
          checks,
          held_fragment: held.fragment,
          requests,
          page_errors: errors,
          scope:
            "Actual native fixture and sandboxed browser; no head receipt or inference claim",
        },
        null,
        2,
      ) + "\n",
    );
  } catch (e) {
    fs.writeFileSync(
      path.join(out, "failure.json"),
      JSON.stringify({ error: e.stack, checks, page_errors: errors }, null, 2),
    );
    throw e;
  } finally {
    if (browser) await browser.close();
    if (server && server.exitCode === null) {
      const exit = new Promise((resolve) => server.once("exit", resolve));
      server.kill("SIGTERM");
      await wait(exit, 5000);
    }
    fs.writeFileSync(
      path.join(out, "cleanup.json"),
      JSON.stringify(
        {
          server_reaped:
            !server || server.exitCode !== null || server.signalCode !== null,
        },
        null,
        2,
      ),
    );
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
