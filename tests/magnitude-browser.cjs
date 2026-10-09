"use strict";
// Guarded, loopback-only static viewer QA. Never starts or calls inference.
const { chromium, expect } = require("playwright/test");
const fs = require("fs"),
  assert = require("assert/strict");
const base = process.env.ATLAS_TEST_URL || "http://127.0.0.1:8798";
assert(/^http:\/\/127\.0\.0\.1:\d+$/.test(base));
const out = process.env.ATLAS_EVIDENCE_DIR || "results/magnitude-browser";
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
    requests = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("request", (r) => requests.push(r.url()));
  const ready = async () => {
    await expect(page.locator("#comparison")).toHaveAttribute(
      "aria-busy",
      "false",
    );
    await page.waitForFunction(
      (__atlasController) => {
        return ["left", "right"].every((s) =>
          __atlasController.state.viewers[s]?.world
            .getItemAt(0)
            ?.getFullyLoaded(),
        );
      },
      await page.evaluateHandle(async () => {
        const __atlasController =
          typeof state === "undefined" ? await import("/app.js") : { state };
        return __atlasController;
      }),
    );
  };
  try {
    await page.goto(base);
    await ready();
    // Select the exact full Qwen attention matrix by native tensor name.
    const name =
      process.env.ATLAS_TEST_TENSOR || "model.layers.0.self_attn.q_proj.weight";
    await page
      .getByRole("button", { name: "Select " + name, exact: true })
      .click();
    await ready();
    await page.locator("#left-rule").selectOption("tensor_magnitude");
    await ready();
    await page.locator("#right-rule").selectOption("tensor_linear");
    await ready();
    await page.locator("#fit").click();
    await ready();
    const geometry = await page.evaluate(async () => {
      const __atlasController =
        typeof state === "undefined" ? await import("/app.js") : { state };
      return {
        shape: __atlasController.state.current.tensor.shape,
        left: __atlasController.state.viewers.left.viewport.getBounds(true),
        right: __atlasController.state.viewers.right.viewport.getBounds(true),
        levels: ["left", "right"].map(
          (s) => document.getElementById(s + "-canvas").dataset.loadedLevels,
        ),
      };
    });
    assert.deepEqual(geometry.shape, [4096, 4096]);
    for (const key of ["x", "y", "width", "height"])
      assert(Math.abs(geometry.left[key] - geometry.right[key]) < 1e-9);
    await expect(page.locator("#left-min")).toHaveText("0");
    await expect(page.locator("#left-mid")).toHaveText("");
    await expect(page.locator("#left-units")).toContainText(
      "mean absolute raw weight",
    );
    await expect(page.locator("#left-formula")).toContainText("abs(x)/M");
    await expect(page.locator("#left-scope")).toContainText(
      "complete original tensor",
    );
    await page.screenshot({
      path: out + "/qwen-magnitude-signed-desktop.png",
      fullPage: true,
    });
    await page.locator("#row").fill("103");
    await page.locator("#col").fill("207");
    await page.locator("#inspect-submit").click();
    await expect(page.locator(".raw-value")).not.toBeEmpty();
    const raw = await page.locator(".raw-value").innerText();
    await page.locator("#right-rule").selectOption("tensor_magnitude");
    await ready();
    await page.locator("#left-rule").selectOption("tensor_linear");
    await ready();
    await page.locator("#inspect-submit").click();
    await expect(page.locator(".raw-value")).toHaveText(raw);
    await expect(page.locator("#right-min")).toHaveText("0");
    await expect(page.locator("#left-mid")).toHaveText("0");
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator("#fit").click();
    await ready();
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    for (const id of [
      "left-rule",
      "right-rule",
      "right-formula",
      "right-units",
    ]) {
      const box = await page.locator("#" + id).boundingBox();
      assert(box.width > 0 && box.x >= 0 && box.x + box.width <= 390);
    }
    const mobile = [];
    for (const side of ["left", "right"]) {
      await page.locator("#" + side + "-canvas").scrollIntoViewIfNeeded();
      await page.waitForFunction(
        (side) => {
          const canvas = document.querySelector("#" + side + "-canvas canvas");
          return (
            canvas &&
            canvas.width > 0 &&
            canvas.height > 0 &&
            canvas
              .getContext("2d")
              .getImageData(
                Math.floor(canvas.width / 2),
                Math.floor(canvas.height / 2),
                1,
                1,
              ).data[3] === 255
          );
        },
        side,
        { timeout: 10000 },
      );
      mobile.push(
        await page.evaluate(async (side) => {
          const __atlasController =
            typeof state === "undefined" ? await import("/app.js") : { state };
          return {
            side,
            bounds:
              __atlasController.state.viewers[side].viewport.getBounds(true),
            levels: document.getElementById(side + "-canvas").dataset
              .loadedLevels,
            canvas: [
              ...document.querySelectorAll("#" + side + "-canvas canvas"),
            ].map((c) => ({
              width: c.width,
              height: c.height,
              center: [
                ...c
                  .getContext("2d")
                  .getImageData(
                    Math.floor(c.width / 2),
                    Math.floor(c.height / 2),
                    1,
                    1,
                  ).data,
              ],
            })),
          };
        }, side),
      );
      await page
        .locator("#" + side + "-canvas")
        .screenshot({ path: out + "/qwen-mobile-" + side + "-canvas.png" });
    }
    await page.screenshot({
      path: out + "/qwen-magnitude-mobile.png",
      fullPage: true,
    });
    assert.deepEqual(errors, []);
    assert(
      !requests.some((u) => /\/api\/inference\/(start|poll|cancel)/.test(u)),
    );
    const report = {
      passed: true,
      browser: browser.version(),
      geometry,
      mobile,
      raw_exact: raw,
      checks: [
        "4096×4096 full fitted view, magnitude and signed at identical geometry",
        "magnitude in both selectors, one-sided legend, formula, scale scope, native/pooled units",
        "raw scalar unchanged across rule swap",
        "390×844 layout without overflow and opaque painted canvas centers after scroll; screenshots require visual review",
      ],
      physical_phone_tested: false,
    };
    fs.writeFileSync(
      out + "/report.json",
      JSON.stringify(report, null, 2) + "\n",
    );
    console.log(JSON.stringify(report, null, 2));
  } finally {
    try {
      await context.tracing.stop({ path: out + "/trace.zip" });
    } finally {
      await browser.close();
    }
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
