"use strict";
// Normal feature acceptance on owned synthetic sources. No model execution or probes.
const { chromium, expect } = require("playwright/test"),
  assert = require("assert/strict"),
  fs = require("fs");
const urls = [process.env.ATLAS_SINGLE_URL, process.env.ATLAS_PAIR_URL];
for (const url of urls) assert(/^http:\/\/127\.0\.0\.1:\d+$/.test(url));
const out = process.env.ATLAS_EVIDENCE_DIR;
fs.mkdirSync(out, { recursive: true });
(async () => {
  let browser, context, page;
  const errors = [],
    requests = [],
    checks = [];
  let browserLifetimes = 0;
  const openBrowser = async () => {
    browser = await chromium.launch({
      headless: true,
      chromiumSandbox: true,
      executablePath: process.env.ATLAS_CHROMIUM,
      args: ["--renderer-process-limit=1", "--disable-gpu"],
    });
    context = await browser.newContext({
      viewport: { width: 1100, height: 850 },
    });
    await context.route("**/*", (r) =>
      urls.some((u) =>
        r
          .request()
          .url()
          .startsWith(u + "/"),
      )
        ? r.continue()
        : r.abort(),
    );
    page = await context.newPage();
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("request", (r) => requests.push(r.url()));
    browserLifetimes++;
  };
  await openBrowser();
  const singleReady = async () => {
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
  const pairReady = async () => {
    await page.waitForFunction(
      (current) => {
        return (
          current.view &&
          ["left", "right"].every((s) =>
            current.viewers[s]?.world.getItemAt(0)?.getFullyLoaded(),
          )
        );
      },
      await page.evaluateHandle(async () => {
        const current =
          typeof state === "undefined"
            ? (await import("/comparison.js")).state
            : state;
        return current;
      }),
    );
    await expect(page.locator("#fit")).toBeEnabled();
  };
  const painted = async (side, comparison = false) => {
    await page.locator("#" + side + "-canvas").scrollIntoViewIfNeeded();
    await page.waitForFunction((side) => {
      const c = document.querySelector("#" + side + "-canvas canvas");
      return (
        c &&
        c.width &&
        c.height &&
        c
          .getContext("2d")
          .getImageData(Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1)
          .data[3] === 255
      );
    }, side);
  };
  try {
    await page.goto(urls[0]);
    await singleReady();
    await page
      .getByRole("button", { name: "Select bf16", exact: true })
      .click();
    await singleReady();
    for (const side of ["left", "right"]) {
      await page.locator("#" + side + "-rule").selectOption("tensor_robust99");
      await singleReady();
      await expect(page.locator("#" + side + "-parameter")).toContainText(
        "Q99",
      );
      await page
        .locator("#" + side + "-rule")
        .selectOption("tensor_signed_percentile");
      await singleReady();
      await expect(page.locator("#" + side + "-units")).toContainText(
        "dimensionless",
      );
    }
    await page.locator("#left-rule").selectOption("tensor_robust99");
    await singleReady();
    await painted("left");
    await page
      .locator("#workspace")
      .screenshot({ path: out + "/rules-desktop.png" });
    checks.push(
      "Both panels select robust99 and signed percentile with correct formula/units; actual desktop paint",
    );
    await page
      .getByRole("button", { name: "Select float_edges", exact: true })
      .click();
    await expect(page.locator("#error")).toContainText("unavailable for F32");
    await expect(page.locator("#inspect-submit")).toBeEnabled();
    await page.locator("#col").fill("1");
    await page.locator("#row").fill("0");
    await page.locator("#inspect-submit").click();
    await expect(page.locator(".raw-value")).toHaveText("-0.0");
    await expect(page.locator("#inspection")).toContainText("F32");
    await expect(page.locator("#inspection")).toContainText(
      "exact absolute-value rank index",
    );
    checks.push(
      "F32 rank refusal preserves exact signed-zero inspection without replacement field",
    );
    await page.locator("#right-rule").selectOption("tensor_linear");
    await singleReady();
    await page
      .getByRole("button", { name: "Select half_all_finite", exact: true })
      .click();
    await singleReady();
    await page.locator("#col").fill("0");
    await page.locator("#row").fill("0");
    await page.locator("#inspect-submit").click();
    await expect(page.locator("#inspection")).toContainText("Original F16");
    checks.push(
      "F16 native scalar and rendering are available after switching dtype",
    );
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator("#fit").click();
    await singleReady();
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    for (const side of ["left", "right"]) {
      await page.locator("#show-" + side).click();
      await expect(page.locator("#comparison")).toHaveAttribute(
        "data-mobile-side",
        side,
      );
      await expect(page.locator("#show-" + side)).toHaveAttribute(
        "aria-pressed",
        "true",
      );
      await expect(page.locator("#" + side + "-panel")).toBeVisible();
      await expect(
        page.locator("#" + (side === "left" ? "right" : "left") + "-panel"),
      ).toBeHidden();
      await painted(side);
      await page
        .locator("#" + side + "-panel")
        .screenshot({ path: out + "/rules-mobile-" + side + ".png" });
    }
    checks.push(
      "390px rules layout fits viewport; A/B selection shows and paints each panel while hiding the inactive panel",
    );
    // These are independent viewers: no assertion depends on shared browser history.
    // Reap the first browser before allocating the second; keep all checks/caps.
    await browser.close();
    await openBrowser();
    await page.goto(urls[1]);
    await page.setViewportSize({ width: 1100, height: 850 });
    await pairReady();
    const moduleBoundary = await page.evaluate(async () => ({
      scriptType: document.querySelector('script[src="/comparison.js"]').type,
      globalState: typeof state,
      exportedViewReady: !!(await import("/comparison.js")).state.view,
    }));
    assert.deepEqual(moduleBoundary, {
      scriptType: "module",
      globalState: "undefined",
      exportedViewReady: true,
    });
    const matrix = await page.evaluate(async () => {
      const current =
        typeof state === "undefined"
          ? (await import("/comparison.js")).state
          : state;
      return current.model.catalog.find((p) => p.name === "matrix").id;
    });
    await page.locator("#tensor").selectOption(String(matrix));
    await pairReady();
    const raw = await page.evaluate(async () => {
      const current =
        typeof state === "undefined"
          ? (await import("/comparison.js")).state
          : state;
      return {
        left: current.view.legends.left,
        right: current.view.legends.right,
      };
    });
    assert.equal(raw.left.bound, raw.right.bound);
    assert.equal(raw.left.calibration_domain, "shared_raw");
    for (const side of ["left", "right"]) {
      for (const q of ["a", "b", "delta", "abs_delta"]) {
        await page.locator("#" + side).selectOption(q);
        await pairReady();
        await expect(page.locator("#" + side + "-scope")).toContainText(
          q === "a" || q === "b" ? "original" : "derived",
        );
      }
    }
    await page.locator("#left").selectOption("delta");
    await pairReady();
    await page.locator("#right").selectOption("abs_delta");
    await pairReady();
    for (const mapping of ["asinh", "magnitude", "linear"]) {
      await page.locator("#mapping").selectOption(mapping);
      await pairReady();
    }
    await page.locator("#fit").click();
    await pairReady();
    await painted("left", true);
    await painted("right", true);
    const geometry = await page.evaluate(async () => {
      const current =
        typeof state === "undefined"
          ? (await import("/comparison.js")).state
          : state;
      return {
        left: current.viewers.left.viewport.getBounds(true),
        right: current.viewers.right.viewport.getBounds(true),
      };
    });
    for (const key of ["x", "y", "width", "height"])
      assert(Math.abs(geometry.left[key] - geometry.right[key]) < 1e-9);
    await page
      .locator(".panels")
      .screenshot({ path: out + "/comparison-desktop.png" });
    checks.push(
      "Comparison both-panel quantities/mappings, shared raw scale, separate derived legends and aligned native geometry paint correctly",
    );
    await page.locator("#row").fill("1");
    await page.locator("#col").fill("2");
    await page.locator("#inspect").click();
    await expect(page.locator("#inspection")).toContainText("Original A");
    await expect(page.locator("#inspection")).toContainText("Original B");
    await expect(page.locator("#inspection")).toContainText("Derived B − A");
    await expect(page.locator("#inspection")).toContainText(
      "no inference edit target",
    );
    await page
      .locator("#inspection")
      .screenshot({ path: out + "/comparison-inspector.png" });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator("#fit").click();
    await pairReady();
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    for (const side of ["left", "right"]) {
      await painted(side, true);
      await page
        .locator("#" + side + "-canvas")
        .locator("..")
        .screenshot({ path: out + "/comparison-mobile-" + side + ".png" });
    }
    checks.push(
      "Comparison original/derived inspector and 390px stacked panels are readable with painted canvases",
    );
    assert.deepEqual(errors, []);
    assert(
      !requests.some((u) => /\/api\/inference\/(start|poll|cancel)/.test(u)),
    );
    const result = {
      passed: true,
      browser: browser.version(),
      browser_lifetimes: browserLifetimes,
      scope:
        "Two sequential browser lifetimes for independent viewers; all original assertions retained; uninterrupted combined-session fit remains unqualified",
      geometry,
      checks,
      physical_phone_tested: false,
      errors,
      request_count: requests.length,
    };
    fs.writeFileSync(
      out + "/browser-report.json",
      JSON.stringify(result, null, 2) + "\n",
    );
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
