"use strict";
// Actual browser/native fixture plus explicitly labelled served helper controls.
const { chromium, expect } = require("playwright/test"),
  assert = require("node:assert/strict"),
  fs = require("node:fs");
const base = process.env.ATLAS_TEST_URL,
  out = process.env.ATLAS_EVIDENCE_DIR;
assert(/^http:\/\/127\.0\.0\.1:\d+$/.test(base));
fs.mkdirSync(out, { recursive: true });
let browser,
  step = "setup";
const requests = [],
  errors = [],
  checks = [];
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    try {
      browser = await chromium.launch({
        headless: true,
        chromiumSandbox: true,
        executablePath: process.env.ATLAS_CHROMIUM,
        args: ["--renderer-process-limit=1", "--disable-gpu"],
      });
      const context = await browser.newContext({
        viewport: { width: 1100, height: 850 },
      });
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
      page.on("request", (request) =>
        requests.push({ url: request.url(), method: request.method() }),
      );
      const ready = async () => {
        await expect(page.locator("#comparison")).toHaveAttribute(
          "aria-busy",
          "false",
        );
        await page.waitForFunction(() =>
          ["left", "right"].every((side) =>
            state.viewers[side]?.world.getItemAt(0)?.getFullyLoaded(),
          ),
        );
      };
      await page.goto(base);
      await page.waitForFunction(() => state.viewers.left?.source);
      step = "P01 tile coordinates";
      for (const side of ["left", "right"]) {
        const params = await page.evaluate(
          (side) => [
            ...new URL(
              state.viewers[side].source.getTileUrl(10, 2, 1),
              location.origin,
            ).searchParams,
          ],
          side,
        );
        const query = new Map(params);
        assert.equal(query.get("x"), "2");
        assert.equal(query.get("y"), "1");
      }
      await ready();
      assert.deepEqual(
        await page.evaluate(() => [state.tensor.rows, state.tensor.cols]),
        [513, 769],
      );
      step = "P13 English count";
      assert(
        (await page.locator("#tensor-shape").textContent()).includes(
          "394,497 values",
        ),
      );
      step = "P14 native indices";
      assert.deepEqual(
        await page.evaluate(() =>
          AtlasTools.nativeIndices(state.tensor, 300, 600),
        ),
        [300, 600],
      );
      step = "P02 revision boundary";
      assert.deepEqual(
        await page.evaluate(() => {
          const value = {
            ...settings(),
            region: [300, 600, 305, 611],
            viewport: [600, 300, 12, 6],
          };
          const bookmark = AtlasTools.parseBookmark(
            AtlasTools.bookmark(value, state.model),
          );
          const healthy = AtlasTools.resolveBookmark(
            bookmark,
            state.model,
          ).tensor;
          let rejected = "";
          try {
            AtlasTools.resolveBookmark(bookmark, {
              ...state.model,
              model_identity: (state.model.model_identity[0] === "a"
                ? "b"
                : "a"
              ).repeat(64),
            });
          } catch (error) {
            rejected = error.message;
          }
          return [healthy === state.tensor.id, rejected];
        }),
        [
          true,
          "Bookmark belongs to a different source identity or model revision",
        ],
      );
      step = "P03 bookmark precision";
      assert.deepEqual(
        await page.evaluate(
          () =>
            AtlasTools.parseBookmark(
              AtlasTools.bookmark(
                {
                  ...settings(),
                  region: [0, 0, 0, 0],
                  viewport: [1e-8, -1e-8, 2, 2],
                },
                state.model,
              ),
            )?.viewport,
        ),
        [1e-8, -1e-8, 2, 2],
      );
      step = "P04 F16 decoding";
      assert.deepEqual(
        await page.evaluate(() => [
          AtlasTools.decodeSource("F16", "003e"),
          AtlasTools.decodeSource("F16", "00c0"),
        ]),
        [1.5, -2],
      );
      step = "P05 region boundary";
      assert.deepEqual(
        await page.evaluate(() => {
          const valid = AtlasTools.region([512, 768, 512, 768], state.tensor);
          let rejected = "";
          try {
            AtlasTools.region([512, 768, 513, 768], state.tensor);
          } catch (error) {
            rejected = error.message;
          }
          return [valid, rejected];
        }),
        [[512, 768, 512, 768], "Region is empty or outside this tensor"],
      );
      step = "P07 rule availability";
      assert.deepEqual(
        await page.evaluate(() => {
          const original = state.tensor,
            result = [];
          try {
            for (const dtype of ["F16", "F32"]) {
              state.tensor = { ...original, dtype };
              updateRuleAvailability();
              result.push(
                ...["left", "right"].map(
                  (side) =>
                    [...document.getElementById(side + "-rule").options].find(
                      (option) => option.value === "tensor_signed_percentile",
                    ).disabled,
                ),
              );
            }
          } finally {
            state.tensor = original;
            updateRuleAvailability();
          }
          return result;
        }),
        [false, false, true, true],
      );
      step = "P08 hover boundary";
      assert.deepEqual(
        await page.evaluate(() => {
          queueHover(512, 100);
          const valid = hover.key !== null;
          queueHover(513, 100);
          const invalid = [
            hover.key,
            document.getElementById("hover-readout").textContent,
          ];
          cancelHover();
          return [valid, ...invalid];
        }),
        [true, null, "Hover over either image to read one original value."],
      );
      step = "P10 CSV text guard";
      assert.equal(
        await page.evaluate(() => AtlasTools.csvCell("=synthetic-label", true)),
        '"\'=synthetic-label"',
      );
      step = "P11 bounded retry";
      assert.deepEqual(
        await page.evaluate(async () => {
          let calls = 0;
          const waits = [];
          try {
            const value = await AtlasTools.readJSON("/api/model", {
              fetchImpl: async () =>
                ++calls === 1
                  ? {
                      ok: false,
                      status: 503,
                      json: async () => ({
                        code: "admission_full",
                        error: "Synthetic busy response",
                      }),
                    }
                  : { ok: true, status: 200, json: async () => state.model },
              wait: async (ms) => waits.push(ms),
            });
            return {
              calls,
              waits,
              identity: value.model_identity === state.model.model_identity,
            };
          } catch (error) {
            return { calls, waits, error: error.message };
          }
        }),
        { calls: 2, waits: [250], identity: true },
      );
      checks.push(
        "Served helpers: revision refusal, eight-decimal bookmark round trip, F16 exact decode, region/hover bounds, dtype rule availability, harmless CSV text, native indices; retry uses an explicit response/wait double and does not exercise backend pressure",
      );
      step = "P06 pending right rule";
      assert.equal(
        await page.evaluate(async () => {
          document.getElementById("right-rule").value = "tensor_linear";
          const opening = loadView();
          document.getElementById("right-rule").value = "tensor_magnitude";
          await opening;
          return state.current;
        }),
        null,
      );
      await page.evaluate(async () => {
        document.getElementById("right-rule").value = "tensor_asinh";
        await loadView();
      });
      await ready();
      checks.push(
        "Pending native viewer open refuses changed right-hand settings; healthy recolor recovers",
      );
      await page.locator("#right-rule").selectOption("tensor_linear");
      await ready();
      assert.equal(
        await page.evaluate(() => state.current.settings.right),
        "tensor_linear",
      );
      await page.locator("#row").fill("300");
      await page.locator("#col").fill("600");
      await page.locator("#inspect-submit").click();
      await expect(page.locator(".raw-value")).toHaveText("10");
      step = "P09 inspector coordinates";
      assert.equal(
        await page.evaluate(
          () =>
            [...document.querySelectorAll("#inspection dt")].find(
              (element) => element.textContent === "Display row / column",
            ).nextElementSibling.textContent,
        ),
        "300 / 600",
      );
      checks.push(
        "Asymmetric native inspector address and independently generated BF16 value",
      );
      await page.locator(".workspace-tools > summary").click();
      for (const [id, value] of [
        ["region-r0", 300],
        ["region-c0", 600],
        ["region-r1", 305],
        ["region-c1", 611],
      ])
        await page.locator("#" + id).fill(String(value));
      await page.locator("#region-focus").click();
      await ready();
      step = "P12 inclusive focus";
      const centers = await page.evaluate(() =>
        ["left", "right"].map((side) => {
          const viewer = state.viewers[side],
            item = viewer.world.getItemAt(0),
            bounds = viewer.viewport.getBounds(true);
          const a = item.viewportToImageCoordinates(bounds.x, bounds.y, true),
            b = item.viewportToImageCoordinates(
              bounds.x + bounds.width,
              bounds.y + bounds.height,
              true,
            );
          return [(a.x + b.x) / 2, (a.y + b.y) / 2];
        }),
      );
      assert(
        centers.every(
          ([x, y]) => Math.abs(x - 606) < 1e-7 && Math.abs(y - 303) < 1e-7,
        ),
      );
      await page.waitForFunction(() =>
        state.viewers.left.world
          .getItemAt(0)
          .lastDrawn.some(
            ({ tile }) =>
              tile.level === state.tensor.max_level &&
              tile.x === 2 &&
              tile.y === 1 &&
              tile.loaded,
          ),
      );
      for (const side of ["left", "right"]) {
        const params = await page.evaluate(
          (side) => [
            ...new URL(
              state.viewers[side].source.getTileUrl(
                state.tensor.max_level,
                2,
                1,
              ),
              location.origin,
            ).searchParams,
          ],
          side,
        );
        const query = new Map(params);
        assert.equal(query.get("x"), "2");
        assert.equal(query.get("y"), "1");
      }
      assert(
        requests.some(({ url }) => {
          const parsed = new URL(url);
          return (
            parsed.pathname === "/tile" &&
            parsed.searchParams.get("x") === "2" &&
            parsed.searchParams.get("y") === "1" &&
            parsed.searchParams.get("level") === "10"
          );
        }),
      );
      checks.push(
        "Inclusive region focus on both real viewers; native tile (2,1) requested and loaded in a 513-by-769 source",
      );
      assert.deepEqual(errors, []);
      assert(requests.every((request) => request.method === "GET"));
      assert(
        !requests.some((request) =>
          request.url.includes("/api/inference/start"),
        ),
      );
      await page.screenshot({ path: out + "/multitile.png" });
      fs.writeFileSync(
        out + "/result.json",
        JSON.stringify(
          {
            status: "PASS",
            checks,
            requests,
            page_errors: errors,
            browser: browser.version(),
            centers,
          },
          null,
          2,
        ) + "\n",
      );
    } catch (error) {
      fs.writeFileSync(
        out + "/failure.json",
        JSON.stringify(
          {
            error: error.stack,
            step,
            assertion_failure: error.code === "ERR_ASSERTION",
            checks,
            requests,
            page_errors: errors,
          },
          null,
          2,
        ) + "\n",
      );
      throw error;
    } finally {
      if (browser) await browser.close();
    }
  })().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  }),
);
