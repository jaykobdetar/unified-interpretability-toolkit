"use strict";

const assert = require("node:assert/strict"),
  { pathToFileURL } = require("node:url"),
  path = require("node:path");

require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    assert.equal(typeof document, "undefined");
    const before = new Set(Object.getOwnPropertyNames(globalThis));
    const module = await import(
      pathToFileURL(path.join(__dirname, "../web/comparison.js"))
    );
    assert.deepEqual(
      Object.getOwnPropertyNames(globalThis).filter((key) => !before.has(key)),
      [],
    );
    assert.equal(module.state.model, null);
    assert.equal(module.state.view, null);
    assert.equal(module.state.controller, null);
    assert.deepEqual(module.state.viewers, {});
    const model = Object.freeze({
      comparison_identity: "ordered & pair",
      sources: Object.freeze({
        a: Object.freeze({ source_identity: "source-a" }),
        b: Object.freeze({ source_identity: "source-b" }),
      }),
    });
    const response = Object.freeze({
      ...model,
      coordinate_space: "checkpoint-comparison-v1",
      inference_editable: false,
    });
    assert.equal(module.envelope(response, null), undefined);
    assert.equal(module.envelope(response, model), undefined);
    assert.throws(
      () =>
        module.envelope(
          { ...response, inference_editable: true, sources: null },
          model,
        ),
      { message: "Invalid comparison coordinate boundary" },
    );
    assert.throws(
      () => module.envelope({ ...response, coordinate_space: "other" }, model),
      { message: "Invalid comparison coordinate boundary" },
    );
    assert.throws(
      () => module.envelope({ ...response, comparison_identity: "" }, model),
      { message: "Missing ordered comparison identities" },
    );
    assert.throws(
      () =>
        module.envelope(
          { ...response, sources: { a: model.sources.a } },
          model,
        ),
      { message: "Missing ordered comparison identities" },
    );
    for (const changed of [
      { ...response, comparison_identity: "other" },
      { ...response, sources: { a: model.sources.b, b: model.sources.a } },
    ]) {
      assert.throws(() => module.envelope(changed, model), {
        message: "Comparison source identity changed; refresh required",
      });
    }
    const pair = Object.freeze({ id: 4, rows: 3, cols: 7, max_level: 3 });
    const tile = module.tileSource(
      pair,
      "delta",
      "linear",
      model.comparison_identity,
    );
    const { getTileUrl, ...geometry } = tile;
    assert.deepEqual(geometry, {
      width: 7,
      height: 3,
      tileSize: 256,
      tileOverlap: 0,
      minLevel: 0,
      maxLevel: 3,
    });
    assert.equal(
      getTileUrl(2, 1, 0),
      "/api/comparison/tile?comparison_identity=ordered+%26+pair&tensor=4&quantity=delta&mapping=linear&level=2&x=1&y=0",
    );
    assert.equal(
      module.tileSource({ ...pair, rows: 1 }, "a", "asinh", "pair").height,
      1,
    );
    console.log(
      "PASS: actual native comparison module import, no page initialization, exact envelope errors and pure tile geometry/URLs",
    );
  })(),
);
