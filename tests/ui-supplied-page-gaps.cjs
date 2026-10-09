"use strict";
require("./support/comparison-controller.cjs").enableModules(__filename);
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    // Whole production scripts with the existing DOM/OSD doubles; no browser or server.
    const assert = require("node:assert/strict"),
      { createFixture } = require("./support/ui-fixture.cjs"),
      { requireCompletion } = require("./support/async-completion.cjs"),
      A = require("../web/atlas-tools.js");

    async function ready() {
      const f = await createFixture();
      Object.assign(f.catalog[0], {
        shape: [513, 769],
        rows: 513,
        cols: 769,
        count: 513 * 769,
        max_level: 10,
        dtype: "BF16",
        element_bytes: 2,
      });
      f.model.parameter_count = f.catalog.reduce((sum, t) => sum + t.count, 0);
      f.model.source_bytes = 2 * f.model.parameter_count;
      f.context.fixtureModel = f.model;
      f.run("bind();populateModel(fixtureModel,{tensor:11})");
      await f.activate();
      return f;
    }

    requireCompletion(
      (async () => {
        const selection = await ready();
        const original = selection.currentSettings();
        selection.context.first = original;
        selection.context.second = { ...original };
        assert.equal(selection.run("sameSettings(first, second)"), true);
        selection.context.second = { ...original, left: "tensor_linear" };
        assert.notEqual(original.left, "tensor_linear");
        assert.equal(
          selection.run("sameSettings(first, second)"),
          false,
          "A left-rule-only change must invalidate the old view",
        );
        selection.context.first = { ...original, slice: "0" };
        selection.context.second = { ...original, slice: "1" };
        assert.equal(
          selection.run("sameSettings(first, second)"),
          false,
          "A slice-only change must invalidate the old view",
        );

        const click = await ready();
        assert.equal(
          click.get("row").max,
          "512",
          "Last valid row is rows minus one",
        );
        click.viewers[0].emit("canvas-click", {
          quick: true,
          position: { x: 7.8, y: 2.3 },
        });
        const request = click.take("/api/inspect");
        const query = new URL(request.url, "http://127.0.0.1").searchParams;
        assert.equal(query.get("row"), "2");
        assert.equal(query.get("col"), "7");
        request.resolve(click.inspect(click.catalog[0], 2, 7, "-0.0"));
        await click.tick();
        assert.deepEqual(click.copy(click.run("state.selected")), [2, 7]);
        for (const viewer of click.viewers) {
          assert.equal(viewer.overlays.length, 1);
          assert.deepEqual(click.copy(viewer.overlays[0].location), {
            x: 7,
            y: 2,
            width: 1,
            height: 1,
          });
        }
        click.run("scalarZoom()");
        for (const viewer of click.viewers) {
          assert.equal(viewer.bounds.x, 7.5);
          assert.equal(viewer.bounds.y, 2.5);
        }

        const pointer = await ready();
        pointer.OSD.Point = function Point(x, y) {
          this.x = x;
          this.y = y;
        };
        pointer.get("left-canvas").getBoundingClientRect = () => ({
          left: 100,
          top: 50,
        });
        pointer.get("left-canvas").listeners.pointermove({
          pointerType: "mouse",
          buttons: 0,
          clientX: 107.8,
          clientY: 52.3,
        });
        assert.equal(
          pointer.run("hover.key"),
          pointer.run("`${state.viewEpoch}:2:7`"),
        );
        const hovering = pointer.timers[pointer.run("hover.timer") - 1]();
        const hoverRequest = pointer.take("/api/inspect");
        const hoverQuery = new URL(hoverRequest.url, "http://127.0.0.1")
          .searchParams;
        assert.equal(hoverQuery.get("row"), "2");
        assert.equal(hoverQuery.get("col"), "7");
        hoverRequest.resolve(pointer.inspect(pointer.catalog[0], 2, 7, "-0.0"));
        await hovering;
        assert.match(
          pointer.get("hover-readout").textContent,
          /^Row 2 · column 7 · native \[2, 7\]/,
        );
        pointer.run("queueHover(2,769)");
        assert.equal(pointer.run("hover.key"), null);
        assert.equal(pointer.run("hover.timer"), null);
        assert.equal(
          pointer.pending.length,
          0,
          "One-past-last hover schedules no request",
        );

        const invalid = await ready();
        const refusal = invalid.run("inspectAt(513,0)");
        assert.equal(
          invalid.pending.length,
          0,
          "One-past-last row must not issue a scalar request",
        );
        await refusal;
        assert.match(invalid.get("error").textContent, /inside this tensor/);

        const bounds = await ready();
        bounds.viewers[0].bounds = { x: 600, y: 300, width: 12, height: 6 };
        bounds.run("updateViewportBounds()");
        assert.equal(
          bounds.get("viewport-bounds").textContent,
          "Visible rows 300–305 of 513 ↓ · columns 600–611 of 769 → · zero-based, inclusive",
        );

        assert.equal(
          A.csvCell('a"b"c'),
          '"a""b""c"',
          "Every embedded CSV quote must be doubled",
        );
        const scope = A.scope(selection.model, selection.catalog[0]);
        const notes = Array.from({ length: 100 }, (_, i) => ({
          id: `n${i}`,
          kind: "row",
          region: [0, 0, 0, 768],
          text: `Note ${i}`,
        }));
        assert.equal(
          A.updateNote(notes, scope, { ...notes[0], text: "Edited" }).length,
          100,
        );
        assert.throws(
          () => A.updateNote(notes, scope, { ...notes[0], id: "extra" }),
          /100-note limit reached/,
        );
        assert.equal(notes.length, 100);
        assert.equal(notes[0].text, "Note 0");

        const layout = require("./fixtures/smollm2-head-layout-v1.json");
        const tensor = {
          name: "model.layers.0.self_attn.q_proj.weight",
          shape: [576, 576],
          rows: 576,
          cols: 576,
        };
        const model = {
          source_identity: "a".repeat(64),
          model_identity: "b".repeat(64),
          head_layout: layout,
          revision: layout.source_model.revision,
        };
        model.head_layout_binding = {
          source_identity: model.source_identity,
          model_identity: model.model_identity,
          weights_sha256: layout.source_model.weights_sha256,
          config_sha256: layout.source_model.config_sha256,
        };
        assert.match(A.hoverHead(model, tensor, 64, 5), /Query head 1/);
        assert.match(
          A.hoverHead(
            { ...model, model_identity: "c".repeat(64) },
            tensor,
            64,
            5,
          ),
          /unavailable/,
          "A descriptor bound to another model must not label this cell",
        );
        console.log(
          "PASS: supplied page gaps: independent selection keys, asymmetric click/hover/marker/zoom, bounds, CSV quotes, notes limit and model-bound labels; inert DOM/OSD only",
        );
      })().catch((error) => {
        console.error(error);
        process.exitCode = 1;
      }),
    );
  })(),
);
