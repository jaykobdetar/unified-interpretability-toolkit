"use strict";
require("./support/comparison-controller.cjs").enableModules(__filename);
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    // Reuse the repository's DOM/OSD doubles; all source-binding checks are real.
    // Extends the independently reviewed first-view schedules to a pending raw read.
    const { createFixture } = require("./support/ui-fixture.cjs");
    const {
      fs,
      vm,
      assert,
      crypto,
      path,
      source,
      Element,
      elements,
      get,
      pending,
      viewers,
      frames,
      timers,
      OSD,
      context,
      run,
      copy,
      tick,
      take,
      tensor,
      catalog,
      model,
      statusUpdate,
      currentSettings,
      view,
      inspect,
      complete,
      activate,
      passed,
    } = await createFixture();
    const scenario = process.argv[2];
    if (
      ![
        "restore",
        "prior-pin",
        "newer-complete",
        "newer-pending",
        "cancel-before",
        "cancel-opening",
        "tensor-change",
        "binding-mismatch",
      ].includes(scenario)
    )
      throw Error("Unknown scenario");
    await require("./support/async-completion.cjs").requireCompletion(
      (async () => {
        vm.runInContext(
          fs.readFileSync(
            path.join(__dirname, "../web/atlas-tools.js"),
            "utf8",
          ),
          context,
        );
        model.source_identity = "a".repeat(64);
        model.model_identity = "b".repeat(64);
        model.calibration_complete = false;
        catalog[0].calibration_complete = false;
        for (const t of catalog) {
          t.dtype = "BF16";
          t.element_bytes = 2;
        }
        context.fixtureModel = model;
        run(
          "bind();populateModel(fixtureModel,{tensor:11});allowRawInspection()",
        );
        const binding = copy(
          run("AtlasTools.sourceBinding(state.model,state.tensor)"),
        );
        const scalar = (r, c, raw) => ({
          ...inspect(catalog[0], r, c, raw),
          source_binding: binding,
        });
        if (scenario === "prior-pin") {
          const prior = run("inspectAt(0,0)");
          take("/api/inspect").resolve(scalar(0, 0, "0.0625"));
          await prior;
        }
        get("row").value = "1";
        get("col").value = "1";
        get("inspect-form").listeners.submit({ preventDefault() {} });
        const raw = take("/api/inspect");
        assert.equal(run('state.pendingInspection.selected.join(",")'), "1,1");
        if (scenario === "cancel-before") run("clearInspection()");
        let opening, metadata;
        if (scenario === "tensor-change") {
          run("selectTensor(12)");
          await tick();
          take("/api/tensor-status").resolve(statusUpdate(catalog[1]));
          await tick();
          metadata = take("/api/view");
        } else {
          run("state.tensor.calibration_complete=true");
          opening = run("loadView()");
          metadata = take("/api/view");
        }
        assert.equal(context.window.atlasInferenceSelection, null);
        assert(raw.options.signal.aborted);
        let newer;
        if (["newer-complete", "newer-pending"].includes(scenario)) {
          context.window.atlasAnalyticsBridge.jump({ axis: "row", index: 2 });
          newer = take("/api/inspect");
          if (scenario === "newer-complete") {
            newer.resolve(scalar(2, 1, "0.25"));
            await tick();
          }
        }
        if (scenario === "cancel-opening") run("clearInspection()");
        const before = viewers.length;
        metadata.resolve({
          ...view(currentSettings()),
          source_binding: copy(
            run("AtlasTools.sourceBinding(state.model,state.tensor)"),
          ),
        });
        await tick();
        for (const v of viewers.slice(before)) v.emit("open");
        await opening;
        await tick();
        if (newer) {
          assert.equal(
            pending.length,
            0,
            "Opening must not queue the superseded raw address",
          );
          assert.equal(newer.options.signal.aborted, false);
          raw.resolve(scalar(1, 1, "0.125"));
          await tick();
          if (scenario === "newer-pending") {
            assert.equal(
              run("state.pendingInspection.selected[0]"),
              2,
              "Late old finally must not clear newer pending intent",
            );
            newer.resolve(scalar(2, 1, "0.25"));
            await tick();
          }
          assert.deepEqual(copy(run("state.selected")), [2, 1]);
          assert.equal(context.window.atlasInferenceSelection.row, 2);
        } else if (
          ["cancel-before", "cancel-opening", "tensor-change"].includes(
            scenario,
          )
        ) {
          raw.resolve(scalar(1, 1, "0.125"));
          await tick();
          assert.equal(pending.length, 0);
          assert.equal(context.window.atlasInferenceSelection, null);
          assert.equal(run("state.pendingInspection"), null);
        } else {
          assert.equal(
            pending.filter((r) => r.url.includes("/api/inspect")).length,
            1,
            "Exactly one bound re-read after ready",
          );
          const reread = take("/api/inspect");
          const address = new URLSearchParams(reread.url.split("?")[1]);
          assert.equal(address.get("row"), "1");
          assert.equal(address.get("col"), "1");
          const epoch = run("state.pendingInspection.epoch");
          raw.resolve(scalar(1, 1, "0.125"));
          await tick();
          assert.equal(
            run("state.pendingInspection.epoch"),
            epoch,
            "Late old finally must not clear the re-read",
          );
          const response = scalar(1, 1, "0.125");
          if (scenario === "binding-mismatch")
            response.source_binding = {
              ...binding,
              source_identity: "c".repeat(64),
            };
          reread.resolve(response);
          await tick();
          if (scenario === "binding-mismatch") {
            assert.equal(context.window.atlasInferenceSelection, null);
            assert(get("error").textContent.includes("binding mismatch"));
          } else {
            assert.deepEqual(copy(run("state.selected")), [1, 1]);
            assert.equal(context.window.atlasInferenceSelection.row, 1);
          }
          assert.equal(run("state.pendingInspection"), null);
        }
        assert.equal(pending.length, 0, "No automatic retry loop");
        console.log(
          JSON.stringify({
            status: "PASS",
            scenario,
            app_sha256: crypto
              .createHash("sha256")
              .update(source)
              .digest("hex"),
            scope:
              "Pending first-view intent, real source binding, cancellation/tensor/newer-selection precedence; pure DOM/transport",
          }),
        );
      })().catch((e) => {
        console.error(e);
        process.exitCode = 1;
      }),
    );
  })(),
);
