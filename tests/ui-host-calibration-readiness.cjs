"use strict";
require("./support/comparison-controller.cjs").enableModules(__filename);
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    // Real prepareTensor with DOM/OSD/transport doubles; no listener or numeric work.
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
    await require("./support/async-completion.cjs").requireCompletion(
      (async () => {
        vm.runInContext(
          fs.readFileSync(
            path.join(__dirname, "../web/atlas-tools.js"),
            "utf8",
          ),
          context,
        );
        const bound = [];
        context.AtlasHost = {
          setProfileSelection: async (value) => bound.push(copy(value)),
          bindRead: (url) => ({ url, assertCurrent() {}, check() {} }),
          url: (value) => value,
        };
        for (const t of catalog)
          Object.assign(t, {
            dtype: "BF16",
            element_bytes: 2,
            calibration_complete: false,
          });
        model.calibration_complete = false;
        model.global_max = null;
        context.initial = model;
        run("bind();populateModel(initial)");
        await run("prepareTensor()");
        assert.equal(bound.length, 1);
        assert.equal(bound[0].tensor, run("state.tensor.id"));
        assert.equal(run("state.current"), null);
        assert.equal(viewers.length, 0);
        assert.equal(get("inspect-submit").disabled, false);
        assert.match(get("status").textContent, /Raw inspection ready/);
        assert.match(
          get("calibration-note").textContent,
          /Color scales are not prepared/,
        );
        assert.equal(
          pending.length,
          0,
          "Uncalibrated hosted preparation must not start a view or calibration",
        );
        await assert.rejects(
          run("calibrationAction('tensor=11')"),
          /owner CLI action/,
        );
        assert.equal(
          pending.length,
          0,
          "Visitor calibration refusal makes no mutation request",
        );
        const t = run("state.tensor"),
          binding = copy(
            run("AtlasTools.sourceBinding(state.model,state.tensor)"),
          );
        const reading = run("inspectAt(0,0)");
        take("/api/inspect").resolve({
          ...inspect(t, 0, 0, "0.125"),
          source_binding: binding,
          transforms_ready: false,
          transformed: { left: null, right: null },
        });
        await reading;
        assert.equal(get("inspection").children[1].textContent, "0.125");
        assert.equal(
          run("state.current"),
          null,
          "Raw inspection is not a color-view readiness claim",
        );
        // An owner-prepared catalog may open the real view path. This toggles only
        // the transport double's calibration metadata; no fake persisted cache exists.
        t.calibration_complete = true;
        model.calibration_complete = true;
        model.global_max = 34;
        const preparing = run("prepareTensor()");
        await tick();
        const request = take("/api/view"),
          before = viewers.length,
          data = view(currentSettings());
        data.source_binding = binding;
        request.resolve(data);
        await tick();
        for (const v of viewers.slice(before)) v.emit("open");
        await tick();
        const reread = pending.findIndex((r) => r.url.includes("/api/inspect"));
        if (reread >= 0)
          pending
            .splice(reread, 1)[0]
            .resolve({ ...inspect(t, 0, 0, "0.125"), source_binding: binding });
        await preparing;
        await tick();
        assert.equal(run("state.current.tensor.id"), t.id);
        assert.equal(run("state.loading"), false);
        assert.equal(viewers.slice(before).length, 2);
        assert.equal(pending.length, 0);
        assert.equal(
          bound.length,
          2,
          "Both states retain verified profile binding setup",
        );
        console.log(
          JSON.stringify({
            status: "PASS",
            checks: 6,
            scope:
              "Uncalibrated hosted raw inspection and owner-only calibration; calibrated metadata opens both real view paths; no automatic calibration or profile work; DOM/transport doubles only",
          }),
        );
      })().catch((e) => {
        console.error(e);
        process.exitCode = 1;
      }),
    );
  })(),
);
