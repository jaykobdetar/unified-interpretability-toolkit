"use strict";
require("./support/comparison-controller.cjs").enableModules(__filename);
require("./support/async-completion.cjs").requireCompletion(
  (async () => {
    // Pure DOM/transport regression for the independent mixed-catalog finding.
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
    const packed = {
      id: 99,
      name: "packed.weight",
      dtype: "I32",
      element_bytes: 4,
      shape: [1],
      rows: 1,
      cols: 1,
      count: 1,
      max_level: 0,
      min_level: 0,
      available: false,
      unavailable_reason: "Quantization semantics unsupported",
      calibration_complete: false,
      max_abs: null,
    };
    const mixed = {
      ...model,
      calibration_complete: false,
      global_max: null,
      catalog: [...catalog, packed],
      parameter_count: model.parameter_count + 1,
      coverage: {
        ...model.coverage,
        statistics_complete: false,
        all_requested: false,
        active_tensor: null,
      },
    };
    context.fixture = mixed;
    await require("./support/async-completion.cjs").requireCompletion(
      (async () => {
        run("bind();populateModel(fixture,null)");
        assert.equal(get("calibrate-all").disabled, true);
        assert.equal(
          get("calibrate-all").textContent,
          "Calibrate supported tensors",
        );
        assert.match(
          get("calibration-note").textContent,
          /Global calibration is unavailable/,
        );
        assert.match(
          get("calibration-note").textContent,
          /All supported tensors are calibrated/,
        );
        assert(
          !get("calibration-note").textContent.includes("enable global rules"),
        );
        assert.match(
          get("global-scale-note").textContent,
          /cannot enable global rules/,
        );
        assert.match(
          get("readiness-summary").textContent,
          /Supported local scales ready/,
        );
        assert.equal(get("left-rule").value, "tensor_linear");
        assert.equal(
          get("left-rule").options.find((o) => o.value === "global_linear")
            .disabled,
          true,
        );
        await get("calibrate-all").listeners.click();
        assert.equal(pending.length, 0);
        await activate();
        assert.equal(run("state.current.tensor.id"), 11);
        // A remaining supported tensor enables only the correctly labeled subset action.
        context.fixture = {
          ...mixed,
          catalog: mixed.catalog.map((t) =>
            t.id === 12 ? { ...t, calibration_complete: false } : t,
          ),
        };
        run("populateModel(fixture,null)");
        assert.equal(get("calibrate-all").disabled, false);
        assert.match(
          get("calibration-note").textContent,
          /remaining supported work/,
        );
        assert(
          !get("calibration-note").textContent.includes("enable global rules"),
        );
        const action = get("calibrate-all").listeners.click(),
          request = take("/api/calibrate");
        assert.match(request.url, /all=1/);
        request.resolve(
          {
            api_version: 1,
            queued: "supported tensors; global calibration unavailable",
          },
          202,
        );
        await action;
        run("deactivate()"); // cancels the mock polling timer before the next model context.
        context.fixture = { ...mixed, catalog: [packed], parameter_count: 1 };
        run("populateModel(fixture,null)");
        assert.equal(get("calibrate-all").disabled, true);
        assert.match(
          get("calibration-note").textContent,
          /no supported tensors/,
        );
        assert.match(
          get("readiness-summary").textContent,
          /No supported color calibration/,
        );
        assert.equal(pending.length, 0);
        console.log(
          "PASS: mixed catalog global unavailability, completed/pending/empty supported subsets, local view remains usable, global rules remain disabled, and no redundant calibration job. Pure DOM doubles only.",
        );
      })().catch((e) => {
        console.error(e);
        process.exitCode = 1;
      }),
    );
  })(),
);
