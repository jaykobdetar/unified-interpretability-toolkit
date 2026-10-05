"use strict";
// Loaded only by the explicit local profile host; private ownership stays in this closure.
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.AtlasProfiles = api;
})(globalThis, () => {
  const check = (ok, message) => {
    if (!ok) throw new Error(message);
  };
  const abort = () => new DOMException("Profile context changed", "AbortError");
  const clone = (value) =>
    value == null ? value : JSON.parse(JSON.stringify(value));
  const stable = (value) =>
    JSON.stringify(value, (_k, v) =>
      v && typeof v === "object" && !Array.isArray(v)
        ? Object.fromEntries(
            Object.keys(v)
              .sort()
              .map((k) => [k, v[k]]),
          )
        : v,
    );
  const validJob = (r) =>
    /^[0-9a-f]{32}$/.test(r.job_id) && /^[0-9a-f]{64}$/.test(r.job_capability);
  function create({ post, render = () => {} }) {
    let admissionsSuspended = false;
    let enabled = false,
      context = null,
      selection = null,
      job = null,
      retiring = null,
      generation = 0,
      pending = false,
      status = null;
    const snapshot = () => ({
      enabled,
      pending,
      admissions_suspended: admissionsSuspended,
      state: retiring
        ? "stopping"
        : status?.state || (pending ? "admitting" : "idle"),
      accepted: retiring ? null : clone(status?.accepted || null),
      control_seed: retiring ? null : (job?.seed ?? null),
      error: status?.error || null,
      cleanup_pending: !!retiring || status?.cleanup_pending === true,
      resume_available: false,
    });
    const show = () => render(snapshot());
    const owner = (ctx = context, owned = job) => {
      check(ctx && owned, "No owned profile");
      return {
        version: 1,
        ...ctx,
        job_id: owned.id,
        job_capability: owned.capability,
      };
    };
    const same = (id) => {
      if (id !== generation) throw abort();
    };
    const request = (action, data) => post("/api/profiles/" + action, data);
    function retire(data) {
      retiring = { data };
      show();
      request("cancel", data).catch(() => {
        show();
      }); // Keep uncertain cleanup visible.
    }
    function configure(capabilities, next, nextBinding = null) {
      const changed =
        stable(context) !== stable(next) ||
        stable(selection) !== stable(nextBinding);
      if (changed) {
        if (job) retire(owner());
        else if (pending && !retiring)
          retiring = { pending: true, context: clone(context), awaiting: true };
        generation++;
        job = null;
        status = null;
        pending = false;
        context = clone(next);
        selection = clone(nextBinding);
      }
      enabled = capabilities?.profiles_enabled === true && !!context;
      show();
    }
    async function start(binding, seed, values, { restart = false } = {}) {
      check(
        enabled &&
          !admissionsSuspended &&
          !pending &&
          !retiring &&
          !status?.cleanup_pending,
        "Profiles unavailable, source transition or cleanup pending",
      );
      check(
        !job ||
          (restart &&
            ["complete", "partial", "error", "cancelled"].includes(
              status?.state,
            )),
        "Explicit Restart after terminal cleanup required",
      );
      check(
        Number.isSafeInteger(seed) && seed >= 0 && seed <= 4294967295,
        "Use a whole-number control seed",
      );
      check(
        Number.isSafeInteger(values) &&
          values > 0 &&
          values <= binding.rows * binding.cols,
        "Invalid requested slice coverage",
      );
      check(
        selection && stable(binding) === stable(selection),
        "Current native binding required",
      );
      // Every admission can replace ownership, including explicit Restart. Invalidate
      // old pages/capabilities before sending, and give the attempt a fresh epoch.
      const id = ++generation,
        ownedContext = clone(context),
        selected = clone(binding);
      job = null;
      status = null;
      pending = true;
      show();
      try {
        const result = await request("start", {
          version: 1,
          ...ownedContext,
          binding: selected,
          seed,
          values,
          restart,
        });
        check(validJob(result), "Invalid private job admission");
        check(
          result.model_id === ownedContext.model_id &&
            result.context_id === ownedContext.context_id,
          "Profile admission context mismatch",
        );
        check(result.resume_available === false, "Resume is unqualified");
        if (id !== generation) {
          retire(
            owner(ownedContext, {
              id: result.job_id,
              capability: result.job_capability,
            }),
          );
          throw abort();
        }
        job = {
          id: result.job_id,
          capability: result.job_capability,
          binding: selected,
          seed,
        };
        status = result;
      } catch (error) {
        // A lost/malformed admission response is not proof that nothing started.
        // With no returned job capability, only trusted host reconciliation or
        // owner lease cleanup can resolve this state; never blindly retry.
        if (!job && !retiring) {
          retiring = { pending: true, context: ownedContext, awaiting: false };
          show();
        }
        if (retiring?.pending) retiring.awaiting = false;
        throw error;
      } finally {
        if (id === generation) {
          pending = false;
          show();
        }
      }
    }
    async function poll() {
      if (pending) return;
      if (retiring) {
        const saved = retiring;
        if (saved.awaiting) return;
        if (!saved.data) {
          pending = true;
          try {
            const result = await request("reconcile", {
              version: 1,
              ...saved.context,
            });
            if (
              retiring === saved &&
              result.state === "cancelled" &&
              result.cleanup_pending === false
            ) {
              retiring = null;
              status = null;
            }
          } finally {
            pending = false;
            show();
          }
          return;
        }
        pending = true;
        try {
          const result = await request("status", saved.data);
          check(
            result.job_id === saved.data.job_id &&
              result.model_id === saved.data.model_id &&
              result.context_id === saved.data.context_id,
            "Cleanup status context mismatch",
          );
          if (
            retiring === saved &&
            !result.cleanup_pending &&
            ["cancelled", "error"].includes(result.state)
          ) {
            retiring = null;
            status = null;
          }
        } finally {
          pending = false;
          show();
        }
        return;
      }
      if (!job) return;
      const id = generation,
        data = owner();
      pending = true;
      try {
        const result = await request("status", data);
        same(id);
        check(
          result.model_id === context.model_id &&
            result.context_id === context.context_id &&
            result.job_id === job.id,
          "Profile status context mismatch",
        );
        check(result.resume_available === false, "Resume is unqualified");
        status = result;
        if (result.state === "cancelled" && !result.cleanup_pending) job = null;
      } finally {
        if (id === generation) {
          pending = false;
          show();
        }
      }
    }
    async function page(axis, start, count) {
      check(
        !retiring && status?.accepted && !status.cleanup_pending,
        "No accepted profile revision",
      );
      const id = generation,
        revision = status.accepted.revision;
      const result = await request("page", {
        ...owner(),
        revision,
        axis,
        start,
        count,
      });
      same(id);
      check(
        result.revision === revision && status.accepted.revision === revision,
        "Stale profile page",
      );
      check(
        stable(result.binding) === stable(job.binding),
        "Profile page binding mismatch",
      );
      check(
        result.axis === axis &&
          result.start === start &&
          result.end <= start + count,
        "Profile page range mismatch",
      );
      return result;
    }
    async function heartbeat() {
      if (!job || retiring) return;
      const id = generation,
        data = owner();
      const result = await request("heartbeat", data);
      same(id);
      check(
        result.job_id === job.id &&
          result.context_id === context.context_id &&
          result.model_id === context.model_id,
        "Heartbeat ownership mismatch",
      );
    }
    async function cancel() {
      if (retiring) return;
      if (job) {
        const data = owner();
        generation++;
        job = null;
        pending = false;
        status = null;
        retire(data);
      } else if (pending) {
        generation++;
        pending = false;
        retiring = { pending: true, context: clone(context), awaiting: true };
        show();
      }
    }
    return {
      configure,
      start,
      poll,
      page,
      cancel,
      heartbeat,
      snapshot,
      suspend(value) {
        admissionsSuspended = !!value;
        show();
      },
    };
  }
  // Status/lease timers never start numerical work or continue a partial grant.
  function mount(
    root,
    { post, setTimer = setTimeout, clearTimer = clearTimeout },
  ) {
    const doc = root.ownerDocument,
      make = (tag, text) => {
        const n = doc.createElement(tag);
        if (text) n.textContent = text;
        return n;
      };
    const title = make("h2", "Whole-slice strengths"),
      scope = make("p"),
      note = make("p"),
      seed = make("input");
    scope.className = "small";
    note.setAttribute("role", "status");
    note.setAttribute("aria-live", "polite");
    seed.type = "number";
    seed.value = "0";
    seed.min = "0";
    seed.max = "4294967295";
    seed.step = "1";
    seed.required = true;
    const explanation = make(
      "p",
      "Compare sums and means of absolute stored weights by row or column with a seeded shuffled control. This is a weight summary, not a measure of functional importance. Partial sums cover only visited values.",
    );
    explanation.className = "small muted";
    const start = make("button", "Start full slice"),
      cancel = make("button", "Cancel / Reset"),
      refresh = make("button", "Refresh status");
    for (const button of [start, cancel, refresh]) button.type = "button";
    const axes = make("select");
    for (const axis of ["rows", "columns"]) {
      const n = make("option", axis);
      n.value = axis;
      axes.append(n);
    }
    axes.value = "rows";
    const offset = make("input");
    offset.type = "number";
    offset.min = "0";
    offset.value = "0";
    offset.step = "1";
    const page = make("button", "Read paired page"),
      previous = make("button", "Previous page"),
      next = make("button", "Next page");
    for (const button of [page, previous, next]) button.type = "button";
    const results = make("div");
    results.className = "profile-results";
    const pageNote = make("p");
    pageNote.className = "small";
    let selected = null,
      disposed = false,
      timer = null,
      round = 0,
      pageEpoch = 0,
      pageEnd = 0,
      axisLength = 0;
    function labeled(text, input) {
      const label = make("label", text + " ");
      label.append(input);
      return label;
    }
    const actions = make("div");
    actions.className = "tool-actions";
    actions.append(labeled("Next control seed", seed), start, cancel, refresh);
    const pages = make("div");
    pages.className = "tool-actions";
    pages.append(
      labeled("Axis", axes),
      labeled("First entry", offset),
      page,
      previous,
      next,
    );
    root.replaceChildren(
      title,
      scope,
      explanation,
      actions,
      note,
      pages,
      pageNote,
      results,
    );
    function clearPage() {
      pageEpoch++;
      results.replaceChildren();
      pageNote.textContent = "";
      pageEnd = axisLength = 0;
      previous.disabled = next.disabled = true;
    }
    const running = new Set([
      "admitting",
      "running",
      "validating",
      "finishing",
      "stopping",
    ]);
    const client = create({
      post,
      render: (s) => {
        if (disposed) return;
        root.hidden = !s.enabled && !s.cleanup_pending;
        root.setAttribute("aria-busy", String(running.has(s.state)));
        start.disabled =
          !s.enabled ||
          s.admissions_suspended ||
          s.pending ||
          s.cleanup_pending ||
          running.has(s.state);
        seed.disabled = start.disabled;
        start.textContent =
          s.accepted || s.state === "error"
            ? "Restart full slice"
            : "Start full slice";
        cancel.disabled = s.state === "idle" || s.state === "cancelled";
        page.disabled = !s.accepted || s.cleanup_pending;
        if (!s.accepted) clearPage();
        note.textContent = s.cleanup_pending
          ? "Cancelling or resolving ownership. No new work can start until cleanup is confirmed."
          : s.admissions_suspended
            ? "Source selection is changing. Start is unavailable until its binding is verified."
            : s.accepted
              ? `${s.accepted.visited_values} / ${s.accepted.total_values} values validated · ${s.accepted.complete ? "complete" : "partial"}. ${s.accepted.complete ? "Paired pages are ready." : "No automatic continuation. Restart begins again; Resume is unavailable."}`
              : s.state === "running"
                ? "Computing the selected slice. Visited counts are unavailable until a snapshot is validated; no percentage or time estimate."
                : s.state === "validating" || s.state === "finishing"
                  ? "Validating the snapshot and confirming worker cleanup. No result is accepted yet."
                  : s.state === "admitting"
                    ? "Starting one bounded operation…"
                    : s.state === "error"
                      ? "Profile stopped without an accepted result. Reset or explicitly restart."
                      : s.state === "cancelled"
                        ? "Cleanup confirmed. Start explicitly to compute again."
                        : "Ready. Start explicitly to request the full selected slice.";
      },
    });
    function error(e) {
      if (!disposed)
        note.textContent =
          e.name === "AbortError"
            ? "Selection changed; stale results were discarded."
            : e.message;
    }
    async function readPage() {
      const epoch = ++pageEpoch;
      try {
        const axis = axes.value,
          first = Number(offset.value);
        check(
          Number.isSafeInteger(first) && first >= 0,
          "Use a nonnegative whole-number entry",
        );
        const result = await client.page(axis, first, 128);
        if (disposed || epoch !== pageEpoch) return;
        check(
          Array.isArray(result.original) &&
            Array.isArray(result.control) &&
            result.original.length === result.end - result.start &&
            result.control.length === result.original.length,
          "Incomplete paired page",
        );
        const table = make("table"),
          head = make("tr");
        for (const heading of [
          "Index",
          "Original sum |w|",
          "Original mean |w|",
          "Original coverage",
          "Control sum |w|",
          "Control mean |w|",
          "Control coverage",
        ])
          head.append(make("th", heading));
        const thead = make("thead");
        thead.append(head);
        table.append(thead);
        const body = make("tbody");
        result.original.forEach((original, i) => {
          const row = make("tr");
          row.append(make("th", String(original.index)));
          for (const item of [original, result.control[i]]) {
            check(
              item.index === result.start + i &&
                Number.isSafeInteger(item.visited_count) &&
                Number.isSafeInteger(item.expected_count) &&
                item.visited_count >= 0 &&
                item.visited_count <= item.expected_count &&
                item.complete === (item.visited_count === item.expected_count),
              "Invalid paired coverage",
            );
            check(
              Number.isFinite(item.sum_abs) &&
                (item.visited_count === 0
                  ? item.mean_abs === null
                  : Number.isFinite(item.mean_abs)),
              "Invalid paired strength",
            );
            row.append(
              make(
                "td",
                item.visited_count ? String(item.sum_abs) : "Not visited",
              ),
              make("td", item.mean_abs === null ? "—" : String(item.mean_abs)),
              make(
                "td",
                `${item.visited_count} / ${item.expected_count} · ${item.complete ? "complete" : "partial"}`,
              ),
            );
          }
          body.append(row);
        });
        table.append(body);
        results.replaceChildren(table);
        pageEnd = result.end;
        axisLength = result.axis_length;
        pageNote.textContent = `${axis} ${result.start}–${result.end - 1} of ${result.axis_length} · control seed ${client.snapshot().control_seed} · paired snapshot ${result.revision.slice(0, 12)} · ${result.visited_values} / ${result.total_values} values visited. Means divide by visited counts.`;
        previous.disabled = result.start === 0;
        next.disabled = pageEnd >= axisLength;
      } catch (e) {
        if (epoch === pageEpoch) error(e);
      }
    }
    async function reset() {
      await client.cancel();
      for (let n = 0; client.snapshot().cleanup_pending && n < 5; n++) {
        await client.poll();
        if (client.snapshot().cleanup_pending)
          await new Promise((resolve) => setTimer(resolve, 500));
      }
      return client.snapshot();
    }
    start.addEventListener("click", () => {
      if (start.disabled) return;
      const value = seed.value.trim();
      if (!/^\d+$/.test(value)) {
        error(new Error("Use a whole-number control seed"));
        return;
      }
      client
        .start(selected, Number(value), selected.rows * selected.cols, {
          restart:
            !!client.snapshot().accepted || client.snapshot().state === "error",
        })
        .catch(error);
    });
    cancel.addEventListener("click", () => reset().catch(error));
    refresh.addEventListener("click", () => client.poll().catch(error));
    page.addEventListener("click", readPage);
    axes.addEventListener("change", () => {
      offset.value = "0";
      clearPage();
    });
    offset.addEventListener("input", clearPage);
    previous.addEventListener("click", () => {
      offset.value = String(Math.max(0, Number(offset.value) - 128));
      readPage();
    });
    next.addEventListener("click", () => {
      offset.value = String(pageEnd);
      readPage();
    });
    async function refreshLoop() {
      if (disposed) return;
      try {
        await client.poll();
        if (++round % 8 === 0) await client.heartbeat();
      } catch (e) {
        if (!disposed) error(e);
      }
      if (!disposed) timer = setTimer(refreshLoop, 500);
    }
    timer = setTimer(refreshLoop, 500);
    return {
      reset,
      client,
      suspend: (value) => client.suspend(value),
      unavailable(message) {
        note.textContent = "Profile binding unavailable: " + message;
        root.hidden = false;
      },
      destroy() {
        disposed = true;
        clearTimer(timer);
        client.cancel().catch(() => {});
      },
      configure(capabilities, context, binding) {
        selected = clone(binding);
        clearPage();
        offset.value = "0";
        scope.textContent = binding
          ? `${binding.name} · ${binding.dtype} · ${binding.rows} rows × ${binding.cols} columns${binding.slice.leading_indices.length ? " · leading indices [" + binding.slice.leading_indices.join(", ") + "]" : ""}. One bounded request; partial coverage is possible.`
          : "Select a supported tensor and every leading slice index.";
        client.configure(capabilities, context, binding);
      },
    };
  }
  return { create, mount };
});
