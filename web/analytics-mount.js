import shared from "./viewer-context.js";
import { mountAnalytics, renderModelOutliers } from "./analytics-panel.js";

// Coordinator-only mount. No inference session adoption, source paths or tokens
// in URLs/storage. One owned analysis at a time; no background/eager model scan.
const host = document.createElement("section");
host.className = "atlas-analytics";
host.setAttribute("aria-label", "Bounded weight analytics");
host.style.margin = "20px auto";
const title = document.createElement("h2");
title.textContent = "Compare a bounded native window";
host.append(title);
const explanation = document.createElement("p");
explanation.textContent =
  "Analysis runs separately from inference. One local compute job at a time. Up to 65,536 values; large tensors and models show explicit partial coverage.";
host.append(explanation);
const fields = document.createElement("div");
fields.className = "analytics-controls";
host.append(fields);
const inputs = {};
for (const [key, label, value] of [
  ["row", "First row", 0],
  ["col", "First column", 0],
  ["rows", "Rows", 64],
  ["cols", "Columns", 64],
]) {
  const container = document.createElement("label");
  container.textContent = label;
  const input = document.createElement("input");
  input.type = "number";
  input.min = key === "row" || key === "col" ? "0" : "1";
  input.step = "1";
  input.value = value;
  inputs[key] = input;
  container.append(input);
  fields.append(container);
}
const button = (label, action) => {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = label;
  b.onclick = action;
  fields.append(b);
  return b;
};
const note = document.createElement("p");
note.setAttribute("role", "status");
host.append(note);
const regionHost = document.createElement("div"),
  modelHost = document.createElement("div");
host.append(regionHost, modelHost);
(document.querySelector("main") || document.body).after(host);
let generation = 0,
  job = null,
  pending = false;
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function api(action, data) {
  const controller = new AbortController(),
    timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(
      `/api/analytics${action ? "/" + action : ""}`,
      action
        ? {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-Atlas-Local": "1",
            },
            body: JSON.stringify(data),
            signal: controller.signal,
          }
        : { signal: controller.signal },
    );
    const payload = await response.json();
    if (!response.ok)
      throw new Error(payload.error || "Analytics request failed");
    return payload;
  } finally {
    clearTimeout(timer);
  }
}
async function cancel() {
  ++generation;
  if (!job) {
    note.textContent = pending
      ? "Cancelling after admission returns…"
      : "No active analysis.";
    return;
  }
  const target = job;
  try {
    const snapshot = await api("cancel", { job: target });
    if (snapshot.status !== "stopping" && job === target) job = null;
    note.textContent =
      snapshot.status === "stopping"
        ? "Owned worker cleanup pending; new compute remains blocked."
        : "Analysis cancelled.";
  } catch (error) {
    note.textContent = `${error.message}. Worker time limit remains active.`;
  }
}
async function run(data) {
  if (pending)
    throw new Error(
      "An analysis request is active; cancel or wait for cleanup.",
    );
  if (job) {
    await cancel();
    if (job) throw new Error("Owned worker cleanup is still pending.");
  }
  pending = true;
  const ownGeneration = ++generation;
  note.textContent = "Starting bounded analysis…";
  let target = null;
  try {
    let snapshot = await api("start", data);
    target = snapshot.job;
    job = target;
    while (true) {
      if (ownGeneration !== generation) {
        await api("cancel", { job: target });
        throw new Error("Analysis superseded or cancelled.");
      }
      if (snapshot.status === "complete") {
        if (job === target) job = null;
        note.textContent =
          "Bounded analysis complete; source weights unchanged.";
        return snapshot.result;
      }
      if (!["starting", "running", "stopping"].includes(snapshot.status)) {
        if (job === target) job = null;
        throw new Error(snapshot.error || `Analysis ${snapshot.status}`);
      }
      await delay(200);
      snapshot = await api("poll", { job: target });
    }
  } catch (error) {
    if (target && job === target) {
      try {
        const stopped = await api("cancel", { job: target });
        if (stopped.status !== "stopping") job = null;
      } catch {}
    }
    throw error;
  } finally {
    pending = false;
  }
}
const widget = mountAnalytics(regionHost, {
  load: ({ seed, svd, scope }) => {
    const tensor = shared.atlasAnalyticsBridge?.selected();
    if (!tensor) throw new Error("Select a native tensor first.");
    if (tensor.available === false || tensor.shape.length > 2)
      throw new Error(
        "Legacy window analytics is unavailable for this tensor mapping; no slice is silently substituted.",
      );
    if (tensor.dtype !== "BF16")
      throw new Error(
        "Bounded analytics currently supports original BF16 sources only; F16/F32 viewing remains available.",
      );
    const region = Object.fromEntries(
      Object.entries(inputs).map(([key, input]) => [key, Number(input.value)]),
    );
    if (scope === "svd_summary") {
      if (
        !Object.values(region).every(Number.isInteger) ||
        region.rows < 1 ||
        region.cols < 1 ||
        region.rows > 128 ||
        region.cols > 128
      )
        throw new Error(
          "SVD summary requires a native window of at most 128 × 128 / 16384 values.",
        );
      return run({ scope: "svd_summary", tensor: tensor.id, region, seed });
    }
    return run({ tensor: tensor.id, region, seed, svd });
  },
  jump: (data) => shared.atlasAnalyticsBridge?.jump(data),
});
function select(tensor) {
  ++generation;
  widget.setReport(null);
  modelHost.replaceChildren();
  if (job || pending) cancel();
  if (!tensor) return;
  inputs.row.value = "0";
  inputs.col.value = "0";
  inputs.rows.value = String(Math.min(64, tensor.rows));
  inputs.cols.value = String(Math.min(64, tensor.cols));
  inputs.row.max = tensor.rows - 1;
  inputs.col.max = tensor.cols - 1;
  inputs.rows.max = Math.min(4096, tensor.rows);
  inputs.cols.max = Math.min(4096, tensor.cols);
  note.textContent =
    tensor.dtype === "BF16"
      ? `Selected ${tensor.name}. Choose a window, then Compare original / shuffle.`
      : `Selected ${tensor.name} (${tensor.dtype}). Analytics supports original BF16 sources only; use the viewer for F16/F32.`;
}
button("128 window", () => {
  const t = shared.atlasAnalyticsBridge?.selected();
  if (t) {
    inputs.row.value = "0";
    inputs.col.value = "0";
    inputs.rows.value = Math.min(128, t.rows);
    inputs.cols.value = Math.min(128, t.cols);
  }
});
button("Tall band", () => {
  const t = shared.atlasAnalyticsBridge?.selected();
  if (t) {
    inputs.row.value = "0";
    inputs.col.value = "0";
    inputs.rows.value = Math.min(4096, t.rows);
    inputs.cols.value = Math.min(16, t.cols);
  }
});
button("Wide band", () => {
  const t = shared.atlasAnalyticsBridge?.selected();
  if (t) {
    inputs.row.value = "0";
    inputs.col.value = "0";
    inputs.rows.value = Math.min(16, t.rows);
    inputs.cols.value = Math.min(4096, t.cols);
  }
});
button("Cancel analysis", cancel);
button("Rank bounded model prefix", async () => {
  try {
    modelHost.replaceChildren();
    const seed = Number(regionHost.querySelector("input[type=number]").value);
    const selection = shared.atlasAnalyticsBridge?.selected()?.id;
    const result = await run({ scope: "model", seed });
    if (selection === shared.atlasAnalyticsBridge?.selected()?.id)
      renderModelOutliers(modelHost, result);
  } catch (error) {
    note.textContent = error.message;
  }
});
window.addEventListener("atlas:tensor", (event) => select(event.detail));
window.addEventListener("pagehide", () => {
  ++generation;
  if (job)
    fetch("/api/analytics/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Atlas-Local": "1" },
      body: JSON.stringify({ job }),
      keepalive: true,
    }).catch(() => {});
});
select(shared.atlasAnalyticsBridge?.selected());
api("").catch(() => {
  note.textContent =
    "Analytics unavailable. Launch the local analytics-enabled coordinator to use this panel.";
});
