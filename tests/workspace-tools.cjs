"use strict";
const assert = require("node:assert/strict"),
  A = require("../web/atlas-tools.js");
const source = "a".repeat(64),
  tensor = {
    id: 7,
    name: 'tensor,"name"\nline',
    dtype: "BF16",
    shape: [2, 4],
    rows: 2,
    cols: 4,
  },
  model = {
    source_identity: source,
    model_identity: "d".repeat(64),
    revision: "fixture-v1",
    catalog: [tensor],
  };
const bookmark = {
  tensor: 7,
  left: "tensor_linear",
  right: "tensor_asinh",
  region: [0, 0, 1, 3],
  viewport: [0, 0, 4, 2],
};
const s = A.scope(model, tensor),
  binding = A.sourceBinding(model, tensor),
  sample = (row, col, hex) => ({
    source_binding: binding,
    tensor: 7,
    row,
    col,
    native_indices: [row, col],
    bf16_hex_le: hex,
    raw_exact: Object.is(A.decodeBF16(hex), -0)
      ? "-0"
      : String(A.decodeBF16(hex)),
  });
const cases = [];
function test(name, f) {
  f();
  cases.push(name);
}
const response = (status, body) => ({
  ok: status === 200,
  status,
  json: async () => body,
});
(async () => {
  test("v2 bookmark roundtrip, no free text", () => {
    const hash = A.bookmark(
        {
          ...bookmark,
          prompt: "secret",
          note: "private",
          path: "/private",
          session: "capability",
        },
        model,
      ),
      b = A.parseBookmark(hash);
    assert.equal(b.source, source);
    assert.deepEqual(b.region, bookmark.region);
    assert.deepEqual(b.viewport, bookmark.viewport);
    for (const text of [
      "secret",
      "private",
      "capability",
      "fixture",
      "tensor%2C",
    ])
      assert(!hash.includes(text));
  });
  test("integrated magnitude bookmarks preserve each unsigned rule selection", () => {
    for (const side of ["left", "right"]) {
      const b = A.parseBookmark(
        A.bookmark({ ...bookmark, [side]: "tensor_magnitude" }, model),
      );
      assert.equal(A.resolveBookmark(b, model)[side], "tensor_magnitude");
    }
  });
  test("all eight integrated rules roundtrip without changing source binding", () => {
    assert.equal(A.RULES.length, 8);
    for (const rule of A.RULES) {
      const b = A.resolveBookmark(
        A.parseBookmark(
          A.bookmark({ ...bookmark, left: rule, right: rule }, model),
        ),
        model,
      );
      assert.equal(b.left, rule);
      assert.equal(b.right, rule);
      assert.equal(b.model, model.model_identity);
    }
  });
  test("strict invalid URL parsing", () => {
    const good = A.bookmark(bookmark, model);
    for (const bad of [
      "",
      good + "&prompt=secret",
      good + "&wa=2",
      good.replace("wa=2", "wa=1"),
      good.replace("t=7", "t=-1"),
      good.replace("t=7", "t=NaN"),
      good.replace("t=7", "t=9007199254740992"),
      good.replace("t=7", "t=7.1"),
      good.replace("tensor_linear", "javascript%3Aalert"),
      good.replace("z=0%2C0%2C4%2C2", "z=0%2C0%2CInfinity%2C2"),
      good.replace("b=0%2C0%2C1%2C3", "b=1%2C0%2C0%2C3"),
      good + "%zz",
      "#wa=1&" + "x".repeat(2048),
    ])
      assert.equal(A.parseBookmark(bad), null, bad);
  });
  test("source mismatch refused; bounded coordinates clamp", () => {
    const b = A.parseBookmark(A.bookmark(bookmark, model));
    assert.throws(() =>
      A.resolveBookmark(b, { ...model, source_identity: "b".repeat(64) }),
    );
    const x = A.resolveBookmark(
      { ...b, region: [0, 0, 999, 999], viewport: [-999, 999, 99999, 0.001] },
      model,
    );
    assert.deepEqual(x.region, [0, 0, 1, 3]);
    assert.deepEqual(x.viewport, [-16, 18, 16, 1 / 24]);
  });
  test("vector viewport preserves blank surrounding space and zoom", () => {
    const t = { ...tensor, shape: [128], rows: 1, cols: 128 },
      m = { ...model, catalog: [t] },
      v = {
        ...bookmark,
        region: [0, 0, 0, 127],
        viewport: [0, -63.5, 128, 128],
      },
      b = A.parseBookmark(A.bookmark(v, m));
    assert.deepEqual(A.resolveBookmark(b, m).viewport, v.viewport);
  });
  test("revision-bound bookmarks and model context reject equal-geometry revision changes", () => {
    const next = {
        ...model,
        revision: "fixture-v2",
        model_identity: "e".repeat(64),
      },
      b = A.parseBookmark(A.bookmark(bookmark, model));
    assert.throws(() => A.resolveBookmark(b, next), /revision/);
    assert.equal(A.sameModelContext(model, next, 7), false);
    assert.equal(
      A.sameModelContext(model, { ...model, revision: "fixture-v2" }, 7),
      false,
    );
    assert.equal(
      A.sameModelContext(
        model,
        { ...model, catalog: [{ ...tensor, name: "other-tensor" }] },
        7,
      ),
      false,
    );
    assert.equal(
      A.sameModelContext(
        model,
        { ...model, coverage: { values_streamed: 8 } },
        7,
      ),
      true,
    );
    assert.equal(
      A.parseBookmark(A.bookmark(bookmark, model).replace("wa=2", "wa=1")),
      null,
    );
  });
  test("tiny signed pans serialize as strict decimals and roundtrip", () => {
    for (const n of [1e-8, -1e-8, 1e-7, -1e-7, 1e-6, 0, -0, 1000]) {
      const t = { ...tensor, rows: 2048, cols: 2048, shape: [2048, 2048] },
        m = { ...model, catalog: [t] },
        b = A.parseBookmark(
          A.bookmark({ ...bookmark, viewport: [n, n, 2, 2] }, m),
        );
      assert(b);
      assert.equal(b.viewport[0], n === 0 ? 0 : n);
      assert.equal(b.viewport[1], n === 0 ? 0 : n);
    }
    const huge = {
        ...tensor,
        shape: [Number.MAX_SAFE_INTEGER],
        rows: 1,
        cols: Number.MAX_SAFE_INTEGER,
      },
      m = { ...model, catalog: [huge] },
      b = A.parseBookmark(
        A.bookmark(
          {
            ...bookmark,
            region: [0, 0, 0, 1],
            viewport: [
              Number.MAX_SAFE_INTEGER * 1.5,
              0,
              Number.MAX_SAFE_INTEGER,
              1,
            ],
          },
          m,
        ),
      );
    assert(b);
    assert(b.viewport[0] <= Number.MAX_SAFE_INTEGER);
  });
  test("scalar binding includes exact tensor identity and excludes filesystem/revision text", () => {
    const withPaths = {
        ...model,
        source_directory: "/private/source",
        revision: "/private/revision",
      },
      b = A.sourceBinding(withPaths, tensor);
    assert.deepEqual(Object.keys(b), [
      "version",
      "model_identity",
      "source_identity",
      "tensor",
      "name",
      "dtype",
      "shape",
      "rows",
      "cols",
      "slice",
    ]);
    assert(!JSON.stringify(b).includes("/private"));
    for (const [key, value] of Object.entries({
      version: 1,
      model_identity: "e".repeat(64),
      source_identity: "f".repeat(64),
      tensor: 8,
      name: "other",
      dtype: "F32",
      shape: [4, 2],
      rows: 4,
      cols: 2,
    }))
      assert.throws(
        () => A.requireBinding({ ...binding, [key]: value }, binding),
        /binding mismatch/,
      );
    assert.throws(() => A.requireBinding(undefined, binding));
  });
  test("empty, fractional, vector and edge regions", () => {
    for (const bad of [
      [],
      [0, 0, -1, 0],
      [1, 0, 0, 0],
      [0, 0, 2, 0],
      [0, 0.1, 0, 1],
    ])
      assert.throws(() => A.region(bad, tensor));
    assert.deepEqual(A.region([1, 3, 1, 3], tensor), [1, 3, 1, 3]);
    assert.deepEqual(
      A.region([0, 127, 0, 127], { rows: 1, cols: 128 }),
      [0, 127, 0, 127],
    );
    assert.throws(() => A.region([0, 0, 0, 0], { rows: 0, cols: 1 }));
  });
  test("private note storage persists, isolates revisions and enforces bounds", () => {
    const data = new Map(),
      store = {
        getItem: (k) => data.get(k) || null,
        setItem: (k, v) => data.set(k, v),
      };
    let notes = A.updateNote([], s, {
      id: "one",
      kind: "row",
      region: [1, 0, 1, 3],
      text: "private <script>text</script>",
    });
    A.saveNotes(store, s, notes);
    assert.deepEqual(A.loadNotes(store, s), notes);
    assert.deepEqual(A.loadNotes(store, { ...s, revision: "v2" }), []);
    assert.deepEqual(
      A.loadNotes(store, { ...s, source_identity: "b".repeat(64) }),
      [],
    );
    notes = A.updateNote(notes, s, { ...notes[0], text: "edited" });
    assert.equal(notes.length, 1);
    assert.equal(notes[0].text, "edited");
    assert.throws(() =>
      A.updateNote(notes, s, { ...notes[0], text: "x".repeat(501) }),
    );
    assert.throws(() =>
      A.updateNote(notes, s, { ...notes[0], region: [0, 1, 0, 2] }),
    );
    A.saveNotes(store, s, []);
    assert.deepEqual(A.loadNotes(store, s), []);
    data.set(A.noteKey(s), "invalid");
    assert.throws(() => A.loadNotes(store, s));
    assert.equal(data.get(A.noteKey(s)), "invalid");
  });
  test("CSV exact BF16 metadata, signed zero, quoting and formula precaution", () => {
    const v = [sample(0, 0, "0080"), sample(0, 1, "0100")],
      csv = A.boundedCSV(s, [0, 0, 0, 1], v, binding);
    assert(csv.includes('"-0"'));
    assert(csv.includes("bf16_hex_le"));
    assert(csv.includes("untransformed"));
    assert(csv.includes('tensor,\\"'));
    assert.equal(A.csvCell("=1+2", true), '"\'=1+2"');
    assert.equal(A.csvCell("\t@sum(1)", true), '"\'\t@sum(1)"');
    assert.equal(A.csvCell("-0"), '"-0"');
    assert.throws(() => A.boundedCSV(s, [0, 0, 0, 1], [v[1], v[0]], binding));
    assert.throws(() =>
      A.boundedCSV(
        s,
        [0, 0, 0, 1],
        [{ ...v[0], raw_exact: "0" }, v[1]],
        binding,
      ),
    );
    assert.throws(() =>
      A.boundedCSV({ ...s, dtype: "F64" }, [0, 0, 0, 1], v, binding),
    );
  });
  let calls = 0,
    states = [],
    delays = [];
  const ok = await A.readJSON("/api/model", {
    fetchImpl: async () => {
      calls++;
      return calls < 3
        ? response(503, { code: "backend_unavailable", error: "offline" })
        : response(200, { api_version: 1 });
    },
    wait: async (ms) => delays.push(ms),
    onState: (s) => states.push(s),
  });
  assert.equal(ok.api_version, 1);
  assert.equal(calls, 3);
  assert.deepEqual(delays, [250, 750]);
  assert.deepEqual(
    states.map((s) => s.phase),
    ["started", "retrying", "retrying", "success"],
  );
  cases.push("bounded coded-503 retry succeeds after 250/750 ms");
  for (const code of [
    "backend_unavailable",
    "admission_full",
    "dispatch_full",
    undefined,
    "cleanup_pending",
    "calibration_not_ready",
  ]) {
    calls = 0;
    await assert.rejects(
      A.readJSON("/api/view", {
        fetchImpl: async () => {
          calls++;
          return response(503, { code, error: "no" });
        },
        wait: async () => {},
      }),
    );
    assert.equal(
      calls,
      ["backend_unavailable", "admission_full", "dispatch_full"].includes(code)
        ? 3
        : 1,
    );
  }
  cases.push("only appropriate coded 503s retry; readiness stays distinct");
  for (const url of [
    "/api/calibrate",
    "/api/inference/start",
    "/api/infer/start",
    "https://other/api/model",
    "/api/model#fragment",
  ])
    await assert.rejects(
      A.readJSON(url, {
        fetchImpl: () => {
          throw new Error("must not fetch");
        },
      }),
    );
  cases.push("closed read route allowlist excludes all mutations");
  states = [];
  await assert.rejects(
    A.readJSON("/api/model", {
      fetchImpl: async () => response(200, null),
      onState: (s) => states.push(s),
    }),
  );
  assert.equal(states.at(-1).code, "contract_error");
  cases.push(
    "malformed success JSON clears retrying state into explicit failure",
  );
  calls = 0;
  await assert.rejects(
    A.readJSON("/api/model", {
      fetchImpl: async () => {
        calls++;
        throw new TypeError("offline");
      },
    }),
  );
  assert.equal(calls, 1);
  cases.push("transport error requires manual recovery");
  const c = new AbortController();
  let wake;
  calls = 0;
  const pending = A.readJSON("/api/model", {
    signal: c.signal,
    fetchImpl: async () => {
      calls++;
      return response(503, { code: "backend_unavailable" });
    },
    wait: () => new Promise((r) => (wake = r)),
  });
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  c.abort();
  wake();
  await assert.rejects(pending, { name: "AbortError" });
  assert.equal(calls, 1);
  cases.push("stale retry timer cannot fetch after navigation abort");
  let deliver;
  states = [];
  const d = new AbortController(),
    late = A.readJSON("/api/model", {
      signal: d.signal,
      fetchImpl: () => new Promise((r) => (deliver = r)),
      onState: (s) => states.push(s),
    });
  d.abort();
  deliver(response(200, { api_version: 1 }));
  await assert.rejects(late, { name: "AbortError" });
  assert.deepEqual(
    states.map((s) => s.phase),
    ["started"],
  );
  cases.push("abort-ignoring transport cannot publish success");
  let reads = [];
  const csv = await A.collectRegion({
    model,
    tensor,
    bounds: [1, 3, 1, 3],
    read: async (url) => {
      reads.push(url);
      return url === "/api/model" ? model : sample(1, 3, "803f");
    },
  });
  assert.equal(reads.length, 3);
  assert(csv.includes('"1","3","[1,3]"'));
  cases.push("edge-cell export reads once with before/after identity");
  let count = 0;
  await assert.rejects(
    A.collectRegion({
      model,
      tensor,
      bounds: [0, 0, 0, 0],
      read: async (url) => {
        count++;
        return url === "/api/model"
          ? count === 3
            ? { ...model, source_identity: "c".repeat(64) }
            : model
          : sample(0, 0, "803f");
      },
    }),
    /Source changed/,
  );
  cases.push("source revision switch discards completed export");
  for (const changed of [
    { ...binding, model_identity: "e".repeat(64) },
    { ...binding, source_identity: "b".repeat(64) },
    { ...binding, name: "other" },
    undefined,
  ]) {
    let n = 0;
    await assert.rejects(
      A.collectRegion({
        model,
        tensor,
        bounds: [0, 0, 0, 1],
        read: async (url) => {
          n++;
          return url === "/api/model"
            ? model
            : { ...sample(0, 0, "803f"), source_binding: changed };
        },
      }),
      /binding mismatch/,
    );
    assert.equal(n, 2);
  }
  cases.push(
    "each scalar rejects temporary foreign source/revision/tensor or missing binding before another read",
  );
  let callbackIds = [];
  await Promise.all(
    [1, 2].map(() =>
      A.readJSON("/api/model", {
        fetchImpl: async () => response(200, { api_version: 1 }),
        onState: (e) => callbackIds.push(e),
      }),
    ),
  );
  assert.equal(new Set(callbackIds.map((e) => e.request)).size, 2);
  assert.equal(callbackIds.filter((e) => e.phase === "started").length, 2);
  cases.push(
    "concurrent identical URLs carry distinct stable request identities",
  );
  const huge = { ...tensor, rows: 17, cols: 17, shape: [17, 17] };
  await assert.rejects(
    A.collectRegion({
      model: { ...model, catalog: [huge] },
      tensor: huge,
      bounds: [0, 0, 16, 16],
      read: () => {
        throw Error("must not read");
      },
    }),
    /256/,
  );
  cases.push("hard cell cap checked before any read");
  const ec = new AbortController();
  count = 0;
  await assert.rejects(
    A.collectRegion({
      model,
      tensor,
      bounds: [0, 0, 0, 1],
      signal: ec.signal,
      read: async (url) => {
        count++;
        if (count === 2) ec.abort();
        return url === "/api/model" ? model : sample(0, 0, "803f");
      },
    }),
    { name: "AbortError" },
  );
  assert.equal(count, 2);
  cases.push("cancelled export discards late scalar and starts no next read");
  console.log(
    JSON.stringify(
      { status: "PASS", checks: cases.length, passed: cases },
      null,
      2,
    ),
  );
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
