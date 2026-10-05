"use strict";
const assert = require("node:assert/strict"),
  A = require("../web/atlas-tools.js");
// Independent IEEE-754 known values, including negative zero/subnormals/extrema.
const fixtures = {
  BF16: [
    ["0000", 0],
    ["0080", -0],
    ["0100", 2 ** -133],
    ["7f7f", 3.3895313892515355e38],
    ["803f", 1],
  ],
  F16: [
    ["0000", 0],
    ["0080", -0],
    ["0100", 2 ** -24],
    ["ff7b", 65504],
    ["003c", 1],
    ["00bc", -1],
  ],
  F32: [
    ["00000000", 0],
    ["00000080", -0],
    ["01000000", 2 ** -149],
    ["ffff7f7f", 3.4028234663852886e38],
    ["0000803f", 1],
  ],
};
for (const [dtype, entries] of Object.entries(fixtures)) {
  const tensor = {
      id: 2,
      name: "test.weight",
      dtype,
      shape: [2, 1, entries.length],
      slice: [1],
      rows: 1,
      cols: entries.length,
    },
    model = {
      source_identity: "a".repeat(64),
      model_identity: "b".repeat(64),
      revision: "fixture",
      catalog: [tensor],
    },
    s = A.scope(model, tensor),
    binding = A.sourceBinding(model, tensor);
  const values = entries.map(([hex, n], col) => ({
    source_binding: binding,
    tensor: 2,
    row: 0,
    col,
    native_indices: [1, 0, col],
    dtype,
    element_bytes: dtype === "F32" ? 4 : 2,
    raw_hex_le: hex,
    raw_exact: Object.is(n, -0) ? "-0" : String(n),
  }));
  entries.forEach(([h, n]) => assert(Object.is(A.decodeSource(dtype, h), n)));
  const csv = A.boundedCSV(s, [0, 0, 0, entries.length - 1], values, binding);
  assert(csv.includes('"-0"'));
  assert(csv.includes(dtype));
  const npy = A.boundedNPY(s, [0, 0, 0, entries.length - 1], values, binding),
    n = npy.data[8] + 256 * npy.data[9],
    header = new TextDecoder().decode(npy.data.slice(10, 10 + n));
  assert.equal((10 + n) % 64, 0);
  assert(header.includes("'fortran_order': False"));
  assert(!header.includes("|O"));
  assert.equal(npy.metadata.numpy.allow_pickle, false);
  const payload = Buffer.from(npy.data.slice(10 + n)),
    expected = Buffer.concat(
      entries.map(([h]) =>
        Buffer.from(dtype === "BF16" ? "0000" + h : h, "hex"),
      ),
    );
  assert.deepEqual(payload, expected);
  assert.deepEqual(npy.metadata.slice, [1]);
  assert.deepEqual(
    npy.metadata.source_hex_le,
    entries.map(([h]) => h),
  );
  for (const changes of [
    { raw_exact: "2" },
    { raw_hex_le: "bad" },
    { dtype: "F64" },
    { element_bytes: 8 },
    { native_indices: [0, 0, 0] },
    { source_binding: { ...binding, model_identity: "c".repeat(64) } },
  ])
    assert.throws(() =>
      A.boundedNPY(
        s,
        [0, 0, 0, entries.length - 1],
        [{ ...values[0], ...changes }, ...values.slice(1)],
        binding,
      ),
    );
  const nonfinite = { BF16: "807f", F16: "007c", F32: "0000807f" }[dtype];
  assert.throws(() =>
    A.boundedCSV(
      s,
      [0, 0, 0, entries.length - 1],
      [
        { ...values[0], raw_hex_le: nonfinite, raw_exact: "Infinity" },
        ...values.slice(1),
      ],
      binding,
    ),
  );
}
console.log(
  "PASS: BF16/F16/F32 CSV and numeric NPY; independent values, exact payload bits, negative zero, higher-rank slice binding, malformed/nonfinite/storage/address refusals.",
);
