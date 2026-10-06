"use strict";

// Initial correctness rules. Deferred rules and qualification status are recorded
// in dev/STATIC-CHECKS.md. This scope includes first-party browser and test code.
// no-promise-executor-return is deferred: 35 existing diagnostics include three
// production sites; this test/tool pass does not rewrite those runtime expressions.
const rules = [
  "constructor-super",
  "for-direction",
  "getter-return",
  "no-async-promise-executor",
  "no-compare-neg-zero",
  "no-cond-assign",
  "no-const-assign",
  "no-dupe-args",
  "no-dupe-class-members",
  "no-dupe-else-if",
  "no-dupe-keys",
  "no-duplicate-case",
  "no-empty-character-class",
  "no-ex-assign",
  "no-func-assign",
  "no-import-assign",
  "no-invalid-regexp",
  "no-irregular-whitespace",
  "no-loss-of-precision",
  "no-obj-calls",
  "no-setter-return",
  "no-this-before-super",
  "no-unexpected-multiline",
  "no-unreachable",
  "no-unsafe-finally",
  "no-unsafe-negation",
  "use-isnan",
  "valid-typeof",
];

module.exports = [
  { ignores: ["web/vendor/**", "dev/node_modules/**"] },
  {
    files: ["web/**/*.js", "tests/**/*.js", "tests/**/*.cjs", "tests/**/*.mjs"],
    languageOptions: { ecmaVersion: 2022, sourceType: "module" },
    rules: Object.fromEntries(rules.map((name) => [name, "error"])),
  },
  { files: ["tests/**/*.cjs"], languageOptions: { sourceType: "commonjs" } },
];
