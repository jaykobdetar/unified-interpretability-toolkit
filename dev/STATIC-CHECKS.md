# Static checks

The exact development tools are Ruff 0.16.8, ESLint 9.15.0 and mypy 1.18.2. They add no application runtime dependencies. `python3 tools/check.py all` requires all three; `static` runs only them. Missing tools, wrong versions, diagnostics and warnings fail. The runner has no optional bypass or automatic installation. The existing no-install `lint` and `contracts` entrypoints remain available. `--dev-python` retains the virtual environment launcher path, and `--ruff` can select another existing exact Ruff executable.

## Python lint scope

Ruff reads all first-party Python under tools/ and tests/, including the immutable lock without writing it. The initial correctness rule families are E9, F63, F7 and F82. F821 has two explicit scope exceptions: observation_reference.py and prompt_pair_reference.py have module-level forward hooks that use dictionaries while the hooks run, then explicitly delete those dictionaries after use for bounded memory. Ruff diagnoses those late-bound names after their final `del`; the oracle arithmetic, assertions and cleanup remain unchanged. No other F821 source is exempted.

The initial E/F inventory contained 480 diagnostics: E401 26, E402 90, E501 294, E731 5, E741 1, F401 54, F821 8 and F841 2. Production tools accounted for E401 7, E402 11, E501 212, E731 1 and F401 25. Style/import-order/unused-name rules are deferred openly. Production edits or unused-code deletion to satisfy them require separate review. No inline suppressions were added to production or tests. The two F821 exceptions and their reasons belong in review's scope/allowance record. A healthy control and twelve benign disposable correctness faults verify the enabled Ruff scope.

## JavaScript lint scope

ESLint checks all first-party .js, .cjs and .mjs under web/ and tests/. Git and source archives use the same first-party file selection. Vendored code and installed development packages are excluded. The explicit 28 initial correctness rules are in eslint.config.cjs. No warning is tolerated in this scope.

The exact ESLint 9.15.0 recommended-preset inventory reported no-empty 14, no-unused-vars 134, no-undef 149, no-control-regex 2 and no-useless-escape 1. Production browser code accounted for 8/2/53/2/1 respectively. These are retained diagnostic leads, not verified production defects; browser globals shared between scripts and deliberate empty callbacks require separate review. The recommended preset is not claimed as passing, and these rules are outside this initial passing-rule scope.

The proposed no-promise-executor-return rule produced 35 diagnostics, including three production expressions that return scheduling handles from Promise executors. It is explicitly deferred because production rewrites for lint rules require separate review. The exact diagnostics remain in the review evidence; the rule was not silently disabled.

The browser CSP assertion previously had a Prettier-formatted newline between headers() and its property lookup. The test now names the unchanged response headers once and preserves the same includes/assert check. Matched old/new controls pass, and both reject an invalid CSP. A healthy control and twenty benign disposable valid-JavaScript faults verify enabled ESLint diagnostics without executing the fault bodies.

## Python type scope

mypy checks the development/CI runner and all 105 qualified modules and implementation clients listed in `dev/mypy-files.txt`, in strict mode using Python 3.12 syntax. The scope includes inference contracts, runtime helpers, compatibility modules, analytics, coordinator/worker implementations, host observation and diagnostics, supervisor and static-model policies, profile routing, snapshots, workers, OS ownership, fixture validation, lifetime and platform services, the asset catalogue, dense admission, static operations, runtime adapter, fixture launcher, hosted runtime, profile HTTP handler, pure work ledger and native progress adapter. Every target passed strict qualification before entering the shared declaration. The same declaration drives local `static`/`all` and CI, with normal transitive checking and no reduced CI scope.

The explicit wider host inventory now passes inside that shared scope. This does not claim whole-repository, external numerical-library implementation or reference-oracle typing. The separately reviewed nineteen-module host import inventory is acyclic, including function-local imports; it is not a proof about arbitrary dynamic imports. No missing-import suppression, generated application stub, plugin or automatic stub installation is introduced.

The default-scope contract pins all 105 targets and the actual mypy command. Each expansion preserves every prior assertion and target order. Original and expanded witnesses detect the same ten omitted original targets; the latest expansion also detects omission of the work ledger, native progress adapter and their implementation client. Four healthy controls and complete disposable-copy restoration pass. Application and checker runtime are unchanged in scope-expansion commits. Existing checker type and correctness faults remain retained.

The type and lint gates preserve existing formatter pins and run in the full development/CI runner. Static checks supplement the existing syntax, portable contracts, Rust, smoke, launcher, behaviour-lock and separately guarded browser/model checks. A passing local run does not claim hosted CI for an unpublished commit.
