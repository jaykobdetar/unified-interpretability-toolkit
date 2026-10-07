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

mypy checks the development/CI runner and every qualified module/client listed in `dev/mypy-files.txt`, in strict mode using Python 3.12 syntax. The 63 explicit targets include the runner plus the 62 files previously checked together during qualification: inference contracts, runtime helpers, compatibility modules, analytics, coordinator/worker implementations and their typed clients. Normal transitive checking remains enabled. The same declaration drives local `static`/`all` and CI; no separate reduced CI scope remains.

This scope does not claim whole-repository typing, external numerical-library implementation typing or reference-oracle typing. The separate wider host closure is deferred with 774 recorded diagnostics in sixteen files. No missing-import suppression, generated application stub, plugin or automatic stub installation is introduced. The existing ten disposable type faults cover the runner's return, command, environment, timeout, version and path/result boundaries. Eighteen original checker correctness faults qualify the scope activation, including unchanged command flags, tool versions and admission. A separate default-scope contract pins all 63 targets and their actual mypy command; all existing command assertions remain intact.

The type and lint gates preserve existing formatter pins and run in the full development/CI runner. Neither selected static scope replaces the existing syntax, portable contract, Rust, smoke, launcher, behavior-lock or separately guarded browser/model checks.
