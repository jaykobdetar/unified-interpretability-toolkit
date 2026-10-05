# Development formatter

Black 26.1.0 and its dependencies are development-only. The hash-pinned wheel set in `requirements-format.txt` was verified against official PyPI release metadata and tested with CPython 3.12 on Linux x86_64. It does not change the application runtime or portable no-install checks.

Create a local environment in the ignored qualification directory:

```sh
python3.12 -m venv qualification/dev/black-26.1.0
qualification/dev/black-26.1.0/bin/python -m pip install --index-url https://pypi.org/simple --only-binary=:all: --require-hashes -r dev/requirements-format.txt
python3 tools/check.py format
```

The supplied `tools/behaviour_lock.py` is excluded and must remain unchanged. Apply formatting with Black's default AST safety check enabled; do not use `--fast`. Formatter-only commits contain formatter output only. Run the full project checks before committing; `src/` changes also require the unchanged behavior lock. Development tooling is optional for application use and existing no-install checks.

The runner accepts Python 3.12 or newer, Node 22 or newer, and Rust/Cargo 1.92 or newer. Formatter versions remain exact: Black 26.1.0, Prettier 3.9.6 and rustfmt 1.8.0-stable. The recorded runtime versions in `versions.json` describe the qualified setup; `runtime_minimums` defines the runtime gate. The hash-pinned Black wheels use a separate CPython 3.12 environment, which can format code while the runner uses a newer Python. See [development setup](../docs/DEVELOPMENT.md) for the pinned npm installation. `--dev-python` selects another existing interpreter with the exact Black version.

The same checks work in a source archive without `.git`. In a Git checkout, formatting selects tracked first-party files. In an archive it selects first-party files under `tools/`, `tests/` and `web/`, with the same immutable-lock and vendored-code exclusions. The no-install `lint` and `contracts` commands still need only the application development runtimes.

The full runner also requires exact development-only Ruff 0.16.8, ESLint 9.15.0 and mypy 1.18.2. Install the Python tools into the same ignored environment, then use the locked npm development setup:

```sh
qualification/dev/black-26.1.0/bin/python -m pip install --only-binary=:all: --no-deps -r dev/requirements-lint.txt
PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm --prefix dev ci --ignore-scripts
python3 tools/check.py all
```

See [the static-check scope](STATIC-CHECKS.md) for selected rules, two explicit oracle-hook scope exceptions and type-check limits. `static` runs these checks alone. `--ruff` can select another existing exact Ruff executable. The no-install `lint` and `contracts` entrypoints remain available.

The full runner builds the release binary before running the complete release test suite. Cold production compilation and test-target compilation/execution therefore use separate existing 140-second command budgets. Both checks remain mandatory; either failure stops the runner. CPU affinity, one build job, offline locked vendor inputs, memory/address-space/disk guards, test threads and smoke/launcher checks are unchanged. A warm retry or cache-assisted run is not a cold-build timing claim.
