# Development formatter

Black 26.1.0 and its dependencies are development-only. The hash-pinned wheel set in `requirements-format.txt` was verified against official PyPI release metadata and tested with CPython 3.12 on Linux x86_64. It does not change the application runtime or portable no-install checks.

Create a local environment in the ignored qualification directory:

```sh
python3 -m venv qualification/dev/format
qualification/dev/format/bin/python -m pip install --index-url https://pypi.org/simple --only-binary=:all: --require-hashes -r dev/requirements-format.txt
qualification/dev/format/bin/python -m black --check --workers 1 tools tests
```

The supplied `tools/behaviour_lock.py` is excluded and must remain unchanged. Apply formatting with Black's default AST safety check enabled; do not use `--fast`. Formatter-only commits contain formatter output only. Run the full project checks before committing; `src/` changes also require the unchanged behavior lock. Development tooling is optional for application use and existing no-install checks.
