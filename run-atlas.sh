#!/usr/bin/env bash
set -euo pipefail
if ! command -v python3 >/dev/null 2>&1; then
  printf '%s\n' 'Python 3 is required for the launcher. Install it, then run ./run-atlas.sh --help.' >&2
  exit 1
fi
exec python3 "$(dirname -- "$0")/tools/launch.py" "$@"
