"""Generate native asset declarations; --check needs only standard Python."""

import argparse
from pathlib import Path

from atlas_host.host_assets import render

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/page_assets.rs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text() != source:
            raise SystemExit(
                "Page asset declarations are stale; run tools/generate_page_assets.py"
            )
        print("PASS: native page assets match the shared catalogue")
    else:
        OUTPUT.write_text(source)


if __name__ == "__main__":
    main()
