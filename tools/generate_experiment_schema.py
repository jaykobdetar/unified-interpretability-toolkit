"""Keep browser field declarations aligned with the shared experiment schema."""

import argparse
import ast
import json
from pathlib import Path
import re

from atlas_host.experiment_schema import BROWSER_SITES, FIELD_GROUPS

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = re.compile(
    r"/\* schema-fields: ([a-z_]+) \*/(.*?)/\* end-schema-fields \*/",
    re.DOTALL,
)


def literal_fields(source: str) -> tuple[str, tuple[str, ...]]:
    value: object = ast.literal_eval(source.strip())
    if isinstance(value, str):
        return "string", tuple(value.split(" "))
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "array", tuple(str(v) for v in value)
    raise ValueError("Schema declarations must be strings or string arrays")


def update(source: str, expected: tuple[tuple[str, str], ...], check: bool) -> str:
    matches = list(DECLARATION.finditer(source))
    if tuple((match[1], literal_fields(match[2])[0]) for match in matches) != expected:
        raise ValueError("Browser schema declaration sites differ from the catalogue")
    for match in reversed(matches):
        fields = FIELD_GROUPS[match[1]]
        form, existing = literal_fields(match[2])
        if existing == fields:
            continue
        if check:
            raise ValueError(f"Stale browser schema declaration: {match[1]}")
        value = " ".join(fields) if form == "string" else list(fields)
        source = (
            source[: match.start(2)]
            + " "
            + json.dumps(value)
            + " "
            + source[match.end(2) :]
        )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for file, sites in BROWSER_SITES.items():
        path = ROOT / file
        original = path.read_text()
        result = update(original, sites, args.check)
        if not args.check and result != original:
            path.write_text(result)
    print("PASS: browser record declarations match the shared schema")


if __name__ == "__main__":
    main()
