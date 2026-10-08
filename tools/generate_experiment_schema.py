"""Keep browser record and layout declarations aligned with shared facts."""

import argparse
import ast
from collections.abc import Mapping
import json
from pathlib import Path
import re

from atlas_host.experiment_schema import BROWSER_SITES, FIELD_GROUPS
from atlas_host.inference_architecture import architecture
from atlas_host.limits import ARCHIVE_MAX_ACTIVATION_VALUES, ARCHIVE_MAX_ARRAY_VALUES

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


LAYOUT_DECLARATION = re.compile(
    r"/\* layout-fact: ([a-z_]+) \*/(.*?)/\* end-layout-fact \*/",
    re.DOTALL,
)
LAYOUT_SITES: Mapping[str, tuple[str, ...]] = {
    "web/inference-import.js": (
        "archive_array_limit",
        "archive_activation_limit",
        "width",
        "layers",
        "query_heads",
        "kv_heads",
        "head_dim",
        "queries_per_kv",
        "vocab_size",
        "intermediate_size",
    ),
    "web/atlas-tools.js": ("q_projection", "k_projection", "norm_vector"),
}


def layout_facts() -> dict[str, int | list[int]]:
    arch = architecture()
    return {
        "archive_array_limit": ARCHIVE_MAX_ARRAY_VALUES,
        "archive_activation_limit": ARCHIVE_MAX_ACTIVATION_VALUES,
        "width": arch.width,
        "layers": arch.layers,
        "query_heads": arch.query_heads,
        "kv_heads": arch.kv_heads,
        "head_dim": arch.head_dim,
        "queries_per_kv": arch.query_heads // arch.kv_heads,
        "vocab_size": arch.vocab_size,
        "intermediate_size": arch.description["intermediate_size"],
        "q_projection": [arch.query_heads * arch.head_dim, arch.width],
        "k_projection": [arch.kv_heads * arch.head_dim, arch.width],
        "norm_vector": [arch.width],
    }


def update_layout(
    source: str,
    expected: tuple[str, ...],
    facts: Mapping[str, int | list[int]],
    check: bool,
) -> str:
    matches = list(LAYOUT_DECLARATION.finditer(source))
    if tuple(match[1] for match in matches) != expected:
        raise ValueError("Browser layout declaration sites differ from the catalogue")
    for match in reversed(matches):
        value = facts[match[1]]
        # JSON spellings distinguish bool/float/int while ignoring source whitespace.
        if json.dumps(ast.literal_eval(match[2].strip())) == json.dumps(value):
            continue
        if check:
            raise ValueError(f"Stale browser layout declaration: {match[1]}")
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
    facts = layout_facts()
    for file in dict.fromkeys((*BROWSER_SITES, *LAYOUT_SITES)):
        path = ROOT / file
        original = path.read_text()
        result = update(original, BROWSER_SITES.get(file, ()), args.check)
        result = update_layout(result, LAYOUT_SITES.get(file, ()), facts, args.check)
        if not args.check and result != original:
            path.write_text(result)
    print("PASS: browser record and layout declarations match shared facts")


if __name__ == "__main__":
    main()
