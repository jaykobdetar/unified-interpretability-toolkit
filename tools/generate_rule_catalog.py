"""Keep native rule declarations and browser ID lists aligned with one catalogue."""

import argparse
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "id",
    "title",
    "formula",
    "scope",
    "transform",
    "statistics",
    "supported_dtypes",
    "availability_note",
    "dtype_refusal",
)
NATIVE = re.compile(
    r"(// generated-rule-catalog\n)(.*?)(// end-generated-rule-catalog)",
    re.DOTALL,
)
BROWSER = re.compile(
    r"/\* rule-ids \*/(\s*\[.*?\])(\s*;?\s*)/\* end-rule-ids \*/", re.DOTALL
)
MAGNITUDE = re.compile(
    r"/\* magnitude-rule-ids \*/(\s*\[.*?\])(\s*;?\s*)/\* end-magnitude-rule-ids \*/",
    re.DOTALL,
)


def read_catalog(path: Path) -> list[dict[str, str]]:
    value: object = json.loads(path.read_text())
    if not isinstance(value, list) or not value:
        raise ValueError("Rule catalogue must be a nonempty ordered list")
    rules: list[dict[str, str]] = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != set(FIELDS):
            raise ValueError("Rule catalogue fields differ from the declaration")
        rule: dict[str, str] = {}
        for field in FIELDS:
            item = entry[field]
            if not isinstance(item, str):
                raise ValueError(f"Rule field must be text: {field}")
            rule[field] = item
        if not re.fullmatch(r"[a-z][a-z0-9_]*", rule["id"]):
            raise ValueError("Rule ID must be a lowercase identifier")
        for field in ("scope", "transform", "statistics", "supported_dtypes"):
            if not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", rule[field]):
                raise ValueError(f"Rule field must name a native symbol: {field}")
        rules.append(rule)
    if len({rule["id"] for rule in rules}) != len(rules):
        raise ValueError("Rule catalogue IDs must be unique")
    return rules


def render_native(rules: list[dict[str, str]]) -> str:
    lines = [f"static DEFINITIONS: [Definition; {len(rules)}] = ["]
    prefixes = {"scope": "Scope", "transform": "Transform", "statistics": "Statistics"}
    for rule in rules:
        lines.append("    Definition {")
        for field in FIELDS:
            value = rule[field]
            if field in prefixes:
                literal = prefixes[field] + "::" + value
            elif field == "supported_dtypes":
                literal = value
            else:
                literal = json.dumps(value, ensure_ascii=False)
            lines.append(f"        {field}: {literal},")
        lines.append("    },")
    lines.extend(
        [
            "];",
            "",
            "/// Public order is part of metadata and saved viewer state.",
            f"pub const IDS: [&str; {len(rules)}] = [",
            *(f"    DEFINITIONS[{i}].id," for i in range(len(rules))),
            "];",
        ]
    )
    percentile = [rule["id"] for rule in rules].index("tensor_signed_percentile")
    lines.append(
        "pub(crate) const SIGNED_PERCENTILE: &Definition = "
        f"&DEFINITIONS[{percentile}];"
    )
    return "\n".join(lines) + "\n"


def native_tokens(source: str) -> list[str]:
    """Compare this generated declaration grammar without formatting/comments."""
    tokens = re.findall(
        r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\s]',
        source,
        re.DOTALL,
    )
    return [token for token in tokens if not token.startswith(("//", "/*"))]


def update_native(source: str, rules: list[dict[str, str]], check: bool) -> str:
    matches = list(NATIVE.finditer(source))
    if len(matches) != 1:
        raise ValueError("Expected one native rule catalogue declaration")
    match = matches[0]
    expected = render_native(rules)
    if native_tokens(match[2]) == native_tokens(expected):
        return source
    if check:
        raise ValueError("Stale native rule catalogue declaration")
    return source[: match.start(2)] + expected + source[match.end(2) :]


def update_browser(
    source: str,
    rules: list[dict[str, str]],
    check: bool,
    pattern: re.Pattern[str] = BROWSER,
    magnitude: bool = False,
) -> str:
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError("Expected one browser rule ID declaration")
    match = matches[0]
    expected = [rule["id"] for rule in rules]
    if magnitude:
        expected = [
            rule["id"]
            for rule in rules
            if rule["transform"] in ("MagnitudeLinear", "MagnitudeAsinh")
        ]
    if ast.literal_eval(match[1].strip()) == expected:
        return source
    if check:
        raise ValueError("Stale browser rule ID declaration")
    return (
        source[: match.start(1)]
        + " "
        + json.dumps(expected)
        + " "
        + source[match.end(1) :]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rules = read_catalog(ROOT / "dev/rule-catalog.json")
    for name in ("src/rules.rs", "web/app.js", "web/atlas-tools.js"):
        path = ROOT / name
        source = path.read_text()
        update = update_native if name.endswith(".rs") else update_browser
        result = update(source, rules, args.check)
        if name == "web/app.js":
            result = update_browser(result, rules, args.check, MAGNITUDE, True)
        if not args.check and result != source:
            path.write_text(result)
    print("PASS: native rule facts and browser IDs match the shared catalogue")


if __name__ == "__main__":
    main()
