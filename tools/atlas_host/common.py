"""Small strict JSON and type helpers shared by local host contracts."""

import hashlib
import json
import re
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, required, optional=()):
    require(type(value) is dict, "Expected an object")
    require(
        set(required) <= value.keys() <= set(required) | set(optional),
        "Missing or unknown fields",
    )


def integer(value, low=0, high=2**63 - 1):
    require(
        type(value) is int and low <= value <= high, "Integer outside allowed range"
    )
    return value


def label(value, maximum=128):
    require(
        type(value) is str
        and 0 < len(value) <= maximum
        and not any(ord(c) < 32 or ord(c) == 127 for c in value),
        "Invalid label",
    )
    return value


def digest(value):
    require(
        type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
        "Expected lowercase SHA-256 digest",
    )
    return value


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def identity(domain, value):
    return hashlib.sha256(canonical([domain, value])).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def read_json(path, limit):
    with Path(path).open("rb") as source:
        raw = source.read(limit + 1)
    require(len(raw) <= limit, "JSON file exceeds byte limit")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON")
            ),
        )
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid bounded JSON") from error
