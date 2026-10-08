#!/usr/bin/env python3
"""Read-only verification of user-selected local Qwen shards against pinned hashes."""

import argparse, hashlib, json, os
from pathlib import Path

from atlas_host.memory import guard

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    root = args.directory.resolve()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    guard()
    manifest = json.loads((ROOT / "docs/models/qwen3-8b.json").read_text())
    records = []
    for shard in manifest["shards"]:
        file = root / shard["shard"]
        assert not file.is_symlink() and file.is_file(), file.name
        assert file.stat().st_size == shard["bytes"], file.name
        digest = hashlib.sha256()
        with file.open("rb") as f:
            while True:
                guard()
                block = f.read(2 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        assert digest.hexdigest() == shard["sha256"], f"SHA-256 mismatch: {file.name}"
        records.append(
            {
                "shard": file.name,
                "bytes": shard["bytes"],
                "sha256": digest.hexdigest(),
                "matched": True,
            }
        )
    print(
        json.dumps(
            {
                "repository": manifest["repository"],
                "revision": manifest["revision"],
                "verified": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
