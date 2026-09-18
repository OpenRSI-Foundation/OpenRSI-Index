#!/usr/bin/env python3
"""Create/check the closed immutable portion of the /workspace starting tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MUTABLE = {"policy", "trials", "submission"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in MUTABLE:
            continue
        if path.is_symlink():
            raise RuntimeError(f"immutable workspace contains symlink: {relative}")
        if path.is_file():
            result[relative.as_posix()] = digest(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["create", "check"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    actual = {"version": 1, "root": str(args.root), "files": inventory(args.root)}
    if args.mode == "create":
        if args.output is None:
            parser.error("--output is required for create")
        args.output.write_text(json.dumps(actual, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    else:
        if args.manifest is None:
            parser.error("--manifest is required for check")
        expected = json.loads(args.manifest.read_text(encoding="utf-8"))
        if expected != actual:
            missing = sorted(set(expected.get("files", {})) - set(actual["files"]))[:20]
            added = sorted(set(actual["files"]) - set(expected.get("files", {})))[:20]
            changed = sorted(k for k in set(expected.get("files", {})) & set(actual["files"]) if expected["files"][k] != actual["files"][k])[:20]
            raise SystemExit(f"workspace integrity failed: missing={missing} added={added} changed={changed}")


if __name__ == "__main__":
    main()
