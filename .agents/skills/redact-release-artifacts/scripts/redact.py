#!/usr/bin/env python3
"""Scan/redact selected text artifacts; report counts without exposing matches."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess


def patterns(policy):
    rules = []
    for item in policy["literals"]:
        expression = re.escape(item["value"])
        if item.get("word_boundary"):
            expression = r"\b" + expression + r"\b"
        rules.append(("private_identifier", re.compile(expression, re.I), item["replacement"]))
    rules.extend([
        ("credential", re.compile(r"\b(?:gh[pousr]_|github_pat_|hf_|sk-proj-|sk-ant-|sk-)[A-Za-z0-9_-]{16,}"), "[REDACTED_CREDENTIAL]"),
        ("credential", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "[REDACTED_CREDENTIAL]"),
        ("private_key", re.compile(r"-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----.*?-----END (?:[A-Z ]+)?PRIVATE KEY-----", re.S), "[REDACTED_PRIVATE_KEY]"),
        ("authorization", re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/-]{16,}={0,2}"), "Bearer [REDACTED_CREDENTIAL]"),
        ("credential_assignment", re.compile(
            r"(?i)(\b(?!(?:SCALE|CACHE|SOURCE|DATASET|RUNTIME)_KEY\b)"
            r"(?:[A-Z][A-Z0-9_]*(?:_KEY|_TOKEN|_PASSWORD)|password|secret)"
            r"[\"']?\s*[:=]\s*[\"']?)(?!REDACTED|PLACEHOLDER|YOUR_)"
            r"[A-Za-z0-9+/=_-]{12,}"), "[REDACTED_CREDENTIAL]"),
        ("email", re.compile(r"\b[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
        ("account_path", re.compile(r"/(?:Users|home)/[^/\s\"'<>\\]+"), "/workspace/private"),
        ("site_path", re.compile(r"/(?:data\d+|proj|gpfs|lustre)/[^/\s\"'<>\\]+"), "/rsi-data/private"),
        ("private_ip", re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"), "private-host"),
        ("private_host", re.compile(r"\b(?:[A-Za-z0-9-]+\.)+(?:internal|private|local)\b", re.I), "private-host"),
    ])
    return rules


def transform_text(text, rules, counts):
    for category, expression, replacement in rules:
        if category == "credential_assignment":
            text, count = expression.subn(lambda match: match.group(1) + replacement, text)
        else:
            text, count = expression.subn(lambda _: replacement, text)
        counts[category] += count
    return text


def transform_json(value, rules, counts):
    if isinstance(value, str):
        return transform_text(value, rules, counts)
    if isinstance(value, list):
        return [transform_json(item, rules, counts) for item in value]
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            updated = transform_text(key, rules, counts)
            if updated in output:
                raise ValueError("Redaction would merge two JSON keys; choose distinct private aliases")
            output[updated] = transform_json(item, rules, counts)
        return output
    return value


def redact(raw, suffix, rules):
    text = raw.decode("utf-8")
    counts = Counter()
    if suffix == ".json":
        data = json.loads(text)
        updated = transform_json(data, rules, counts)
        content = (json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode()
    elif suffix == ".jsonl":
        rows = [transform_json(json.loads(line), rules, counts) for line in text.splitlines() if line.strip()]
        content = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    else:
        content = transform_text(text, rules, counts).encode()
    return content if any(counts.values()) else raw, +counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*")
    parser.add_argument("--policy", type=Path, default=Path.home() / ".codex/private/release-redaction.json")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    if not args.policy.is_file():
        parser.error("Private policy is missing; create it outside the publication tree")
    if args.staged and (args.write or args.paths):
        parser.error("--staged is a read-only scan of the index and takes no paths")
    if not args.staged and not args.paths:
        parser.error("Provide selected artifact paths or --staged")
    rules = patterns(json.loads(args.policy.read_text()))
    if args.staged:
        files = [Path(p) for p in subprocess.check_output([
            "git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"
        ]).decode().split("\0") if p]
    else:
        files = sorted({p for root in args.paths for p in (root.rglob("*") if root.is_dir() else [root])
                        if p.is_file() and not any(part in {".git", "__pycache__"} for part in p.parts)})
    findings = 0
    for path in files:
        name_counts = Counter()
        safe_name = transform_text(str(path), rules, name_counts)
        raw = subprocess.check_output(["git", "show", ":" + path.as_posix()]) if args.staged else path.read_bytes()
        try:
            result, counts = redact(raw, path.suffix, rules)
        except (UnicodeDecodeError, json.JSONDecodeError):
            print(json.dumps({"file": safe_name, "needs_manual_review": "binary or invalid structured text"}))
            findings += 1
            continue
        if counts or any(name_counts.values()):
            findings += 1
            print(json.dumps({"file": safe_name, "content_counts": counts,
                              "filename_counts": +name_counts, "rewritten": bool(args.write and counts)}))
            if args.write and counts:
                temporary = path.with_name(path.name + ".redaction-part")
                temporary.write_bytes(result)
                temporary.chmod(path.stat().st_mode & 0o777)
                temporary.replace(path)
    print(json.dumps({"scanned_files": len(files), "files_with_findings": findings,
                      "mode": "write" if args.write else "scan"}))
    if findings and not args.write:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
