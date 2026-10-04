"""Public train-only development CLI; no reward or formal inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--batch-size", type=int, choices=(8, 32), default=8)
    parser.add_argument("--env-shards", type=int, choices=(2, 8), default=2)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).parent
    episodes = json.loads((root / "dev_manifest.json").read_text())["episodes"]
    if args.episodes <= 0 or args.episodes > len(episodes) or args.episodes % args.batch_size:
        parser.error("episodes must be a positive multiple of batch-size, at most 24")
    from runtime import run_pass
    with tempfile.TemporaryDirectory(prefix="rt-", dir="/tmp") as scratch:
        result = run_pass(episodes[:args.episodes], episodes[:8], Path(args.policy).read_text(),
                          (root / "reference/policy.py").read_text(), args.batch_size, args.env_shards, scratch)
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
