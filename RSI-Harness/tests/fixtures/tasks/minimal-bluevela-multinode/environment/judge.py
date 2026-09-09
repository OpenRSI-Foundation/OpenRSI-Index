from __future__ import annotations

import json
import os
from pathlib import Path

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
if world_size != 8:
    raise RuntimeError(f"expected 8 Judge ranks, received {world_size}")

smoke = json.loads(Path("/workspace/smoke.json").read_text())
if smoke != {"started": True}:
    raise RuntimeError("Work smoke marker is invalid")

work_ranks = sorted(
    json.loads(path.read_text())["rank"]
    for path in Path("/workspace/ranks").glob("*.json")
)
if work_ranks != list(range(16)):
    raise RuntimeError(f"Work ranks are incomplete: {work_ranks!r}")

judge_root = Path("/logs/verifier/judge-ranks")
judge_root.mkdir(parents=True, exist_ok=True)
(judge_root / f"{rank:03d}.json").write_text(
    json.dumps({"rank": rank, "world_size": world_size}) + "\n"
)
