from __future__ import annotations

import json
import os
from pathlib import Path

rank = int(os.environ["RANK"])
root = Path("/workspace/ranks")
root.mkdir(parents=True, exist_ok=True)
(root / f"{rank:03d}.json").write_text(json.dumps({"rank": rank}) + "\n")
if rank == 0:
    Path("/workspace/smoke.json").write_text('{"started": true}\n')
