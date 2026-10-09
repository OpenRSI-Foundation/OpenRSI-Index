#!/bin/bash
set -euo pipefail
task-python --version
child=$(rsi-sandbox create offline --lifetime 20 --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["child_id"])')
rsi-sandbox exec "$child" --json -- python3 -c 'from pathlib import Path; assert not Path("/workspace/marker").exists()' > /tmp/child-result.json
python3 -c 'import json; assert json.load(open("/tmp/child-result.json"))["exit_code"] == 0'
answer=$(cat /workspace/answer.txt)
rsi-sandbox exec "$child" --json -- python3 -c 'import sys; print(int(sys.argv[1] == "42"))' "$answer" > /tmp/child-result.json
python3 -c 'import json; r=json.load(open("/tmp/child-result.json")); assert r["exit_code"] == 0; json.dump({"reward": float(r["stdout"])}, open("/logs/verifier/reward.json", "w"))'
# Leave the child alive: the outer Judge must reclaim it before Work resumes.
