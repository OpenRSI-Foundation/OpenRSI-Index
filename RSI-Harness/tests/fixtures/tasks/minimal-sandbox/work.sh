#!/bin/bash
set -euo pipefail
task-python --version
child=$(rsi-sandbox create offline --lifetime 100 --json |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["child_id"])')
rsi-sandbox exec "$child" --json -- python3 -c \
    'from pathlib import Path; Path("/workspace/marker").write_text("preserved")' \
    > /tmp/created.json
python3 -c 'import json; assert json.load(open("/tmp/created.json"))["exit_code"] == 0'
printf wrong > answer.txt
rsi-submit
rsi-sandbox exec "$child" --json -- python3 -c \
    'from pathlib import Path; assert Path("/workspace/marker").read_text() == "preserved"' \
    > /tmp/preserved.json
python3 -c 'import json; assert json.load(open("/tmp/preserved.json"))["exit_code"] == 0'
printf 42 > answer.txt
rsi-submit
rsi-sandbox destroy "$child" --json
echo preserved-across-two-rounds
