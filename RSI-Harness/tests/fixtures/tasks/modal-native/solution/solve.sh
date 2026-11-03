#!/bin/bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import os, socket
assert os.getuid() == 60000
assert Path.home() == Path('/home/agent')
for path in ('/tests', '/rsi-state', '/opt/rsi-task', '/opt/rsi-runtime', '/dev/nvidia0'):
    assert not Path(path).exists(), path
s = socket.socket()
try:
    s.settimeout(.5)
    s.connect(('1.1.1.1', 443))
except OSError:
    pass
else:
    raise AssertionError('Work escaped its private network')
finally:
    s.close()
Path('/tmp/work-only').write_text('private scratch')
Path('/home/agent/persistent').write_text('persisted')
Path('/workspace/value').write_text('0')
PY
rsi-submit
test -f /home/agent/persistent
test -f /tmp/work-only
test -f /run/rsi-harness/feedback/agent-1.log
printf 1 > /workspace/value
rsi-submit
