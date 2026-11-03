import json
import os
import socket
import subprocess
from pathlib import Path

assert os.getuid() == 0
for path in ('/rsi-state', '/opt/rsi-task', '/home/agent/persistent', '/tmp/work-only', '/tmp/judge-only'):
    assert not Path(path).exists(), path
Path('/tmp/judge-only').write_text('must not survive')
try:
    Path('/workspace/illegal').write_text('mutation')
except OSError:
    pass
else:
    raise AssertionError('candidate was writable in Judge')

# A local broker passes one connected descriptor to isolated candidate code.
# This checks transport confinement, not model inference quality.
left, right = socket.socketpair()
left.sendall(b'local-broker')
code = '''
import os,socket
from pathlib import Path
assert os.getuid() >= 62000
for p in ('/tests/verify.py','/logs/verifier/reward.txt','/dev/nvidia0'):
    try:
        open(p,'rb')
    except OSError:
        pass
    else:
        raise AssertionError(p)
try:
    socket.socket(socket.AF_INET)
except OSError:
    pass
else:
    raise AssertionError('candidate can create internet sockets')
s=socket.socket(fileno=int(os.environ['BROKER_FD']))
assert s.recv(64)==b'local-broker'
s.sendall(b'ok')
print(Path('/workspace/value').read_text())
'''
result = subprocess.run([
    'rsi-isolate', 'run', '--no-network', '--read', '/workspace',
    '--timeout', '10', '--pass-fd', str(right.fileno()),
    '--env', f'BROKER_FD={right.fileno()}', '--', 'python3', '-c', code,
], pass_fds=(right.fileno(),), capture_output=True, text=True)
assert result.returncode == 0, result.stderr
left.settimeout(1)
assert left.recv(64) == b'ok'
score = float(result.stdout.strip())
Path('/logs/verifier/reward.txt').write_text(str(score))
print(json.dumps({'fixture_score': score, 'isolation': 'passed'}))
