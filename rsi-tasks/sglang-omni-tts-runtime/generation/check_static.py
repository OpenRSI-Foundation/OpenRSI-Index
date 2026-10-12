"""Read-only authoring preflight. Parses files; never imports or runs task code."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
assert ROOT.name == 'sglang-omni-tts-runtime'
assert not ROOT.is_symlink()
files = sorted(p for p in ROOT.rglob('*') if p.is_file())
assert all(p.stat().st_size <= 100 * 1024**2 for p in files)
assert not any(p.is_symlink() for p in ROOT.rglob('*'))
assert not any(p.name in ('__pycache__', 'OVER_BUDGET.flag') for p in ROOT.rglob('*'))
test_files = [p for p in files if p.is_relative_to(ROOT / 'tests')]
assert len(test_files) < 100000 and sum(p.stat().st_size for p in test_files) < 1024**3
for p in files:
    if p.suffix == '.py':
        ast.parse(p.read_text(), filename=str(p))
    elif p.suffix == '.json':
        json.loads(p.read_text())
    elif p.suffix == '.sh':
        subprocess.run(['/bin/bash', '-n', str(p)], check=True)
for name in ('tests/test.sh', 'solution/solve.sh'):
    assert os.access(ROOT / name, os.X_OK)
for name in ('source.tar.gz', 'source-manifest.json', 'source-deps.json', 'assets.json', 'requirements.lock', 'python-artifacts.json'):
    assert (ROOT / 'environment' / name).read_bytes() == (ROOT / 'tests' / name).read_bytes(), name
protocol = json.loads((ROOT / 'tests/protocol.json').read_text())
archive = ROOT / 'tests/source.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == protocol['archive_sha256']
inventory = {}
with tarfile.open(archive) as t:
    for m in t:
        name = m.name.partition('/')[2]
        if m.isfile():
            inventory[name] = dict(kind='file', sha256=hashlib.sha256(t.extractfile(m).read()).hexdigest(), mode=(m.mode & 0o755) | 0o600)
        elif m.issym():
            inventory[name] = dict(kind='symlink', target=m.linkname)
        else:
            assert m.isdir()
assert inventory == json.loads((ROOT / 'tests/source-manifest.json').read_text())
metadata = json.loads((ROOT / 'environment/python-artifacts.json').read_text())['packages']
lines = (ROOT / 'environment/requirements.lock').read_text().splitlines()
assert len(lines) == len(metadata)
source_builds = []
for line in lines:
    requirement, *hashes = line.split()
    name, version = requirement.split('==')
    assert metadata[name]['version'] == version
    assert set(hashes) == {'--hash=sha256:' + f['sha256'] for f in metadata[name]['files']}
    assert hashes
    # Metadata-only wheel-family inspection; binary ABI/imports remain pending.
    wheels = [f['filename'] for f in metadata[name]['files'] if f['filename'].endswith('.whl')]
    compatible = any((('manylinux' in w or 'linux_x86_64' in w) and 'aarch64' not in w and ('cp312' in w or '-py3-' in w or '-py2.py3-' in w or '-abi3-' in w)) or w.endswith('-none-any.whl') for w in wheels)
    if not compatible:
        assert any(f['packagetype'] == 'sdist' for f in metadata[name]['files']), name
        source_builds.append(name)
task = tomllib.loads((ROOT / 'task.toml').read_text())
assert task['task']['name'] == 'rsi/' + ROOT.name
assert task['environment']['workdir'] == '/workspace'
assert task['environment']['gpus'] == 1 and task['metadata']['rsi_harness']['verifier']['gpus'] == 3
assert task['environment']['cpus'] == 48 and task['environment']['memory_mb'] == 393216
assert task['verifier']['timeout_sec'] == 7200 and task['agent']['timeout_sec'] == 86400
assert [lane['name'] for lane in protocol['lanes']] == ['base', 'moss', 'cosy']
assert protocol['rows'] == 1088 and protocol['repetitions'] == 3
assert protocol['version'] == 5 and protocol['repetition_seeds'] == [42, 43, 44]
assets = json.loads((ROOT / 'tests/assets.json').read_text())
assert not any('CustomVoice' in asset['repo'] for asset in assets)
assert protocol['benchmark_engine_overrides'] == dict(max_running_requests=64, cuda_graph_max_bs=64)
assert (ROOT / 'tests/calibration.schema.json').is_file()
print(json.dumps(dict(status='PASS', files=len(files), pristine_source_entries=len(inventory), pinned_distributions=len(metadata), source_builds=source_builds, tests_bytes=sum(p.stat().st_size for p in test_files), execution='not performed'), sort_keys=True))
