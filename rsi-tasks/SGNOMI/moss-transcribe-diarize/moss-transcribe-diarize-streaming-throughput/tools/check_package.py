"""Read-only generation preflight: parse/hash files, never import task code."""
import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import tomllib


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    assert root.name == 'moss-transcribe-diarize-streaming-throughput'
    assert not root.is_symlink()
    files = [p for p in root.rglob('*') if p.is_file()]
    assert not any(p.is_symlink() for p in root.rglob('*'))
    for p in files:
        assert p.stat().st_size <= 100 * 1024**2, p
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(p))
        if p.suffix == '.json':
            json.loads(p.read_text())
        if p.suffix == '.sh':
            assert p.stat().st_mode & stat.S_IXUSR
            subprocess.run(['/bin/bash', '-n', str(p)], check=True)
    config = tomllib.loads((root/'task.toml').read_text())
    assert config['task']['name'] == 'rsi/' + root.name
    assert config['environment']['workdir'] == '/workspace'
    assert config['environment']['gpus'] == config['metadata']['rsi_harness']['verifier']['gpus'] == 1
    assert config['agent']['timeout_sec'] == 86400 and config['verifier']['timeout_sec'] == 10800
    assert all(config[k]['network_mode'] == 'no-network' for k in ('environment', 'agent', 'verifier'))
    assert not (root/'OVER_BUDGET.flag').exists()
    assert sha(root/'environment/source.tar.gz') == 'da4c7acd9b895494fe5c9aa7a7884c3c181af6f69b438fa68acf7ad4920b1298'
    source_manifest = json.loads((root/'environment/source-manifest.json').read_text())
    seen = set()
    with tarfile.open(root/'environment/source.tar.gz') as archive:
        for member in archive:
            if member.isfile():
                name = member.name.split('/', 1)[1]
                assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == source_manifest[name]
                seen.add(name)
    assert seen == set(source_manifest)
    data = json.loads((root/'tests/data-manifest.json').read_text())
    for lang, count in [('en', 1088), ('zh', 2020)]:
        rows = 0
        for shard in data[lang]['shards']:
            assert shard['start'] == rows
            rows += shard['rows']
            assert sha(root/'tests'/shard['path']) == shard['sha256']
        assert rows == count == data[lang]['rows']
    tests = [p for p in (root/'tests').rglob('*')]
    assert len(tests) < 100000
    test_bytes = sum(p.stat().st_size for p in tests if p.is_file())
    assert test_bytes < 1024**3
    for manifest_path, base in [('tests/immutable-manifest.json', root/'tests'),
                                ('tools/package-manifest.json', root)]:
        manifest = json.loads((root/manifest_path).read_text())
        for rel, expected in manifest.items():
            assert sha(base/rel) == expected, (manifest_path, rel)
    package = json.loads((root/'tools/package-manifest.json').read_text())
    assert set(package) | {'tools/package-manifest.json', 'tests/build-inventory.json', 'tests/comparison-state.json'} == {p.relative_to(root).as_posix() for p in files}
    lock = (root/'environment/requirements.lock').read_text()
    versions = dict(re.findall(r'^([A-Za-z0-9_.-]+)==([^\s;]+)', lock, flags=re.M))
    for name, value in {'torch':'2.13.0', 'torchvision':'0.28.0', 'sglang':'0.5.21',
                         'transformers':'5.12.1', 'flashinfer-python':'0.6.18',
                         'torchaudio':'2.11.0', 'torchcodec':'0.15.0', 'numpy':'2.3.4',
                         'scipy':'1.16.3', 'pyarrow':'22.0.0', 'aiohttp':'3.13.1',
                         'requests':'2.32.5', 'jiwer':'4.0.0', 'zhon':'2.1.1'}.items():
        assert versions[name] == value, name
    source_lock = (root/'environment/source-requirements.lock').read_text()
    assert 'openai-whisper==20250625 --hash=sha256:' in source_lock
    assert all('--hash=sha256:' in row for row in source_lock.splitlines() if row.strip())
    evidence = json.loads((root/'tools/dependency-evidence.json').read_text())
    for name, version in versions.items():
        assert evidence['distributions'][name]['version'] == version, name
    assert 'protobuf' not in (root/'environment/overrides.txt').read_text()
    for name in ('grpcio', 'grpcio-health-checking', 'grpcio-reflection'):
        assert versions[name] == '1.81.1'
    assert versions['protobuf'] == '6.33.6'

    # Static regression checks inspect syntax and literal data, never execute
    # a candidate gate, scorer, extracted function, or other operational code.
    gates = ast.parse((root/'tests/gates.py').read_text())
    byte_fields = next(n.value for n in gates.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'BYTE_CAPACITY_KEYS' for t in n.targets))
    assert ast.literal_eval(byte_fields) == {'asr.engine.kv_cache_bytes'}
    upper = next(n.value for n in ast.walk(gates) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'upper' for t in n.targets))
    assert ast.unparse(upper) == '2 ** 63 if key in BYTE_CAPACITY_KEYS else 2 ** 31'
    assert 2**31 < 8589934592 < 2**63  # documented 8 GiB example
    assert any(isinstance(n, ast.Compare) and ast.unparse(n) == 'lower <= value < upper' for n in ast.walk(gates))

    evaluator = ast.parse((root/'tests/evaluate.py').read_text())
    request = next(n for n in evaluator.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'request')
    appended = [ast.unparse(n.args[0]) for n in ast.walk(request) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == 'protected.append']
    assert 'delta' not in appended and "''.join(deltas)" in appended
    scrub = next(n for n in evaluator.body if isinstance(n, ast.FunctionDef) and n.name == 'scrub')
    fields = next(n.value.args[0].value for n in ast.walk(scrub) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'fields' for t in n.targets))
    pattern = re.compile(fields)
    assert pattern.search('{"delta": "a"}') and pattern.search("transcript='a'")
    assert not pattern.search('File "/workspace/sglang-omni/transcript.py", line 4')
    assert not pattern.search('RuntimeError: CUDA allocation failed at /workspace/cache.py')
    assert any(isinstance(n, ast.Compare) and ast.unparse(n) == 'len(s) >= 8' for n in ast.walk(scrub))
    codes = {n.args[1].value for tree in (gates, evaluator) for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'EvaluationFailure'
             and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant)}
    assert {'comparison_preparation_required', 'comparison_identity_mismatch',
            'hardware_affinity_mismatch', 'build_inventory_unprepared',
            'build_inventory_identity_mismatch', 'comparison_aggregate_schema',
            'comparison_aggregate_mismatch', 'fixed_asset_unreadable',
            'fixed_asset_digest_mismatch', 'gpu_count_mismatch',
            'cuda_preflight_failed', 'service_launch_failed'} <= codes
    assert any(isinstance(n, ast.Constant) and n.value == 'dependency_import_failed' for n in ast.walk(evaluator))
    print(f'PASS: {len(files)} files; syntax/shell parsing, manifests, pinned source/dependencies, dataset shard hashes, limits and correction AST/literal checks; tests bytes={test_bytes}. No operational task code executed.')


if __name__ == '__main__':
    main()
