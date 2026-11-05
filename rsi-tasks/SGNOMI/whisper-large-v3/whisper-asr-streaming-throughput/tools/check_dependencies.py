"""Generation-only metadata validation; never install/import selected packages.

Run with the supplied Harness Python (packaging is a compiler dependency).
--record refreshes the task-owned evidence from official distribution metadata.
The default validates the lock against fresh metadata and the recorded evidence.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.request import urlopen
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(implementation_name='cpython', implementation_version='3.12.0',
           os_name='posix', platform_machine='x86_64', platform_python_implementation='CPython',
           platform_release='', platform_system='Linux', platform_version='',
           python_full_version='3.12.0', python_version='3.12', sys_platform='linux')
WHEEL_CHECKS = {'grpcio', 'grpcio-health-checking', 'grpcio-reflection',
                'smg-grpc-proto', 'smg-grpc-servicer', 'protobuf'}


def fetch(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def active(req, extras=('',)):
    return req.marker is None or any(req.marker.evaluate({**ENV, 'extra': e}) for e in extras)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    locks = [(ROOT/'environment'/name).read_text() for name in ('requirements.lock', 'source-requirements.lock')]
    versions = {canonicalize_name(n): v for lock in locks
                for n, v in re.findall(r'^([A-Za-z0-9_.-]+)==([^\s;]+)', lock, re.M)}
    # The bound CUDA Tile wheel replaces its PyPI downloader stub without changing version.
    versions['cuda-tile'] = '1.6.0rc5'
    def metadata(pair):
        name, version = pair
        url = f'https://pypi.org/pypi/{name}/{version}/json'
        return name, json.loads(fetch(url))
    with ThreadPoolExecutor(max_workers=12) as pool:
        distributions = dict(pool.map(metadata, sorted(versions.items())))
    extras = {n: {''} for n in versions}
    roots = []
    for line in (ROOT/'environment/requirements.in').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            req = Requirement(line)
            if active(req):
                roots.append(req)
                extras[canonicalize_name(req.name)].update(req.extras)
    requirements = {n: [Requirement(s) for s in d['info'].get('requires_dist') or []]
                    for n, d in distributions.items()}
    changed = True
    while changed:
        changed = False
        for name, reqs in requirements.items():
            for req in reqs:
                if active(req, extras[name]):
                    target = canonicalize_name(req.name)
                    assert target in versions, (name, 'missing dependency', str(req))
                    prior = len(extras[target])
                    extras[target].update(req.extras)
                    changed |= len(extras[target]) != prior
    for req in roots:
        assert Version(versions[canonicalize_name(req.name)]) in req.specifier, str(req)
    evidence = {'target': ENV, 'distributions': {}, 'wheels': {}}
    for name, data in distributions.items():
        info = data['info']
        assert Version(versions[name]) == Version(info['version']), name
        assert not info.get('requires_python') or Version('3.12.0') in SpecifierSet(info['requires_python']), name
        for req in requirements[name]:
            if active(req, extras[name]):
                assert Version(versions[canonicalize_name(req.name)]) in req.specifier, (name, str(req))
        evidence['distributions'][name] = {
            'version': versions[name], 'metadata_url': f'https://pypi.org/pypi/{name}/{versions[name]}/json',
            'requires_python': info.get('requires_python'),
            'requires_dist': sorted(info.get('requires_dist') or []),
        }
    for name in sorted(WHEEL_CHECKS):
        wheels = [f for f in distributions[name]['urls'] if f['filename'].endswith('.whl')]
        wheel = next(f for f in wheels if f['filename'].endswith('-py3-none-any.whl') or
                     ('manylinux' in f['filename'] and 'x86_64' in f['filename'] and
                      ('-cp312-cp312-' in f['filename'] or '-cp310-abi3-' in f['filename'])) )
        content = fetch(wheel['url'])
        digest = hashlib.sha256(content).hexdigest()
        assert digest == wheel['digests']['sha256']
        assert '--hash=sha256:' + digest in '\n'.join(locks), (name, 'wheel hash absent from lock')
        archive = zipfile.ZipFile(io.BytesIO(content))
        meta = BytesParser().parsebytes(archive.read(next(p for p in archive.namelist() if p.endswith('.dist-info/METADATA'))))
        reqs = meta.get_all('Requires-Dist', [])
        assert sorted(reqs) == evidence['distributions'][name]['requires_dist'], name
        generated = {}
        for path in archive.namelist():
            if path.endswith('_pb2.py') or path.endswith('_pb2_grpc.py'):
                tree = ast.parse(archive.read(path), filename=path)
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == 'ValidateProtobufRuntimeVersion'):
                        needed = tuple(ast.literal_eval(a) for a in node.args[1:4])
                        runtime = Version(versions['protobuf']).release
                        assert runtime[0] == needed[0] and runtime >= needed, (name, path, needed)
                        generated[path] = {'protobuf': list(needed)}
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'GRPC_GENERATED_VERSION' for t in node.targets):
                        needed = ast.literal_eval(node.value)
                        assert Version(versions['grpcio']) >= Version(needed), (name, path, needed)
                        generated[path] = {'grpcio': needed}
        evidence['wheels'][name] = {'url': wheel['url'], 'sha256': digest,
                                   'requires_dist': sorted(reqs), 'generated_requirements': generated}
    path = ROOT/'tools/dependency-evidence.json'
    if args.record:
        path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    else:
        assert json.loads(path.read_text()) == evidence, 'Dependency evidence changed; inspect official metadata before refreshing.'
    print(f'PASS: {len(versions)} distribution metadata records; active constraints and Python 3.12 compatibility; six wheel hashes/metadata and generated protobuf/gRPC requirements. No distribution code executed.')


if __name__ == '__main__':
    main()
