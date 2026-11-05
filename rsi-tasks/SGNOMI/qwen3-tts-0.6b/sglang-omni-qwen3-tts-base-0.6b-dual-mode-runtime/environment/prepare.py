"""Image-build preparation only. No GPU use, serving, training or evaluation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

BUILD = Path('/opt/rsi-build')


def digest(path, git=False):
    h = hashlib.sha1() if git else hashlib.sha256()
    if git:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert os.uname().machine == 'x86_64'
    import sys
    assert sys.version_info[:2] == (3, 12)
    packages = json.loads((BUILD / 'python-artifacts.json').read_text())['packages']
    for name, spec in packages.items():
        assert importlib.metadata.version(name).split('+')[0] == spec['version'], name
    for spec in json.loads((BUILD / 'source-deps.json').read_text()):
        dest = Path('/opt/rsi-deps') / spec['name']
        subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/' + spec['repo'] + '.git', str(dest)], check=True)
        subprocess.run(['git', '-C', str(dest), 'checkout', spec['revision']], check=True)
        assert subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip() == spec['revision']
        shutil.rmtree(dest / '.git')
    archive = BUILD / 'source.tar.gz'
    assert digest(archive) == 'c5a1d842729a0d431b3b139e0d1ccbc40d161c8e77aad3234d82a8c742454357'
    with tarfile.open(archive) as t:
        t.extractall('/opt/rsi-unpack', filter='data')
    shutil.move('/opt/rsi-unpack/sglang-omni-1523543056bf334bc2f4d69db23aad0a002cedf2', '/opt/rsi-source')
    shutil.rmtree('/opt/rsi-unpack')
    # Install entry-point metadata from a disposable copy; never an editable install.
    shutil.copytree('/opt/rsi-source', '/tmp/rsi-install', symlinks=True)
    subprocess.run(['/opt/sglang/bin/python3', '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '/tmp/rsi-install'], check=True)
    shutil.rmtree('/tmp/rsi-install')
    from huggingface_hub import snapshot_download
    for spec in json.loads((BUILD / 'assets.json').read_text()):
        dest = Path(snapshot_download(repo_id=spec['repo'], repo_type=spec['kind'], revision=spec['revision']))
        assert dest.name == spec['revision']
        for item in spec['files']:
            p = dest / item['path']
            assert p.stat().st_size == item['size'], str(p)
            assert digest(p, git=not item['sha256']) == (item['sha256'] or item['git_blob']), str(p)
        refs = dest.parent.parent / 'refs'
        refs.mkdir(exist_ok=True)
        (refs / 'main').write_text(spec['revision'])
    from datasets import load_dataset, Audio
    ds = load_dataset('zhaochenyang20/seed-tts-eval-arrow', revision='27f4c1adee83b5b29b7c4b375f6b976324bda308', split='en').cast_column('ref_audio', Audio(decode=False))
    assert len(ds) == 1088
    assert len({r['sample_id'] for r in ds}) == 1088
    assert all(r['target_text'].strip() and r['ref_text'].strip() and r['ref_audio']['bytes'] for r in ds)
    # Final closure is after every install/init operation. Nothing installs in WORKDIR.
    if Path('/workspace').is_symlink():
        Path('/workspace').unlink()
    elif Path('/workspace').exists():
        shutil.rmtree('/workspace')
    Path('/workspace').mkdir()
    shutil.copytree('/opt/rsi-source', '/workspace/sglang-omni', symlinks=True)
    Path('/workspace/research').mkdir()
    expected = json.loads((BUILD / 'source-manifest.json').read_text())
    for root in [Path('/opt/rsi-source'), Path('/workspace/sglang-omni')]:
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}
        assert actual == set(expected)
        for name, item in expected.items():
            p = root / name
            if item['kind'] == 'symlink':
                assert p.is_symlink() and os.readlink(p) == item['target']
            else:
                assert not p.is_symlink() and digest(p) == item['sha256'] and p.stat().st_mode & 511 == item['mode']
    manifest = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())
    (BUILD / 'realized-distributions.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
