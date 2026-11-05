"""Image-build preparation only; no serving, inference or evaluation."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path('/opt/asr-reference')
SOURCE = ROOT / 'source'
ARCHIVE_SHA = 'da4c7acd9b895494fe5c9aa7a7884c3c181af6f69b438fa68acf7ad4920b1298'
SURFACES = ['models/moss_transcribe_diarize', 'scheduling', 'comm', 'http', 'serve',
            'model_runner', 'pipeline', 'relay', 'sampling', 'utils', 'kernels']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def inventory(root):
    result = {}
    for p in sorted(root.rglob('*')):
        key = p.relative_to(root).as_posix()
        if p.is_symlink():
            result[key] = {'type': 'symlink', 'target': os.readlink(p)}
        elif p.is_file():
            result[key] = {'type': 'file', 'sha256': digest(p)}
        elif p.is_dir():
            result[key] = {'type': 'directory'}
        else:
            raise RuntimeError('unexpected build product: ' + str(p))
    return result


def prepare():
    archive = ROOT / 'source.tar.gz'
    assert digest(archive) == ARCHIVE_SHA
    SOURCE.mkdir()
    with tarfile.open(archive) as t:
        members = t.getmembers()
        prefix = members[0].name.split('/')[0] + '/'
        for m in members:
            if not m.name.startswith(prefix):
                continue
            m.name = m.name[len(prefix):]
            if m.name:
                t.extract(m, SOURCE, filter='data')
    expected = json.loads((ROOT / 'source-manifest.json').read_text())
    for rel, sha in expected.items():
        assert digest(SOURCE / rel) == sha, rel
    subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'adapt_baseline.py'), str(SOURCE)], check=True)
    model = Path('/opt/models/moss-transcribe-diarize')
    model.mkdir(parents=True)
    for entry in json.loads((ROOT / 'model-manifest.json').read_text()):
        p = model / entry['path']
        p.parent.mkdir(parents=True, exist_ok=True)
        url = ('https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/resolve/'
               '704aa4a9c304e8520be88901e0d1960158ef5b15/' + entry['path'])
        with urllib.request.urlopen(url, timeout=300) as src, p.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8 * 1024**2)
        assert p.stat().st_size == entry['size']
        if entry['sha256']:
            assert digest(p) == entry['sha256']
        else:
            b = p.read_bytes()
            assert hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == entry['git_sha1']


def close():
    # Installation used the reference source, never the candidate tree.
    for p in list(SOURCE.rglob('__pycache__')) + list(SOURCE.glob('*.egg-info')) + list(SOURCE.glob('build')):
        shutil.rmtree(p)
    work = Path('/workspace')
    work.mkdir(exist_ok=True)
    shutil.copytree(SOURCE, work / 'sglang-omni', symlinks=True)
    shutil.copyfile(ROOT / 'candidate.yaml', work / 'candidate.yaml')
    (work / 'research').mkdir()
    # Captured by the host before any Agent runs, then injected under /tests.
    record = {'workspace': inventory(work),
              'assets': {str(p): inventory(p) for p in
                         [ROOT, Path('/opt/models/moss-transcribe-diarize'), Path('/opt/asr-venv')]}}
    (ROOT / 'build-inventory.json').write_text(json.dumps(record, sort_keys=True))


if __name__ == '__main__':
    import sys
    {'prepare': prepare, 'close': close}[sys.argv[1]]()
