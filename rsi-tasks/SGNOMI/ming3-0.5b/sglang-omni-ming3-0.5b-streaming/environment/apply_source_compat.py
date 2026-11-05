"""Install the complete declared Omni archive in a verified cached environment."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,tarfile,tempfile
build=Path('/opt/rsi-build');metadata=json.loads((build/'source-compatibility.json').read_text());expected=json.loads((build/'source-manifest.json').read_text())
assert hashlib.sha256((build/'source.tar.gz').read_bytes()).hexdigest()==metadata['adapted_archive_sha256']
previous_path=build/'previous-source-manifest.json';previous=json.loads(previous_path.read_text())
accepted=[x['manifest_sha256'] for x in metadata.get('accepted_archive_inputs',[])]+[metadata['adapted_manifest_sha256'],metadata['upstream_manifest_sha256']]
assert hashlib.sha256(previous_path.read_bytes()).hexdigest() in accepted
for root in (Path('/opt/rsi-source'),Path('/workspace/sglang-omni')):
 observed={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()};assert observed==set(previous)
 for name,spec in previous.items():
  p=root/name
  if spec['kind']=='file':assert not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256'],name
  else:assert p.is_symlink() and os.readlink(p)==spec['target'],name
 # The entire previous immutable inventory matched. Replace only these two task roots.
 shutil.rmtree(root);root.mkdir(parents=True)
 with tarfile.open(build/'source.tar.gz') as archive:
  for m in archive:
   n=m.name.partition('/')[2]
   if not n:continue
   p=root/n;assert p.resolve().is_relative_to(root)
   if m.isdir():p.mkdir(parents=True,exist_ok=True)
   elif m.isfile():
    data=archive.extractfile(m).read();assert n in expected and hashlib.sha256(data).hexdigest()==expected[n]['sha256']
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(m.mode & 0o755 | 0o600)
   elif m.issym():
    assert n in expected and expected[n]['target']==m.linkname
    p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(m.linkname)
   else:raise ValueError(m.name)
 observed={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()};assert observed==set(expected)
shutil.copytree('/opt/rsi-source','/tmp/rsi-install',symlinks=True)
try:subprocess.run(['/opt/sglang/bin/python3','-m','pip','install','--no-deps','--no-build-isolation','/tmp/rsi-install'],check=True)
finally:shutil.rmtree('/tmp/rsi-install')
print(json.dumps(dict(status='declared_latest_omni_installed',upstream=metadata['upstream_revision'],archive_sha256=metadata['adapted_archive_sha256'])))
