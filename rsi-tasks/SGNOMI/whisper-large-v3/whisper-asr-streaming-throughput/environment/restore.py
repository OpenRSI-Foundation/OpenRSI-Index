"""Idempotently materialize only the public baseline candidate surfaces."""
from pathlib import Path
import shutil

src = Path('/opt/asr-reference/source/sglang_omni')
dst = Path('/workspace/sglang-omni/sglang_omni')
for rel in ['models/whisper_asr', 'scheduling', 'comm', 'http', 'serve',
            'model_runner', 'pipeline', 'relay', 'sampling', 'utils', 'kernels']:
    target = dst / rel
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    if (src / rel).exists():
        shutil.copytree(src / rel, target, symlinks=True)
shutil.copyfile('/opt/asr-reference/candidate.yaml', '/workspace/candidate.yaml')
