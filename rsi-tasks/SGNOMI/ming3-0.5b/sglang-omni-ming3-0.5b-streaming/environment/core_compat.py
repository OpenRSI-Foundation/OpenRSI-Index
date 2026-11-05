"""Declared API bridge for Omni b14c0bf with SGLang core b8100b74b.

Core PR #47503 replaced rmsnorm_hf with norm.rmsnorm and its explicit
cast_x_before_out_mul flag. This bridge preserves Omni's old return contract
without changing upstream source files or restoring an older kernel.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

SHIM = '''"""Task-declared bridge to core's unified HF-rounding RMSNorm API."""
import torch
from sglang.kernels.ops.layernorm.norm import (
    is_supported_rmsnorm_hidden_size,
    rmsnorm,
)

def is_supported_rmsnorm_hf_hidden_size(hidden_size):
    return hidden_size % 512 == 0 and is_supported_rmsnorm_hidden_size(hidden_size)

def rmsnorm_hf(input, weight, eps=1e-6, out=None):
    output = torch.empty_like(input) if out is None else out
    rmsnorm(input, weight, out=output, eps=eps, cast_x_before_out_mul=True)
    return output
'''

def install():
    spec = importlib.util.find_spec('sglang')
    assert spec and spec.submodule_search_locations
    target = Path(next(iter(spec.submodule_search_locations))) / 'kernels/ops/layernorm/rmsnorm_hf.py'
    assert target.parent.joinpath('norm.py').is_file()
    if target.exists():
        assert target.read_text() == SHIM, 'Refuse to overwrite an upstream or unknown module'
    else:
        target.write_text(SHIM)
        target.chmod(0o644)
    proof = dict(upstream_core='b8100b74bf4f6563f402c20a1e1af685cbd2246e',
                 upstream_omni='0c663861d7d4a558e4f53258c4d48cfce577b47e',
                 reason='Core #47503 unified RMSNorm; Omni retains its old import and returning API',
                 path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                 semantics='latest norm.rmsnorm, explicit output allocation, cast_x_before_out_mul=True')
    Path('/opt/rsi-build/core-compatibility.json').write_text(json.dumps(proof, indent=2))
    return proof

if __name__ == '__main__':
    print(json.dumps(install()))
