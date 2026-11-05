"""Preserve batch context when the ASR attention bridge takes native fallbacks."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tarfile
import unittest

def load_bridge(torch, fused):
    archive=Path(__file__).resolve().parents[1]/'environment/source.tar.gz'
    with tarfile.open(archive) as source:
        member=next(m for m in source if m.name.endswith('/sglang_omni/models/qwen3_asr/sglang_model.py'))
        tree=ast.parse(source.extractfile(member).read().decode())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                  and n.name=='fused_asr_forward_prepare_native')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,function],type_ignores=[]))
    namespace=dict(torch=torch,fused_qk_norm_rope=fused)
    exec(compile(module,str(archive),'exec'),namespace)
    return namespace[function.name]

class AsrAttentionCompatibilityTests(unittest.TestCase):
    def check_fallback(self, *, bfloat16_input, fused):
        bf16=object()
        tensor=SimpleNamespace(dtype=bf16 if bfloat16_input else object())
        positions,batch,result=object(),object(),object()
        calls=[]
        def native(actual_positions,actual_states,forward_batch):
            self.assertIs(actual_positions,positions)
            self.assertIs(actual_states,tensor)
            self.assertIs(forward_batch,batch)
            calls.append(batch)
            return result
        attention=SimpleNamespace(_asr_unfused_forward_prepare_native=native)
        bridge=load_bridge(SimpleNamespace(bfloat16=bf16),fused)
        self.assertIs(bridge(attention,positions=positions,hidden_states=tensor,
                             forward_batch=batch),result)
        self.assertEqual(calls,[batch])

    def test_non_bfloat16_native_fallback_retains_batch_context(self):
        self.check_fallback(bfloat16_input=False,fused=object())

    def test_unavailable_fused_kernel_retains_batch_context(self):
        self.check_fallback(bfloat16_input=True,fused=None)

if __name__=='__main__':
    unittest.main()
