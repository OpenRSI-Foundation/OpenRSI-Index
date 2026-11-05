"""Validate the declared RMSNorm bridge on the assigned accelerator."""
import json
import torch
from sglang.kernels.ops.layernorm.rmsnorm_hf import (
    is_supported_rmsnorm_hf_hidden_size,
    rmsnorm_hf,
)

assert torch.cuda.device_count() == 1
assert [is_supported_rmsnorm_hf_hidden_size(n) for n in (0, 32, 511, 512, 1024, 16384, 16896)] == [False, False, False, True, True, True, False]
torch.manual_seed(17)
results = []
for dtype in (torch.bfloat16, torch.float16):
    for width in (512, 1024, 2048):
        for rows in (1, 16):
            input = torch.randn(rows, width, device='cuda', dtype=dtype)
            weight = torch.randn(width, device='cuda', dtype=dtype)
            original = input.clone()
            eps = 1e-5
            x = input.float()
            expected = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(dtype) * weight
            output = rmsnorm_hf(input, weight, eps)
            assert output.data_ptr() != input.data_ptr()
            assert torch.equal(input, original)
            torch.testing.assert_close(output, expected, rtol=.02 if dtype == torch.bfloat16 else .005,
                                       atol=.02 if dtype == torch.bfloat16 else .005)
            provided = torch.empty_like(input)
            assert rmsnorm_hf(input, weight, eps, out=provided) is provided
            assert torch.equal(output, provided)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                rmsnorm_hf(input, weight, eps, out=provided)
            provided.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output, provided) and torch.equal(input, original)
            results.append(dict(dtype=str(dtype), rows=rows, width=width,
                                max_abs_error=(output.float()-expected.float()).abs().max().item()))
print(json.dumps(dict(status='passed', cases=results, cuda_graph_replay=True,
                     input_preserved=True, explicit_out_contract=True)))
