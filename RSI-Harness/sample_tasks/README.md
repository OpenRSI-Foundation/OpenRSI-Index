# Sample tasks

| Task | What it shows |
| --- | --- |
| [`harbor-in-judge`](harbor-in-judge) | A Judge whose fixed procedure runs Harbor over a Terminal-Bench 2 subset, a Compose sidecar and a Dockerfile-built task, each environment through the sandbox broker |
| [`vllm-in-judge`](vllm-in-judge) | Work leaves a model checkpoint; the Judge serves it with vLLM on its GPU and runs an LLM agent over Harbor tasks in brokered environments |
| [`swebench-in-judge`](swebench-in-judge) | Three SWE-bench Verified tasks run offline through the sandbox broker |
| [`huggingface__transformers…test_serve…lv1`](huggingface__transformers.e2e8dbed.test_serve.4e7860c7.lv1) | GPU research task: implement `transformers` OpenAI-compatible model serving (2 GPUs) |
| [`linkedin__liger-kernel…test_fused_neighborhood_attention…lv2`](linkedin__liger-kernel.c856fbab.test_fused_neighborhood_attention.78217be4.lv2) | GPU research task: implement `liger-kernel` fused neighborhood attention (2 GPUs) |
| [`linkedin__liger-kernel…test_poly_norm…lv1`](linkedin__liger-kernel.c856fbab.test_poly_norm.7b0e3399.lv1) | GPU research task: implement `liger-kernel` polynomial normalization (2 GPUs) |

The `*-in-judge` samples carry the operator policy that approves them
(`operator-policy.toml`) and, where images must be pre-pulled, an
`images.manifest`; see [the operator guide](../docs/sandbox-operator-guide.md).
