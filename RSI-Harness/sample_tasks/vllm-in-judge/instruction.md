This is a demo of a model served by vLLM inside an RSI Harness Judge. Nothing
needs to be trained: put a Hugging Face format checkpoint of a small
instruction-tuned model (for example `Qwen/Qwen2.5-1.5B-Instruct`) into
`/workspace/checkpoint`, then submit with `rsi-submit`.

The Judge receives `/workspace` as a read-only snapshot. Its fixed procedure
starts vLLM serving `/workspace/checkpoint` on the Judge's GPU and runs
Harbor's `terminus-2` agent with that model over the tasks shipped in
`/tests`, each task in an environment the host broker creates. The reward is
the fraction of those tasks the agent solves; a metering proxy in front of
vLLM records the tokens each task's agent spends, reported next to it.
