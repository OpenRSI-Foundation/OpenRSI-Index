Execute exactly this one foreground command, once:

```bash
torchrun --nnodes 2 --nproc-per-node 8 /task-tools/smoke.py
```

Do not add flags, node ranks, rendezvous options, background processes, or a
second launcher. After the command returns successfully, confirm that
`/workspace/smoke.json` exists and call `rsi-submit`.
