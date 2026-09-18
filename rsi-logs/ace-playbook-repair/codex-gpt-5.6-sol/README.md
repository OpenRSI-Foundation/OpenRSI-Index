# Agent transcript

`agent_output.txt.gz` is the complete sanitized terminal transcript, compressed deterministically for publication. Restore the text without changing its order:

```sh
gzip -dc agent_output.txt.gz > agent_output.txt
```

The adjacent manifest records the compressed and restored sizes and SHA-256 digests.
