# 🚀 Quick Start

You need a Linux machine with Docker, NVIDIA Container Toolkit, the GPUs
required by your task, and [uv](https://docs.astral.sh/uv/).

## 1 — Install RSI-Harness

```bash
git clone https://github.com/RSI-Index/RSIs-First-Exam.git
cd RSIs-First-Exam
uv tool install ./RSI-Harness
```

## 2 — Pick a task

```bash
find rsi-tasks -name task.toml -print

export RSI_TASK="$PWD/rsi-tasks/minference-sparse-prefill"
export RSI_GPU_POOL="0"
export RSI_MODEL="gpt-5.6-sol"
```

Check the task's `README.md` and `task.toml` for its GPU and storage
requirements. `RSI_GPU_POOL` accepts comma-separated GPU indexes or UUIDs.

## 3 — Run it

Sign in to Codex on the host, then run:

```bash
sudo -E "$(command -v rsi-harness)" run "$RSI_TASK" \
  --agent codex \
  --agent-auth local \
  --model "$RSI_MODEL" \
  --reasoning-effort xhigh \
  --gpus "$RSI_GPU_POOL" \
  --primary-reward reward \
  --verbose
```

To use an API key instead, export `RSI_AGENT_API_KEY` and optionally
`RSI_AGENT_API_BASE_URL`, then remove `--agent-auth local`. For more information about RSI
Harness commands and configuration options, see
[RSI Harness Quick Commands](../../RSI-Harness/README.md#quick-commands).

## 4 — View results

```bash
sudo -E "$(command -v rsi-harness)" visualize
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Raw logs are under `logs/runs/`.

These tasks use the Harbor task format but must be run with `rsi-harness`, not
`harbor run`, to support long-horizon RSI.
