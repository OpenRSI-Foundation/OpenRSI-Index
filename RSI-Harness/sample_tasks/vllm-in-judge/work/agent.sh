#!/bin/bash
# The demo's scripted Work agent (no model CLI; run through
# tests/acceptance/scripted_cli.py). It trains nothing: it places a small
# existing checkpoint into the WORKDIR, which the Judge receives as a
# read-only snapshot, then submits once. Work has public egress, so the
# checkpoint comes straight from the Hugging Face Hub at a pinned revision.
#
#   RSI_VLLM_MODEL     Hub repository (default Qwen/Qwen2.5-1.5B-Instruct)
#   RSI_VLLM_REVISION  its revision (default: the pinned commit of the
#                      default model, else main)
# Results are "RSI-ACCEPTANCE <key> <value>" lines in the Agent output.
set -uo pipefail
default_model=Qwen/Qwen2.5-1.5B-Instruct
model=${RSI_VLLM_MODEL:-$default_model}
if [[ -n ${RSI_VLLM_REVISION:-} ]]; then
    revision=$RSI_VLLM_REVISION
elif [[ $model == "$default_model" ]]; then
    revision=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
else
    revision=main
fi
checkpoint=${RSI_VLLM_CHECKPOINT:-/workspace/checkpoint}

note() {
    printf 'RSI-ACCEPTANCE %s\n' "$*"
}

# The Hub cache lives outside the WORKDIR and is dropped afterwards: only
# the checkpoint itself goes to the Judge.
started=$(date +%s)
HF_HOME=/tmp/rsi-hf HF_HUB_DISABLE_TELEMETRY=1 \
    hf download "$model" --revision "$revision" --local-dir "$checkpoint"
status=$?
rm -rf /tmp/rsi-hf
# A whole checkpoint: its config and weights.
if ((status == 0)) && ! compgen -G "$checkpoint/*.safetensors" > /dev/null \
    || [[ ! -f $checkpoint/config.json ]]; then
    status=1
fi
note "checkpoint-download $status $model@$revision $(( $(date +%s) - started ))s"
note "checkpoint-files $(cd "$checkpoint" 2> /dev/null && ls | tr '\n' ' ')"
note "checkpoint-bytes $(du -sb "$checkpoint" 2> /dev/null | cut -f1)"
if ((status != 0)); then
    exit 1
fi
rsi-submit
note "submit-exit $?"
