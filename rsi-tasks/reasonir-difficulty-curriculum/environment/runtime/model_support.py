"""Fixed ReasonIR model and LoRA helpers shared by Work-side trusted tools."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path("/opt/reasonir-task/base/ReasonIR-8B")
BASE_REVISION = "c3d0690370ff4a8c3d3882d8dfa85c43650034fa"
TARGET_MODULES = ["q_proj", "o_proj", "v_proj", "k_proj", "w1", "w2", "w3"]
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "inference_mode": False,
    "target_modules": TARGET_MODULES,
    "bias": "none",
    "task_type": "FEATURE_EXTRACTION",
}


class AliasedMLP(nn.Module):
    """Names Llama projections w1/w2/w3 without changing its computation."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.w1 = original.gate_proj
        self.w2 = original.down_proj
        self.w3 = original.up_proj
        self.act_fn = original.act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states))


def install_projection_aliases(model: nn.Module) -> None:
    layers = model.layers
    for layer in layers:
        if not isinstance(layer.mlp, AliasedMLP):
            layer.mlp = AliasedMLP(layer.mlp)


def load_base(training: bool) -> tuple[nn.Module, object]:
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        BASE_DIR,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    install_projection_aliases(model)
    if training:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
        model.train()
    else:
        model.eval()
    return model, tokenizer


def canonical_adapter_config() -> dict:
    return {
        "base_model_name_or_path": str(BASE_DIR),
        "base_revision": BASE_REVISION,
        **LORA_CONFIG,
    }


def write_canonical_config(path: Path) -> None:
    path.write_text(json.dumps(canonical_adapter_config(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
