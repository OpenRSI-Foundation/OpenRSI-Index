# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Candidate-owned optimizer factory.

The checked-in starter deliberately reproduces the locked AdamH control.  It
exists so the environment and checkpoint contract can be smoke-tested before
research begins; an eligible submission must replace it with a genuinely new
optimizer and provide the required novelty/ablation evidence.
"""

from __future__ import annotations

from typing import Any

from marin_adamh import build_megatron_optimizer as build_adamh_optimizer


def install_runtime() -> None:
    """Install optional candidate-owned process-wide behavior."""


def build_megatron_optimizer(config: Any, model_chunks: Any, **kwargs: Any) -> Any:
    """Return the starter AdamH control through the candidate optimizer hook."""

    return build_adamh_optimizer(config, model_chunks, **kwargs)


__all__ = ["build_megatron_optimizer", "install_runtime"]
