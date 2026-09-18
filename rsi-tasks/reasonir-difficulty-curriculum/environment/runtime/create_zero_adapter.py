#!/usr/bin/env python3
"""Materialize the canonical all-zero adapter used by Solution."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model_support import load_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model, _ = load_base(training=True)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            with torch.no_grad():
                parameter.zero_()
    args.output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.output, safe_serialization=True)


if __name__ == "__main__":
    main()
