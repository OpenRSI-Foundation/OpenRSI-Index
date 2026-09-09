"""Validation for immutable images granted root helper authority."""

from __future__ import annotations

import re

_IMMUTABLE_IMAGE_REF = re.compile(
    r"(?:sha256:[0-9a-fA-F]{64}|[^@\s]+@sha256:[0-9a-fA-F]{64})\Z"
)


def is_immutable_image_ref(value: str) -> bool:
    """Return whether *value* is an exact Docker image ID or repo digest."""

    return _IMMUTABLE_IMAGE_REF.fullmatch(value) is not None


__all__ = ["is_immutable_image_ref"]
