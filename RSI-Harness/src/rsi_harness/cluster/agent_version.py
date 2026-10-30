"""The version-output contract shared by submit-host and container probes."""

from __future__ import annotations

import re

from rsi_harness.errors import SetupError

# Accept a complete three-part CLI version, with optional v prefix,
# prerelease and build identifiers. Never match a prefix of another version.
_VERSION = re.compile(
    r"(?<![0-9A-Za-z_.+-])v?"
    r"(\d+\.\d+\.\d+(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?)"
    r"(?![0-9A-Za-z_.+-])"
)


def parse_agent_version(output: str) -> str:
    """Require an unambiguous CLI version; banners and repetition are allowed.

    Unknown or multiple distinct versions are refused rather than guessing
    which executable a wrapper or banner describes.
    """
    versions = set(_VERSION.findall(output))
    if len(versions) != 1:
        raise SetupError(
            "Agent --version must report exactly one distinct three-part version "
            "(optional prerelease/build suffix); output is missing or ambiguous"
        )
    return versions.pop()
