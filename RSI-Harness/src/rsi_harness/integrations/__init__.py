"""Adapters for pinned third-party integration surfaces.

Exports load lazily (PEP 562): importing a sandbox module from this package,
as the Harbor plugin does inside a Judge, must not pull in ``rsi_loop``.
"""

from importlib import import_module

_EXPORTS = {
    "RSILoopAgentAdapter": "rsi_harness.integrations.rsi_loop",
    "generate_submit_client": "rsi_harness.integrations.submit_client",
}

__all__ = ["RSILoopAgentAdapter", "generate_submit_client"]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
