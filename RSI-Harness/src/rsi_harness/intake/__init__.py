"""Runtime Task Intake API (``ale.task.intake.v1``) for RSI-Harness.

Lets an upstream task brewer (``ale-brew-science``) push reviewed, sealed task
bundles that RSI-Harness then runs natively, as OpenRSI-X runs its own tasks.
See ``docs/task-intake.md``.
"""

from rsi_harness.intake.protocol import (
    DEFAULT_PORT,
    PROTOCOL,
    ExecutionMode,
    IntakeStatus,
)

__all__ = ["DEFAULT_PORT", "PROTOCOL", "ExecutionMode", "IntakeStatus"]
