"""Stable public errors raised by RSI Harness."""

from enum import StrEnum


class ErrorCode(StrEnum):
    SETUP_ERROR = "setup_error"
    SUBMISSION_ERROR = "submission_error"
    VERIFIER_ERROR = "verifier_error"
    VERIFIER_TIMEOUT = "verifier_timeout"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    NO_VALID_SUBMISSION = "no_valid_submission"


class HarnessError(Exception):
    """A public error with a stable machine-readable classification."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class SetupError(HarnessError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.SETUP_ERROR, message)


class SubmissionError(HarnessError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.SUBMISSION_ERROR, message)


class RetryableSubmissionError(SubmissionError):
    """A safe preflight rejection that does not consume a Judge round."""


class VerifierError(HarnessError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.VERIFIER_ERROR, message)


class VerifierTimeoutError(HarnessError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.VERIFIER_TIMEOUT, message)


class InfrastructureError(HarnessError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.INFRASTRUCTURE_ERROR, message)


class ContainerExecNotStartedError(InfrastructureError):
    """Docker rejected an exec before any output-capture obligation existed."""


class UnsupportedTaskError(SetupError):
    """The Harbor task uses a capability outside the supported contract."""


class StateTransitionError(RuntimeError):
    """An impossible coordinator lifecycle transition was attempted."""
