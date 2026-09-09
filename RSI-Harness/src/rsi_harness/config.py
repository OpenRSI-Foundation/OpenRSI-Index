"""Mutable runtime-only configuration, including secret values."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EngineConfig(BaseModel):
    """Runtime settings that must never be serialized into a :class:`RunPlan`."""

    model_config = ConfigDict(extra="forbid")

    data_root: Path = Path(".rsi-harness")
    logs_root: Path = Path("logs")
    secret_env: dict[str, str] = Field(default_factory=dict)
    verifier_secret_env: dict[str, str] = Field(default_factory=dict)

    @field_validator("data_root", "logs_root", mode="after")
    @classmethod
    def resolve_runtime_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()
