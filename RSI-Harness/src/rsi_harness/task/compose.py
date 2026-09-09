"""Strict extraction of the supported Docker Compose main service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from rsi_harness.errors import UnsupportedTaskError
from rsi_harness.models import GPURequirement, MainServiceConfig


class ComposeMainServiceParser:
    """Translate the supported single-service Compose subset."""

    _allowed_service_keys = {
        "build",
        "deploy",
        "environment",
        "image",
        "shm_size",
        "user",
        "working_dir",
    }
    _rejected_service_keys = {
        "cap_add": "cap_add",
        "devices": "device",
        "extra_hosts": "extra_hosts",
        "ipc": "ipc",
        "links": "links",
        "network_mode": "host network",
        "pid": "pid",
        "privileged": "privileged",
        "volumes": "host volume",
    }

    def parse(self, path: Path) -> MainServiceConfig:
        path = Path(path).resolve()
        try:
            document = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as error:
            raise UnsupportedTaskError(f"invalid Compose file: {error}") from error
        if not isinstance(document, dict) or set(document) != {"services"}:
            raise UnsupportedTaskError(
                "Compose must contain only a services mapping"
            )
        services = document["services"]
        if not isinstance(services, dict) or "main" not in services:
            raise UnsupportedTaskError(
                "Compose requires exactly one service named main"
            )
        if set(services) != {"main"}:
            raise UnsupportedTaskError("Compose sidecar services are unsupported")
        main = services["main"]
        if not isinstance(main, dict):
            raise UnsupportedTaskError("Compose main service must be a mapping")
        self._reject_service_keys(main)
        if "build" in main and "image" in main:
            raise UnsupportedTaskError(
                "Compose main cannot declare both build and image"
            )

        build_context = self._build_context(path, main.get("build"))
        environment = self._environment(main.get("environment"))
        gpu_requirement = self._gpu_requirement(main.get("deploy"))
        try:
            return MainServiceConfig(
                build_context=build_context,
                image=self._optional_string(main.get("image"), "image"),
                workdir=main.get("working_dir"),
                user=self._optional_string(main.get("user"), "user"),
                environment=environment,
                shm_size=self._optional_string(main.get("shm_size"), "shm_size"),
                gpu_requirement=gpu_requirement,
            )
        except ValidationError as error:
            raise UnsupportedTaskError(
                f"invalid Compose WORKDIR or main service: {error}"
            ) from error

    def _reject_service_keys(self, main: dict[str, Any]) -> None:
        for key, message in self._rejected_service_keys.items():
            if key in main:
                raise UnsupportedTaskError(f"Compose {message} is unsupported")
        unknown = sorted(set(main) - self._allowed_service_keys)
        if unknown:
            raise UnsupportedTaskError(
                f"Compose main setting {unknown[0]!r} is unsupported"
            )

    def _build_context(self, compose_path: Path, value: Any) -> Path | None:
        if value is None:
            return None
        if isinstance(value, str):
            raw_context = value
        elif isinstance(value, dict) and set(value) == {"context"}:
            raw_context = value["context"]
        else:
            raise UnsupportedTaskError("Compose build configuration is unsupported")
        if not isinstance(raw_context, str) or not raw_context:
            raise UnsupportedTaskError("Compose build context must be a path")
        if "://" in raw_context or raw_context.startswith("git@"):
            raise UnsupportedTaskError("Compose remote build contexts are unsupported")
        context = (compose_path.parent / raw_context).resolve()
        if not context.exists() or not context.is_dir():
            raise UnsupportedTaskError(
                "Compose build context must be an existing local directory"
            )
        if context != compose_path.parent.resolve():
            raise UnsupportedTaskError(
                "Compose build context must be exactly the environment directory"
            )
        return context

    def _environment(self, value: Any) -> tuple[tuple[str, str], ...]:
        if value is None:
            return ()
        entries: dict[str, str] = {}
        if isinstance(value, dict):
            for raw_name, raw_value in value.items():
                name = self._environment_name(raw_name)
                entries[name] = (
                    f"${{{name}}}" if raw_value is None else str(raw_value)
                )
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, str):
                    raise UnsupportedTaskError(
                        "Compose environment list entries must be strings"
                    )
                name, separator, raw_value = item.partition("=")
                name = self._environment_name(name)
                entries[name] = raw_value if separator else f"${{{name}}}"
        else:
            raise UnsupportedTaskError("Compose environment must be a mapping or list")
        return tuple(sorted(entries.items()))

    @staticmethod
    def _environment_name(value: Any) -> str:
        if not isinstance(value, str) or not value or "=" in value:
            raise UnsupportedTaskError("Compose environment variable name is invalid")
        return value

    def _gpu_requirement(self, deploy: Any) -> GPURequirement | None:
        if deploy is None:
            return None
        devices = self._reservation_devices(deploy)
        if len(devices) != 1 or not isinstance(devices[0], dict):
            raise UnsupportedTaskError(
                "Compose requires exactly one NVIDIA GPU reservation"
            )
        device = devices[0]
        unknown = set(device) - {"driver", "count", "capabilities", "device_ids"}
        if unknown:
            raise UnsupportedTaskError(
                f"Compose GPU reservation setting {sorted(unknown)[0]!r} is unsupported"
            )
        if "device_ids" in device:
            raise UnsupportedTaskError(
                "Compose device_ids are unsupported; GPU allocation belongs "
                "to the caller"
            )
        if device.get("driver") != "nvidia":
            raise UnsupportedTaskError("only NVIDIA GPU reservations are supported")
        if device.get("capabilities") != ["gpu"]:
            raise UnsupportedTaskError(
                "Compose NVIDIA reservation capabilities must be [gpu]"
            )
        count = device.get("count")
        if isinstance(count, bool):
            raise UnsupportedTaskError(
                "Compose NVIDIA reservation count must not be boolean"
            )
        if count != "all" and (not isinstance(count, int) or count <= 0):
            raise UnsupportedTaskError(
                "Compose NVIDIA reservation count must be positive or 'all'"
            )
        return GPURequirement(count=count)

    @staticmethod
    def _reservation_devices(deploy: Any) -> list[Any]:
        current = deploy
        for expected_key in ("resources", "reservations"):
            if not isinstance(current, dict) or set(current) != {expected_key}:
                raise UnsupportedTaskError(
                    "unsupported Compose resource reservation data"
                )
            current = current[expected_key]
        if not isinstance(current, dict) or set(current) != {"devices"}:
            raise UnsupportedTaskError("unsupported Compose resource reservation data")
        devices = current["devices"]
        if not isinstance(devices, list):
            raise UnsupportedTaskError("Compose reservation devices must be a list")
        return devices

    @staticmethod
    def _optional_string(value: Any, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, (str, int)):
            raise UnsupportedTaskError(f"Compose {field} must be a string")
        return str(value)
