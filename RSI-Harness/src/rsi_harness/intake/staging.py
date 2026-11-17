"""Stage a verified intake task as a native RSI-Harness (OpenRSI-X) task.

Brewer tasks declare ``schema_version = "1.4+ale+openrsi"``: Harbor 1.4 plus an
``[rsi]`` table (domain, evaluation, runtime, resources) and a ``[comment]``
table. Harbor ignores unknown tables, so left alone the ``[rsi]`` settings
would be dropped without a word. Staging copies the task and:

- rewrites ``schema_version`` to ``"1.4"``, the version the compiler accepts;
- maps the ``[rsi]`` fields the Harness supports natively:
  ``evaluation.max_submissions`` and ``runtime.clean_image`` into
  ``[metadata.rsi_harness.verifier]``, ``evaluation.score_direction``,
  ``cooldown_sec`` and ``primary_reward`` into the run options, and
  ``domain.category`` / ``subcategory`` into ``[metadata]``;
- refuses what it cannot honour yet (``rsi.runtime.sandbox``,
  ``rsi.resources.world``) with a clear reason;
- keeps ``[rsi]`` and ``[comment]`` verbatim under ``[metadata.openrsi]``.

``reward.txt`` needs no bridge: the Harness reads ``reward.json`` first and
then Harbor's scalar ``reward.txt``.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi_harness.intake.bundle import BundleError

NATIVE_SCHEMA = "1.4"
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_SLUG = re.compile(r"[^a-z0-9_-]+")


class StagingError(BundleError):
    """The task cannot be run natively by the Harness as declared."""


@dataclass(frozen=True)
class RunOptions:
    """Per-task ``rsi-harness run`` options taken from the task's ``[rsi]``."""

    score_direction: str | None = None
    cooldown_seconds: float | None = None
    primary_reward: str | None = None


@dataclass(frozen=True)
class StagedTask:
    task_dir: Path
    task_id: str
    adapted: bool
    notices: tuple[str, ...]
    run_options: RunOptions = field(default_factory=RunOptions)


# --- task ids: one canonical id per task, whichever runtime reports it ---


def _slug(raw: str, fallback: str) -> str:
    cleaned = _SLUG.sub("_", raw.strip().lower()).strip("_-")
    if not cleaned:
        return fallback
    if not cleaned[0].isalnum():
        cleaned = f"t_{cleaned}"
    return cleaned[:100]


def default_org(task_repo: str, domain: str) -> str:
    repo = task_repo.lower()
    if "openrsi-x" in repo:
        return "openrsi-x"
    if "openrsi-index" in repo:
        return "openrsi-index"
    if domain:
        head = _slug(domain.split("/")[0], "")
        if head:
            return head
    return "openrsi"


def canonical_task_id(name: str, *, task_repo: str = "", domain: str = "") -> str:
    org = default_org(task_repo, domain)
    if "/" in name:
        raw_org, _, raw_name = name.partition("/")
        return f"{_slug(raw_org, org)}/{_slug(raw_name, 'task')}"
    return f"{org}/{_slug(name, 'task')}"


# --- TOML output (the stdlib reads TOML but cannot write it) ---


def _key(key: str) -> str:
    return key if _BARE_KEY.fullmatch(key) else json.dumps(key, ensure_ascii=False)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items())
        return "{ " + inner + " }" if inner else "{}"
    raise StagingError(f"cannot write TOML value of type {type(value).__name__}")


def _is_table_array(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) for item in value)
    )


def _emit(lines: list[str], table: dict[str, Any], path: tuple[str, ...]) -> None:
    for key, value in table.items():
        if not isinstance(value, dict) and not _is_table_array(value):
            lines.append(f"{_key(key)} = {_value(value)}")
    for key, value in table.items():
        sub = (*path, key)
        header = ".".join(_key(part) for part in sub)
        if isinstance(value, dict):
            lines.extend(["", f"[{header}]"])
            _emit(lines, value, sub)
        elif _is_table_array(value):
            for item in value:
                lines.extend(["", f"[[{header}]]"])
                _emit(lines, item, sub)


def dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    _emit(lines, data, ())
    text = "\n".join(lines).lstrip("\n") + "\n"
    tomllib.loads(text)  # never stage a file the Harness cannot read back
    return text


# --- [rsi] mapping ---


def _table(parent: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = parent.setdefault(key, {})
    if not isinstance(value, dict):
        raise StagingError(f"{where} must be a table")
    return value


def _apply_rsi(
    config: dict[str, Any], rsi: dict[str, Any], notices: list[str]
) -> RunOptions:
    metadata = _table(config, "metadata", "metadata")
    options: dict[str, Any] = {}

    domain = rsi.get("domain")
    if isinstance(domain, dict):
        for name in ("category", "subcategory"):
            if isinstance(domain.get(name), str) and name not in metadata:
                metadata[name] = domain[name]
                notices.append(f"mapped rsi.domain.{name} to [metadata].{name}")

    evaluation = rsi.get("evaluation")
    if evaluation is not None and not isinstance(evaluation, dict):
        raise StagingError("rsi.evaluation must be a table")
    evaluation = evaluation or {}
    verifier_meta: dict[str, Any] | None = None

    def verifier() -> dict[str, Any]:
        nonlocal verifier_meta
        if verifier_meta is None:
            harness = _table(metadata, "rsi_harness", "metadata.rsi_harness")
            verifier_meta = _table(harness, "verifier", "metadata.rsi_harness.verifier")
        return verifier_meta

    max_sub = evaluation.get("max_submissions")
    if max_sub is not None:
        if type(max_sub) is not int or max_sub < 1:
            raise StagingError(
                "rsi.evaluation.max_submissions must be a positive integer"
            )
        declared = verifier().get("max_submissions")
        if declared is not None and declared != max_sub:
            raise StagingError(
                "rsi.evaluation.max_submissions disagrees with "
                "metadata.rsi_harness.verifier.max_submissions"
            )
        verifier()["max_submissions"] = max_sub
        notices.append("mapped rsi.evaluation.max_submissions to the Judge budget")

    direction = evaluation.get("score_direction")
    if direction is not None:
        if direction not in ("maximize", "minimize"):
            raise StagingError(
                "rsi.evaluation.score_direction must be maximize or minimize"
            )
        options["score_direction"] = direction
        notices.append(f"mapped rsi.evaluation.score_direction={direction} to the run")

    cooldown = evaluation.get("cooldown_sec")
    if cooldown is not None:
        if (
            isinstance(cooldown, bool)
            or not isinstance(cooldown, int | float)
            or cooldown < 0
        ):
            raise StagingError(
                "rsi.evaluation.cooldown_sec must be a non-negative number"
            )
        if cooldown > 0:
            options["cooldown_seconds"] = float(cooldown)
            notices.append("mapped rsi.evaluation.cooldown_sec to the run's cooldown")

    primary = evaluation.get("primary_reward")
    if primary is not None:
        if not isinstance(primary, str) or not primary.strip():
            raise StagingError("rsi.evaluation.primary_reward must be a reward name")
        options["primary_reward"] = primary.strip()
        notices.append("mapped rsi.evaluation.primary_reward to the run")

    runtime = rsi.get("runtime")
    if runtime is not None and not isinstance(runtime, dict):
        raise StagingError("rsi.runtime must be a table")
    runtime = runtime or {}
    sandbox = runtime.get("sandbox")
    if sandbox is not None and sandbox is not False:
        raise StagingError(
            "rsi.runtime.sandbox is not supported by RSI intake yet; "
            "declare a sandbox through [metadata.rsi_harness] instead"
        )
    clean_image = runtime.get("clean_image")
    if clean_image is not None:
        if type(clean_image) is not bool:
            raise StagingError("rsi.runtime.clean_image must be a boolean")
        verifier()["clean_image"] = clean_image
        notices.append("mapped rsi.runtime.clean_image to the Judge image")

    resources = rsi.get("resources")
    if isinstance(resources, dict) and resources.get("world") is not None:
        raise StagingError("rsi.resources.world is not supported by RSI intake yet")

    return RunOptions(**options)


def stage_task(
    source: Path,
    destination: Path,
    *,
    fallback_name: str,
    task_repo: str = "",
    domain: str = "",
) -> StagedTask:
    """Copy ``source`` to ``destination`` and make it a native Harness task."""
    config_path = source / "task.toml"
    if not config_path.is_file():
        raise StagingError("task has no task.toml")
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise StagingError(f"task.toml is not valid TOML: {error}") from error

    shutil.copytree(source, destination)
    notices: list[str] = []
    changed = False

    version = config.get("schema_version", config.get("version", NATIVE_SCHEMA))
    if not isinstance(version, str):
        raise StagingError("schema_version must be a string")
    if version != NATIVE_SCHEMA:
        if not version.startswith(NATIVE_SCHEMA + "+"):
            raise StagingError(f"unsupported task schema_version {version!r}")
        config.pop("version", None)
        config["schema_version"] = NATIVE_SCHEMA
        changed = True
        notices.append(f"rewrote schema_version {version!r} to {NATIVE_SCHEMA!r}")

    run_options = RunOptions()
    preserved: dict[str, Any] = {}
    for name in ("rsi", "comment"):
        if name in config:
            preserved[name] = config.pop(name)
            changed = True
    if "rsi" in preserved:
        if not isinstance(preserved["rsi"], dict):
            raise StagingError("[rsi] must be a table")
        run_options = _apply_rsi(config, preserved["rsi"], notices)
    if preserved:
        metadata = _table(config, "metadata", "metadata")
        if "openrsi" in metadata:
            raise StagingError(
                "[metadata.openrsi] is reserved for the staged [rsi] table"
            )
        metadata["openrsi"] = preserved
        notices.append(
            "kept "
            + " and ".join(f"[{name}]" for name in preserved)
            + " under [metadata.openrsi]"
        )

    # Harbor defaults to no GPUs; the Harness compiler wants that said
    # explicitly, so a Brewer task that leaves it out gets Harbor's default.
    environment = _table(config, "environment", "environment")
    if "gpus" not in environment:
        environment["gpus"] = 0
        changed = True
        notices.append("set [environment].gpus = 0 (Harbor's default; none declared)")

    if changed:
        target = destination / "task.toml"
        target.chmod(target.stat().st_mode | 0o600)  # a read-only source stays writable
        target.write_text(dump_toml(config), encoding="utf-8")

    task_table = config.get("task")
    raw_name = (
        task_table.get("name")
        if isinstance(task_table, dict) and isinstance(task_table.get("name"), str)
        else fallback_name
    )
    return StagedTask(
        task_dir=destination,
        task_id=canonical_task_id(raw_name, task_repo=task_repo, domain=domain),
        adapted=changed,
        notices=tuple(notices),
        run_options=run_options,
    )
