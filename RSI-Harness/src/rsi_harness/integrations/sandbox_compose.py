"""Compose files to one brokered EnvSpec, translated on the caller side.

The broker never runs ``docker compose`` and never parses YAML (spec 5 M-2):
this front-end runs inside Work or Judge, next to its caller, and sends the
broker only a strict EnvSpec, which cannot express host authority. Anything
compose can say that an EnvSpec cannot is refused here with its key path;
every silent-looking normalization is returned as a note instead.

Stdlib and PyYAML only (JSON compose files load without PyYAML). The harness
injects this file into every sandbox endpoint as ``py/rsi_sandbox_compose.py``,
so it must never import ``rsi_harness``.
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import math
import os
import posixpath
import re
import shlex
import stat
import tarfile
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

MIB = 1024**2
MAX_VOLUMES = 8
# The EnvSpec's reserved mount roots (sandbox_env_contracts): refused here
# too, so such a compose fails at definition time, before any pull.
RESERVED_MOUNT_ROOTS = ("/proc", "/sys", "/dev", "/run/rsi-harness")
MAX_STOP_GRACE_SEC = 30
DEFAULT_TMPFS_MB = 64
# ``rsi-sandbox up`` / ``compose up``: the default readiness wait, capped at
# what is left of the env's lifetime. The wait is measured here, the broker
# checks it on arrival: this much of the lifetime is left for the way there.
DEFAULT_START_WAIT_SEC = 300.0
START_SLACK_SEC = 1.0
# Docker's default capability set: ``cap_drop: [ALL]`` drops exactly these.
DEFAULT_CAPABILITIES = (
    "AUDIT_WRITE",
    "CHOWN",
    "DAC_OVERRIDE",
    "FOWNER",
    "FSETID",
    "KILL",
    "MKNOD",
    "NET_BIND_SERVICE",
    "NET_RAW",
    "SETFCAP",
    "SETGID",
    "SETPCAP",
    "SETUID",
    "SYS_CHROOT",
)
_TOP_LEVEL = frozenset(
    {"version", "name", "services", "volumes", "networks", "configs", "secrets"}
)
# Every service key the front-end understands; anything else is refused.
_SERVICE_KEYS = frozenset(
    {
        "image",
        "build",
        "pull_policy",
        "platform",
        "command",
        "entrypoint",
        "environment",
        "env_file",
        "working_dir",
        "user",
        "group_add",
        "hostname",
        "networks",
        "network_mode",
        "links",
        "container_name",
        "extra_hosts",
        "healthcheck",
        "depends_on",
        "init",
        "tty",
        "stdin_open",
        "read_only",
        "privileged",
        "cap_drop",
        "security_opt",
        "cpus",
        "mem_limit",
        "memswap_limit",
        "mem_reservation",
        "cpu_shares",
        "cpu_percent",
        "pids_limit",
        "deploy",
        "shm_size",
        "tmpfs",
        "ulimits",
        "volumes",
        "configs",
        "secrets",
        "stop_signal",
        "stop_grace_period",
        "profiles",
        "labels",
        "annotations",
        "logging",
        "develop",
        "restart",
        "ports",
        "expose",
        "scale",
        "ipc",
        "cgroup",
        "oom_kill_disable",
    }
)
# domainname is refused (it changes the FQDN); these only decorate.
_IGNORED = ("labels", "annotations", "logging", "develop", "stdin_open")
_UNION_LISTS = frozenset({"cap_drop", "group_add", "security_opt", "ports", "expose"})
_BUILD_KEYS = frozenset(
    {"context", "dockerfile", "dockerfile_inline", "args", "target", "network"}
    | {"no_cache", "labels", "pull"}
)
_PULL_POLICIES = {
    "always": "always",
    "missing": "missing",
    "if_not_present": "missing",
    "never": "missing",
}
_CONDITIONS = {
    "service_started": "started",
    "service_healthy": "healthy",
    "service_completed_successfully": "completed_successfully",
}
_NO_NEW_PRIVILEGES = frozenset(
    {"no-new-privileges", "no-new-privileges:true", "no-new-privileges=true"}
)
_VOLUME_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(us|µs|ms|h|m|s)")
_SIZE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?$", re.IGNORECASE)
_UNITS = {"": 1, "k": 1024, "m": MIB, "g": 1024**3, "t": 1024**4}


class ComposeError(ValueError):
    """A compose file this front-end cannot translate; names the key path."""


# -- loading --------------------------------------------------------------------


def _safe_loader():
    import yaml

    class Loader(yaml.SafeLoader):
        """Plain YAML only: no ``!reset``/``!override`` or any other tag."""

    def refuse(loader, suffix, node):
        raise ComposeError(f"YAML tag {node.tag!r} is unsupported")

    Loader.add_multi_constructor("!", refuse)
    Loader.add_multi_constructor("tag:", refuse)
    return Loader


def load_document(path: Path) -> dict:
    """One compose file as a mapping (JSON when PyYAML is unavailable)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise ComposeError(
                f"{path}: PyYAML is unavailable and the file is not JSON"
            ) from error
    else:
        try:
            document = yaml.load(text, Loader=_safe_loader())
        except yaml.YAMLError as error:
            raise ComposeError(f"{path}: invalid YAML: {error}") from None
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ComposeError(f"{path}: a compose file must be a mapping")
    return document


def read_dotenv(path: Path, environ: Mapping[str, str] | None = None) -> dict:
    """A compose ``.env`` or ``env_file``: ``KEY=VALUE`` lines.

    Single quotes are literal; double-quoted and bare values interpolate
    against ``environ`` and the keys read so far, as compose does.
    """
    result: dict[str, str] = {}
    scope = dict(environ or {})
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not _ENV_NAME.match(key):
            raise ComposeError(f"{path}:{number}: expected KEY=VALUE")
        value = value.strip()
        if value[:1] == "'" and value.endswith("'") and len(value) > 1:
            value = value[1:-1]
        else:
            if value[:1] == '"' and value.endswith('"') and len(value) > 1:
                value = (
                    value[1:-1]
                    .replace("\\n", "\n")
                    .replace('\\"', '"')
                    .replace("\\\\", "\\")
                )
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0]
            value = interpolate(value, scope, f"{path}:{number}")
        result[key] = value
        # Lookups prefer the process environment, as compose does.
        scope.setdefault(key, value)
    return result


def _find_close(text: str, start: int) -> int:
    depth = 1
    index = start
    while index < len(text):
        if text.startswith("${", index):
            depth += 1
            index += 2
            continue
        if text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def interpolate(value: str, environ: Mapping[str, str], where: str) -> str:
    """Compose variable syntax: ``$$``, ``$V``, ``${V}``, ``${V:-d}``,
    ``${V-d}``, ``${V:?e}``, ``${V?e}``, ``${V:+a}`` and ``${V+a}``; defaults
    and alternatives nest. An unset variable without a default is empty."""
    out = []
    index = 0
    while index < len(value):
        char = value[index]
        if char != "$":
            out.append(char)
            index += 1
            continue
        following = value[index + 1 : index + 2]
        if following == "$":
            out.append("$")
            index += 2
            continue
        if following == "{":
            end = _find_close(value, index + 2)
            if end < 0:
                raise ComposeError(f"{where}: unterminated ${{ in {value!r}")
            out.append(_expand(value[index + 2 : end], environ, where))
            index = end + 1
            continue
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", value[index + 1 :])
        if match is None:
            out.append("$")
            index += 1
            continue
        out.append(environ.get(match.group(0), ""))
        index += 1 + match.end()
    return "".join(out)


def _expand(body: str, environ: Mapping[str, str], where: str) -> str:
    match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", body)
    if match is None:
        raise ComposeError(f"{where}: invalid variable ${{{body}}}")
    name, rest = match.group(0), body[match.end() :]
    present = name in environ
    current = environ.get(name, "")
    if not rest:
        return current
    for operator in (":-", ":?", ":+", "-", "?", "+"):
        if rest.startswith(operator):
            word = rest[len(operator) :]
            break
    else:
        raise ComposeError(f"{where}: invalid variable ${{{body}}}")
    usable = bool(current) if operator.startswith(":") else present
    if operator.endswith("-"):
        return current if usable else interpolate(word, environ, where)
    if operator.endswith("+"):
        return interpolate(word, environ, where) if usable else ""
    if not usable:
        message = interpolate(word, environ, where) or "is required"
        raise ComposeError(f"{where}: variable {name}: {message}")
    return current


def _interpolate_tree(node, environ, where):
    if isinstance(node, str):
        return interpolate(node, environ, where)
    if isinstance(node, list):
        return [_interpolate_tree(item, environ, where) for item in node]
    if isinstance(node, dict):
        return {
            key: _interpolate_tree(value, environ, f"{where}.{key}")
            for key, value in node.items()
        }
    return node


# -- normalization and merge ------------------------------------------------------


def _pairs(value, where, separators=("=",)):
    """A compose mapping written as a mapping or as ``K=V`` strings."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if not isinstance(value, list):
        raise ComposeError(f"{where}: expected a mapping or a list")
    result = {}
    for item in value:
        if not isinstance(item, str):
            raise ComposeError(f"{where}: expected strings")
        for separator in separators:
            if separator in item:
                key, _, rest = item.partition(separator)
                result[key.strip()] = rest
                break
        else:
            result[item.strip()] = None
    return result


def _short_volume(item, where):
    """``[SOURCE:]TARGET[:MODE]`` in compose's long form."""
    parts = item.split(":")
    if len(parts) == 1:
        return {"type": "volume", "target": parts[0]}
    if len(parts) > 3:
        raise ComposeError(f"{where}: cannot parse volume {item!r}")
    source, target = parts[0], parts[1]
    mount = {"source": source, "target": target}
    if len(parts) == 3:
        modes = set(parts[2].split(","))
        if "ro" in modes:
            mount["read_only"] = True
        if "nocopy" in modes:
            mount["volume"] = {"nocopy": True}
    mount["type"] = "bind" if source.startswith((".", "/", "~")) else "volume"
    return mount


def _normalize_service(raw, where):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ComposeError(f"{where}: a service must be a mapping")
    service = dict(raw)
    for key in ("environment", "labels", "annotations"):
        if key in service:
            service[key] = _pairs(service[key], f"{where}.{key}")
    if "extra_hosts" in service:
        service["extra_hosts"] = _pairs(
            service["extra_hosts"], f"{where}.extra_hosts", ("=", ":")
        )
    if "depends_on" in service:
        value = service["depends_on"]
        if isinstance(value, list):
            value = {str(name): {} for name in value}
        if not isinstance(value, dict):
            raise ComposeError(f"{where}.depends_on: expected a list or a mapping")
        service["depends_on"] = {
            str(name): dict(item or {}) for name, item in value.items()
        }
    if "networks" in service:
        value = service["networks"]
        if isinstance(value, list):
            value = {str(name): None for name in value}
        if not isinstance(value, dict):
            raise ComposeError(f"{where}.networks: expected a list or a mapping")
        service["networks"] = dict(value)
    if "volumes" in service:
        mounts = {}
        for index, item in enumerate(service["volumes"] or []):
            path = f"{where}.volumes[{index}]"
            mount = _short_volume(item, path) if isinstance(item, str) else item
            if not isinstance(mount, dict) or not isinstance(mount.get("target"), str):
                raise ComposeError(f"{path}: expected a mount with a target")
            mounts[mount["target"]] = dict(mount)
        service["volumes"] = mounts
    if "ulimits" in service and not isinstance(service["ulimits"], dict):
        raise ComposeError(f"{where}.ulimits: expected a mapping")
    if isinstance(service.get("build"), str):
        service["build"] = {"context": service["build"]}
    if isinstance(service.get("build"), dict) and "args" in service["build"]:
        service["build"] = dict(service["build"])
        service["build"]["args"] = _pairs(service["build"]["args"], f"{where}.build")
    if isinstance(service.get("tmpfs"), str):
        service["tmpfs"] = [service["tmpfs"]]
    if isinstance(service.get("env_file"), (str, dict)):
        service["env_file"] = [service["env_file"]]
    return service


def _merge_mapping(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge_mapping(result[key], value)
        else:
            result[key] = value
    return result


def _merge_service(base, override):
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if key in ("command", "entrypoint"):
            result[key] = value
        elif key in (
            "environment",
            "labels",
            "annotations",
            "extra_hosts",
            "volumes",
            "ulimits",
        ) and isinstance(current, dict):
            result[key] = {**current, **value}
        elif key in ("depends_on", "networks") and isinstance(current, dict):
            merged = dict(current)
            for name, item in value.items():
                if isinstance(merged.get(name), dict) and isinstance(item, dict):
                    merged[name] = _merge_mapping(merged[name], item)
                else:
                    merged[name] = item
            result[key] = merged
        elif key in _UNION_LISTS | {"env_file", "tmpfs"} and isinstance(current, list):
            result[key] = current + [item for item in value if item not in current]
        elif isinstance(current, dict) and isinstance(value, dict):
            result[key] = _merge_mapping(current, value)
        else:
            result[key] = value
    return result


def load_project(
    layers: Sequence[Path | Mapping],
    *,
    environ: Mapping[str, str],
    project_dir: Path | None = None,
) -> dict:
    """Merge compose layers in ``-f`` order after interpolating each.

    A layer is a compose file path or an already-built mapping (the Harbor
    plugin's resources/image/env layers). ``environ`` is the whole
    interpolation set; ``project_dir/.env`` sits below it, as for the CLI.
    """
    scope = {}
    if project_dir is not None and (Path(project_dir) / ".env").is_file():
        scope.update(read_dotenv(Path(project_dir) / ".env", environ))
    scope.update(environ)
    project: dict = {"services": {}}
    for layer in layers:
        where = str(layer) if isinstance(layer, Path) else "compose"
        document = (
            load_document(layer)
            if isinstance(layer, (str, Path))
            else json.loads(json.dumps(layer))
        )
        # Compose has no egress allowlist; this extension carries one (the
        # entries of every layer, in order; translate needs network allowlist).
        entries = document.get("x-rsi-allowlist", [])
        if not isinstance(entries, list) or not all(
            isinstance(entry, str) for entry in entries
        ):
            raise ComposeError("x-rsi-allowlist: expected a list of strings")
        project.setdefault("allowlist", []).extend(entries)
        for key in document:
            if key.startswith("x-"):
                continue
            if key not in _TOP_LEVEL:
                raise ComposeError(f"{key}: unsupported top-level key")
        document = _interpolate_tree(
            {key: value for key, value in document.items() if not key.startswith("x-")},
            scope,
            where,
        )
        services = document.get("services") or {}
        if not isinstance(services, dict):
            raise ComposeError("services: expected a mapping")
        for name, raw in services.items():
            service = _normalize_service(raw, f"services.{name}")
            current = project["services"].get(name)
            project["services"][name] = (
                service if current is None else _merge_service(current, service)
            )
        for key in ("volumes", "networks", "configs", "secrets"):
            value = document.get(key)
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ComposeError(f"{key}: expected a mapping")
            merged = project.setdefault(key, {})
            for name, item in value.items():
                merged[name] = _merge_mapping(merged.get(name) or {}, item or {})
        if "name" in document:
            project["name"] = document["name"]
    project["environ"] = scope
    return project


# -- values ----------------------------------------------------------------------


def parse_duration(value, where) -> float:
    """Compose durations (``1m30s``, ``500ms``); a bare number is seconds."""
    if isinstance(value, bool):
        raise ComposeError(f"{where}: expected a duration")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        position, seconds = 0, 0.0
        for match in _DURATION.finditer(text):
            if match.start() != position:
                break
            number, unit = float(match.group(1)), match.group(2)
            seconds += (
                number
                * {
                    "h": 3600,
                    "m": 60,
                    "s": 1,
                    "ms": 1e-3,
                    "us": 1e-6,
                    "µs": 1e-6,
                }[unit]
            )
            position = match.end()
        if not text or position != len(text):
            raise ComposeError(f"{where}: cannot parse duration {value!r}")
    if not math.isfinite(seconds) or seconds < 0:
        raise ComposeError(f"{where}: expected a nonnegative duration")
    return seconds


def parse_bytes(value, where) -> int:
    """Compose byte sizes: an integer is bytes; ``512m``, ``2g``, ``1GiB``."""
    if isinstance(value, bool):
        raise ComposeError(f"{where}: expected a byte size")
    if isinstance(value, int):
        return value
    match = _SIZE.match(str(value).strip())
    if match is None:
        raise ComposeError(f"{where}: cannot parse size {value!r}")
    return int(float(match.group(1)) * _UNITS[match.group(2).lower()])


def _mebibytes(value, where) -> int:
    return max(1, math.ceil(parse_bytes(value, where) / MIB))


def _text(value, where) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ComposeError(f"{where}: expected a scalar")


def _argv(value, where):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return shlex.split(value)
        except ValueError as error:
            raise ComposeError(f"{where}: {error}") from None
    if isinstance(value, list):
        return [_text(item, where) for item in value]
    raise ComposeError(f"{where}: expected a string or a list")


def _signal(value, where) -> str:
    text = str(value).strip().upper()
    if not text.startswith("SIG"):
        text = "SIG" + text
    if not re.fullmatch(r"SIG[A-Z0-9+-]{2,12}", text):
        raise ComposeError(f"{where}: invalid signal {value!r}")
    return text


def _absolute(value, where) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ComposeError(f"{where}: expected an absolute container path")
    return posixpath.normpath(value).replace("//", "/")


def _mount_target(value, where) -> str:
    """A volume, bind or tmpfs target the broker's EnvSpec check admits."""
    clean = _absolute(value, where)
    if clean == "/" or any(
        clean == root or clean.startswith(root + "/") for root in RESERVED_MOUNT_ROOTS
    ):
        raise ComposeError(f"{where}: reserved mount target {clean}")
    return clean


def _digest(*parts) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:12]


# -- translation -------------------------------------------------------------------


@dataclass(frozen=True)
class ImagePull:
    ref: str
    policy: str = "missing"


@dataclass(frozen=True)
class ImageBuild:
    """A compose ``build:`` (or Harbor's build layer), realized through
    ``image_build`` from a staged context (spec 3.3)."""

    context: str
    dockerfile: str | None = None
    dockerfile_inline: str | None = None
    target: str | None = None
    args: tuple[tuple[str, str], ...] = ()
    network: str | None = None
    no_cache: bool = False


@dataclass(frozen=True)
class Seed:
    """Content copied into a created, not yet started, service (M-6).

    ``kind="dir"`` copies the contents of ``source`` into ``dest_dir``;
    ``kind="file"`` writes one entry ``name`` from ``source`` or ``content``.
    """

    service: str
    dest_dir: str
    kind: str
    source: str | None = None
    name: str | None = None
    content: bytes | None = None
    mode: int | None = None


@dataclass
class Translation:
    """``spec`` is an EnvSpec whose service images are still unset: realize
    ``images`` first, then call ``env_spec`` with the resulting handles."""

    spec: dict
    images: dict
    seeds: tuple = ()
    notes: tuple = ()
    services: tuple = field(default=())

    def env_spec(self, handles: Mapping[str, str]) -> dict:
        spec = json.loads(json.dumps(self.spec))
        for name, service in spec["services"].items():
            service["image"] = handles[name]
        return spec


class _Translator:
    def __init__(
        self,
        project,
        *,
        project_dir,
        network,
        disk_mb,
        lifetime_sec,
        default_cpus,
        default_memory_mb,
        profiles,
        drop_targets,
    ):
        self.project = project
        self.project_dir = Path(project_dir).resolve()
        self.network = network
        self.disk_mb = disk_mb
        self.lifetime_sec = lifetime_sec
        self.default_cpus = default_cpus
        self.default_memory_mb = default_memory_mb
        self.drop_targets = frozenset(drop_targets)
        self.environ = project.get("environ") or {}
        wanted = set(profiles)
        wanted.update(
            item for item in self.environ.get("COMPOSE_PROFILES", "").split(",") if item
        )
        self.profiles = wanted
        self.notes: list[str] = []
        self.volumes: dict[str, dict] = {}
        self.volume_names: dict[str, str] = {}
        self.seeds: list[Seed] = []
        self.images: dict = {}
        self.aliases: dict[str, list[str]] = {}
        self.healthy_targets: set[str] = set()
        # Seeded volume -> (local directory, [(service, target), ...]).
        self.pending_dirs: dict[str, tuple[Path, list]] = {}

    def note(self, message):
        if message not in self.notes:
            self.notes.append(message)

    # -- project-level ------------------------------------------------------------

    def active_services(self):
        services = self.project.get("services") or {}
        if not services:
            raise ComposeError("services: a compose project needs a service")
        active = {}
        for name, service in services.items():
            enabled = service.get("profiles")
            if enabled and not self.profiles.intersection(enabled):
                continue
            active[name] = service
        return active

    def local_path(self, source, where) -> Path:
        if not isinstance(source, str) or source.startswith(("/", "~")):
            raise ComposeError(
                f"{where}: host path {source!r} is refused; only paths relative "
                "to the project are copied"
            )
        path = (self.project_dir / source).resolve()
        if path != self.project_dir and self.project_dir not in path.parents:
            raise ComposeError(f"{where}: {source!r} leaves the project directory")
        return path

    def declared_volume(self, name, where) -> str:
        volumes = self.project.get("volumes") or {}
        if name not in volumes:
            raise ComposeError(f"{where}: undefined volume {name!r}")
        config = volumes[name] or {}
        for key in config:
            if key in ("name", "labels") or key.startswith("x-"):
                continue
            if key == "driver" and config[key] in (None, "local"):
                continue
            raise ComposeError(f"volumes.{name}.{key}: unsupported volume option")
        spec_name = name.lower()
        if not _VOLUME_NAME.match(spec_name):
            raise ComposeError(
                f"volumes.{name}: volume name is not a valid DNS-safe key"
            )
        owner = self.volume_names.setdefault(spec_name, name)
        if owner != name:
            raise ComposeError(f"volumes.{name}: collides with volume {owner!r}")
        self.volumes.setdefault(spec_name, {"seeded": False})
        return spec_name

    def new_volume(self, name, *, seeded) -> str:
        self.volumes.setdefault(name, {"seeded": seeded})
        return name

    def effective_network(self, name, service):
        if service.get("network_mode") is not None:
            mode = service["network_mode"]
            if mode != "none":
                raise ComposeError(
                    f"services.{name}.network_mode: {mode!r} is refused (only none)"
                )
            if service.get("networks"):
                raise ComposeError(
                    f"services.{name}.networks: conflicts with network_mode none"
                )
            return None
        networks = service.get("networks") or {"default": None}
        if len(networks) > 1:
            raise ComposeError(
                f"services.{name}.networks: more than one network is unsupported"
            )
        return next(iter(networks))

    def check_networks(self, services):
        used = {}
        for name, service in services.items():
            network = self.effective_network(name, service)
            if network is not None:
                used.setdefault(network, name)
        if len(used) > 1:
            raise ComposeError(
                "networks: services use different networks "
                f"({', '.join(sorted(used))}); one env has one network"
            )
        declared = self.project.get("networks") or {}
        for network in used:
            if network == "default" and network not in declared:
                continue
            if network not in declared:
                raise ComposeError(f"networks.{network}: undefined network")
            for key, value in (declared[network] or {}).items():
                if key in ("name", "labels", "attachable") or key.startswith("x-"):
                    continue
                if key == "driver" and value in (None, "bridge"):
                    continue
                if key in ("internal", "enable_ipv6", "external") and not value:
                    continue
                raise ComposeError(
                    f"networks.{network}.{key}: unsupported network option"
                )
        return used

    # -- per service --------------------------------------------------------------

    def translate_service(self, name, service, all_services):
        where = f"services.{name}"
        for key in service:
            if key.startswith("x-"):
                continue
            if key not in _SERVICE_KEYS:
                raise ComposeError(f"{where}.{key}: unsupported compose key")
        for key in _IGNORED:
            if key in service:
                self.note(f"{where}.{key}: ignored")
        if service.get("privileged"):
            raise ComposeError(f"{where}.privileged: privileged services are refused")
        if service.get("oom_kill_disable"):
            raise ComposeError(f"{where}.oom_kill_disable: refused")
        if service.get("ipc") not in (None, "private"):
            raise ComposeError(f"{where}.ipc: only private IPC is supported")
        if service.get("cgroup") not in (None, "private"):
            raise ComposeError(f"{where}.cgroup: only a private cgroup is supported")
        scale = service.get("scale")
        if scale is not None and scale != 1:
            raise ComposeError(f"{where}.scale: only one replica is supported")
        restart = service.get("restart")
        if restart not in (None, "no"):
            self.note(f"{where}.restart: {restart!r} becomes 'no'")
        for key in ("ports", "expose"):
            if service.get(key):
                self.note(
                    f"{where}.{key}: dropped; services reach each other on "
                    "container ports"
                )
        if service.get("init") is False:
            self.note(f"{where}.init: an init process always runs")
        platform = service.get("platform")
        if platform is not None and platform not in ("linux/amd64", "linux/x86_64"):
            raise ComposeError(f"{where}.platform: only linux/amd64 is supported")

        spec: dict = {}
        self.image(name, service, where)
        entrypoint = _argv(service.get("entrypoint"), f"{where}.entrypoint")
        command = _argv(service.get("command"), f"{where}.command")
        if entrypoint is not None:
            spec["entrypoint"] = entrypoint
        if command is not None:
            spec["command"] = command
        environment = self.environment(name, service, where)
        if environment:
            spec["env"] = environment
        if service.get("working_dir") is not None:
            spec["working_dir"] = _absolute(
                service["working_dir"], f"{where}.working_dir"
            )
        if service.get("user") is not None:
            spec["user"] = _text(service["user"], f"{where}.user")
        if service.get("group_add"):
            spec["group_add"] = [
                _text(item, f"{where}.group_add") for item in service["group_add"]
            ]
        if service.get("hostname") is not None:
            spec["hostname"] = _text(service["hostname"], f"{where}.hostname")
        hosts = []
        for host, address in (service.get("extra_hosts") or {}).items():
            address = (address or "").strip().strip("[]")
            if address == "host-gateway":
                raise ComposeError(
                    f"{where}.extra_hosts.{host}: host-gateway is refused"
                )
            # The EnvSpec admits IP literals only: fail before any pull.
            try:
                ipaddress.ip_address(address)
            except ValueError:
                raise ComposeError(
                    f"{where}.extra_hosts.{host}: expected an IP literal, "
                    f"not {address!r}"
                ) from None
            if "%" in address:
                raise ComposeError(
                    f"{where}.extra_hosts.{host}: scoped addresses are refused"
                )
            hosts.append([host, address])
        if hosts:
            spec["extra_hosts"] = hosts
        network = self.effective_network(name, service)
        if network is None:
            spec["network"] = "none"
            if service.get("container_name"):
                self.note(f"{where}.container_name: no network, alias dropped")
        else:
            config = (service.get("networks") or {}).get(network) or {}
            for key in config:
                if key == "aliases" or key.startswith("x-"):
                    continue
                if key == "priority":
                    continue
                raise ComposeError(
                    f"{where}.networks.{network}.{key}: fixed addresses and "
                    "network options are refused"
                )
            for alias in config.get("aliases") or []:
                self.alias(name, _text(alias, f"{where}.networks.aliases"))
            if service.get("container_name"):
                self.alias(name, _text(service["container_name"], where))
        for link in service.get("links") or []:
            target, _, alias = str(link).partition(":")
            if target not in all_services:
                raise ComposeError(f"{where}.links: unknown service {target!r}")
            if alias and alias != target:
                self.alias(target, alias)
        if service.get("read_only"):
            spec["read_only"] = True
        if service.get("tty"):
            spec["tty"] = True
        caps = []
        for item in service.get("cap_drop") or []:
            cap = str(item).upper().removeprefix("CAP_")
            if cap == "ALL":
                self.note(f"{where}.cap_drop: ALL drops Docker's default set")
                caps.extend("CAP_" + value for value in DEFAULT_CAPABILITIES)
            else:
                caps.append("CAP_" + cap)
        if caps:
            spec["cap_drop"] = sorted(set(caps))
        for item in service.get("security_opt") or []:
            if str(item) not in _NO_NEW_PRIVILEGES:
                raise ComposeError(
                    f"{where}.security_opt: only no-new-privileges is supported"
                )
        self.resources(name, service, spec, where)
        mounts, tmpfs = self.mounts(name, service, spec, where)
        if mounts:
            spec["mounts"] = mounts
        if tmpfs:
            spec["tmpfs"] = tmpfs
        self.files(name, service, where)
        healthcheck = self.healthcheck(service.get("healthcheck"), where)
        if healthcheck is not None:
            spec["healthcheck"] = healthcheck
        depends = {}
        links = [str(item).partition(":")[0] for item in service.get("links") or []]
        for target in links:
            depends[target] = {"condition": "started", "required": True}
        for target, config in (service.get("depends_on") or {}).items():
            condition = config.get("condition", "service_started")
            if condition not in _CONDITIONS:
                raise ComposeError(
                    f"{where}.depends_on.{target}.condition: unknown condition"
                )
            required = config.get("required", True)
            if target not in all_services:
                if required:
                    raise ComposeError(
                        f"{where}.depends_on: service {target!r} is not enabled"
                    )
                self.note(f"{where}.depends_on.{target}: optional, not enabled")
                continue
            depends[target] = {
                "condition": _CONDITIONS[condition],
                "required": bool(required),
            }
            if condition == "service_healthy":
                self.healthy_targets.add(target)
        if depends:
            spec["depends_on"] = depends
        if service.get("stop_signal") is not None:
            spec["stop_signal"] = _signal(
                service["stop_signal"], f"{where}.stop_signal"
            )
        if service.get("stop_grace_period") is not None:
            grace = parse_duration(
                service["stop_grace_period"], f"{where}.stop_grace_period"
            )
            if grace > MAX_STOP_GRACE_SEC:
                raise ComposeError(
                    f"{where}.stop_grace_period: at most {MAX_STOP_GRACE_SEC}s"
                )
            spec["stop_grace_sec"] = math.ceil(grace)
        return spec

    def alias(self, service, alias):
        aliases = self.aliases.setdefault(service, [])
        if alias != service and alias not in aliases:
            aliases.append(alias)

    def image(self, name, service, where):
        build = service.get("build")
        if build is not None:
            if not isinstance(build, dict):
                raise ComposeError(f"{where}.build: expected a mapping")
            for key in build:
                if key not in _BUILD_KEYS:
                    raise ComposeError(f"{where}.build.{key}: unsupported build option")
            context = build.get("context", ".")
            if (
                not isinstance(context, str)
                or "://" in context
                or context.startswith("git@")
            ):
                raise ComposeError(f"{where}.build.context: only local contexts")
            if Path(context).is_absolute():
                # Harbor's build layer names the project itself (CONTEXT_DIR).
                local = Path(context).resolve()
                if local != self.project_dir and self.project_dir not in local.parents:
                    raise ComposeError(
                        f"{where}.build.context: {context!r} leaves the project"
                    )
            else:
                local = self.local_path(context, f"{where}.build.context")
            network = build.get("network")
            if network not in (None, "default", "none"):
                raise ComposeError(f"{where}.build.network: only default or none")
            if build.get("labels"):
                self.note(f"{where}.build.labels: ignored")
            self.images[name] = ImageBuild(
                context=str(local),
                dockerfile=build.get("dockerfile"),
                dockerfile_inline=build.get("dockerfile_inline"),
                target=build.get("target"),
                args=tuple(
                    sorted(
                        (key, _text(value, f"{where}.build.args"))
                        for key, value in (build.get("args") or {}).items()
                        if value is not None
                    )
                ),
                network="none" if network == "none" else None,
                no_cache=bool(build.get("no_cache", False)),
            )
            if service.get("pull_policy") not in (None, "build", "missing", "always"):
                self.note(f"{where}.pull_policy: builds ignore it")
            return
        image = service.get("image")
        if not isinstance(image, str) or not image:
            raise ComposeError(f"{where}: a service needs an image or a build")
        policy = service.get("pull_policy") or "missing"
        if policy == "build":
            raise ComposeError(f"{where}.pull_policy: build needs a build section")
        mapped = _PULL_POLICIES.get(policy)
        if mapped is None:
            self.note(f"{where}.pull_policy: {policy!r} becomes 'missing'")
            mapped = "missing"
        elif policy == "never":
            self.note(f"{where}.pull_policy: never becomes 'missing'")
        self.images[name] = ImagePull(image, mapped)

    def environment(self, name, service, where):
        merged: dict[str, str] = {}
        for index, entry in enumerate(service.get("env_file") or []):
            path, required = entry, True
            if isinstance(entry, dict):
                path, required = entry.get("path"), entry.get("required", True)
            local = self.local_path(path, f"{where}.env_file[{index}]")
            if not local.is_file():
                if required:
                    raise ComposeError(f"{where}.env_file: {path} does not exist")
                continue
            merged.update(read_dotenv(local, self.environ))
        for key, value in (service.get("environment") or {}).items():
            if value is None:
                if key in self.environ:
                    merged[key] = self.environ[key]
                else:
                    merged.pop(key, None)
                continue
            merged[key] = _text(value, f"{where}.environment.{key}")
        for key in merged:
            if key.upper().startswith("NVIDIA_"):
                raise ComposeError(f"{where}.environment.{key}: reserved key")
        return merged

    def resources(self, name, service, spec, where):
        deploy = service.get("deploy") or {}
        for key in deploy:
            if key in ("resources", "replicas", "restart_policy", "labels", "mode"):
                continue
            if key.startswith("x-"):
                continue
            self.note(f"{where}.deploy.{key}: ignored")
        if deploy.get("replicas") not in (None, 1):
            raise ComposeError(f"{where}.deploy.replicas: only one replica")
        if deploy.get("mode") not in (None, "replicated"):
            raise ComposeError(f"{where}.deploy.mode: only replicated")
        if deploy.get("restart_policy"):
            self.note(f"{where}.deploy.restart_policy: restart is always 'no'")
        if deploy.get("labels"):
            self.note(f"{where}.deploy.labels: ignored")
        resources = deploy.get("resources") or {}
        limits = resources.get("limits") or {}
        reservations = resources.get("reservations") or {}
        for key in reservations:
            if key in ("devices", "generic_resources"):
                raise ComposeError(
                    f"{where}.deploy.resources.reservations.{key}: "
                    "device reservations are refused"
                )
        if reservations or service.get("mem_reservation") is not None:
            self.note(f"{where}: reservations are ignored")
        for key in ("cpu_shares", "cpu_percent"):
            if service.get(key) is not None:
                self.note(f"{where}.{key}: ignored")
        for key in limits:
            if key not in ("cpus", "memory", "pids"):
                raise ComposeError(f"{where}.deploy.resources.limits.{key}: refused")
        cpus = service.get("cpus", limits.get("cpus"))
        if cpus is None:
            self.note(f"{where}: no cpus limit; using {self.default_cpus}")
            cpus = self.default_cpus
        try:
            value = float(cpus)
        except (TypeError, ValueError):
            raise ComposeError(f"{where}.cpus: expected a number") from None
        rounded = math.ceil(round(value * 100, 6)) / 100
        if rounded != value:
            self.note(f"{where}.cpus: {value} rounds up to {rounded}")
        if rounded <= 0:
            raise ComposeError(f"{where}.cpus: expected a positive limit")
        spec["cpus"] = rounded
        memory = service.get("mem_limit", limits.get("memory"))
        if memory is None:
            self.note(f"{where}: no memory limit; using {self.default_memory_mb} MiB")
            spec["memory_mb"] = self.default_memory_mb
        else:
            spec["memory_mb"] = _mebibytes(memory, f"{where}.mem_limit")
        if service.get("memswap_limit") is not None:
            self.note(f"{where}.memswap_limit: swap is the operator's swap_ratio")
        pids = service.get("pids_limit", limits.get("pids"))
        if pids is not None:
            if type(pids) is not int:
                raise ComposeError(f"{where}.pids_limit: expected an integer")
            if pids <= 0:
                self.note(f"{where}.pids_limit: unlimited becomes the grant default")
            else:
                spec["pids"] = pids
        if service.get("shm_size") is not None:
            spec["shm_mb"] = _mebibytes(service["shm_size"], f"{where}.shm_size")
        for key, value in (service.get("ulimits") or {}).items():
            if key == "nproc":
                self.note(f"{where}.ulimits.nproc: bounded by the pids limit")
                continue
            if key != "nofile":
                raise ComposeError(f"{where}.ulimits.{key}: unsupported ulimit")
            if isinstance(value, dict):
                soft, hard = value.get("soft"), value.get("hard")
            else:
                soft = hard = value
            if type(hard) is not int or type(soft) is not int:
                raise ComposeError(f"{where}.ulimits.nofile: expected integers")
            if soft != hard:
                self.note(f"{where}.ulimits.nofile: soft becomes the hard limit")
            spec["nofile"] = hard

    def mounts(self, name, service, spec, where):
        mounts, tmpfs = [], {}
        for item in service.get("tmpfs") or []:
            path, _, options = str(item).partition(":")
            size = None
            for option in options.split(","):
                key, _, value = option.partition("=")
                if key == "size":
                    size = _mebibytes(value, f"{where}.tmpfs")
            if size is None:
                self.note(f"{where}.tmpfs {path}: size {DEFAULT_TMPFS_MB} MiB")
                size = DEFAULT_TMPFS_MB
            tmpfs[_mount_target(path, f"{where}.tmpfs")] = size
        for target, mount in (service.get("volumes") or {}).items():
            path = f"{where}.volumes.{target}"
            kind = mount.get("type", "volume")
            for key in mount:
                if key not in (
                    "type",
                    "source",
                    "target",
                    "read_only",
                    "volume",
                    "bind",
                    "tmpfs",
                    "consistency",
                ):
                    raise ComposeError(f"{path}.{key}: unsupported mount option")
            clean = _mount_target(target, path)
            if clean in self.drop_targets:
                self.note(f"{path}: log mount dropped; logs are downloaded")
                continue
            read_only = bool(mount.get("read_only", False))
            if kind == "tmpfs":
                size = (mount.get("tmpfs") or {}).get("size")
                tmpfs[clean] = (
                    DEFAULT_TMPFS_MB
                    if size is None
                    else max(1, math.ceil(parse_bytes(size, path) / MIB))
                )
                continue
            if kind == "bind":
                self.bind(name, mount.get("source"), clean, path)
                continue
            if kind != "volume":
                raise ComposeError(f"{path}: {kind} mounts are refused")
            options = mount.get("volume") or {}
            if "subpath" in options:
                raise ComposeError(f"{path}.volume.subpath: refused")
            source = mount.get("source")
            if source:
                volume = self.declared_volume(source, path)
            else:
                volume = self.new_volume("anon-" + _digest(name, clean), seeded=False)
            entry = {"volume": volume, "target": clean}
            if read_only:
                entry["read_only"] = True
            mounts.append(entry)
        for target, size in tmpfs.items():
            if target in {mount["target"] for mount in mounts}:
                raise ComposeError(f"{where}.tmpfs: {target} is also a volume")
        return mounts, tmpfs

    def bind(self, name, source, target, where):
        local = self.local_path(source, where)
        if local.is_file():
            self.note(f"{where}: file bind becomes a copy before start")
            self.seeds.append(
                Seed(
                    service=name,
                    dest_dir=posixpath.dirname(target) or "/",
                    kind="file",
                    source=str(local),
                    name=posixpath.basename(target),
                )
            )
            return
        if local.exists() and not local.is_dir():
            raise ComposeError(f"{where}: {source!r} is not a file or directory")
        if not local.exists():
            self.note(f"{where}: {source!r} is missing; an empty volume replaces it")
        relative = local.relative_to(self.project_dir).as_posix()
        volume = self.new_volume("seed-" + _digest(relative), seeded=True)
        self.note(
            f"{where}: directory bind becomes a volume seeded once (not "
            "read-only, not synced back)"
        )
        self.pending_dirs.setdefault(volume, (local, []))[1].append((name, target))

    def files(self, name, service, where):
        for kind in ("configs", "secrets"):
            declared = self.project.get(kind) or {}
            for index, item in enumerate(service.get(kind) or []):
                path = f"{where}.{kind}[{index}]"
                if isinstance(item, str):
                    item = {"source": item}
                source = item.get("source")
                if source not in declared:
                    raise ComposeError(f"{path}: undefined {kind[:-1]} {source!r}")
                config = declared[source] or {}
                target = item.get("target") or source
                if not target.startswith("/"):
                    target = (
                        "/run/secrets/" + target if kind == "secrets" else "/" + target
                    )
                target = _absolute(target, path)
                mode = item.get("mode")
                if isinstance(mode, str):
                    mode = int(mode, 8)
                if config.get("external"):
                    raise ComposeError(f"{kind}.{source}: external {kind} are refused")
                seed = {
                    "service": name,
                    "dest_dir": posixpath.dirname(target),
                    "kind": "file",
                    "name": posixpath.basename(target),
                    "mode": mode,
                }
                if "file" in config:
                    local = self.local_path(config["file"], f"{kind}.{source}.file")
                    if not local.is_file():
                        raise ComposeError(f"{kind}.{source}.file: not a file")
                    self.seeds.append(Seed(source=str(local), **seed))
                elif "content" in config:
                    self.seeds.append(
                        Seed(content=str(config["content"]).encode(), **seed)
                    )
                elif "environment" in config:
                    value = self.environ.get(config["environment"])
                    if value is None:
                        raise ComposeError(f"{kind}.{source}.environment: unset")
                    self.seeds.append(Seed(content=value.encode(), **seed))
                else:
                    raise ComposeError(f"{kind}.{source}: needs file or content")
                self.note(f"{path}: copied into the service before start")

    def healthcheck(self, check, where):
        if check is None:
            return None
        if not isinstance(check, dict):
            raise ComposeError(f"{where}.healthcheck: expected a mapping")
        if check.get("disable"):
            return "none"
        test = check.get("test")
        if test is None:
            self.note(f"{where}.healthcheck: without test the image check is used")
            return "image"
        if isinstance(test, str):
            test = ["CMD-SHELL", test]
        test = [_text(item, f"{where}.healthcheck.test") for item in test]
        if test and test[0] == "NONE":
            return "none"
        if not test or test[0] not in ("CMD", "CMD-SHELL"):
            raise ComposeError(f"{where}.healthcheck.test: expected CMD or CMD-SHELL")
        result: dict = {"test": test}
        for key, spec_key in (
            ("interval", "interval_sec"),
            ("timeout", "timeout_sec"),
            ("start_period", "start_period_sec"),
            ("start_interval", "start_interval_sec"),
        ):
            if check.get(key) is not None:
                result[spec_key] = parse_duration(
                    check[key], f"{where}.healthcheck.{key}"
                )
        if check.get("retries") is not None:
            retries = check["retries"]
            if type(retries) is not int or not 0 <= retries <= 100:
                raise ComposeError(f"{where}.healthcheck.retries: 0..100")
            result["retries"] = retries
        return result

    # -- whole project -------------------------------------------------------------

    def run(self) -> Translation:
        services = self.active_services()
        self.check_networks(services)
        specs = {}
        for name in sorted(services):
            specs[name] = self.translate_service(name, services[name], services)
        for target in sorted(self.healthy_targets):
            if specs[target].get("healthcheck") == "none":
                raise ComposeError(
                    f"services.{target}.healthcheck: disabled, but a service "
                    "depends on it being healthy"
                )
            if "healthcheck" not in specs[target]:
                specs[target]["healthcheck"] = "image"
                self.note(
                    f"services.{target}.healthcheck: the image healthcheck "
                    "gates service_healthy"
                )
        owners: dict[str, str] = {}
        for name in specs:
            owners[name.lower()] = name
        for name in sorted(self.aliases):
            aliases = self.aliases[name]
            if specs[name].get("network") == "none":
                continue
            for alias in aliases:
                owner = owners.setdefault(alias.lower(), name)
                if owner != name:
                    raise ComposeError(
                        f"services.{name}: alias {alias!r} already names {owner!r}"
                    )
            if aliases:
                specs[name]["aliases"] = list(aliases)
        for volume in sorted(self.pending_dirs):
            local, consumers = self.pending_dirs[volume]
            for service, target in consumers:
                mounts = specs[service].setdefault("mounts", [])
                if any(mount["target"] == target for mount in mounts):
                    raise ComposeError(
                        f"services.{service}.volumes: duplicate target {target}"
                    )
                mounts.append({"volume": volume, "target": target})
            if local.is_dir():
                service, target = sorted(consumers)[0]
                self.seeds.append(
                    Seed(
                        service=service, dest_dir=target, kind="dir", source=str(local)
                    )
                )
        for name, spec in specs.items():
            used = spec.get("shm_mb", 64) + sum((spec.get("tmpfs") or {}).values())
            if used > spec["memory_mb"]:
                raise ComposeError(
                    f"services.{name}: tmpfs and shm_size exceed the memory limit"
                )
            if "mounts" in spec:
                spec["mounts"].sort(key=lambda mount: mount["target"])
        if len(self.volumes) > MAX_VOLUMES:
            raise ComposeError(
                f"volumes: an env holds at most {MAX_VOLUMES} volumes "
                f"({len(self.volumes)} needed)"
            )
        spec = {
            "version": 1,
            "network": self.network,
            "lifetime_sec": self.lifetime_sec,
            "disk_mb": self.disk_mb,
            "volumes": {name: self.volumes[name] for name in sorted(self.volumes)},
            "services": {name: {"image": None, **specs[name]} for name in specs},
        }
        return Translation(
            spec=spec,
            images=dict(self.images),
            seeds=tuple(self.seeds),
            notes=tuple(self.notes),
            services=tuple(specs),
        )


def translate(
    project: Mapping,
    *,
    project_dir: Path,
    network: str,
    disk_mb: int,
    lifetime_sec: float | None = None,
    default_cpus: float = 1.0,
    default_memory_mb: int = 1024,
    profiles: Sequence[str] = (),
    drop_targets: Sequence[str] = (),
    allowlist: Sequence[str] = (),
) -> Translation:
    """Translate a loaded project (``load_project``) into one EnvSpec.

    ``network`` is the env's egress (``public``, ``none`` or ``allowlist``);
    ``disk_mb`` its soft disk limit. An allowlist env's entries (hostnames,
    IPv4 addresses or CIDRs, each optionally ``:port``; the broker validates
    them) are the caller's ``allowlist`` (Harbor's ``allowed_hosts``) and
    the project's top-level ``x-rsi-allowlist``. Services
    without limits get ``default_cpus`` and ``default_memory_mb`` (the
    broker requires finite limits). Mounts whose target is in
    ``drop_targets`` (log directories the caller downloads) are dropped with
    a note.
    """
    if network not in ("public", "none", "allowlist"):
        raise ComposeError("network: expected public, none or allowlist")
    entries = list(dict.fromkeys([*allowlist, *project.get("allowlist", ())]))
    if entries and network != "allowlist":
        raise ComposeError(
            f"x-rsi-allowlist: entries need network allowlist, not {network}"
        )
    translation = _Translator(
        project,
        project_dir=project_dir,
        network=network,
        disk_mb=disk_mb,
        lifetime_sec=lifetime_sec,
        default_cpus=default_cpus,
        default_memory_mb=default_memory_mb,
        profiles=profiles,
        drop_targets=drop_targets,
    ).run()
    if network == "allowlist":
        translation.spec["allowlist"] = entries
    return translation


def translation_json(translation: Translation) -> dict:
    """A stable, JSON-ready view (golden tests and ``compose config``)."""

    def image(request):
        if isinstance(request, ImagePull):
            return {"pull": request.ref, "policy": request.policy}
        return {
            "build": request.context,
            "dockerfile": request.dockerfile,
            "dockerfile_inline": request.dockerfile_inline,
            "target": request.target,
            "args": dict(request.args),
            "network": request.network,
            "no_cache": request.no_cache,
        }

    def seed(item):
        return {
            key: value
            for key, value in (
                ("service", item.service),
                ("dest_dir", item.dest_dir),
                ("kind", item.kind),
                ("source", item.source),
                ("name", item.name),
                ("content", None if item.content is None else item.content.decode()),
                ("mode", item.mode),
            )
            if value is not None
        }

    return {
        "spec": translation.spec,
        "images": {name: image(item) for name, item in translation.images.items()},
        "seeds": [seed(item) for item in translation.seeds],
        "notes": list(translation.notes),
    }


def write_seed_archive(seed: Seed, target) -> None:
    """A tar of one seed for ``copy_in(seed.dest_dir)``: the directory's
    contents, or one file entry. Regular files, directories and symlinks
    only; never a hardlink entry (the broker accepts nothing else)."""
    with tarfile.open(fileobj=target, mode="w", format=tarfile.PAX_FORMAT) as archive:
        if seed.kind == "file":
            entry = tarfile.TarInfo(seed.name)
            if seed.content is not None:
                entry.size = len(seed.content)
                entry.mode = 0o644 if seed.mode is None else seed.mode
                archive.addfile(entry, io.BytesIO(seed.content))
                return
            info = os.stat(seed.source)
            entry.size = info.st_size
            entry.mode = stat.S_IMODE(info.st_mode) if seed.mode is None else seed.mode
            entry.mtime = int(info.st_mtime)
            with open(seed.source, "rb") as data:
                archive.addfile(entry, data)
            return
        add_tree(archive, seed.source, "")


def add_tree(archive, root, prefix) -> None:
    """Add the contents of ``root`` under ``prefix`` (``""``: at the top)."""
    for child in sorted(os.listdir(root)):
        source = os.path.join(root, child)
        name = prefix + child
        info = os.lstat(source)
        entry = tarfile.TarInfo(name)
        entry.mode = stat.S_IMODE(info.st_mode)
        entry.mtime = int(info.st_mtime)
        if stat.S_ISDIR(info.st_mode):
            entry.type = tarfile.DIRTYPE
            archive.addfile(entry)
            add_tree(archive, source, name + "/")
        elif stat.S_ISLNK(info.st_mode):
            entry.type = tarfile.SYMTYPE
            entry.linkname = os.readlink(source)
            archive.addfile(entry)
        elif stat.S_ISREG(info.st_mode):
            entry.size = info.st_size
            with open(source, "rb") as data:
                archive.addfile(entry, data)
        else:
            raise ComposeError(f"{source}: special files are not copied")


def build_args(items, environ=None):
    """``--build-arg`` values as ``docker build`` reads them: ``NAME=VALUE``,
    or ``NAME`` alone for this environment's NAME (omitted when unset)."""
    environ = os.environ if environ is None else environ
    found = []
    for item in items:
        name, separator, value = item.partition("=")
        if separator:
            found.append((name, value))
        elif name in environ:
            found.append((name, environ[name]))
    return tuple(found)


def start_build(client, request: ImageBuild, timeout_sec: float, what: str) -> str:
    """Stage ``request.context`` and start its ``image_build``; the job ID.

    Shared by ``rsi-sandbox compose up`` (a service's ``build:``) and
    ``rsi-sandbox build``. The network defaults to public when the build
    grant offers it; the timeout is capped by the grant's ``max_build_sec``.
    """
    build = (client.capabilities().get("environments") or {}).get("build")
    if build is None:
        error = ComposeError(
            f"{what} needs an image build, which this sandbox grant does not offer"
        )
        error.code, error.field = "unsupported", "build"
        raise error
    with tempfile.TemporaryFile() as spool:
        with tarfile.open(fileobj=spool, mode="w", format=tarfile.PAX_FORMAT) as tar:
            add_tree(tar, request.context, "")
        stage_id = client.upload_stage(spool)["stage_id"]
    return client.image_build(
        stage_id,
        dockerfile=request.dockerfile,
        dockerfile_inline=request.dockerfile_inline,
        target=request.target,
        build_args=dict(request.args),
        no_cache=request.no_cache,
        network=request.network
        or ("public" if "public" in build["network"] else "none"),
        timeout_sec=min(timeout_sec, build["max_build_sec"]),
    )


def lifetime_refusal(error: BaseException) -> bool:
    """The broker refused an ``env_start`` wait beyond the env's remaining
    lifetime. Nothing started, so a shorter wait may be sent again."""
    return (
        getattr(error, "code", None) == "quota"
        and getattr(error, "field", None) == "wait_timeout_sec"
        and "remaining lifetime" in str(error)
    )


def start_env(
    client, created: Mapping, sent: float, wait_timeout: float | None
) -> dict:
    """Start the env ``created`` by an ``env_create`` sent at ``sent``
    (``time.monotonic()``) and wait until it is no longer starting.

    Shared by ``rsi-sandbox up`` and ``rsi-sandbox compose up``. The broker
    refuses a wait beyond what is left of the env's lifetime (spec A8: a
    Judge whose verifier timeout is short), counted here from before the
    create, so never later than the broker's. The default wait is capped at
    it, and resent once shorter if slot queueing spent the slack; an explicit
    ``wait_timeout`` that does not fit is refused. A refused env is destroyed.
    """
    env_id = created["env_id"]
    expires_at = sent + float(created["expires_in_sec"])
    retried = False
    while True:
        left = expires_at - time.monotonic() - START_SLACK_SEC
        wait = (
            min(DEFAULT_START_WAIT_SEC, left) if wait_timeout is None else wait_timeout
        )
        if left <= 0 or wait > left:
            client.env_destroy(env_id)
            error = ComposeError(
                f"--wait-timeout {wait:g} s exceeds the env's remaining lifetime "
                f"({max(0.0, left):.0f} s); env {env_id} was destroyed"
                if wait_timeout is not None
                else f"env {env_id}'s lifetime ended before it could start; "
                "it was destroyed"
            )
            error.code, error.field = "quota", "wait_timeout_sec"
            raise error
        try:
            client.env_start(env_id, wait)
        except ValueError as error:  # the client's ProtocolError
            if retried or wait_timeout is not None or not lifetime_refusal(error):
                raise
            retried = True
            continue
        return client.wait_env(env_id, wait + 30)


__all__ = [
    "ComposeError",
    "ImageBuild",
    "ImagePull",
    "Seed",
    "Translation",
    "add_tree",
    "build_args",
    "interpolate",
    "load_document",
    "load_project",
    "parse_bytes",
    "parse_duration",
    "lifetime_refusal",
    "read_dotenv",
    "start_build",
    "start_env",
    "translate",
    "translation_json",
    "write_seed_archive",
]
