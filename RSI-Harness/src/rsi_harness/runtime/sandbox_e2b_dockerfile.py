"""Single-stage Dockerfiles as E2B template steps (``image_build`` on e2b),
and checked for Modal's builder (on modal).

E2B builds a template from a base image and a list of steps. The SDK's own
``from_dockerfile`` is not used: it drops COPY flags, turns ARG into ENV and
CMD into a start command. ``parse`` reads the Dockerfile when the build is
admitted and refuses what the steps cannot express as Docker builds it;
``steps`` runs in the build job, once the base image's config is known, and
returns the steps and the config env_create applies (as for a pull).

A step is a tuple: ``("user", name)``, ``("workdir", path)``, ``("env",
{name: value})``, ``("run", command)`` or ``("copy"|"add", source, dest,
owner, mode)``. A RUN exports the tracked ENV and the ARGs itself, so E2B's
login shell cannot change them and ARGs never reach the image's ENV. COPY
and ADD without ``--chown`` are owned by root, as in Docker. E2B resolves
neither a numeric step user nor a ``--chown`` user name without a group as
Docker does: a RUN as a numeric USER (other than 0) and such a ``--chown``
are refused.

Modal's builder runs the Dockerfile's own instructions after FROM
(``lines``) as Docker does, so only what the config overlay or a single
stage on the pulled base cannot follow is refused there (``backend``).

Imported only by remote builds (``dockerfile-parse``, the ``e2b`` and
``modal`` extras).
"""

from __future__ import annotations

import io
import json
import posixpath
import re
import shlex
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from dockerfile_parse import DockerfileParser
from dockerfile_parse.util import WordSplitter, extract_key_values

from rsi_harness.runtime.build_context import parser_directives
from rsi_harness.runtime.sandbox_contracts import SandboxError
from rsi_harness.runtime.sandbox_images import pull_reference, reference_text

# BuildKit's automatic platform ARGs; E2B builds linux/amd64.
PLATFORM_ARGS = {
    f"{prefix}{name}": value
    for prefix in ("TARGET", "BUILD")
    for name, value in (
        ("PLATFORM", "linux/amd64"),
        ("OS", "linux"),
        ("ARCH", "amd64"),
        ("VARIANT", ""),
    )
}
# Not applied, as for pulls (labels and the healthcheck are not kept).
IGNORED = frozenset({"EXPOSE", "LABEL", "MAINTAINER", "STOPSIGNAL", "HEALTHCHECK"})
SUPPORTED = IGNORED | {
    "ARG",
    "ENV",
    "RUN",
    "COPY",
    "ADD",
    "WORKDIR",
    "USER",
    "CMD",
    "ENTRYPOINT",
    "VOLUME",
}
# Instructions whose words Docker expands ($VAR, ${VAR}).
_EXPANDED = frozenset({"ARG", "ENV", "COPY", "ADD", "WORKDIR", "USER", "VOLUME"})
# ${VAR:-word}, ${VAR#x} and other modifiers.
_MODIFIER = re.compile(r"\$\{[A-Za-z0-9_]*[^A-Za-z0-9_}]")
# A word BuildKit reads as a heredoc (<<NAME, <<-NAME, 2<<NAME), not a
# here-string or a shift.
_HEREDOC = re.compile(r"[0-9]*<<-?\s*([^<]*)")
_REMOTE = re.compile(r"^([a-z][a-z0-9+.-]*://|git@)", re.IGNORECASE)
_COPY_FLAGS = ("--chown", "--chmod", "--link")
_MODE = re.compile(r"[0-7]{3,4}")
# A --chown E2B applies as Docker does: user:group, or a numeric uid.
_OWNER = re.compile(r"[^:]+:[^:]+|[0-9]+")


def unsupported(what: str, backend: str = "e2b") -> SandboxError:
    return SandboxError(
        "unsupported", "dockerfile", f"{what} unsupported on the {backend} backend"
    )


def _invalid(message: str) -> SandboxError:
    return SandboxError("invalid", "dockerfile", message)


@dataclass(frozen=True, slots=True)
class Plan:
    """A checked Dockerfile: its base reference (domain/remote:tag), the
    caller's build args, the ARGs declared before FROM and every later
    instruction as (INSTRUCTION, value)."""

    base: str
    build_args: Mapping[str, str]
    global_args: Mapping[str, str]
    instructions: tuple[tuple[str, str], ...]


def _expand(word: str, values: Mapping[str, str]) -> str:
    """Docker's word expansion: $VAR and ${VAR}, then quote removal."""
    return WordSplitter(word, args={}, envs=dict(values)).dequote()


def _words(value: str) -> list[str]:
    return list(WordSplitter(value).split(dequote=False))


def _exec_form(value: str) -> list[str] | None:
    if not value.lstrip().startswith("["):
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        return None  # Docker runs it as the shell form
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    return None


def _flags(
    instruction: str, words: list[str], backend: str = "e2b"
) -> tuple[dict[str, str], list[str]]:
    flags: dict[str, str] = {}
    while words and words[0].startswith("--"):
        name, _, value = words.pop(0).partition("=")
        allowed = _COPY_FLAGS if instruction in ("COPY", "ADD") else ()
        if name not in allowed:
            raise unsupported(f"{instruction} {name}", backend)
        flags[name] = value
    return flags, words


def _arg_pairs(value: str, values: Mapping[str, str]) -> list[tuple[str, str | None]]:
    pairs: list[tuple[str, str | None]] = []
    for word in _words(value):
        name, equals, default = word.partition("=")
        pairs.append((_expand(name, {}), _expand(default, values) if equals else None))
    return pairs


def _check(instruction: str, value: str, backend: str) -> None:
    if instruction not in SUPPORTED:
        raise unsupported(f"the {instruction} instruction", backend)
    if instruction in _EXPANDED and _MODIFIER.search(value):
        raise unsupported(f"a ${{VAR:...}} modifier in {instruction}", backend)
    if instruction in ("RUN", "COPY", "ADD") and any(
        (found := _HEREDOC.fullmatch(word)) and found[1]
        for word in WordSplitter(value).split()
    ):
        raise unsupported(f"a heredoc in {instruction}", backend)
    if instruction == "RUN" and _exec_form(value) is None:
        _flags(instruction, _words(value), backend)
    if instruction in ("COPY", "ADD"):
        sources = _sources(instruction, value, backend)[0]
        if instruction == "ADD" and any(_REMOTE.match(item) for item in sources):
            raise unsupported("ADD of a URL or git source", backend)
    if instruction == "USER" and ":" in value:
        raise unsupported("USER with a group", backend)


def parse(
    dockerfile: bytes,
    build_args: Mapping[str, str],
    target: str | None,
    backend: str = "e2b",
) -> Plan:
    """The checked Dockerfile, or a SandboxError naming what is refused."""
    if "escape" in parser_directives(dockerfile):
        raise unsupported("the escape directive", backend)
    parser = DockerfileParser(
        fileobj=io.BytesIO(dockerfile.removeprefix(b"\xef\xbb\xbf"))
    )
    try:
        structure = [
            (item["instruction"], item["value"])
            for item in parser.structure
            if item["instruction"] != "COMMENT"
        ]
    except UnicodeError:
        raise _invalid("the Dockerfile is not UTF-8") from None
    froms = [index for index, (name, _) in enumerate(structure) if name == "FROM"]
    if len(froms) > 1:
        raise unsupported("a multi-stage Dockerfile", backend)
    if not froms:
        raise _invalid("the Dockerfile has no FROM")
    global_args: dict[str, str] = {}
    for instruction, value in structure[: froms[0]]:
        if instruction != "ARG":
            raise _invalid(f"{instruction} before FROM")
        _check(instruction, value, backend)
        for name, default in _arg_pairs(value, {**PLATFORM_ARGS, **global_args}):
            if name in build_args:
                global_args[name] = build_args[name]
            elif default is not None:
                global_args[name] = default
    value = structure[froms[0]][1]
    if _MODIFIER.search(value):
        raise unsupported("a ${VAR:...} modifier in FROM", backend)
    scope = {**PLATFORM_ARGS, **global_args}
    words = _words(value)
    while words and words[0].startswith("--"):
        name, _, platform = words.pop(0).partition("=")
        if name != "--platform" or _expand(platform, scope) != "linux/amd64":
            raise unsupported(f"FROM {name} other than linux/amd64", backend)
    if len(words) not in (1, 3) or (len(words) == 3 and words[1].lower() != "as"):
        raise _invalid("FROM takes an image and an optional AS name")
    stage = words[2] if len(words) == 3 else None
    if target is not None and target != stage:
        raise SandboxError("invalid", "target", f"no build stage {target}")
    image = _expand(words[0], scope)
    if image == "scratch":
        raise unsupported("FROM scratch", backend)
    try:
        base = reference_text(*pull_reference(image))
    except SandboxError:
        raise _invalid(f"FROM {image} is not an image reference") from None
    instructions = tuple(structure[froms[0] + 1 :])
    for instruction, value in instructions:
        _check(instruction, value, backend)
    return Plan(base, dict(build_args), global_args, instructions)


def _sources(
    instruction: str, value: str, backend: str = "e2b"
) -> tuple[list[str], str, dict[str, str]]:
    """COPY/ADD sources and dest (unexpanded) and flags."""
    flags, words = _flags(instruction, _words(value), backend)
    words = _exec_form(" ".join(words)) or words
    if len(words) < 2:
        raise _invalid(f"{instruction} needs a source and a destination")
    return words[:-1], words[-1], flags


def _source(path: str) -> str:
    """A context path: absolute sources are relative to the context, as in
    Docker; none may leave it."""
    normal = posixpath.normpath(path.lstrip("/") or ".")
    if normal == ".." or normal.startswith("../"):
        raise _invalid(f"source {path} is outside the build context")
    return normal


def _dest(path: str, workdir: str) -> str:
    """Absolute, against the WORKDIR; a directory keeps its trailing /."""
    directory = path.endswith("/") or posixpath.basename(path) in (".", "..")
    dest = posixpath.normpath(posixpath.join(workdir or "/", path))
    return dest + "/" if directory and dest != "/" else dest


def _command(value: str, shell: list[str]) -> list[str]:
    return _exec_form(value) or [*shell, value]


def _run(value: str, exported: Mapping[str, str], shell: list[str]) -> str:
    command = "exec " + shlex.join(_command(value, shell))
    if not exported:
        return command
    pairs = " ".join(f"{name}={shlex.quote(item)}" for name, item in exported.items())
    return f"export {pairs} && {command}"


def _user(user: str, backend: str = "e2b") -> str:
    if ":" in user:
        raise unsupported("USER with a group", backend)
    return "root" if user == "0" else user


def steps(
    plan: Plan, base_config: Mapping[str, Any], backend: str = "e2b"
) -> tuple[tuple[tuple[Any, ...], ...], dict[str, Any]]:
    """The template steps after the base image, and its config overlay
    (WorkingDir, User, Entrypoint, Cmd, Volumes, Env), from the base
    image's config (its registry's). A shell form runs in the base
    image's SHELL, as in Docker; its ONBUILD is refused. On a backend
    other than e2b the steps only check the build (``lines`` is what
    runs), so E2B's own limits on users are not applied."""
    e2b = backend == "e2b"
    if base_config.get("OnBuild"):
        raise unsupported("a base image's ONBUILD", backend)
    shell = list(base_config.get("Shell") or ("/bin/sh", "-c"))
    env = dict(
        item.partition("=")[::2] for item in base_config.get("Env") or () if item
    )
    args: dict[str, str] = {}
    user = _user(base_config.get("User") or "", backend)
    workdir = base_config.get("WorkingDir") or ""
    entrypoint = base_config.get("Entrypoint")
    cmd = base_config.get("Cmd")
    cmd_set = False
    volumes = dict(base_config.get("Volumes") or {})
    # Docker runs the steps as the base image's USER, in its WORKDIR.
    out: list[tuple[Any, ...]] = [("user", user or "root"), ("workdir", workdir or "/")]
    for instruction, value in plan.instructions:
        values = {**args, **env}  # an ENV hides an ARG of its name
        if instruction == "ARG":
            for name, default in _arg_pairs(value, values):
                if name in plan.build_args:
                    args[name] = plan.build_args[name]
                elif default is not None:
                    args[name] = default
                elif name in plan.global_args:
                    args[name] = plan.global_args[name]
                elif name in PLATFORM_ARGS:
                    args[name] = PLATFORM_ARGS[name]
        elif instruction == "ENV":
            try:
                pairs = dict(extract_key_values(True, {}, values, value))
            except (ValueError, IndexError):
                raise _invalid(f"ENV {value} is not name=value") from None
            env.update(pairs)
            out.append(("env", pairs))
        elif instruction == "RUN":
            if e2b and (user or "root").isdigit():
                raise unsupported("RUN as a numeric USER")
            out.append(("run", _run(value, {**args, **env}, shell)))
        elif instruction in ("COPY", "ADD"):
            sources, dest, flags = _sources(instruction, value, backend)
            if len(sources) > 1 and not dest.endswith("/"):
                raise _invalid(f"{instruction} of several sources needs a dest/")
            mode = flags.get("--chmod")
            if mode is not None and not _MODE.fullmatch(mode):
                raise unsupported(f"{instruction} --chmod other than octal", backend)
            owner = _expand(flags.get("--chown") or "root", values)
            if e2b and owner != "root" and not _OWNER.fullmatch(owner):
                raise unsupported(f"{instruction} --chown of a user name alone")
            target = _dest(_expand(dest, values), workdir)
            for source in sources:
                out.append(
                    (
                        instruction.lower(),
                        _source(_expand(source, values)),
                        target,
                        owner,
                        None if mode is None else int(mode, 8),
                    )
                )
        elif instruction == "WORKDIR":
            workdir = posixpath.normpath(
                posixpath.join(workdir or "/", _expand(value, values))
            )
            out.append(("workdir", workdir))
        elif instruction == "USER":
            user = _user(_expand(value, values), backend)
            out.append(("user", user))
        elif instruction == "CMD":
            cmd, cmd_set = _command(value, shell), True
        elif instruction == "ENTRYPOINT":
            entrypoint = _command(value, shell)
            if not cmd_set:
                cmd = None  # an inherited Cmd is reset, as in Docker
        elif instruction == "VOLUME":
            paths = _exec_form(value) or _words(value)
            volumes.update({_expand(path, values): {} for path in paths})
    config = {
        "Env": [f"{name}={item}" for name, item in env.items()],
        "WorkingDir": workdir,
        "User": user,
        "Entrypoint": entrypoint,
        "Cmd": cmd,
        "Volumes": volumes,
    }
    return tuple(out), {key: item for key, item in config.items() if item}


def _word(text: str) -> str:
    """A COPY word as is, quoted only when the shell form needs it: a
    builder may not glob a quoted wildcard."""
    return text if re.fullmatch(r"[^\s'\"\\$]+", text) else shlex.quote(text)


# A RUN as USER on a builder that runs it as root: the user's uid, gid,
# groups and HOME, as Docker gives them; a uid without a passwd entry gets
# gid 0, a name without one fails.
_AS_USER = (
    'u={user}; if id "$u" >/dev/null 2>&1; then'
    ' set -- --reuid="$(id -u "$u")" --regid="$(id -g "$u")" --init-groups;'
    " HOME=$(awk -F: -v u=\"$u\" '$1==u||$3==u{{print $6; exit}}' /etc/passwd);"
    ' else set -- --reuid="$u" --regid=0 --clear-groups; HOME=; fi;'
    ' HOME=${{HOME:-/}} exec setpriv "$@" -- /bin/sh -c {command}'
)


def lines(plan: Plan, found: tuple[tuple[Any, ...], ...]) -> tuple[str, ...]:
    """The instructions after FROM as one-line Dockerfile instructions, for
    a builder that runs them on the pulled base (``FROM base``) without
    build args: each stage ARG gets the build arg of its name as its
    default, or without one and without a default the value Docker would
    give it from the global ARGs or the platform.

    COPY and ADD are written from their ``found`` steps, as a plain COPY of
    the resolved sources (``.`` for the context root) to the resolved dest:
    Modal uploads only the context files a COPY's sources name, and reads
    no ADD, exec form or $VAR there. Modal's builder runs every RUN as
    root, so USER is left out and a RUN under a non-root USER (the base
    image's or a USER's) runs its ``found`` command through ``setpriv``."""
    inherited = {**PLATFORM_ARGS, **plan.global_args}
    copies = iter(step for step in found if step[0] in ("copy", "add"))
    users = iter(step[1] for step in found if step[0] == "user")
    runs = iter(step[1] for step in found if step[0] == "run")
    user = next(users)
    out = []
    for instruction, value in plan.instructions:
        if instruction == "USER":
            user = next(users)
            continue
        if instruction == "RUN":
            command = next(runs)
            if user != "root":
                value = _AS_USER.format(
                    user=shlex.quote(user), command=shlex.quote(command)
                )
        elif instruction in ("COPY", "ADD"):
            sources, _, flags = _sources(instruction, value)
            group = [next(copies) for _ in sources]
            words = [f"--chown={group[0][3]}"] if "--chown" in flags else []
            if "--chmod" in flags:
                words.append(f"--chmod={flags['--chmod']}")
            words += [_word(step[1]) for step in group] + [_word(group[0][2])]
            value, instruction = " ".join(words), "COPY"
        elif instruction == "ARG":
            words = []
            for word in _words(value):
                name = _expand(word.partition("=")[0], {})
                if name in plan.build_args:
                    word = f"{name}={shlex.quote(plan.build_args[name])}"
                elif "=" not in word and name in inherited:
                    word = f"{name}={shlex.quote(inherited[name])}"
                words.append(word)
            value = " ".join(words)
        out.append(f"{instruction} {value}")
    return tuple(out)


def _matches(name: str, pattern: str) -> bool:
    """Docker's source match: a wildcard does not cross /."""
    names, patterns = name.split("/"), pattern.split("/")
    return len(names) == len(patterns) and all(map(fnmatchcase, names, patterns))


def refuse_archives(
    found: tuple[tuple[Any, ...], ...],
    input_path: Path,
    directory: str,
    backend: str = "e2b",
) -> None:
    """ADD of a local tar archive, which Docker would extract, is refused,
    by the build input and the steps: at admission (without the base
    image's config, which is not known yet) and again in the build job."""
    added = [step[1] for step in found if step[0] == "add"]
    if not added:
        return
    prefix = f"{directory}/ctx/"
    with tarfile.open(input_path) as archive:
        for member in archive:
            name = member.name.removeprefix(prefix)
            if not member.isfile() or name == member.name:
                continue
            if any(_matches(name, pattern) for pattern in added):
                reader = archive.extractfile(member)
                if reader is not None and tarfile.is_tarfile(reader):
                    raise unsupported("ADD of a local archive", backend)


def _keep_mode(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo:
    return tarfile.tar_filter(member, path).replace(
        mode=member.mode & 0o7777, uid=None, gid=None, uname=None, gname=None
    )


def extract_context(input_path: Path, directory: str, dest: Path) -> None:
    """The build input's context (already validated: regular files and
    directories only) into ``dest``, for the SDK's COPY uploads. Modes are
    kept whole (the tar filter would drop setuid and group/other write
    bits, which Docker keeps); owners are not."""
    prefix = f"{directory}/ctx/"
    dest.mkdir(mode=0o700)
    with tarfile.open(input_path) as archive:
        members = []
        for member in archive:
            name = member.name.removeprefix(prefix)
            if name != member.name:
                members.append(member.replace(name=name))
        archive.extractall(dest, members=members, filter=_keep_mode)


__all__ = ["Plan", "extract_context", "lines", "parse", "refuse_archives", "steps"]
