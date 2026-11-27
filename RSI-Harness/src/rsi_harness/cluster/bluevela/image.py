"""Content-addressed Apptainer image planning and build payloads."""

from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass
from pathlib import Path

from pydantic import field_validator

from rsi_harness.cluster import podman_bundle
from rsi_harness.cluster.bluevela.preflight import (
    apptainer_preflight,
    node_tmp_assignment,
)
from rsi_harness.cluster.config import ClusterProfile
from rsi_harness.models import PersistedModel
from rsi_harness.task.digest import hash_tree
from rsi_loop.harness.agent.base import Agent
from rsi_loop.harness.agent.factory import get_agent_class

_CACHE_SCHEMA = b"rsi-harness-bluevela-sif-v3\0"
_AGENT_LAYER_SCHEMA = b"rsi-harness-bluevela-agent-layer-v1\0"
# Where the definition's %files stage the Harness's Agent files, which the
# build section installs in order and then removes.
AGENT_FILES = "/opt/rsi-harness/.agent-files"
_AGENT_SCRIPT = "install-agent.sh"
# The local Work image's prerequisite step (bash, curl, tar, xz ...), as a
# command: the RUN instruction without its keyword.
_AGENT_PREREQUISITES_SCRIPT = (
    "if command -v apt-get >/dev/null 2>&1; then "
    "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
    "bash ca-certificates curl passwd sudo tar xz-utils && "
    "rm -rf /var/lib/apt/lists/*; "
    "elif command -v apk >/dev/null 2>&1; then "
    "apk add --no-cache bash ca-certificates curl shadow sudo tar xz; "
    "else echo 'setup command failed: supported package manager' >&2; "
    "exit 86; fi"
)


def agent_layer_required(agent_name: str) -> bool:
    """Whether the Agent's Work files must be installed into the image.

    Codex and Claude Code run as one native binary bound from shared
    storage. An Agent with packaged image files (the Agent driver, pi's
    extension) needs its pinned CLI, node and those files in the image, as
    on local Docker: the cluster image gets an Agent layer on top of the
    task's own SIF.
    """
    return agent_name != "oracle" and bool(get_agent_class(agent_name).image_files)


@dataclass(frozen=True)
class AgentLayer:
    """An Agent's install as an Apptainer build section on the task's SIF.

    ``script`` runs the same steps as the local Work image's Dockerfile;
    ``files`` are the Harness's packaged Agent files, staged under
    ``AGENT_FILES`` by the definition's ``%files`` and installed by the
    script at their targets in the Dockerfile's order.
    """

    agent_name: str
    script: str
    files: tuple[tuple[str, bytes], ...]

    @property
    def digest(self) -> str:
        digest = hashlib.sha256(_AGENT_LAYER_SCHEMA)
        for value in (self.agent_name, self.script):
            encoded = value.encode()
            digest.update(len(encoded).to_bytes(8, "big") + encoded)
        for name, content in self.files:
            digest.update(len(name).to_bytes(8, "big") + name.encode())
            digest.update(len(content).to_bytes(8, "big") + content)
        return digest.hexdigest()

    def definition(self, task_sif: Path, directory: Path) -> str:
        """The Apptainer definition, with the layer written in ``directory``.

        The script runs as a staged file, so its text needs no quoting or
        indentation for the definition format.
        """
        names = (*(name for name, _content in self.files), _AGENT_SCRIPT)
        files = "".join(
            f"    {directory / name} {AGENT_FILES}/{name}\n" for name in names
        )
        return (
            f"Bootstrap: localimage\nFrom: {task_sif}\n\n"
            f"%files\n{files}\n"
            f"%post\n    /bin/sh -eu {AGENT_FILES}/{_AGENT_SCRIPT}\n"
        )

    def write(self, directory: Path, task_sif: Path) -> Path:
        """Write the files, the script and the definition into a new directory."""
        directory.mkdir(mode=0o700)
        for name, content in self.files:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (directory / _AGENT_SCRIPT).write_text(self.script)
        definition = directory / "agent.def"
        definition.write_text(self.definition(task_sif, directory))
        return definition


def agent_layer(agent: Agent) -> AgentLayer:
    """The layer that installs the Agent as the local Work image does."""
    # Here, not at import: the image module loads the Docker SDK, which the
    # CLI and the cluster job otherwise never need.
    from rsi_harness.runtime.images import (
        agent_build_args,
        agent_install_steps,
        chmod_command,
    )

    arguments = agent_build_args(agent)
    # Its own command, as its own RUN on Docker: `sh -e` would not stop on a
    # failure inside its `if ... && ...` list.
    lines = ["set -eu", shlex.join(("/bin/sh", "-c", _AGENT_PREREQUISITES_SCRIPT))]
    files: list[tuple[str, bytes]] = []
    for step in agent_install_steps(agent):
        if step.copy is not None:
            files.append((step.copy.context_name, step.copy.content))
            lines.append(
                "install -D -m 0644 "
                f"{shlex.quote(f'{AGENT_FILES}/{step.copy.context_name}')} "
                f"{shlex.quote(str(step.copy.target))}"
            )
        elif step.chmod:
            lines.append(shlex.join(("/bin/bash", "-lc", chmod_command(step.chmod))))
        elif step.arg is not None:
            # A Dockerfile ARG without a value leaves the variable unset.
            value = arguments.get(step.arg)
            if value:
                lines.append(f"export {step.arg}={shlex.quote(value)}")
        else:
            lines.append(shlex.join(step.argv))
    lines.append(f"rm -rf {AGENT_FILES}")
    return AgentLayer(
        agent_name=agent.name, script="\n".join(lines) + "\n", files=tuple(files)
    )


class ImagePlan(PersistedModel):
    cache_key: str
    sif_path: Path
    sha256_path: Path
    cache_hit: bool
    # An Agent layer's record of the task SIF digest it was built on.
    base_path: Path | None = None

    @field_validator("sif_path", "sha256_path")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("SIF cache paths must be absolute")
        return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cached_image(plan: ImagePlan) -> bool:
    """Require a nonempty SIF and a sidecar that verifies its exact bytes."""
    if not plan.sif_path.is_file() or plan.sif_path.stat().st_size == 0:
        return False
    if not plan.sha256_path.is_file():
        return False
    try:
        fields = plan.sha256_path.read_text().strip().split()
        if len(fields) != 2 or fields[1].lstrip("*") != plan.sif_path.name:
            return False
        expected = fields[0]
        return len(expected) == 64 and _file_sha256(plan.sif_path) == expected
    except OSError:
        return False


def plan_image(
    build_context: Path, cache_root: Path, agent: AgentLayer | None = None
) -> ImagePlan:
    context = Path(build_context).resolve()
    cache = Path(cache_root).resolve()
    context_digest = hash_tree(context, excluded_roots=(".git",))
    key_input = _CACHE_SCHEMA + context_digest.encode("ascii")
    if agent is not None:
        # The task's image alone keeps its key; each Agent layer has its own.
        key_input += b"\0agent\0" + agent.digest.encode("ascii")
    cache_key = hashlib.sha256(key_input).hexdigest()
    sif_path = cache / f"{cache_key}.sif"
    plan = ImagePlan(
        cache_key=cache_key,
        sif_path=sif_path,
        sha256_path=cache / f"{cache_key}.sif.sha256",
        cache_hit=False,
        base_path=None if agent is None else cache / f"{cache_key}.sif.base",
    )
    hit = validate_cached_image(plan)
    if agent is not None and hit:
        # Only on the task SIF the cache holds now: one evicted and rebuilt
        # since would leave the layer on bytes the Oracle no longer uses.
        task = plan_image(context, cache)
        hit = task.cache_hit and _recorded_digest(plan.base_path) == (
            _recorded_digest(task.sha256_path)
        )
    return plan.model_copy(update={"cache_hit": hit})


def _recorded_digest(path: Path | None) -> str | None:
    try:
        fields = path.read_text().split() if path is not None else []
    except OSError:
        return None
    return fields[0] if fields else None


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def render_build_driver(
    plan: ImagePlan,
    profile: ClusterProfile,
    frozen_context: Path,
    output_path: Path,
    minimum_tmp_mb: int,
) -> None:
    """Render the compute-node-only OCI archive to SIF build."""
    output = Path(output_path)
    cache_dir = plan.sif_path.parent
    image_name = f"localhost/rsi-harness-{plan.cache_key}:latest"
    apptainer_args = " ".join(_quote(arg) for arg in profile.apptainer.build_args)
    dockerfile = Path(frozen_context) / "Dockerfile"
    rootless_apt = (
        profile.builder.rootless_apt_sandbox
        and dockerfile.is_file()
        and "apt-get" in dockerfile.read_text(errors="replace")
    )
    builder = profile.builder
    builder_tmp = node_tmp_assignment(
        "builder_tmp_root", builder.temp_root, profile.resources.node_tmp_variable
    )
    podman_options = [
        f"--storage-driver={builder.storage_driver}",
        "--storage-opt",
        # VFS accepts only the prefixed option name.
        "vfs.ignore_chown_errors=true"
        if builder.storage_driver == "vfs"
        else "ignore_chown_errors=true",
    ]
    for option, path in (("--conmon", builder.conmon), ("--runtime", builder.runtime)):
        if path is not None:
            podman_options += [option, str(path)]
    podman_text = " ".join(_quote(arg) for arg in podman_options)
    host_namespaces = "".join(f" --{name}=host" for name in builder.host_namespaces)
    # Namespaces each RUN step still creates on this node.
    created_namespaces = " ".join(
        kernel
        for name, kernel in (("network", "net"), ("uts", "uts"))
        if name not in builder.host_namespaces
    )
    helpers = " ".join(
        _quote(path) for path in (builder.conmon, builder.runtime) if path is not None
    )
    # The bundle's own configuration, not the node's or the user's.
    bundle_env = ""
    bundle_preflight = ""
    signature_policy = ""
    if builder.podman_bundle is not None:
        arch = podman_bundle.architecture()
        tree = podman_bundle.bundle_dir(builder.podman_bundle, arch)
        machine = podman_bundle.MACHINES[arch]
        # The bundle follows the submit host; a build node must match it.
        bundle_preflight = (
            'case "$(uname -m)" in\n'
            f"  {machine} | {arch}) ;;\n"
            f'  *) echo "compute-node preflight: the Podman bundle is for {machine},"'
            ' "this node is $(uname -m)" >&2\n'
            "    exit 1 ;;\n"
            "esac\n"
        )
        bundle_env = (
            f"export CONTAINERS_CONF={_quote(tree / podman_bundle.CONTAINERS_CONF)}\n"
            "export CONTAINERS_STORAGE_CONF="
            f"{_quote(tree / podman_bundle.STORAGE_CONF)}\n"
        )
        signature_policy = f" --signature-policy {_quote(tree / podman_bundle.POLICY)}"
    build_flags = host_namespaces + signature_policy
    script = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077

{apptainer_preflight(profile.apptainer.binary)}
builder={_quote(profile.builder.binary)}
frozen_context={_quote(Path(frozen_context).resolve())}
cache_dir={_quote(cache_dir)}
final_sif={_quote(plan.sif_path)}
final_sha={_quote(plan.sha256_path)}
image_name={_quote(image_name)}
{builder_tmp}
faked={_quote(profile.builder.faked_binary or "")}
fakeroot_library={_quote(profile.builder.fakeroot_library or "")}
for tool in "$builder" {helpers}; do
  if ! command -v -- "$tool" >/dev/null; then
    echo "compute-node preflight: builder tool is missing or not executable: $tool" >&2
    exit 1
  fi
done
{bundle_preflight}for ns in {created_namespaces}; do
  if [[ "$(cat "/proc/sys/user/max_${{ns}}_namespaces" 2>/dev/null)" == 0 ]]; then
    echo "compute-node preflight: this node cannot create $ns namespaces" \
      "(max_${{ns}}_namespaces=0); add it to builder.host_namespaces" >&2
    exit 1
  fi
done
if [[ ! -d "$builder_tmp_root" ]]; then
  echo "builder temp_root does not exist on this node: $builder_tmp_root" >&2
  exit 1
fi
available_tmp_kb="$(df -Pk -- "$builder_tmp_root" | awk 'NR == 2 {{print $4}}')"
required_tmp_kb=$(({minimum_tmp_mb} * 1024))
if [[ ! "$available_tmp_kb" =~ ^[0-9]+$ ]] || \
   (( available_tmp_kb < required_tmp_kb )); then
  echo "insufficient node-local build scratch:" \
    "need {minimum_tmp_mb} MiB under $builder_tmp_root," \
    "have ${{available_tmp_kb:-unknown}} KiB" >&2
  exit 1
fi

mkdir -p "$cache_dir"
exec 9>"$cache_dir/.{plan.cache_key}.build.lock"
/usr/bin/flock 9

if [[ -s "$final_sif" && -s "$final_sha" ]] && \
   (cd "$cache_dir" && sha256sum -c "$(basename "$final_sha")"); then
  exit 0
fi

local_root="$(mktemp -d "$builder_tmp_root/rsi-harness-sif.XXXXXX")"
# Podman 4.x to 5.0 refuse a runroot longer than 50 characters.
podman_runroot="$(mktemp -d /tmp/rsi-run.XXXXXX)"
# Named after the build job (a job sees only its own scheduler's variables),
# never a PID another node may reuse.
job_tag="${{SLURM_JOB_ID:-${{LSB_JOBID:-${{PBS_JOBID:-${{PJM_JOBID:-$$}}}}}}}}"
partial_sif="$cache_dir/.{plan.cache_key}.$job_tag.sif.partial"
partial_sha="$cache_dir/.{plan.cache_key}.$job_tag.sha256.partial"
cleanup() {{
  if [[ -n "${{faked_pid:-}}" ]]; then
    kill "$faked_pid" >/dev/null 2>&1 || true
  fi
  if declare -p podman >/dev/null 2>&1; then
    "${{podman[@]}}" system reset --force >/dev/null 2>&1 || true
  fi
  rm -rf "$local_root" "$podman_runroot"
  rm -f "$partial_sif" "$partial_sha"
}}
trap cleanup EXIT

local_context="$local_root/context"
local_archive="$local_root/image.tar"
local_sif="$local_root/image.sif"
export XDG_RUNTIME_DIR="$local_root/xdg-runtime"
export CONTAINERS_REGISTRIES_CONF="$local_root/registries.conf"
{bundle_env}# Image copies stage under TMPDIR (default /var/tmp).
export TMPDIR="$local_root/tmp"
unset DBUS_SESSION_BUS_ADDRESS
mkdir -p "$local_context" "$local_root/podman-root" \
  "$local_root/apptainer-tmp" "$local_root/apptainer-cache" \
  "$XDG_RUNTIME_DIR" "$TMPDIR"
chmod 0700 "$XDG_RUNTIME_DIR"
printf '%s\n' \
  'unqualified-search-registries = ["docker.io"]' \
  'short-name-mode = "disabled"' > "$CONTAINERS_REGISTRIES_CONF"
cp -a "$frozen_context"/. "$local_context"/

rootless_build_args=()
fakeroot_args=()
if [[ -n "$faked" ]]; then
  export FAKED_MODE=unknown-is-root
  key_pid="$("$faked")"
  if [[ ! "$key_pid" =~ ^([0-9]+):([0-9]+)$ ]]; then
    echo "cannot start fakeroot daemon: $key_pid" >&2
    exit 1
  fi
  fake_key="${{BASH_REMATCH[1]}}"
  faked_pid="${{BASH_REMATCH[2]}}"
  fakeroot_args+=(
    --ipc host
    --volume "$fakeroot_library:/usr/lib/libfakeroot.so:ro"
    --env "LD_PRELOAD=/usr/lib/libfakeroot.so"
    --env "FAKEROOTKEY=$fake_key"
    --env "FAKEROOTDONTTRYCHOWN=1"
    --env "FAKED_MODE=unknown-is-root"
    --unsetenv LD_PRELOAD
    --unsetenv FAKEROOTKEY
    --unsetenv FAKEROOTDONTTRYCHOWN
    --unsetenv FAKED_MODE
  )
fi
if {_quote(str(rootless_apt).lower())}; then
  apt_config="$local_root/apt-root.conf"
  printf '%s\n' 'APT::Sandbox::User "root";' > "$apt_config"
  rootless_build_args+=(
    --volume
    "$apt_config:/etc/apt/apt.conf.d/99-rsi-rootless.conf:ro"
  )
fi

podman=(
  "$builder"
  --root "$local_root/podman-root"
  --runroot "$podman_runroot"
  {podman_text}
  --cgroup-manager=cgroupfs
  --events-backend=file
)
build_log="$local_root/build.log"
# ${{a[@]+"${{a[@]}}"}}: bash 4.2 (CentOS 7) treats an empty array as unset.
build_image() {{
  "${{podman[@]}}" build --isolation rootless --format docker{build_flags} \\
    ${{rootless_build_args[@]+"${{rootless_build_args[@]}}"}} "$@" \\
    -t "$image_name" "$local_context" 2>&1 | tee "$build_log" >&2
  return "${{PIPESTATUS[0]}}"
}}
build_hint() {{
  local ownership='lchown|error setting ownership|chown.*invalid argument'
  if grep -qiE "$ownership" "$build_log"; then
    echo "build hint: the Dockerfile changes file ownership, which needs" \\
      "/etc/subuid ranges or builder.faked_binary and fakeroot_library" >&2
  fi
  if grep -qiE '(unshare|clone).*no space left on device' "$build_log"; then
    echo "build hint: a namespace limit is 0 on this node;" \\
      "see builder.host_namespaces" >&2
  fi
  if grep -qiE 'mount_program is required|overlay.*not supported' "$build_log"; then
    echo "build hint: rootless overlay is unavailable;" \\
      'set builder.storage_driver = "vfs"' >&2
  fi
  if grep -qiE 'could not find a working conmon|conmon.*not found' "$build_log"; then
    echo "build hint: set builder.conmon to the conmon executable" >&2
  fi
}}
if ! build_image ${{fakeroot_args[@]+"${{fakeroot_args[@]}}"}}; then
  # A glibc libfakeroot does not load in every image (musl, older glibc).
  # Not Podman's echo of the ENV step that sets LD_PRELOAD to it.
  if [[ -n "$faked" ]] && grep -v '^STEP [0-9]*/[0-9]*: ' "$build_log" \\
      | grep -q /usr/lib/libfakeroot.so; then
    echo "build note: libfakeroot does not load in this image;" \\
      "rebuilding without it" >&2
    build_image || {{ build_hint; exit 1; }}
  else
    build_hint
    exit 1
  fi
fi
"${{podman[@]}}" save{signature_policy} --format docker-archive \\
  -o "$local_archive" "$image_name"

export APPTAINER_BIND={_quote(profile.apptainer.dns_bind)}
export APPTAINER_TMPDIR="$local_root/apptainer-tmp"
export APPTAINER_CACHEDIR="$local_root/apptainer-cache"
squashfs_args=(--mksquashfs-args {_quote(f"-processors {profile.builder.cpu_slots}")})
if (( singularity_ce )); then
  # SingularityCE reads only its own names and has no --mksquashfs-args.
  export SINGULARITY_BIND="$APPTAINER_BIND"
  export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
  export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
  squashfs_args=()
fi
"$apptainer" build {apptainer_args} ${{squashfs_args[@]+"${{squashfs_args[@]}}"}} \
  "$local_sif" "docker-archive://$local_archive"
"$apptainer" inspect "$local_sif" >/dev/null
test -s "$local_sif"

digest="$(sha256sum "$local_sif" | awk '{{print $1}}')"
/usr/bin/dd if="$local_sif" of="$partial_sif" bs=16M status=none conv=fsync
printf '%s  %s\n' "$digest" "$(basename "$final_sif")" > "$partial_sha"
mv "$partial_sif" "$final_sif"
mv "$partial_sha" "$final_sha"
(cd "$cache_dir" && sha256sum -c "$(basename "$final_sha")")
"""
    output.write_text(script)
    output.chmod(0o700)


def render_agent_build_driver(
    plan: ImagePlan,
    task: ImagePlan,
    profile: ClusterProfile,
    task_driver: Path,
    definition: Path,
    output_path: Path,
    minimum_tmp_mb: int,
) -> None:
    """Render the build of an Agent layer's SIF on the task's own SIF.

    The task driver builds the task's SIF first or finds it cached, so the
    Oracle, Codex, Claude Code and every Agent layer of a task share its
    exact bytes. The layer is an Apptainer build from that local image.
    """
    if plan.base_path is None:
        raise ValueError("an Agent layer plan records the task SIF it is built on")
    base_path = plan.base_path
    output = Path(output_path)
    cache_dir = plan.sif_path.parent
    apptainer_args = " ".join(_quote(arg) for arg in profile.apptainer.build_args)
    builder_tmp = node_tmp_assignment(
        "builder_tmp_root",
        profile.builder.temp_root,
        profile.resources.node_tmp_variable,
    )
    script = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077

{apptainer_preflight(profile.apptainer.binary)}
task_driver={_quote(Path(task_driver).resolve())}
definition={_quote(Path(definition).resolve())}
task_sif={_quote(task.sif_path)}
task_sha={_quote(task.sha256_path)}
final_base={_quote(base_path)}
cache_dir={_quote(cache_dir)}
final_sif={_quote(plan.sif_path)}
final_sha={_quote(plan.sha256_path)}
{builder_tmp}
# The task's own SIF: built now, or the cached one the Oracle, Codex,
# Claude Code and the other Agent layers of this task use.
"$task_driver"
(cd "$(dirname "$task_sif")" && sha256sum -c "$(basename "$task_sha")") >/dev/null
task_digest="$(awk '{{print $1}}' "$task_sha")"
if [[ ! -d "$builder_tmp_root" ]]; then
  echo "builder temp_root does not exist on this node: $builder_tmp_root" >&2
  exit 1
fi
available_tmp_kb="$(df -Pk -- "$builder_tmp_root" | awk 'NR == 2 {{print $4}}')"
required_tmp_kb=$(({minimum_tmp_mb} * 1024))
if [[ ! "$available_tmp_kb" =~ ^[0-9]+$ ]] || \\
   (( available_tmp_kb < required_tmp_kb )); then
  echo "insufficient node-local build scratch:" \\
    "need {minimum_tmp_mb} MiB under $builder_tmp_root," \\
    "have ${{available_tmp_kb:-unknown}} KiB" >&2
  exit 1
fi

mkdir -p "$cache_dir"
exec 9>"$cache_dir/.{plan.cache_key}.build.lock"
/usr/bin/flock 9

if [[ -s "$final_sif" && -s "$final_sha" ]] && \\
   [[ "$(cat "$final_base" 2>/dev/null)" == "$task_digest" ]] && \\
   (cd "$cache_dir" && sha256sum -c "$(basename "$final_sha")"); then
  exit 0
fi

local_root="$(mktemp -d "$builder_tmp_root/rsi-harness-agent-sif.XXXXXX")"
job_tag="${{SLURM_JOB_ID:-${{LSB_JOBID:-${{PBS_JOBID:-${{PJM_JOBID:-$$}}}}}}}}"
partial_sif="$cache_dir/.{plan.cache_key}.$job_tag.sif.partial"
partial_sha="$cache_dir/.{plan.cache_key}.$job_tag.sha256.partial"
partial_base="$cache_dir/.{plan.cache_key}.$job_tag.base.partial"
cleanup() {{
  rm -rf "$local_root"
  rm -f "$partial_sif" "$partial_sha" "$partial_base"
}}
trap cleanup EXIT
local_sif="$local_root/image.sif"
export APPTAINER_BIND={_quote(profile.apptainer.dns_bind)}
export APPTAINER_TMPDIR="$local_root/apptainer-tmp"
export APPTAINER_CACHEDIR="$local_root/apptainer-cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
squashfs_args=(--mksquashfs-args {_quote(f"-processors {profile.builder.cpu_slots}")})
if (( singularity_ce )); then
  export SINGULARITY_BIND="$APPTAINER_BIND"
  export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
  export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
  squashfs_args=()
fi
"$apptainer" build {apptainer_args} ${{squashfs_args[@]+"${{squashfs_args[@]}}"}} \\
  "$local_sif" "$definition"
"$apptainer" inspect "$local_sif" >/dev/null
test -s "$local_sif"

digest="$(sha256sum "$local_sif" | awk '{{print $1}}')"
/usr/bin/dd if="$local_sif" of="$partial_sif" bs=16M status=none conv=fsync
printf '%s  %s\\n' "$digest" "$(basename "$final_sif")" > "$partial_sha"
printf '%s\\n' "$task_digest" > "$partial_base"
# The digest last: a valid sidecar implies the SIF and its base record.
rm -f "$final_sha"
mv "$partial_sif" "$final_sif"
mv "$partial_base" "$final_base"
mv "$partial_sha" "$final_sha"
(cd "$cache_dir" && sha256sum -c "$(basename "$final_sha")")
"""
    output.write_text(script)
    output.chmod(0o700)
