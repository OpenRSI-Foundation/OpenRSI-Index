"""Content-addressed Apptainer image planning and build payloads."""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

from pydantic import field_validator

from rsi_harness.cluster.config import ClusterProfile
from rsi_harness.models import PersistedModel
from rsi_harness.task.digest import hash_tree

_CACHE_SCHEMA = b"rsi-harness-bluevela-sif-v3\0"


class ImagePlan(PersistedModel):
    cache_key: str
    sif_path: Path
    sha256_path: Path
    cache_hit: bool

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


def plan_image(build_context: Path, cache_root: Path) -> ImagePlan:
    context = Path(build_context).resolve()
    cache = Path(cache_root).resolve()
    context_digest = hash_tree(context, excluded_roots=(".git",))
    cache_key = hashlib.sha256(
        _CACHE_SCHEMA + context_digest.encode("ascii")
    ).hexdigest()
    sif_path = cache / f"{cache_key}.sif"
    plan = ImagePlan(
        cache_key=cache_key,
        sif_path=sif_path,
        sha256_path=cache / f"{cache_key}.sif.sha256",
        cache_hit=False,
    )
    return plan.model_copy(update={"cache_hit": validate_cached_image(plan)})


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
    temp_template = profile.builder.temp_root / "rsi-harness-sif.XXXXXX"
    dockerfile = Path(frozen_context) / "Dockerfile"
    rootless_apt = (
        profile.builder.rootless_apt_sandbox
        and dockerfile.is_file()
        and "apt-get" in dockerfile.read_text(errors="replace")
    )
    script = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077

builder={_quote(profile.builder.binary)}
apptainer={_quote(profile.apptainer.binary)}
frozen_context={_quote(Path(frozen_context).resolve())}
cache_dir={_quote(cache_dir)}
final_sif={_quote(plan.sif_path)}
final_sha={_quote(plan.sha256_path)}
image_name={_quote(image_name)}
builder_tmp_root={_quote(profile.builder.temp_root)}
faked={_quote(profile.builder.faked_binary or '')}
fakeroot_library={_quote(profile.builder.fakeroot_library or '')}
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

local_root="$(mktemp -d {_quote(temp_template)})"
partial_sif="$cache_dir/.{plan.cache_key}.${{LSB_JOBID:-$$}}.sif.partial"
partial_sha="$cache_dir/.{plan.cache_key}.${{LSB_JOBID:-$$}}.sha256.partial"
cleanup() {{
  if [[ -n "${{faked_pid:-}}" ]]; then
    kill "$faked_pid" >/dev/null 2>&1 || true
  fi
  if declare -p podman >/dev/null 2>&1; then
    "${{podman[@]}}" system reset --force >/dev/null 2>&1 || true
  fi
  rm -rf "$local_root"
  rm -f "$partial_sif" "$partial_sha"
}}
trap cleanup EXIT

local_context="$local_root/context"
local_archive="$local_root/image.tar"
local_sif="$local_root/image.sif"
export XDG_RUNTIME_DIR="$local_root/xdg-runtime"
export CONTAINERS_REGISTRIES_CONF="$local_root/registries.conf"
unset DBUS_SESSION_BUS_ADDRESS
mkdir -p "$local_context" "$local_root/podman-root" \
  "$local_root/podman-runroot" "$local_root/apptainer-tmp" \
  "$local_root/apptainer-cache" "$XDG_RUNTIME_DIR"
chmod 0700 "$XDG_RUNTIME_DIR"
printf '%s\n' \
  'unqualified-search-registries = ["docker.io"]' \
  'short-name-mode = "disabled"' > "$CONTAINERS_REGISTRIES_CONF"
cp -a "$frozen_context"/. "$local_context"/

rootless_build_args=()
if [[ -n "$faked" ]]; then
  export FAKED_MODE=unknown-is-root
  key_pid="$("$faked")"
  if [[ ! "$key_pid" =~ ^([0-9]+):([0-9]+)$ ]]; then
    echo "cannot start fakeroot daemon: $key_pid" >&2
    exit 1
  fi
  fake_key="${{BASH_REMATCH[1]}}"
  faked_pid="${{BASH_REMATCH[2]}}"
  rootless_build_args+=(
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
  --runroot "$local_root/podman-runroot"
  --storage-driver=overlay
  --storage-opt ignore_chown_errors=true
  --cgroup-manager=cgroupfs
  --events-backend=file
)
"${{podman[@]}}" build --isolation rootless --format docker \
  "${{rootless_build_args[@]}}" \
  -t "$image_name" "$local_context"
"${{podman[@]}}" save --format docker-archive -o "$local_archive" "$image_name"

export APPTAINER_BIND={_quote(profile.apptainer.dns_bind)}
export APPTAINER_TMPDIR="$local_root/apptainer-tmp"
export APPTAINER_CACHEDIR="$local_root/apptainer-cache"
"$apptainer" build {apptainer_args} \
  --mksquashfs-args {_quote(f"-processors {profile.builder.cpu_slots}")} \
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
