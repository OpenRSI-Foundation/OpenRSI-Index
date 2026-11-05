#!/usr/bin/env bash
# Build the static Podman bundle: build.sh RELEASE OUT_DIR [ARCH], RELEASE as
# in rsi_harness.cluster.podman_bundle (for example 5.8.9-1), ARCH amd64
# (default) or arm64; a foreign ARCH needs QEMU binfmt on the build host.
# Writes podman-static-RELEASE-linux-ARCH.tar.gz and
# podman-static-RELEASE-sources.tar.gz, the same sources for every ARCH.
set -euo pipefail
release=$1
arch=${3:-amd64}
case "$arch" in
  amd64 | arm64) ;;
  *) echo "unsupported architecture: $arch (amd64 or arm64)" >&2; exit 2 ;;
esac
here="$(cd "$(dirname "$0")" && pwd)"
out="$(mkdir -p "$2" && cd "$2" && pwd)"
builder=rsi-podman-bundle
docker buildx inspect "$builder" >/dev/null 2>&1 \
  || docker buildx create --name "$builder" --driver docker-container >/dev/null
tree="$(mktemp -d)"
trap 'rm -rf "$tree"' EXIT
docker buildx build --builder "$builder" --platform "linux/$arch" \
  --output "type=local,dest=$tree" "$here"
name="podman-static-$release"
mv "$tree/bundle" "$tree/$name"
mv "$tree/sources" "$tree/$name-sources"
pack() {
  tar -C "$tree" --sort=name --owner=0 --group=0 --numeric-owner \
    --mtime=@0 -cf - "$1" | gzip -n -9 > "$out/$2"
}
pack "$name" "$name-linux-$arch.tar.gz"
pack "$name-sources" "$name-sources.tar.gz"
(cd "$out" && sha256sum "$name-linux-$arch.tar.gz" "$name-sources.tar.gz")
