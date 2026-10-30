#!/usr/bin/env bash
# Build the static Podman bundle: build.sh RELEASE OUT_DIR, RELEASE as in
# rsi_harness.cluster.podman_bundle (for example 5.8.9-1). Writes
# podman-static-RELEASE-linux-amd64.tar.gz and podman-static-RELEASE-sources.tar.gz.
set -euo pipefail
release=$1
here="$(cd "$(dirname "$0")" && pwd)"
out="$(mkdir -p "$2" && cd "$2" && pwd)"
builder=rsi-podman-bundle
docker buildx inspect "$builder" >/dev/null 2>&1 \
  || docker buildx create --name "$builder" --driver docker-container >/dev/null
tree="$(mktemp -d)"
trap 'rm -rf "$tree"' EXIT
docker buildx build --builder "$builder" --platform linux/amd64 \
  --output "type=local,dest=$tree" "$here"
name="podman-static-$release"
mv "$tree/bundle" "$tree/$name"
mv "$tree/sources" "$tree/$name-sources"
pack() {
  tar -C "$tree" --sort=name --owner=0 --group=0 --numeric-owner \
    --mtime=@0 -cf - "$1" | gzip -n -9 > "$out/$2"
}
pack "$name" "$name-linux-amd64.tar.gz"
pack "$name-sources" "$name-sources.tar.gz"
(cd "$out" && sha256sum "$name-linux-amd64.tar.gz" "$name-sources.tar.gz")
