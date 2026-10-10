#!/usr/bin/env bash
# Pre-pull a large image set by digest, ahead of runs, with plain
# `docker pull` (docs/sandbox-operator-guide.md, "Pre-pulling large image sets").
#
#   scripts/operator/prepull_images.sh [--dry-run] [--retries N] MANIFEST
#
# MANIFEST has one `name@sha256:<64 hex>` per line; blank lines and `#`
# comments are skipped (sample_tasks/harbor-in-judge/images.manifest is an example). An
# image already on the host under that reference is skipped. A failed pull
# is retried N times (default 3) after a pause growing by
# RSI_PREPULL_PAUSE_SEC (default 10) each time; the images that
# still failed are listed at the end and the exit status is 1. Run it as a
# user that may use Docker (root or the docker group); a `docker login`
# of that user applies to these pulls, never to the broker's. --dry-run
# checks the manifest and prints what would be pulled, pulling nothing.
set -euo pipefail

retries=3
pause=${RSI_PREPULL_PAUSE_SEC:-10}
dry_run=0
manifest=""
while (($#)); do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --retries)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo "--retries needs a number" >&2; exit 2; }
      retries=$2
      shift
      ;;
    -h | --help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *)
      [[ -z "$manifest" ]] || { echo "one MANIFEST only" >&2; exit 2; }
      manifest=$1
      ;;
  esac
  shift
done
[[ -n "$manifest" && -f "$manifest" ]] || { echo "usage: $0 [--dry-run] [--retries N] MANIFEST" >&2; exit 2; }

refs=()
line_no=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line_no=$((line_no + 1))
  line=${line%%#*}
  line=${line//[[:space:]]/}
  [[ -z "$line" ]] && continue
  if [[ ! "$line" =~ ^[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}$ ]]; then
    echo "$manifest:$line_no: not name@sha256:<64 hex>: $line" >&2
    exit 2
  fi
  refs+=("$line")
done <"$manifest"
echo "${#refs[@]} images in $manifest"

if ((dry_run)); then
  ((${#refs[@]} == 0)) || printf 'would pull %s\n' "${refs[@]}"
  exit 0
fi

root=$(docker info --format '{{.DockerRootDir}}')
df -h "$root" | sed "s|^|docker root $root: |"

pulled=0 present=0 failed=()
for ref in "${refs[@]}"; do
  if docker image inspect "$ref" >/dev/null 2>&1; then
    present=$((present + 1))
    continue
  fi
  attempt=0
  until docker pull --quiet "$ref" >/dev/null; do
    attempt=$((attempt + 1))
    if ((attempt > retries)); then
      failed+=("$ref")
      break
    fi
    echo "retry $attempt/$retries in $((attempt * pause)) s: $ref" >&2
    sleep $((attempt * pause))
  done
  if ((attempt <= retries)); then
    pulled=$((pulled + 1))
    echo "pulled $ref"
  fi
done

echo "pulled $pulled, already present $present, failed ${#failed[@]}"
df -h "$root" | sed "s|^|docker root $root: |"
if ((${#failed[@]})); then
  printf 'failed %s\n' "${failed[@]}" >&2
  exit 1
fi
