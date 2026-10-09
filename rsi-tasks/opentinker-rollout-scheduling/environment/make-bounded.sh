#!/bin/bash
# Bound native dependency builds even when upstream setup.py asks for all host CPUs.
set -euo pipefail
args=()
possible_count=0
for arg in "$@"; do
    if (( possible_count )); then
        possible_count=0
        if [[ "$arg" =~ ^[0-9]+$ ]]; then
            continue
        fi
    fi
    case "$arg" in
        -j|--jobs) possible_count=1 ;;
        -j[0-9]*|--jobs=*) ;;
        *) args+=("$arg") ;;
    esac
done
export MAKEFLAGS=-j8
exec /usr/bin/make -j8 "${args[@]}"
