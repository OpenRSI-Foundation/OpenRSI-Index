"""Generate the archive-free control client installed in Work containers."""

from __future__ import annotations


def generate_submit_client() -> str:
    """Return a synchronous, curl-free client that carries no workspace data."""
    return r'''#!/bin/sh
set -eu

mode=submit
if [ "$#" -gt 1 ]; then
    printf 'Usage: rsi-submit [--list|--help]\n' >&2
    exit 2
fi
if [ "$#" -eq 1 ]; then
    case "$1" in
        --list) mode=list ;;
        --help)
            printf '%s\n' \
                'Usage: rsi-submit [--list|--help]' \
                '' \
                'Options:' \
                '  --list  Show previous submissions for this run without submitting' \
                '  --help  Show this help message'
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$1" >&2
            exit 2
            ;;
    esac
fi

: "${RSI_JUDGE_URL:?RSI_JUDGE_URL is required}"
: "${RSI_TOKEN:?RSI_TOKEN is required}"
base_url=${RSI_JUDGE_URL%/}

if command -v python3 >/dev/null 2>&1; then
    python=python3
elif command -v python >/dev/null 2>&1; then
    python=python
elif command -v curl >/dev/null 2>&1; then
    if [ "$mode" = list ]; then
        exec curl --fail-with-body --silent --show-error \
            -H "Authorization: Bearer ${RSI_TOKEN}" \
            "${base_url}/api/v1/history"
    fi
    exec curl --fail-with-body --silent --show-error \
        -X POST \
        -H "Authorization: Bearer ${RSI_TOKEN}" \
        "${base_url}/api/v1/submit"
else
    printf 'rsi-submit requires Python 3 or curl in the task image\n' >&2
    exit 127
fi

exec "$python" - "$mode" "$base_url" <<'PY'
import http.client
import os
import sys
import urllib.parse

mode, base_url = sys.argv[1:]
parsed = urllib.parse.urlsplit(base_url)
if parsed.scheme not in {"http", "https"} or not parsed.hostname:
    print("RSI_JUDGE_URL must be an absolute HTTP(S) URL", file=sys.stderr)
    raise SystemExit(2)

connection_type = (
    http.client.HTTPSConnection
    if parsed.scheme == "https"
    else http.client.HTTPConnection
)
connection = connection_type(parsed.hostname, parsed.port)
suffix = "/api/v1/history" if mode == "list" else "/api/v1/submit"
path = parsed.path.rstrip("/") + suffix
headers = {"Authorization": "Bearer " + os.environ["RSI_TOKEN"]}
try:
    connection.request("GET" if mode == "list" else "POST", path, headers=headers)
    response = connection.getresponse()
    body = response.read()
finally:
    connection.close()

sys.stdout.buffer.write(body)
sys.stdout.buffer.flush()
if response.status >= 400:
    print(
        f"rsi-submit: judge returned HTTP {response.status}",
        file=sys.stderr,
    )
    raise SystemExit(22)
PY
'''
