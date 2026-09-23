# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28,<1"]
# ///
"""Publish only the standalone dashboard to its dedicated Cloudflare Worker.

Uses Cloudflare's Workers Static Assets direct-upload API. Credentials come
from environment variables or an explicitly supplied private JSON file;
the token and upload JWTs are never printed or saved into the repository.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.cloudflare.com/client/v4"
WORKER = "openrsi-logs"
TAG = "openrsi-index-logs-dashboard"
STATIC_FILES = (
    "index.html", "assets/logs.css", "assets/logs.js", "assets/favicon.png",
    "assets/openrsi-mark.png", "assets/logos/claude.png", "assets/logos/openai.svg",
)
MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8", ".json": "application/json",
        ".gz": "application/gzip", ".svg": "image/svg+xml", ".png": "image/png"}


def dashboard_files(root=ROOT):
    """Export the UI and referenced trace pages, excluding tools and local files."""
    data = root / "assets/logs-data"
    index = json.loads((data / "index.json").read_text())
    if not index.get("runs"):
        raise ValueError("The archive is empty; run dashboard/tools/build.py first.")
    names = list(STATIC_FILES) + ["assets/logs-data/index.json"]
    for run in index["runs"]:
        ident = run["id"]
        if not re.fullmatch(r"[0-9a-f]{16}", ident):
            raise ValueError("Invalid run ID in the generated archive.")
        prefix = f"assets/logs-data/{ident}/"
        names.append(prefix + "index.json")
        detail = json.loads((root / prefix / "index.json").read_text())
        for page in detail["pages"]:
            if not re.fullmatch(r"[0-9]{4,}\.json\.gz", page["file"]):
                raise ValueError("Invalid trace page in the generated archive.")
            names.append(prefix + page["file"])
    paths = []
    for name in sorted(set(names)):
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid dashboard asset: {name}")
        paths.append(path)
    return index, paths


def manifest(paths, root=ROOT):
    assets, by_hash = {}, {}
    for path in paths:
        raw = path.read_bytes()
        # Include the extension so identical bytes with different MIME types
        # remain separate assets. See Cloudflare's direct-upload example.
        digest = hashlib.sha256(base64.b64encode(raw) + path.suffix[1:].encode()).hexdigest()[:32]
        assets["/" + path.relative_to(root).as_posix()] = {"hash": digest, "size": len(raw)}
        by_hash[digest] = path
    return assets, by_hash


def api(client, method, path, allow_missing=False, **kwargs):
    for attempt in range(3):
        try:
            response = client.request(method, API + path, **kwargs)
        except httpx.TransportError:
            if attempt == 2:
                raise RuntimeError("Cloudflare could not be reached; retry deployment.") from None
            time.sleep(2 ** attempt)
            continue
        if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
            time.sleep(2 ** attempt)
            continue
        if allow_missing and response.status_code == 404:
            return None
        try:
            value = response.json()
        except ValueError:
            raise RuntimeError(f"Cloudflare returned HTTP {response.status_code} with an unexpected response.") from None
        if response.is_error or not value.get("success"):
            codes = ", ".join(str(error.get("code", "unknown")) for error in value.get("errors", []))
            raise RuntimeError(f"Cloudflare request failed (HTTP {response.status_code}, codes: {codes}).")
        return value.get("result") or {}


def deploy(credentials_file=None, prepare_only=False):
    index, paths = dashboard_files()
    assets, by_hash = manifest(paths)
    size = sum(asset["size"] for asset in assets.values())
    print(f"Prepared {len(assets)} assets ({size / 1024**2:.1f} MiB) for {WORKER}.", flush=True)
    if prepare_only:
        return
    saved = json.loads(credentials_file.read_text()) if credentials_file else {}
    token = os.environ.get("CLOUDFLARE_API_TOKEN") or saved.get("api_token")
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID") or saved.get("account_id")
    if not token or not account or not re.fullmatch(r"[0-9a-f]{32}", account):
        raise ValueError("Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID, or supply a private --credentials-file.")
    base = f"/accounts/{account}/workers"
    endpoint = base + f"/scripts/{WORKER}"
    with httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=180) as client:
        existing = api(client, "GET", endpoint + "/settings", allow_missing=True)
        if existing is not None and TAG not in existing.get("tags", []):
            raise ValueError(f"The existing {WORKER} Worker is not marked as this dashboard; refusing to overwrite it.")
        subdomain = api(client, "GET", base + "/subdomain").get("subdomain")
        if not subdomain:
            raise ValueError("Set up the account's workers.dev subdomain in Cloudflare before deployment.")
        upload = api(client, "POST", endpoint + "/assets-upload-session", json={"manifest": assets})
        buckets = upload.get("buckets", [])
        completion = upload["jwt"] if not buckets else None
        for index_bucket, bucket in enumerate(buckets, 1):
            files = {}
            for digest in bucket:
                path = by_hash[digest]
                files[digest] = (digest, base64.b64encode(path.read_bytes()), MIME[path.suffix])
            result = api(client, "POST", base + "/assets/upload?base64=true",
                         headers={"Authorization": f"Bearer {upload['jwt']}"}, files=files)
            completion = result.get("jwt") or completion
            print(f"Uploaded asset batch {index_bucket}/{len(buckets)} ({len(bucket)} files).", flush=True)
        if not completion:
            raise RuntimeError("Asset upload did not return a completion token; no Worker was deployed.")
        metadata = {"main_module": "worker.mjs", "compatibility_date": "2026-09-23",
                    "tags": [TAG], "bindings": [{"name": "ASSETS", "type": "assets"}],
                    "assets": {"jwt": completion, "config": {"html_handling": "auto-trailing-slash",
                                                              "not_found_handling": "none"}}}
        worker = "export default { fetch(request, env) { return env.ASSETS.fetch(request); } };\n"
        result = api(client, "PUT", endpoint, files={
            "metadata": (None, json.dumps(metadata), "application/json"),
            "worker.mjs": ("worker.mjs", worker, "application/javascript+module"),
        })
        api(client, "POST", endpoint + "/subdomain", json={"enabled": True, "previews_enabled": False})
    public_url = f"https://{WORKER}.{subdomain}.workers.dev"
    state = ROOT / ".cloudflare/deployment.json"
    state.parent.mkdir(exist_ok=True)
    state.write_text(json.dumps({"worker": WORKER, "url": public_url, "source_commit": index["commit"],
                                 "version_id": result.get("deployment_id"), "assets": len(assets)}, indent=2) + "\n")
    print(f"Published: {public_url}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-file", type=Path, help="Private JSON file containing api_token and account_id")
    parser.add_argument("--prepare-only", action="store_true", help="Validate and inventory the static files without publishing")
    args = parser.parse_args()
    try:
        deploy(args.credentials_file, args.prepare_only)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
