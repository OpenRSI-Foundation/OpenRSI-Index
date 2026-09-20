#!/usr/bin/env python3
"""Mirror, localize, and verify the complete dependency closure of five USD roots."""
import argparse
import hashlib
import json
import pathlib
import re
import urllib.parse
import urllib.request

from pxr import Sdf, UsdUtils

BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
ROOTS = [
    "Isaac/IsaacLab/Factory/franka_mimic.usd",
    "Isaac/IsaacLab/Factory/factory_peg_8mm.usd",
    "Isaac/IsaacLab/Factory/factory_hole_8mm.usd",
    "Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    "Isaac/Environments/Grid/default_environment.usd",
]
USD_SUFFIXES = {".usd", ".usda", ".usdc"}
# Bare MDL module names (for example OmniPBR.mdl) are Kit core materials resolved from its
# built-in MDL search path, never files beside the layer; they are neither mirrored nor rewritten.
BUILTIN_MDL = re.compile(r"[A-Za-z0-9_]+\.mdl")


def builtin_mdl(authored):
    return BUILTIN_MDL.fullmatch(authored or "") is not None


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_url(owner, authored):
    if not authored or "<UDIM>" in authored or "%(UDIM)d" in authored:
        raise RuntimeError(f"unsupported unresolved asset pattern: {authored!r}")
    parsed = urllib.parse.urlparse(authored)
    if parsed.query or parsed.fragment:
        raise RuntimeError(f"unsupported queried or fragmented asset path: {authored}")
    if parsed.scheme == "omniverse":
        relative_path = parsed.path.lstrip("/")
        if not relative_path:
            raise RuntimeError(f"empty omniverse asset path: {authored}")
        return urllib.parse.urljoin(BASE, relative_path)
    if parsed.scheme in {"http", "https"}:
        return authored
    if parsed.scheme:
        raise RuntimeError(f"unsupported asset scheme: {authored}")
    if authored.startswith("/"):
        return urllib.parse.urljoin(BASE, authored.lstrip("/"))
    return urllib.parse.urljoin(owner, authored)


def relative(url):
    base = urllib.parse.urlparse(BASE)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != base.scheme or parsed.netloc != base.netloc or not parsed.path.startswith(base.path):
        raise RuntimeError(f"dependency outside approved asset origin: {url}")
    rel = urllib.parse.unquote(parsed.path[len(base.path) :].lstrip("/"))
    parts = pathlib.PurePosixPath(rel).parts
    if not rel or ".." in parts or any(part in {"", "."} for part in parts):
        raise RuntimeError(f"unsafe asset path: {url}")
    return pathlib.PurePosixPath(*parts).as_posix()


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as source, open(destination, "xb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)


def localize_layer(layer_path, owner_url, output_root, queue):
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None:
        raise RuntimeError(f"cannot inspect USD layer: {layer_path}")

    def rewrite(authored):
        if builtin_mdl(authored):
            return authored
        resolved = canonical_url(owner_url, authored)
        dependency = (output_root / relative(resolved)).resolve()
        if not dependency.is_relative_to(output_root.resolve()):
            raise RuntimeError(f"localized dependency escaped output root: {authored}")
        queue.append(resolved)
        return str(dependency)

    UsdUtils.ModifyAssetPaths(layer, rewrite)
    if not layer.Save():
        raise RuntimeError(f"cannot save localized USD layer: {layer_path}")


def verify_closure(output_root):
    root = output_root.resolve()
    discovered = set()
    for rel in ROOTS:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(root / rel))
        unresolved = [item for item in unresolved if not builtin_mdl(pathlib.PurePosixPath(item).name) or "/" in item]
        if unresolved:
            raise RuntimeError(f"unresolved USD dependencies for {rel}: {sorted(set(unresolved))}")
        for layer in layers:
            path = pathlib.Path(layer.realPath or layer.identifier).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise RuntimeError(f"USD layer escaped offline closure: {path}")
            discovered.add(path)
        for asset in assets:
            path = pathlib.Path(asset).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise RuntimeError(f"asset escaped offline closure: {asset}")
            discovered.add(path)
    actual = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if discovered != actual:
        missing = sorted(str(path.relative_to(root)) for path in actual - discovered)
        raise RuntimeError(f"downloaded files are not in the verified USD closure: {missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    queue = [urllib.parse.urljoin(BASE, rel) for rel in ROOTS]
    seen = set()
    records = []
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        rel = relative(url)
        destination = output / rel
        download(url, destination)
        if destination.suffix.lower() in USD_SUFFIXES:
            localize_layer(destination, url, output, queue)
        records.append({"path": rel, "url": url})
    verify_closure(output)
    for record in records:
        path = output / record["path"]
        record.update({"size": path.stat().st_size, "sha256": file_sha(path)})
    manifest = {
        "schema_version": 2,
        "base_url": BASE,
        "closure_verified": True,
        "roots": {rel: file_sha(output / rel) for rel in ROOTS},
        "files": sorted(records, key=lambda item: item["path"]),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
