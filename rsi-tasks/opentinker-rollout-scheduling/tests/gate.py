"""Judge-owned scope and fixed-asset identities. Does not execute the candidate."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import stat
from adapter.policy_diagnostics import exception_detail, repair_detail


class GateError(Exception):
    def __init__(self, code, field, condition, path="/workspace/candidate/policy.py", detail=None):
        self.detail = {"code": code, "path": path, "field": field, "condition": condition,
                       **repair_detail(detail)}
        super().__init__(code)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def candidate(root):
    workspace = Path("/workspace")
    required = {"candidate", "candidate/policy.py", "candidate/source_manifest.json"}
    allowed = required | {"notes.md"}
    actual = set()
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace).as_posix()
        actual.add(relative)
        mode = path.lstat().st_mode
        if relative not in allowed or (relative == "candidate" and not stat.S_ISDIR(mode)) or (
                relative != "candidate" and not stat.S_ISREG(mode)):
            raise GateError("workspace_scope", "workspace", "only the declared regular files and candidate directory may exist", "/workspace")
    if not required <= actual:
        missing = sorted(required - actual)[0]
        raise GateError("missing_artifact", "workspace", "required file or directory is missing", str(workspace / missing))
    for name, bound in (("candidate/policy.py", 1048576), ("notes.md", 4194304), ("candidate/source_manifest.json", 65536)):
        path = workspace / name
        if path.exists() and path.stat().st_size > bound:
            raise GateError("artifact_size", "bytes", f"file must be at most {bound} bytes", str(path))
    if (workspace / "candidate/source_manifest.json").read_bytes() != (root / "reference/source_manifest.json").read_bytes():
        raise GateError("fixed_manifest", "source_manifest", "restore the fixed source manifest", "/workspace/candidate/source_manifest.json")
    try:
        source = (workspace / "candidate/policy.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="/workspace/candidate/policy.py", feature_version=(3, 10))
    except (UnicodeError, SyntaxError, ValueError, RecursionError) as exc:
        raise GateError("policy_syntax", "source", "provide valid UTF-8 Python 3.10 source",
                        detail=exception_detail(exc, "compile")) from None
    if not any(isinstance(node, ast.ClassDef) and node.name == "Policy" for node in tree.body):
        raise GateError("policy_class", "Policy", "define a top-level Policy class with reset and schedule methods")
    return source


def fixed(root):
    expected = json.loads((root / "fixed_identity.json").read_text())
    for name, value in expected.items():
        path = Path(name)
        if not path.is_file() or path.is_symlink() or digest(path) != value:
            raise RuntimeError("fixed_asset_integrity")
    # Extra Python/config files in source roots could shadow checked imports.
    for source_root in ("/opt/src/opentinker", "/opt/src/verl", "/opt/src/alfworld", "/opt/opentinker_task"):
        for path in Path(source_root).rglob("*"):
            if path.is_symlink() or (path.is_file() and str(path) not in expected):
                raise RuntimeError("fixed_source_inventory")
    for name, spec in json.loads((root / "model_identity.json").read_text()).items():
        path = Path("/opt/models/qwen2.5-3b-instruct") / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != spec["size"] or digest(path) != spec["sha256"]:
            raise RuntimeError("model_integrity")
    model_names = {p.name for p in Path("/opt/models/qwen2.5-3b-instruct").iterdir()}
    if model_names != set(json.loads((root / "model_identity.json").read_text())):
        raise RuntimeError("model_inventory")
    for name, value in json.loads((root / "data_identity.json").read_text()).items():
        path = Path("/opt/data/alfworld") / name
        if path.is_symlink() or not path.is_file() or digest(path) != value:
            raise RuntimeError("data_integrity")
    for name, spec in json.loads((root / "dependency_metadata.json").read_text())["packages"].items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError("dependency_missing") from None
        if version != spec["version"]:
            raise RuntimeError("dependency_version")
