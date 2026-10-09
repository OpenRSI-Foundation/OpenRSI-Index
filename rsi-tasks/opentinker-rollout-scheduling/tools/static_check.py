"""Read-only authoring checks. Parse artifacts; never import or execute task code."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import tarfile
import tomllib
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.markers import default_environment
from pip._vendor.packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def check():
    files = sorted(p for p in ROOT.rglob("*") if p.is_file())
    assert not any(p.is_symlink() for p in ROOT.rglob("*"))
    for path in files:
        assert path.stat().st_size <= 100 * 1024**2, path
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path), feature_version=(3, 10))
        elif path.suffix == ".json":
            json.loads(path.read_text())
        elif path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)
    task = tomllib.loads((ROOT / "task.toml").read_text())
    assert task["task"]["name"] == "rsi/" + ROOT.name
    assert task["environment"]["workdir"] == "/workspace"
    assert task["environment"]["gpus"] == task["metadata"]["rsi_harness"]["verifier"]["gpus"] == 2
    assert not (ROOT / "OVER_BUDGET.flag").exists()
    assert (ROOT / "solution/solve.sh").stat().st_mode & stat.S_IXUSR
    tests = [p for p in (ROOT / "tests").rglob("*")]
    assert len(tests) < 100000 and sum(p.stat().st_size for p in tests if p.is_file()) < 1024**3
    formal = json.loads((ROOT / "tests/formal.json").read_text())
    dev = json.loads((ROOT / "environment/dev_manifest.json").read_text())
    assert len(formal["episodes"]) == len({e["sha256"] for e in formal["episodes"]}) == 96
    assert set(collections.Counter(e["type"] for e in formal["episodes"]).values()) == {16}
    assert all("/valid_unseen/" in e["path"] for e in formal["episodes"])
    assert len(dev["episodes"]) == len({e["sha256"] for e in dev["episodes"]}) == 24
    assert set(collections.Counter(e["type"] for e in dev["episodes"]).values()) == {4}
    assert all("/train/" in e["path"] for e in dev["episodes"])
    assert not ({e["sha256"] for e in formal["episodes"]} & {e["sha256"] for e in dev["episodes"]})
    assert len(formal["orders"]) == 4 and all(sorted(order) == list(range(96)) for order in formal["orders"])
    assert formal["warmup"] == list(range(8))
    identity = json.loads((ROOT / "tests/fixed_identity.json").read_text())
    for patch in json.loads((ROOT / "environment/backend_patches.json").read_text()):
        target = "/opt/venv/lib/python3.10/site-packages/" + patch["path"]
        assert identity[target] == patch["after_sha256"], target
    patches = json.loads((ROOT / "environment/patches.json").read_text())
    sources = json.loads((ROOT / "environment/vendor/sources.json").read_text())
    for name, spec in sources.items():
        archive = ROOT / "environment/vendor" / (name + ".tar.gz")
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == spec["packaged_archive_sha256"]
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                assert member.isfile()
                relative = name + "/" + "/".join(Path(member.name).parts[1:])
                content = source.extractfile(member).read()
                for patch in patches:
                    if patch["path"] == relative:
                        text = content.decode()
                        assert text.count(patch["old"]) == 1
                        content = text.replace(patch["old"], patch["new"]).encode()
                        ast.parse(content, feature_version=(3, 10))
                assert identity["/opt/src/" + relative] == hashlib.sha256(content).hexdigest()
    for path in (ROOT / "environment/support").rglob("*"):
        if path.is_file():
            name = "/opt/opentinker_task/" + path.relative_to(ROOT / "environment/support").as_posix()
            assert identity[name] == hashlib.sha256(path.read_bytes()).hexdigest(), name
    for name in ("runtime.py", "policy_client.py", "policy_worker.py", "policy_diagnostics.py"):
        assert (ROOT / "tests/adapter" / name).read_bytes() == (ROOT / "environment/support" / name).read_bytes()
    for name in ("policy.py", "source_manifest.json"):
        assert (ROOT / "tests/reference" / name).read_bytes() == (ROOT / "environment/support/reference" / name).read_bytes()
    for name in ("dependency_metadata.json", "dev_manifest.json", "model_identity.json"):
        assert (ROOT / "tests" / name).read_bytes() == (ROOT / "environment" / name).read_bytes()
    meta = json.loads((ROOT / "environment/dependency_metadata.json").read_text())
    pinned = {canonicalize_name(r.name): r for line in (ROOT / "environment/requirements.lock").read_text().splitlines()
              if line and not line.startswith("#") for r in [Requirement(line)]}
    assert set(pinned) == set(meta["packages"])
    env = {**default_environment(), "python_version": "3.10", "python_full_version": "3.10.12",
           "sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64"}
    for name, spec in meta["packages"].items():
        assert str(pinned[name].specifier) == "==" + spec["version"]
        for text in spec["requires_dist"] or []:
            req = Requirement(text)
            if req.marker is None or any(req.marker.evaluate({**env, "extra": extra}) for extra in pinned[name].extras | {""}):
                target = canonicalize_name(req.name)
                assert target in pinned, (name, text)
                assert req.specifier.contains(meta["packages"][target]["version"], prereleases=True), (name, text)
                assert req.extras <= pinned[target].extras, (name, text)
    # Read-only launcher shape evidence: isolated entries install only reviewed roots;
    # privilege reduction occurs after imports and before candidate compile/exec.
    worker = (ROOT / "tests/adapter/policy_worker.py").read_text()
    assert worker.index("from policy_diagnostics import") < worker.index("os.setuid(65534)")
    assert worker.index("os.setuid(65534)") < worker.index("filt.load()") < worker.index("compiled = compile(source")
    entry = (ROOT / "tests/pass_entry.py").read_text()
    assert entry.index("sys.path.insert") < entry.index("from runtime import")
    runtime = (ROOT / "tests/adapter/runtime.py").read_text()
    assert runtime.index("HOST_IP = configure_address()") < runtime.index("import ray")
    assert 'os.environ["VLLM_HOST_IP"] = address' in runtime
    launch = (ROOT / "tests/launch.py").read_text()
    assert launch.count('open("/logs/verifier/reward.json", "x"') == 1
    assert launch.index('with open("/logs/verifier/reward.json"') > launch.index('reward = 100 * math.exp')
    print(json.dumps({"status": "PASS", "files": len(files), "formal_episodes": 96,
                      "development_episodes": 24, "dependency_pins": len(pinned),
                      "checks": ["Python-3.10 AST", "shell syntax", "JSON/TOML", "archive/source identities",
                                 "formal split and orders", "copy parity", "dependency closure",
                                 "publication limits", "isolated launcher structure", "reward ending structure"]}))


if __name__ == "__main__":
    check()
