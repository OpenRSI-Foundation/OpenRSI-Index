"""Authoring utility: pin registry metadata without importing or building packages."""
import concurrent.futures
import argparse
import json
from pathlib import Path
import urllib.request
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import Version, InvalidVersion
from pip._vendor.packaging.specifiers import SpecifierSet, InvalidSpecifier
from pip._vendor.packaging.utils import canonicalize_name

ROOTS = ("vllm==0.12.0 torch==2.9.0 ray[default,cgraph]==2.48.0 transformers==4.57.3 "
         "tokenizers==0.22.1 tensordict==0.10.0 numpy==1.26.4 textworld[pddl]==1.6.2 "
         "torchdata==0.11.0 opencv-python-headless==4.11.0.86 "
         "peft==0.17.1 accelerate==1.10.1 hydra-core==1.3.2 omegaconf==2.3.0 datasets==3.6.0 "
         "pandas==2.2.3 pyarrow==20.0.0 pydantic==2.12.5 fastapi[standard]==0.124.4 "
         "uvicorn==0.35.0 aiohttp==3.12.15 pyseccomp==0.1.2 setuptools==78.1.1 wheel==0.45.1 "
         "pip==25.2 gym==0.25.2 gym-notices==0.1.0 opencv-python==4.11.0.86 opentelemetry-api==1.37.0 opentelemetry-sdk==1.37.0 "
         "opentelemetry-exporter-prometheus==0.58b0 spacy==3.7.5 thinc==8.2.5 "
         "codetiming dill pybind11 pylatexenc wandb tensorboard termcolor pyyaml scipy psutil "
         "packaging tqdm cloudpickle").split()
ENV = {"python_version": "3.10", "python_full_version": "3.10.12", "os_name": "posix",
       "sys_platform": "linux", "platform_machine": "x86_64", "platform_system": "Linux",
       "platform_release": "", "platform_version": "", "implementation_name": "cpython",
       "implementation_version": "3.10.12", "platform_python_implementation": "CPython", "extra": ""}
CACHE, VERSIONS = {}, {}
CUTOFF = "2025-12-31"


def get(url):
    with urllib.request.urlopen(url, timeout=60) as stream:
        return json.load(stream)


def info(name, version):
    key = (name, version)
    if key not in VERSIONS:
        VERSIONS[key] = get(f"https://pypi.org/pypi/{name}/{version}/json")
        if key == ("gym", "0.25.2"):
            # Official sdist setup.py was inspected as data; PyPI JSON omits this list.
            VERSIONS[key]["info"]["requires_dist"] = ["numpy>=1.18.0", "cloudpickle>=1.2.0",
                "gym_notices>=0.0.4", "importlib_metadata>=4.8.0; python_version<'3.10'",
                "dataclasses==0.8; python_version=='3.6'"]
    return VERSIONS[key]


def python_compatible(spec):
    try:
        return not spec or Version("3.10.12") in SpecifierSet(spec)
    except InvalidSpecifier:
        # Malformed historical registry metadata cannot qualify a candidate.
        return False


def choose(name, requirements):
    spec = SpecifierSet(",".join(str(r.specifier) for r in requirements))
    exact = [s.version for s in spec if s.operator == "==" and "*" not in s.version]
    if exact:
        candidates = exact
    else:
        if name not in CACHE:
            CACHE[name] = get(f"https://pypi.org/pypi/{name}/json")
        candidates = []
        for version, files in CACHE[name]["releases"].items():
            try:
                parsed = Version(version)
            except InvalidVersion:
                continue
            if not spec.contains(parsed, prereleases=True):
                continue
            if any(not f.get("yanked") and f["upload_time_iso_8601"] < CUTOFF
                   and python_compatible(f.get("requires_python"))
                   for f in files):
                candidates.append(version)
        candidates.sort(key=lambda v: (not Version(v).is_prerelease, Version(v)), reverse=True)
    for version in candidates:
        if not spec.contains(Version(version), prereleases=True):
            continue
        data = info(name, version)
        py = data["info"].get("requires_python")
        if python_compatible(py):
            return version
    raise RuntimeError("No version " + name + " " + str(spec))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path,
                        default=Path(__file__).resolve().parents[1] / "environment")
    args = parser.parse_args()
    selected = {}
    for iteration in range(35):
        requirements, extras, visited = {}, {}, set()
        queue = [Requirement(r) for r in ROOTS]
        while queue:
            requirement = queue.pop(0)
            name = canonicalize_name(requirement.name)
            requirements.setdefault(name, []).append(requirement)
            extras.setdefault(name, set()).update(requirement.extras)
            if name not in selected:
                continue
            key = (name, selected[name], tuple(sorted(extras[name])))
            if key in visited:
                continue
            visited.add(key)
            for text in info(name, selected[name])["info"].get("requires_dist") or []:
                child = Requirement(text)
                if child.marker is None or any(child.marker.evaluate({**ENV, "extra": extra}) for extra in extras[name] | {""}):
                    if canonicalize_name(child.name) == "numpy" and ">=2" in str(child.specifier):
                        print("NumPy-2 constraint from", name, selected[name], str(child), flush=True)
                    queue.append(child)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            chosen = dict(zip(requirements, pool.map(lambda item: choose(*item), requirements.items())))
        print("metadata round", iteration, len(chosen), flush=True)
        if chosen == selected:
            break
        selected = chosen
    else:
        raise RuntimeError("metadata resolution did not converge")
    root = args.output_directory
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "requirements.lock"
    metadata = root / "dependency_metadata.json"
    assert not lock.exists() and not metadata.exists()
    with lock.open("x") as stream:
        stream.write("# CPython 3.10 Linux x86_64 registry metadata closure; builds remain pending.\n")
        for name, version in sorted(selected.items()):
            suffix = "[" + ",".join(sorted(extras[name])) + "]" if extras[name] else ""
            stream.write(name + suffix + "==" + version + "\n")
    with metadata.open("x") as stream:
        json.dump({"python": "3.10.12", "platform": "linux-x86_64", "selection_cutoff": CUTOFF,
                   "roots": ROOTS, "packages": {n: {"version": v, "requires_python": info(n, v)["info"].get("requires_python"),
                                                     "requires_dist": info(n, v)["info"].get("requires_dist")}
                                                  for n, v in sorted(selected.items())}}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
