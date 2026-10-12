"""Authoring-only PyPI metadata resolver. Never imports/builds target distributions."""
import functools, hashlib, json, tarfile, tomllib, urllib.request
from dataclasses import dataclass
from pathlib import Path
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import Version, InvalidVersion
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.resolvelib import Resolver, BaseReporter
from pip._vendor.resolvelib.providers import AbstractProvider
ROOT = Path(__file__).resolve().parents[1]
ENV = dict(implementation_name="cpython", implementation_version="3.12.0", os_name="posix", platform_machine="x86_64", platform_release="", platform_system="Linux", platform_version="", python_full_version="3.12.0", python_version="3.12", sys_platform="linux", extra="")
# Explicit upstream/contract overrides, including qwen-tts's no-deps install.
OVERRIDES = {"torch":"2.13.0", "torchvision":"0.28.0", "torchaudio":"2.11.0", "transformers":"5.12.1", "torchcodec":"0.15.0", "sglang":"0.5.20", "flashinfer-python":"0.6.18", "protobuf":"6.33.5", "setuptools":"79.0.1"}
@functools.lru_cache(None)
def metadata(name, version=""):
    url = f"https://pypi.org/pypi/{name}/" + (version+"/" if version else "") + "json"
    with urllib.request.urlopen(url, timeout=90) as f: return json.load(f)
def active(r, extras=("",)):
    return not r.marker or any(r.marker.evaluate(dict(ENV,extra=e)) for e in extras)
def req(text):
    r=Requirement(text); name=canonicalize_name(r.name)
    if name in OVERRIDES:
        r=Requirement(name+("["+",".join(sorted(r.extras))+"]" if r.extras else "")+"=="+OVERRIDES[name])
    return r
@dataclass(frozen=True)
class Candidate:
    name: str
    version: str
    extras: frozenset
class Provider(AbstractProvider):
    def identify(self,requirement_or_candidate):return canonicalize_name(requirement_or_candidate.name)
    def get_preference(self,identifier,resolutions,candidates,information,backtrack_causes):
        return (not any("==" in str(i.requirement.specifier) for i in information[identifier]), identifier)
    def find_matches(self,identifier,requirements,incompatibilities):
        rs=list(requirements[identifier]); bad=set(incompatibilities[identifier]); extras=frozenset(e for r in rs for e in r.extras)
        def gen():
            versions=[]
            for v,files in metadata(identifier)["releases"].items():
                try: parsed=Version(v)
                except InvalidVersion: continue
                if files and not all(f.get("yanked") for f in files) and all(r.specifier.contains(parsed,prereleases=True) for r in rs):versions.append(parsed)
            versions.sort(key=lambda v:(not v.is_prerelease,v),reverse=True)
            for v in versions:
                c=Candidate(identifier,str(v),extras)
                if c in bad:continue
                m=metadata(identifier,str(v))
                if not SpecifierSet(m["info"].get("requires_python") or "").contains("3.12.0"):continue
                yield c
        return gen
    def is_satisfied_by(self,requirement,candidate):
        return requirement.specifier.contains(candidate.version,prereleases=True) and requirement.extras <= candidate.extras
    def get_dependencies(self,candidate):
        if candidate.name=="qwen-tts":return []
        return [req(s) for s in metadata(candidate.name,candidate.version)["info"].get("requires_dist") or [] if active(Requirement(s),candidate.extras|{""})]
with tarfile.open(ROOT/"environment/source.tar.gz") as t:
    doc=tomllib.loads(t.extractfile("sglang-omni-1523543056bf334bc2f4d69db23aad0a002cedf2/pyproject.toml").read().decode())
initial=doc["project"]["dependencies"]+doc["project"]["optional-dependencies"]["fun-cosyvoice3"]+["qwen-tts==0.1.1","psutil","wheel","Cython","ninja","packaging"]
rs=[req(s) for s in initial if active(Requirement(s))]
print("Resolving official metadata",flush=True)
result=Resolver(Provider(),BaseReporter()).resolve(rs,max_rounds=3000)
lock={}
for name,c in sorted(result.mapping.items()):
    m=metadata(name,c.version)
    lock[name]={"version":c.version,"requires_python":m["info"].get("requires_python"),"requires_dist":m["info"].get("requires_dist"),"files":[{"filename":f["filename"],"url":f["url"],"sha256":f["digests"]["sha256"],"packagetype":f["packagetype"]} for f in m["urls"] if not f.get("yanked")]}
# The PyPI cuda-tile sdist is a downloader stub. Resolve its official NVIDIA
# binary explicitly instead of executing that backend during generation/build.
wheel_url="https://pypi.nvidia.com/cuda-tile/cuda_tile-1.6.0rc5-cp312-cp312-manylinux2014_x86_64.whl"
wheel_sha="b74c20348210d2182cd998a0ecb60c518989a79b28592d72eb8294b38ddb93d7"
with urllib.request.urlopen(wheel_url,timeout=90) as f: assert hashlib.sha256(f.read()).hexdigest()==wheel_sha
assert lock["cuda-tile"]["version"]=="1.6.0rc5"
lock["cuda-tile"].update(metadata_source="https://pypi.nvidia.com/cuda-tile/",requires_dist=["typing-extensions"],files=[dict(filename=wheel_url.rsplit("/",1)[1],url=wheel_url,sha256=wheel_sha,packagetype="bdist_wheel")])
requirements="".join(n+"=="+v["version"]+" "+" ".join("--hash=sha256:"+f["sha256"] for f in v["files"])+"\n" for n,v in lock.items())
for name,data in (("environment/requirements.lock",requirements),("environment/python-artifacts.json",json.dumps({"overrides":OVERRIDES,"packages":lock},indent=2)+"\n")):
    p=ROOT/name
    assert not p.exists(),p
    with p.open("x") as f:f.write(data)
print("Locked",len(lock),"distributions",flush=True)
