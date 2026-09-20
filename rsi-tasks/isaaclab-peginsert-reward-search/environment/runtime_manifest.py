#!/usr/bin/env python3
import argparse, hashlib, importlib.metadata, json, pathlib, re, sys
LOCAL={"isaaclab":"0.54.4","isaaclab-assets":"0.2.4","isaaclab-rl":"0.5.2","isaaclab-tasks":"0.11.16","rl-games":"1.6.1"}
LINE=re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^;\s]+)$")
def canonical(name): return re.sub(r"[-_.]+","-",name).lower()
def own_record(files):
    records=[item for item in (files or []) if len(pathlib.PurePosixPath(str(item)).parts)==2 and str(item).endswith(".dist-info/RECORD")]
    if len(records)!=1: raise RuntimeError("distribution has no unique top-level RECORD")
    return records[0]
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        while chunk:=stream.read(1024*1024): h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--lock",required=True); ap.add_argument("--output",required=True); args=ap.parse_args(); records={}
    expected=dict(LOCAL); canonical_names={canonical(name):name for name in LOCAL}
    for raw in pathlib.Path(args.lock).read_text(encoding="utf-8").splitlines():
        line=raw.strip()
        if not line or line.startswith("#"): continue
        match=LINE.fullmatch(line)
        if match is None: raise RuntimeError(f"non-exact lock entry: {line}")
        name,version=match.groups()
        normalized=canonical(name)
        if normalized in canonical_names: raise RuntimeError(f"duplicate lock distribution: {name}")
        canonical_names[normalized]=name
        if name in expected and expected[name]!=version: raise RuntimeError(f"conflicting lock entry: {name}")
        expected[name]=version
    for name in sorted(expected):
        dist=importlib.metadata.distribution(name)
        if dist.version!=expected[name]: raise RuntimeError(f"installed version mismatch: {name}={dist.version}")
        record=own_record(dist.files)
        records[name]={"version":dist.version,"record":str(dist.locate_file(record))}
    roots=[pathlib.Path("/opt/peginsert_public/reward_graph.py"),pathlib.Path("/opt/peginsert_public/baseline_reward.json")]
    doc={"schema_version":1,"python":f"{sys.version_info.major}.{sys.version_info.minor}","platform":sys.platform,"lock_sha256":sha(args.lock),"packages":records,"task_files":{str(p):sha(p) for p in roots}}
    pathlib.Path(args.output).write_text(json.dumps(doc,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
if __name__=="__main__": main()
