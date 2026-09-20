#!/usr/bin/env python3
"""Authenticate and restrict-load the official checkpoint, then export player tensors only."""
import argparse, hashlib, json, pathlib
import numpy as np
import torch
from safetensors.torch import save_file

EXPECTED_SIZE=212_889_115
EXPECTED_SHA256="1a1388408d119959a19d28efd1e56c3d498cd6eed1d259c48e9319c756081496"
EXPECTED_TENSORS=28
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        while chunk:=stream.read(1024*1024): h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("source"); ap.add_argument("output"); args=ap.parse_args(); source=pathlib.Path(args.source); out=pathlib.Path(args.output)
    if source.stat().st_size!=EXPECTED_SIZE or sha(source)!=EXPECTED_SHA256: raise RuntimeError("official checkpoint identity mismatch")
    safe=[np.core.multiarray.scalar,np.dtype,np.dtypes.Float32DType]
    with torch.serialization.safe_globals(safe): data=torch.load(source,map_location="cpu",weights_only=True)
    model=data.get("model")
    if not isinstance(model,dict) or len(model)!=EXPECTED_TENSORS: raise RuntimeError("unexpected player tensor inventory")
    tensors={}
    for name,value in sorted(model.items()):
        if not isinstance(name,str) or not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.is_sparse or not torch.isfinite(value).all(): raise RuntimeError(f"invalid player tensor: {name}")
        tensors[name]=value.detach().cpu().contiguous()
    if sum(value.dtype==torch.float64 for value in tensors.values())!=6: raise RuntimeError("expected six float64 normalization tensors")
    out.mkdir(parents=True,exist_ok=False); policy=out/"policy.safetensors"; save_file(tensors,str(policy),metadata={"format":"pt","source_sha256":EXPECTED_SHA256})
    inventory={name:{"shape":list(value.shape),"dtype":str(value.dtype),"bytes":value.numel()*value.element_size()} for name,value in tensors.items()}
    provenance={"schema_version":1,"source":{"size":EXPECTED_SIZE,"sha256":EXPECTED_SHA256},"derived":{"file":"policy.safetensors","size":policy.stat().st_size,"sha256":sha(policy),"tensor_count":len(tensors),"inventory":inventory},"excluded":["optimizer","training critic","training return scalar","environment state"],"historical_training_protocol":"unknown"}
    (out/"provenance.json").write_text(json.dumps(provenance,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8"); source.unlink()
if __name__=="__main__": main()
