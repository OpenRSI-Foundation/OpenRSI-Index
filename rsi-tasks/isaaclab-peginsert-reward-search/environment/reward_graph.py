#!/usr/bin/env python3
"""Public reward-graph validator/interpreter. Judge uses its own implementation."""

from __future__ import annotations
import argparse, json, math, os, re, sys

MAX_BYTES, MAX_NODES, MAX_DEPTH, MAX_MEMORY = 65536, 1024, 32, 32
INPUT_TYPES = {
    "peg_position":"vec3","peg_quaternion":"quat","peg_linear_velocity":"vec3","peg_angular_velocity":"vec3",
    "hole_position":"vec3","hole_quaternion":"quat","hole_linear_velocity":"vec3","hole_angular_velocity":"vec3","fingertip_position":"vec3","fingertip_quaternion":"quat",
    "fingertip_linear_velocity":"vec3","fingertip_angular_velocity":"vec3","joint_position":"vec7","joint_velocity":"vec7",
    "action":"vec6","previous_action":"vec6","peg_diameter":"scalar","peg_height":"scalar","hole_diameter":"scalar",
    "hole_height":"scalar","episode_progress":"scalar","keypoint_distance":"scalar","engaged":"scalar","success":"scalar",
}
VECTORS={"vec3","vec6","vec7","quat"}; DIMS={"vec3":3,"vec6":6,"vec7":7,"quat":4}; COMPONENTS={f"component_{i}" for i in range(7)}; IDS=re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
UNARY={"neg","abs","square","sqrt","tanh","sigmoid","exp_clamped","log1p_abs"}
REDUCE={"norm","sum_components","mean_components"}; COMPARE={"lt","le","gt","ge","eq"}

class Invalid(ValueError): pass
def need(ok, message):
    if not ok: raise Invalid(message)
def pairs(items):
    out={}
    for key,value in items:
        need(key not in out, f"duplicate key: {key}"); out[key]=value
    return out
def finite(value):
    need(not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(float(value)) and abs(float(value))<=1e6,"constant must be finite and bounded")
    return float(value)
def closed(obj, allowed, required):
    need(isinstance(obj,dict),"object required"); need(not(set(obj)-allowed),f"unknown keys: {sorted(set(obj)-allowed)}"); need(required<=set(obj),f"missing keys: {sorted(required-set(obj))}")

def load_recipe(path):
    flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0); fd=os.open(path,flags)
    try:
        st=os.fstat(fd); need((st.st_mode&0o170000)==0o100000,"recipe must be a regular file"); need(st.st_size<=MAX_BYTES,"recipe exceeds 65536 bytes")
        raw=os.read(fd,MAX_BYTES+1); need(len(raw)==st.st_size,"recipe changed while read")
    finally: os.close(fd)
    text=raw.decode("utf-8"); depth=0; quote=False; escape=False
    for char in text:
        if quote:
            if escape: escape=False
            elif char=="\\": escape=True
            elif char=='"': quote=False
        elif char=='"': quote=True
        elif char in "[{": depth+=1; need(depth<=MAX_DEPTH+8,"JSON nesting is too deep")
        elif char in "]}": depth-=1; need(depth>=0,"unbalanced JSON")
    doc=json.loads(text,object_pairs_hook=pairs,parse_constant=lambda token: (_ for _ in ()).throw(Invalid(f"nonfinite number: {token}")))
    return validate(doc)

def infer(op, types, node):
    if op=="const": need(types==[] and "value" in node,"const signature"); return "scalar"
    if op in UNARY: need(types==["scalar"],f"{op} requires scalar"); return "scalar"
    if op in REDUCE: need(len(types)==1 and types[0] in VECTORS,f"{op} requires vector"); return "scalar"
    if op in {"x","y","z"}: need(len(types)==1 and types[0] in VECTORS,f"{op} requires vector"); return "scalar"
    if op in COMPONENTS:
        index=int(op.rsplit("_",1)[1]); need(len(types)==1 and types[0] in VECTORS and index<DIMS[types[0]],f"{op} exceeds vector dimension"); return "scalar"
    if op=="w": need(types==["quat"],"w requires quaternion"); return "scalar"
    if op in {"add","mul","min","max"}: need(len(types)>=2 and len(set(types))==1 and types[0]!="bool",f"{op} requires equal numeric operands"); return types[0]
    if op in {"sub","div"}: need(len(types)==2 and types[0]==types[1] and types[0]!="bool",f"{op} requires equal numeric operands"); return types[0]
    if op=="scale": need(len(types)==2 and types[0]=="scalar" and types[1] in VECTORS,"scale requires scalar, vector"); return types[1]
    if op=="dot": need(len(types)==2 and types[0]==types[1] and types[0] in VECTORS,"dot requires equal vectors"); return "scalar"
    if op=="clamp": need(len(types)==1 and types[0]!="bool" and {"min","max"}<=set(node),"clamp signature"); return types[0]
    if op=="keypoint_squash": need(types==["scalar"] and {"a","b"}<=set(node),"keypoint_squash signature"); return "scalar"
    if op in COMPARE: need(types==["scalar","scalar"],f"{op} requires scalars"); return "bool"
    if op in {"and","or"}: need(types==["bool","bool"],f"{op} requires booleans"); return "bool"
    if op=="not": need(types==["bool"],"not requires boolean"); return "bool"
    if op=="where": need(len(types)==3 and types[0]=="bool" and types[1]==types[2] and types[1]!="bool","where signature"); return types[1]
    if op=="quat_conjugate": need(types==["quat"],"quat_conjugate signature"); return "quat"
    if op=="quat_multiply": need(types==["quat","quat"],"quat_multiply signature"); return "quat"
    if op=="quat_rotate": need(types==["quat","vec3"],"quat_rotate signature"); return "vec3"
    raise Invalid(f"unsupported operation: {op}")

def validate(doc):
    closed(doc,{"schema_version","selected_epoch","memory_size","nodes","output","memory_updates"},{"schema_version","selected_epoch","memory_size","nodes","output","memory_updates"})
    need(type(doc["schema_version"]) is int and doc["schema_version"]==1,"schema_version must equal integer 1"); need(type(doc["selected_epoch"]) is int and 1<=doc["selected_epoch"]<=50,"selected_epoch must be 1..50")
    size=doc["memory_size"]; need(type(size) is int and 0<=size<=MAX_MEMORY,"memory_size must be 0..32")
    nodes=doc["nodes"]; need(isinstance(nodes,list) and 1<=len(nodes)<=MAX_NODES,"nodes must contain 1..1024 entries")
    types=dict(INPUT_TYPES); depths={k:0 for k in types}
    for slot in range(size): types[f"memory_{slot}"]="scalar"; depths[f"memory_{slot}"]=0
    for node in nodes:
        need(isinstance(node,dict),"node must be an object"); op=node.get("op"); allowed={"id","op","args"}|({"value"} if op=="const" else set())|({"min","max"} if op=="clamp" else set())|({"a","b"} if op=="keypoint_squash" else set())
        closed(node,allowed,{"id","op","args"}); need(isinstance(op,str),"operation must be a string"); ident=node["id"]; need(isinstance(ident,str) and IDS.fullmatch(ident) and ident not in types,"node id must be unique")
        args=node["args"]; need(isinstance(args,list) and all(isinstance(x,str) and x in types for x in args),"args must reference public inputs or prior nodes")
        typ=infer(op,[types[x] for x in args],node); depth=1+max([depths[x] for x in args] or [0]); need(depth<=MAX_DEPTH,"graph depth exceeds 32")
        if op=="const": node["value"]=finite(node["value"])
        if op=="clamp": node["min"],node["max"]=finite(node["min"]),finite(node["max"]); need(node["min"]<=node["max"],"clamp bounds out of order")
        if op=="keypoint_squash": node["a"],node["b"]=finite(node["a"]),finite(node["b"]); need(node["a"]>=0 and node["b"]>=0,"squash parameters must be nonnegative")
        types[ident]=typ; depths[ident]=depth
    need(isinstance(doc["output"],str) and types.get(doc["output"])=="scalar","output must reference a scalar")
    updates=doc["memory_updates"]; need(isinstance(updates,list) and len(updates)<=size,"too many memory updates"); seen=set()
    for update in updates:
        closed(update,{"index","value"},{"index","value"}); slot=update["index"]; need(type(slot) is int and 0<=slot<size and slot not in seen,"invalid or duplicate memory index"); need(isinstance(update["value"],str) and types.get(update["value"])=="scalar","memory value must be scalar"); seen.add(slot)
    return doc

class Runtime:
    def __init__(self,recipe,torch,num_envs,device): self.recipe=recipe; self.torch=torch; self.memory=torch.zeros((num_envs,recipe["memory_size"]),device=device)
    def reset(self,ids):
        if self.memory.shape[1]: self.memory[ids]=0
    def evaluate(self,inputs):
        t=self.torch; values={k:v.detach().clone() for k,v in inputs.items()}; exemplar=values["episode_progress"]
        for i in range(self.memory.shape[1]): values[f"memory_{i}"]=self.memory[:,i].clone()
        for node in self.recipe["nodes"]:
            a=[values[x] for x in node["args"]]; op=node["op"]
            if op=="const": v=t.full_like(exemplar,node["value"])
            elif op=="neg": v=-a[0]
            elif op=="abs": v=t.abs(a[0])
            elif op=="square": v=t.square(a[0])
            elif op=="sqrt": v=t.sqrt(t.clamp(a[0],min=0))
            elif op=="tanh": v=t.tanh(a[0])
            elif op=="sigmoid": v=t.sigmoid(a[0])
            elif op=="exp_clamped": v=t.exp(t.clamp(a[0],-40,40))
            elif op=="log1p_abs": v=t.log1p(t.abs(a[0]))
            elif op=="norm": v=t.linalg.vector_norm(a[0],dim=-1)
            elif op=="sum_components": v=t.sum(a[0],dim=-1)
            elif op=="mean_components": v=t.mean(a[0],dim=-1)
            elif op in {"x","y","z","w"}:
                index={"w":0,"x":1 if a[0].shape[-1]==4 else 0,"y":2 if a[0].shape[-1]==4 else 1,"z":3 if a[0].shape[-1]==4 else 2}[op]; v=a[0][...,index]
            elif op in COMPONENTS: v=a[0][...,int(op.rsplit("_",1)[1])]
            elif op=="add": v=sum(a[1:],a[0])
            elif op=="mul":
                v=a[0]
                for x in a[1:]: v=v*x
            elif op=="min": v=t.stack(a).amin(0)
            elif op=="max": v=t.stack(a).amax(0)
            elif op=="sub": v=a[0]-a[1]
            elif op=="div": v=a[0]/t.where(t.abs(a[1])<1e-6,t.full_like(a[1],1e-6),a[1])
            elif op=="scale": v=a[0].unsqueeze(-1)*a[1]
            elif op=="dot": v=t.sum(a[0]*a[1],dim=-1)
            elif op=="clamp": v=t.clamp(a[0],node["min"],node["max"])
            elif op=="keypoint_squash": x=t.clamp(node["a"]*a[0],-40,40); v=1/(t.exp(-x)+node["b"]+t.exp(x))
            elif op in COMPARE: v={"lt":t.lt,"le":t.le,"gt":t.gt,"ge":t.ge,"eq":t.eq}[op](a[0],a[1])
            elif op=="and": v=t.logical_and(a[0],a[1])
            elif op=="or": v=t.logical_or(a[0],a[1])
            elif op=="not": v=t.logical_not(a[0])
            elif op=="where": v=t.where(a[0],a[1],a[2])
            elif op=="quat_conjugate": v=t.cat((a[0][...,:1],-a[0][...,1:]),-1)
            elif op=="quat_multiply": v=qmul(t,a[0],a[1])
            elif op=="quat_rotate": v=qmul(t,qmul(t,a[0],t.cat((t.zeros_like(a[1][...,:1]),a[1]),-1)),t.cat((a[0][...,:1],-a[0][...,1:]),-1))[...,1:]
            need(bool(t.isfinite(v).all()),f"nonfinite value at {node['id']}"); values[node["id"]]=v
        nxt=self.memory.clone()
        for update in self.recipe["memory_updates"]: nxt[:,update["index"]]=values[update["value"]]
        need(bool(t.isfinite(nxt).all()),"nonfinite memory"); self.memory=nxt; reward=values[self.recipe["output"]]; need(reward.ndim==1 and bool(t.isfinite(reward).all()),"nonfinite reward"); return reward
def qmul(t,a,b):
    aw,ax,ay,az=a.unbind(-1); bw,bx,by,bz=b.unbind(-1); return t.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--json",required=True); args=parser.parse_args()
    try: recipe=load_recipe(args.json)
    except Exception as exc: print(json.dumps({"status":"candidate_invalid","path":args.json,"error":str(exc)})); return 2
    print(json.dumps({"status":"valid","selected_epoch":recipe["selected_epoch"],"nodes":len(recipe["nodes"]),"memory_size":recipe["memory_size"]})); return 0
if __name__=="__main__": sys.exit(main())
