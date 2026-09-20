"""Judge-owned parser, type checker, and tensor interpreter for reward graphs."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

MAX_BYTES = 65_536
MAX_NODES = 1_024
MAX_DEPTH = 32
MAX_MEMORY = 32

INPUT_TYPES = {
    "peg_position": "vec3",
    "peg_quaternion": "quat",
    "peg_linear_velocity": "vec3",
    "peg_angular_velocity": "vec3",
    "hole_position": "vec3",
    "hole_quaternion": "quat",
    "hole_linear_velocity": "vec3",
    "hole_angular_velocity": "vec3",
    "fingertip_position": "vec3",
    "fingertip_quaternion": "quat",
    "fingertip_linear_velocity": "vec3",
    "fingertip_angular_velocity": "vec3",
    "joint_position": "vec7",
    "joint_velocity": "vec7",
    "action": "vec6",
    "previous_action": "vec6",
    "peg_diameter": "scalar",
    "peg_height": "scalar",
    "hole_diameter": "scalar",
    "hole_height": "scalar",
    "episode_progress": "scalar",
    "keypoint_distance": "scalar",
    "engaged": "scalar",
    "success": "scalar",
}

IDS = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
UNARY_SCALAR = {"neg", "abs", "square", "sqrt", "tanh", "sigmoid", "exp_clamped", "log1p_abs"}
REDUCTIONS = {"norm", "sum_components", "mean_components"}
COMPARE = {"lt", "le", "gt", "ge", "eq"}
BOOL_OPS = {"and", "or"}
VECTOR_TYPES = {"vec3", "vec6", "vec7", "quat"}
VECTOR_DIMS = {"vec3": 3, "vec6": 6, "vec7": 7, "quat": 4}
COMPONENT_OPS = {f"component_{index}" for index in range(7)}


@dataclass(frozen=True)
class CandidateInvalid(Exception):
    code: str
    field: str
    condition: str
    expected: Any
    actual: Any
    hint: str

    def envelope(self) -> dict[str, Any]:
        return {
            "status": "candidate_invalid",
            "code": self.code,
            "path": "/workspace/reward.json",
            "field": self.field,
            "condition": self.condition,
            "expected": self.expected,
            "actual": self.actual,
            "hint": self.hint,
        }


def invalid(code: str, field: str, condition: str, expected: Any, actual: Any, hint: str) -> None:
    raise CandidateInvalid(code, field, condition, expected, actual, hint)


def _bounded_json_bytes(path: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        st = os.fstat(fd)
        if not os.path.isfile(path) or not (st.st_mode & 0o170000) == 0o100000:
            invalid("candidate_type", "$", "regular_file", "regular file", "other", "Replace the path with one closed regular file.")
        if st.st_size > MAX_BYTES:
            invalid("candidate_size", "$", "maximum_bytes", MAX_BYTES, st.st_size, "Reduce the recipe size.")
        data = os.read(fd, MAX_BYTES + 1)
        if len(data) != st.st_size:
            invalid("candidate_changed", "$", "stable_read", st.st_size, len(data), "Close and flush the file before submission.")
        return data
    finally:
        os.close(fd)


def _precheck_nesting(text: str) -> None:
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_DEPTH + 8:
                invalid("json_depth", "$", "maximum_container_depth", MAX_DEPTH + 8, depth, "Flatten the JSON document.")
        elif char in "]}":
            depth -= 1
            if depth < 0:
                invalid("json_syntax", "$", "balanced_containers", True, False, "Repair the JSON syntax.")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            invalid("duplicate_key", "$", "unique_object_keys", True, key, "Remove the duplicate key.")
        out[key] = value
    return out


def load_recipe(path: str) -> dict[str, Any]:
    raw = _bounded_json_bytes(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        invalid("encoding", "$", "utf8", True, str(exc), "Encode the recipe as UTF-8.")
    _precheck_nesting(text)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object,
            parse_constant=lambda token: invalid("number", "$", "finite_json_number", True, token, "Use finite JSON numbers."),
        )
    except CandidateInvalid:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        invalid("json_syntax", "$", "valid_json", True, str(exc), "Repair the JSON syntax.")
    return validate_recipe(value)


def _keys(value: dict[str, Any], allowed: set[str], required: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        invalid("unknown_key", field, "closed_object", sorted(allowed), unknown, "Remove unknown keys.")
    if missing:
        invalid("missing_key", field, "required_keys", sorted(required), missing, "Add all required keys.")


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        invalid("number", field, "finite_number", True, value, "Use a finite JSON number.")
    if abs(float(value)) > 1_000_000.0:
        invalid("number_range", field, "absolute_maximum", 1_000_000.0, value, "Use a bounded constant.")
    return float(value)


def validate_recipe(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict):
        invalid("root_type", "$", "object", "object", type(doc).__name__, "Use one JSON object.")
    _keys(doc, {"schema_version", "selected_epoch", "memory_size", "nodes", "output", "memory_updates"}, {"schema_version", "selected_epoch", "memory_size", "nodes", "output", "memory_updates"}, "$")
    if type(doc["schema_version"]) is not int or doc["schema_version"] != 1:
        invalid("schema_version", "$.schema_version", "equals", 1, doc["schema_version"], "Use schema version 1.")
    if isinstance(doc["selected_epoch"], bool) or not isinstance(doc["selected_epoch"], int) or not 1 <= doc["selected_epoch"] <= 50:
        invalid("selected_epoch", "$.selected_epoch", "integer_range", "1..50", doc["selected_epoch"], "Precommit a completed PPO epoch.")
    memory_size = doc["memory_size"]
    if isinstance(memory_size, bool) or not isinstance(memory_size, int) or not 0 <= memory_size <= MAX_MEMORY:
        invalid("memory_size", "$.memory_size", "integer_range", f"0..{MAX_MEMORY}", memory_size, "Use at most 32 scalar registers.")
    nodes = doc["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= MAX_NODES:
        invalid("node_count", "$.nodes", "array_length", f"1..{MAX_NODES}", len(nodes) if isinstance(nodes, list) else type(nodes).__name__, "Provide a bounded nonempty graph.")

    types = dict(INPUT_TYPES)
    depths = {name: 0 for name in INPUT_TYPES}
    for index in range(memory_size):
        types[f"memory_{index}"] = "scalar"
        depths[f"memory_{index}"] = 0

    normalized = []
    for index, node in enumerate(nodes):
        field = f"$.nodes[{index}]"
        if not isinstance(node, dict):
            invalid("node_type", field, "object", "object", type(node).__name__, "Use an operation object.")
        op = node.get("op")
        if not isinstance(op, str):
            invalid("operation", field + ".op", "string_operation", True, type(op).__name__, "Use a named supported operation.")
        allowed = {"id", "op", "args"}
        if op == "const":
            allowed.add("value")
        if op == "clamp":
            allowed.update({"min", "max"})
        if op == "keypoint_squash":
            allowed.update({"a", "b"})
        _keys(node, allowed, {"id", "op", "args"}, field)
        node_id = node["id"]
        if not isinstance(node_id, str) or not IDS.fullmatch(node_id) or node_id in types:
            invalid("node_id", field + ".id", "new_identifier", "unique ASCII identifier", node_id, "Choose a unique letter-led identifier.")
        args = node["args"]
        if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
            invalid("args", field + ".args", "reference_array", "array of strings", type(args).__name__, "Reference only public inputs or prior nodes.")
        missing = [arg for arg in args if arg not in types]
        if missing:
            invalid("reference", field + ".args", "prior_reference", True, missing, "Topologically order the graph and use declared inputs.")
        arg_types = [types[arg] for arg in args]
        out_type = _infer_type(op, arg_types, node, field)
        depth = 1 + max((depths[arg] for arg in args), default=0)
        if depth > MAX_DEPTH:
            invalid("graph_depth", field, "maximum_depth", MAX_DEPTH, depth, "Flatten the reward graph.")
        copy = dict(node)
        if op == "const":
            copy["value"] = _finite(node["value"], field + ".value")
        if op == "clamp":
            copy["min"] = _finite(node["min"], field + ".min")
            copy["max"] = _finite(node["max"], field + ".max")
            if copy["min"] > copy["max"]:
                invalid("clamp", field, "ordered_bounds", "min <= max", [copy["min"], copy["max"]], "Order the clamp bounds.")
        if op == "keypoint_squash":
            copy["a"] = _finite(node["a"], field + ".a")
            copy["b"] = _finite(node["b"], field + ".b")
            if copy["a"] < 0.0 or copy["b"] < 0.0:
                invalid("squash_parameters", field, "nonnegative", True, [copy["a"], copy["b"]], "Use nonnegative source-squash parameters.")
        normalized.append(copy)
        types[node_id] = out_type
        depths[node_id] = depth

    output = doc["output"]
    if not isinstance(output, str) or output not in types or types.get(output) != "scalar":
        invalid("output", "$.output", "scalar_reference", True, output, "Reference a scalar input or node.")
    updates = doc["memory_updates"]
    if not isinstance(updates, list) or len(updates) > memory_size:
        invalid("memory_updates", "$.memory_updates", "bounded_array", f"0..{memory_size}", len(updates) if isinstance(updates, list) else type(updates).__name__, "Update each register at most once.")
    seen = set()
    for index, update in enumerate(updates):
        field = f"$.memory_updates[{index}]"
        if not isinstance(update, dict):
            invalid("memory_update", field, "object", "object", type(update).__name__, "Use index/value objects.")
        _keys(update, {"index", "value"}, {"index", "value"}, field)
        slot, value = update["index"], update["value"]
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < memory_size or slot in seen:
            invalid("memory_index", field + ".index", "unique_integer_range", f"0..{memory_size - 1}", slot, "Update an existing register once.")
        if not isinstance(value, str) or types.get(value) != "scalar":
            invalid("memory_value", field + ".value", "scalar_reference", True, value, "Reference a scalar value.")
        seen.add(slot)
    return {**doc, "nodes": normalized}


def _infer_type(op: Any, types: list[str], node: dict[str, Any], field: str) -> str:
    def exact(count: int, accepted: Any, result: str) -> str:
        if len(types) != count or (callable(accepted) and not accepted(types)) or (not callable(accepted) and types != accepted):
            invalid("operation_types", field, "signature", accepted if not callable(accepted) else op, types, "Use the documented operation signature.")
        return result

    if op == "const":
        if "value" not in node:
            invalid("missing_key", field, "required_keys", ["value"], [], "Add the constant value.")
        return exact(0, [], "scalar")
    if op in UNARY_SCALAR:
        return exact(1, ["scalar"], "scalar")
    if op in REDUCTIONS:
        return exact(1, lambda ts: ts[0] in VECTOR_TYPES, "scalar")
    if op in {"x", "y", "z"}:
        return exact(1, lambda ts: ts[0] in VECTOR_TYPES, "scalar")
    if op in COMPONENT_OPS:
        component = int(op.rsplit("_", 1)[1])
        return exact(1, lambda ts: ts[0] in VECTOR_TYPES and component < VECTOR_DIMS[ts[0]], "scalar")
    if op == "w":
        return exact(1, ["quat"], "scalar")
    if op in {"add", "mul", "min", "max"}:
        if len(types) < 2 or len(set(types)) != 1 or types[0] == "bool":
            invalid("operation_types", field, "same_numeric_variadic", "at least two equal numeric types", types, "Use equal scalar or vector operands.")
        return types[0]
    if op in {"sub", "div"}:
        return exact(2, lambda ts: ts[0] == ts[1] and ts[0] != "bool", types[0] if types else "scalar")
    if op == "scale":
        return exact(2, lambda ts: ts[0] == "scalar" and ts[1] in VECTOR_TYPES, types[1] if len(types) > 1 else "vec3")
    if op == "dot":
        return exact(2, lambda ts: ts[0] == ts[1] and ts[0] in VECTOR_TYPES, "scalar")
    if op == "clamp":
        if "min" not in node or "max" not in node:
            invalid("missing_key", field, "required_keys", ["min", "max"], [], "Add both clamp bounds.")
        return exact(1, lambda ts: ts[0] != "bool", types[0] if types else "scalar")
    if op == "keypoint_squash":
        if "a" not in node or "b" not in node:
            invalid("missing_key", field, "required_keys", ["a", "b"], [], "Add source-squash parameters.")
        return exact(1, ["scalar"], "scalar")
    if op in COMPARE:
        return exact(2, lambda ts: ts == ["scalar", "scalar"], "bool")
    if op in BOOL_OPS:
        return exact(2, ["bool", "bool"], "bool")
    if op == "not":
        return exact(1, ["bool"], "bool")
    if op == "where":
        return exact(3, lambda ts: ts[0] == "bool" and ts[1] == ts[2] and ts[1] != "bool", types[1] if len(types) > 1 else "scalar")
    if op == "quat_conjugate":
        return exact(1, ["quat"], "quat")
    if op == "quat_multiply":
        return exact(2, ["quat", "quat"], "quat")
    if op == "quat_rotate":
        return exact(2, ["quat", "vec3"], "vec3")
    invalid("operation", field + ".op", "closed_operation_set", sorted(UNARY_SCALAR | REDUCTIONS | COMPARE | BOOL_OPS | COMPONENT_OPS | {"const", "x", "y", "z", "w", "add", "mul", "min", "max", "sub", "div", "scale", "dot", "clamp", "keypoint_squash", "not", "where", "quat_conjugate", "quat_multiply", "quat_rotate"}), op, "Choose a supported operation.")


class RewardRuntime:
    def __init__(self, recipe: dict[str, Any], torch_module: Any, num_envs: int, device: Any):
        self.recipe = recipe
        self.torch = torch_module
        self.memory = torch_module.zeros((num_envs, recipe["memory_size"]), dtype=torch_module.float32, device=device)

    def reset(self, env_ids: Any) -> None:
        if self.memory.shape[1]:
            self.memory[env_ids] = 0.0

    def evaluate(self, inputs: dict[str, Any]) -> Any:
        torch = self.torch
        values = {name: value.detach().clone() for name, value in inputs.items()}
        for index in range(self.memory.shape[1]):
            values[f"memory_{index}"] = self.memory[:, index].clone()
        exemplar = values["episode_progress"]
        for node in self.recipe["nodes"]:
            args = [values[name] for name in node["args"]]
            op = node["op"]
            if op == "const": value = torch.full_like(exemplar, node["value"])
            elif op == "neg": value = -args[0]
            elif op == "abs": value = torch.abs(args[0])
            elif op == "square": value = torch.square(args[0])
            elif op == "sqrt": value = torch.sqrt(torch.clamp(args[0], min=0.0))
            elif op == "tanh": value = torch.tanh(args[0])
            elif op == "sigmoid": value = torch.sigmoid(args[0])
            elif op == "exp_clamped": value = torch.exp(torch.clamp(args[0], -40.0, 40.0))
            elif op == "log1p_abs": value = torch.log1p(torch.abs(args[0]))
            elif op == "norm": value = torch.linalg.vector_norm(args[0], dim=-1)
            elif op == "sum_components": value = torch.sum(args[0], dim=-1)
            elif op == "mean_components": value = torch.mean(args[0], dim=-1)
            elif op == "x": value = args[0][..., 1 if args[0].shape[-1] == 4 else 0]
            elif op == "y": value = args[0][..., 2 if args[0].shape[-1] == 4 else 1]
            elif op == "z": value = args[0][..., 3 if args[0].shape[-1] == 4 else 2]
            elif op == "w": value = args[0][..., 0]
            elif op in COMPONENT_OPS: value = args[0][..., int(op.rsplit("_", 1)[1])]
            elif op == "add": value = sum(args[1:], args[0])
            elif op == "mul":
                value = args[0]
                for arg in args[1:]: value = value * arg
            elif op == "min": value = torch.stack(args).amin(dim=0)
            elif op == "max": value = torch.stack(args).amax(dim=0)
            elif op == "sub": value = args[0] - args[1]
            elif op == "div": value = args[0] / torch.where(torch.abs(args[1]) < 1e-6, torch.full_like(args[1], 1e-6), args[1])
            elif op == "scale": value = args[0].unsqueeze(-1) * args[1]
            elif op == "dot": value = torch.sum(args[0] * args[1], dim=-1)
            elif op == "clamp": value = torch.clamp(args[0], node["min"], node["max"])
            elif op == "keypoint_squash":
                ax = torch.clamp(node["a"] * args[0], -40.0, 40.0)
                value = 1.0 / (torch.exp(-ax) + node["b"] + torch.exp(ax))
            elif op == "lt": value = args[0] < args[1]
            elif op == "le": value = args[0] <= args[1]
            elif op == "gt": value = args[0] > args[1]
            elif op == "ge": value = args[0] >= args[1]
            elif op == "eq": value = args[0] == args[1]
            elif op == "and": value = torch.logical_and(args[0], args[1])
            elif op == "or": value = torch.logical_or(args[0], args[1])
            elif op == "not": value = torch.logical_not(args[0])
            elif op == "where": value = torch.where(args[0], args[1], args[2])
            elif op == "quat_conjugate": value = torch.cat((args[0][..., :1], -args[0][..., 1:]), dim=-1)
            elif op == "quat_multiply": value = _quat_multiply(torch, args[0], args[1])
            elif op == "quat_rotate":
                pure = torch.cat((torch.zeros_like(args[1][..., :1]), args[1]), dim=-1)
                value = _quat_multiply(torch, _quat_multiply(torch, args[0], pure), torch.cat((args[0][..., :1], -args[0][..., 1:]), dim=-1))[..., 1:]
            else: raise RuntimeError("validated operation dispatch is incomplete")
            if not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(f"nonfinite reward value at node {node['id']}")
            values[node["id"]] = value
        new_memory = self.memory.clone()
        for update in self.recipe["memory_updates"]:
            new_memory[:, update["index"]] = values[update["value"]]
        if not bool(torch.isfinite(new_memory).all().item()):
            raise FloatingPointError("nonfinite reward memory update")
        self.memory = new_memory
        reward = values[self.recipe["output"]]
        if reward.ndim != 1 or not bool(torch.isfinite(reward).all().item()):
            raise FloatingPointError("reward output is not one finite scalar per environment")
        return reward


def _quat_multiply(torch: Any, first: Any, second: Any) -> Any:
    aw, ax, ay, az = first.unbind(-1)
    bw, bx, by, bz = second.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), dim=-1)
