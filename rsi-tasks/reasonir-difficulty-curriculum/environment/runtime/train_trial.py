#!/usr/bin/env python3
"""Fixed four-process ReasonIR LoRA trainer with durable global accounting."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel
from transformers import get_constant_schedule_with_warmup

from model_support import BASE_REVISION, LORA_CONFIG, load_base

UPDATES = 1000
WORLD_SIZE = 4
PER_DEVICE_BATCH = 4
ACCUMULATION = 4
LOCAL_LOGICAL_QUERIES = PER_DEVICE_BATCH * ACCUMULATION
GLOBAL_EXPOSURES = WORLD_SIZE * LOCAL_LOGICAL_QUERIES
GROUP_SIZE = 2
QUERY_MAX_LENGTH = 2048
PASSAGE_MAX_LENGTH = 2048
TEMPERATURE = 0.02
WARMUP_STEPS = 60
OPTIMIZER = {"lr": 2e-5, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
BASE_BOS = "<s>"
USER_BOS = "<|user|>\n"
USER_EOS = ""
EMBED_BOS = "\n<|embed|>\n"

# Static invariants for the confirmed four-process GritLM trajectory.
assert LOCAL_LOGICAL_QUERIES == 16
assert GLOBAL_EXPOSURES == 64
assert GROUP_SIZE == 2
assert WARMUP_STEPS == int(UPDATES * 0.06)
assert OPTIMIZER["weight_decay"] == 0.0


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def journal_record(record: dict) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def append_journal(path: Path, record: dict) -> None:
    with path.open("ab") as stream:
        stream.write(journal_record(record))
        stream.flush()
        os.fsync(stream.fileno())


def journal_state(path: Path) -> tuple[int, int, str]:
    admitted: set[int] = set()
    completed: set[int] = set()
    raw = path.read_bytes() if path.exists() else b""
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError("accounting journal is not UTF-8") from exc
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError("accounting journal has a torn final record")
    for line in lines:
        row = json.loads(line)
        allowed = ({"event", "update", "first_exposure", "exposures"}, {"event", "update"})
        if not isinstance(row, dict) or set(row) not in allowed:
            raise RuntimeError("invalid accounting journal record shape")
        update = int(row["update"])
        if not 0 <= update < UPDATES:
            raise RuntimeError("accounting journal update is out of range")
        if row["event"] == "admitted":
            expected = {"event", "update", "first_exposure", "exposures"}
            if set(row) != expected or row["first_exposure"] != update * GLOBAL_EXPOSURES or row["exposures"] != GLOBAL_EXPOSURES or update in admitted:
                raise RuntimeError("invalid admitted journal record")
            admitted.add(update)
        elif row["event"] == "completed":
            if set(row) != {"event", "update"} or update not in admitted or update in completed:
                raise RuntimeError("invalid completed journal record")
            completed.add(update)
        else:
            raise RuntimeError("unknown journal event")
    if admitted - completed:
        raise RuntimeError("a prior post-admission failure consumed exposures; this trial cannot be scoreable")
    if completed != set(range(len(completed))) or admitted != completed:
        raise RuntimeError("accounting journal updates are not contiguous")
    return len(completed), len(admitted) * GLOBAL_EXPOSURES, sha256_bytes(raw)


def format_embedding_pair(instruction: str, text: str) -> tuple[str, str]:
    clean = instruction[: QUERY_MAX_LENGTH * 10].strip("\t\n :")
    prefix = BASE_BOS + USER_BOS + clean + USER_EOS + EMBED_BOS if clean else BASE_BOS + EMBED_BOS.lstrip()
    return prefix + text[: QUERY_MAX_LENGTH * 10], prefix


def tokenize_pairs(tokenizer, pairs: list[tuple[str, str]], maximum: int) -> dict[str, torch.Tensor]:
    sentences: list[str] = []
    instruction_lens: list[int] = []
    for instruction, text in pairs:
        sentence, prefix = format_embedding_pair(instruction, text)
        sentences.append(sentence)
        instruction_lens.append(len(tokenizer.tokenize(prefix)))
    encoded = tokenizer(sentences, padding=True, truncation=True, max_length=maximum, return_tensors="pt", add_special_tokens=False)
    if any(length >= maximum for length in instruction_lens):
        raise RuntimeError("fixed training instruction leaves no text token to embed")
    encoded["instruction_lens"] = torch.tensor(instruction_lens, dtype=torch.long)
    return encoded


def encode_batch(model, encoded: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    instruction_lens = encoded["instruction_lens"]
    inputs = {key: value.to(device) for key, value in encoded.items() if key != "instruction_lens"}
    inputs["is_causal"] = False
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**inputs)[0]
    mask = inputs["attention_mask"].clone()
    for index, length in enumerate(instruction_lens):
        mask[index, : int(length)] = 0
        if not bool(mask[index].sum()):
            raise RuntimeError("fixed training example has no unmasked text token")
    summed = torch.sum(output * mask.unsqueeze(-1).float(), dim=1)
    embedding = summed / mask.sum(dim=1, keepdim=True).float()
    return functional.normalize(embedding, dim=-1).contiguous().to(output.dtype)


def choose_pair(row: dict, ordinal: int) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    positives = row["positive"]
    negatives = row["negative"]
    if not positives or not negatives:
        raise RuntimeError(f"fixed row {row['row_id']} has no positive or negative")
    query = (row["query_instruction"], row["query"])
    positive = positives[ordinal % len(positives)]
    negative = negatives[(ordinal // max(1, len(positives))) % len(negatives)]
    return query, (positive["instruction"], positive["text"]), (negative["instruction"], negative["text"])


def rng_snapshot(device: torch.device) -> dict:
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}


def restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)


def checkpoint_path(trial: Path) -> Path:
    return trial / "checkpoint.pt"


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_checkpoint(path: Path, model, optimizer, scheduler, completed: int, schedule_digest: str, expected_journal_digest: str, device: torch.device) -> None:
    from peft import get_peft_model_state_dict

    local_rng = rng_snapshot(device)
    rng_states: list[dict | None] = [None] * WORLD_SIZE
    dist.all_gather_object(rng_states, local_rng)
    if dist.get_rank() != 0:
        return
    payload = {
        "version": 1,
        "completed_updates": completed,
        "schedule_sha256": schedule_digest,
        "accounting_sha256": expected_journal_digest,
        "adapter": cpu_tree(get_peft_model_state_dict(model.module)),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "rng_by_rank": rng_states,
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def restore_checkpoint(path: Path, model, optimizer, scheduler, completed: int, schedule_digest: str, journal_digest: str, rank: int, device: torch.device) -> None:
    if completed == 0:
        if path.exists():
            raise RuntimeError("checkpoint exists for an empty accounting journal")
        return
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("completed accounting journal has no closed checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"version", "completed_updates", "schedule_sha256", "accounting_sha256", "adapter", "optimizer", "scheduler", "rng_by_rank"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("checkpoint field set is invalid")
    if payload["version"] != 1 or payload["completed_updates"] != completed or payload["schedule_sha256"] != schedule_digest or payload["accounting_sha256"] != journal_digest:
        raise RuntimeError("checkpoint identity does not match schedule and accounting journal")
    if not isinstance(payload["rng_by_rank"], list) or len(payload["rng_by_rank"]) != WORLD_SIZE:
        raise RuntimeError("checkpoint does not contain one RNG state per process")
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    expected_adapter_keys = set(get_peft_model_state_dict(model.module))
    if set(payload["adapter"]) != expected_adapter_keys:
        raise RuntimeError("checkpoint adapter tensor inventory does not match the fixed model")
    result = set_peft_model_state_dict(model.module, payload["adapter"])
    if result.unexpected_keys:
        raise RuntimeError(f"checkpoint adapter key mismatch: unexpected={result.unexpected_keys[:5]}")
    optimizer.load_state_dict(payload["optimizer"])
    optimizer_to_device(optimizer, device)
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng_by_rank"][rank], device)


def logical_batch(model, tokenizer, table: list[dict], schedule: list[int], update: int, rank: int, device: torch.device) -> torch.Tensor:
    query_batches: list[dict[str, torch.Tensor]] = []
    passage_batches: list[dict[str, torch.Tensor]] = []
    local_start = update * GLOBAL_EXPOSURES + rank * LOCAL_LOGICAL_QUERIES
    for chunk in range(ACCUMULATION):
        offset = local_start + chunk * PER_DEVICE_BATCH
        queries: list[tuple[str, str]] = []
        passages: list[tuple[str, str]] = []
        for local, pool_index in enumerate(schedule[offset : offset + PER_DEVICE_BATCH]):
            query, positive, negative = choose_pair(table[pool_index], offset + local)
            queries.append(query)
            passages.extend((positive, negative))
        query_batches.append(tokenize_pairs(tokenizer, queries, QUERY_MAX_LENGTH))
        passage_batches.append(tokenize_pairs(tokenizer, passages, PASSAGE_MAX_LENGTH))

    q_reps: list[torch.Tensor] = []
    p_reps: list[torch.Tensor] = []
    q_rng: list[dict] = []
    p_rng: list[dict] = []
    with torch.no_grad():
        for query, passage in zip(query_batches, passage_batches):
            q_rng.append(rng_snapshot(device))
            q_reps.append(encode_batch(model, query, device))
            p_rng.append(rng_snapshot(device))
            p_reps.append(encode_batch(model, passage, device))

    local_q = torch.cat(q_reps).detach().requires_grad_()
    local_p = torch.cat(p_reps).detach().requires_grad_()
    gathered_q = [torch.empty_like(local_q) for _ in range(WORLD_SIZE)]
    gathered_p = [torch.empty_like(local_p) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_q, local_q.detach())
    dist.all_gather(gathered_p, local_p.detach())
    gathered_q[rank] = local_q
    gathered_p[rank] = local_p
    global_q = torch.cat(gathered_q)
    global_p = torch.cat(gathered_p)
    assert global_q.shape[0] == GLOBAL_EXPOSURES
    assert global_p.shape[0] == GLOBAL_EXPOSURES * GROUP_SIZE
    scores = global_q.float() @ global_p.float().T / TEMPERATURE
    targets = torch.arange(GLOBAL_EXPOSURES, device=device, dtype=torch.long) * GROUP_SIZE
    loss = functional.cross_entropy(scores, targets)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at update {update}")
    loss.backward()
    q_grad = local_q.grad.detach().chunk(ACCUMULATION)
    p_grad = local_p.grad.detach().chunk(ACCUMULATION)

    operations = []
    for index in range(ACCUMULATION):
        operations.append((query_batches[index], q_rng[index], q_grad[index]))
        operations.append((passage_batches[index], p_rng[index], p_grad[index]))
    for index, (batch, rng, gradient) in enumerate(operations):
        restore_rng(rng, device)
        context = nullcontext() if index == len(operations) - 1 else model.no_sync()
        with context:
            representation = encode_batch(model, batch, device)
            torch.sum(representation * gradient).backward()
    return loss.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError("fixed trainer requires exactly four processes")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    schedule_doc = json.loads(args.schedule.read_text(encoding="utf-8"))
    schedule = schedule_doc["schedule"]
    if len(schedule) != UPDATES * GLOBAL_EXPOSURES:
        raise RuntimeError("fixed schedule does not contain exactly 64,000 exposures")
    table = pq.read_table(args.pool).to_pylist()
    if schedule_doc.get("pool_rows") != len(table) or any(not isinstance(index, int) or not 0 <= index < len(table) for index in schedule):
        raise RuntimeError("compiled schedule does not match immutable pool row identities")
    schedule_digest = sha256(args.schedule)
    model, tokenizer = load_base(training=True)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, **OPTIMIZER)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS)

    journal = args.trial / "accounting.jsonl"
    if rank == 0:
        start, exposures, journal_digest = journal_state(journal)
        state = torch.tensor([start, exposures], device=device, dtype=torch.long)
    else:
        journal_digest = ""
        state = torch.zeros(2, device=device, dtype=torch.long)
    dist.broadcast(state, src=0)
    start = int(state[0].item())
    digests: list[str | None] = [journal_digest if rank == 0 else None]
    dist.broadcast_object_list(digests, src=0)
    journal_digest = str(digests[0])
    if int(state[1].item()) != start * GLOBAL_EXPOSURES:
        raise RuntimeError("durable exposure/update state is inconsistent")
    restore_checkpoint(checkpoint_path(args.trial), model, optimizer, scheduler, start, schedule_digest, journal_digest, rank, device)

    for update in range(start, UPDATES):
        admitted = {"event": "admitted", "update": update, "first_exposure": update * GLOBAL_EXPOSURES, "exposures": GLOBAL_EXPOSURES}
        if rank == 0:
            append_journal(journal, admitted)
        dist.barrier()
        optimizer.zero_grad(set_to_none=True)
        loss = logical_batch(model, tokenizer, table, schedule, update, rank, device)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        completed = {"event": "completed", "update": update}
        if rank == 0:
            current = journal.read_bytes()
            expected_journal_digest = sha256_bytes(current + journal_record(completed))
        else:
            expected_journal_digest = ""
        expected: list[str | None] = [expected_journal_digest if rank == 0 else None]
        dist.broadcast_object_list(expected, src=0)
        save_checkpoint(checkpoint_path(args.trial), model, optimizer, scheduler, update + 1, schedule_digest, str(expected[0]), device)
        dist.barrier()
        if rank == 0:
            append_journal(journal, completed)
            if (update + 1) % 10 == 0:
                print(json.dumps({"update": update + 1, "exposures": (update + 1) * GLOBAL_EXPOSURES, "loss": float(loss)}), flush=True)
        dist.barrier()

    if rank == 0:
        adapter = args.trial / "adapter"
        staging = args.trial / "adapter.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        model.module.save_pretrained(staging, safe_serialization=True)
        if adapter.exists():
            shutil.rmtree(adapter)
        staging.replace(adapter)
        adapter_file = adapter / "adapter_model.safetensors"
        config_file = adapter / "adapter_config.json"
        manifest = {
            "schema_version": 1,
            "base_revision": BASE_REVISION,
            "adapter_file": "adapter/adapter_model.safetensors",
            "adapter_sha256": sha256(adapter_file),
            "adapter_config_file": "adapter/adapter_config.json",
            "adapter_config_sha256": sha256(config_file),
            "lora": LORA_CONFIG,
            "training": {
                "updates": UPDATES,
                "exposures": UPDATES * GLOBAL_EXPOSURES,
                "world_size": WORLD_SIZE,
                "per_device_batch": PER_DEVICE_BATCH,
                "gradient_accumulation": ACCUMULATION,
                "schedule_sha256": schedule_digest,
                "policy_sha256": schedule_doc["policy_sha256"],
                "accounting_sha256": sha256(journal),
            },
        }
        temporary = args.trial / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.trial / "manifest.json")
        (args.trial / "COMPLETE").write_text("complete\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
