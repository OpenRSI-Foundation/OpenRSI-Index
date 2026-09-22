# Vendored from jaredpalmer/kev@c096660c8da20a80ce7c61c63d224960f497623a.
# Task changes are limited to distributed execution, exact record weighting,
# collective accounting and rank-zero artifact publication; model/loss definitions
# and the upstream reference architecture remain unchanged. See upstream LICENSE.
import argparse, contextlib, json, math, os, random, resource, sys, time
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from .data import EVAL_ONLY, build, augment, load_records, materialize, none_pair, source_seed
from .suite import digest, load_split, write_json
from .model import MAX_BRANCH, MAX_STATE, DecisionModel, load_tokenizer, encode
from .distributed import DistributedContext, iter_global_batches, zero_loss


def permuted_copy(rec, rng):
    """Re-shuffle options of every Choice question with K>=3; return (record, perms) with perms[q] = new->old index or None."""
    out, perms = {"state": rec["state"], "questions": []}, []
    for q in rec["questions"]:
        if q["qtype"] == "choice" and len(q["options"]) >= 3:
            perm = list(range(len(q["options"]))); rng.shuffle(perm)
            out["questions"].append({**q, "options": [q["options"][j] for j in perm], "label": perm.index(q["label"])}); perms.append(perm)
        else:
            out["questions"].append(q); perms.append(None)
    return out, perms


def question_loss(z, q, dev, ord_w):
    """Cross-entropy (or cross-entropy against a soft target when the question carries one), optionally plus the
    normalized ranked probability score for ordered levels."""
    if q.get("target") is not None:
        t = torch.tensor(q["target"], device=dev, dtype=z.dtype)
        return -(t * F.log_softmax(z, -1)).sum()
    y = torch.tensor([q["label"]], device=dev)
    loss = F.cross_entropy(z[None], y)
    if q["qtype"] == "score" and ord_w > 0:
        p = F.softmax(z, -1)
        observed_cdf = (torch.arange(len(p) - 1, device=dev) >= q["label"]).to(p.dtype)
        loss = loss + ord_w * (p.cumsum(-1)[:-1] - observed_cdf).square().mean()
    return loss


def anchor_loss(z, q, target, dev):
    """KL(teacher || student) for one question, teacher = frozen base zero-shot distribution keyed by option key.
    Skips (returns None) when the current option set is not exactly the teacher's (e.g. a none-option was inserted)."""
    if target is None or set(target) != set(q["keys"]): return None
    t = torch.tensor([target[k] for k in q["keys"]], device=dev, dtype=torch.float32).clamp_min(1e-6); t = t / t.sum()
    return F.kl_div(F.log_softmax(z, -1), t, reduction="sum")


def fits_context(tok, req):
    """True when the clean record encodes within the training limits (the rule frozen suites are filtered by)."""
    try:
        return len(encode(tok, materialize(req), strict=True)["ids"]) <= 2048
    except ValueError:
        return False


def accumulation_records(n, batch, accum, microbatch):
    start = (microbatch // accum) * accum * batch
    return min(accum * batch, n - start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--n_per_source", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head_lr", type=float, default=0.0, help="separate learning rate for the pointer head (0 = same as --lr); the head trains from scratch")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay on LoRA and head parameters")
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--holdout", default="", help="comma-separated sources excluded from training (evaluated as out-of-source)")
    ap.add_argument("--perm_kl", type=float, default=0.0, help="weight of symmetric KL between predictions under two option orders")
    ap.add_argument("--perm_frac", type=float, default=0.3, help="fraction of records that get the second permuted forward pass")
    ap.add_argument("--ord_w", type=float, default=0.0, help="weight of ranked probability score for Score questions")
    ap.add_argument("--suite", help="frozen suite directory; train only on its training partition")
    ap.add_argument("--train_sources", default="", help="comma-separated subset of the suite's trainable sources (ablations); default all")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    ap.add_argument("--batch", type=int, default=1, help="records per forward pass (padded batch); optimizer step every --accum micro-batches")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32", help="bf16 = autocast forward with fp32 master weights (CUDA only)")
    ap.add_argument("--weights_dtype", choices=["fp32", "bf16"], default="fp32", help="dtype of the frozen backbone weights. bf16 halves memory and is required by the fused MoE experts "
                                                                                        "(torch._grouped_mm wants bf16); LoRA and head stay fp32 (peft upcasts adapters). Evaluation of such a run must also load bf16 (KEV_DTYPE=bf16).")
    ap.add_argument("--checkpointing", type=int, choices=[0, 1], default=0)
    ap.add_argument("--option_isolation", type=int, choices=[0, 1], default=0, help="option spans are isolated sub-branches with shared positions (exact permutation invariance)")
    ap.add_argument("--special_embeddings", type=int, choices=[0, 1], default=0, help="also train the embeddings of the 5 delimiter tokens")
    ap.add_argument("--head_dim", type=int, default=256, help="pointer head dimension")
    ap.add_argument("--lora_targets", choices=["all", "dense", "attn", "qv"], default="all", help="LoRA module set; fewer modules = less drift from the base; dense = all minus the DeltaNet projections on hybrid bases")
    ap.add_argument("--base_revision", default="", help="pin the base commit when the suite manifest does not pin this base")
    ap.add_argument("--p_none", type=float, default=0.1)
    ap.add_argument("--p_none_distract", type=float, default=0.12)
    ap.add_argument("--p_distract", type=float, default=0.15)
    ap.add_argument("--p_none_pair", type=float, default=0.0, help="fraction of Choice records that additionally emit a none-present/none-absent minimal pair")
    ap.add_argument("--synthetic_repeat", type=int, default=1, help="oversample synthetic policy sources (legacy_policy, compositional, contrastive) this many times per epoch")
    ap.add_argument("--public_frac", type=float, default=1.0, help="deterministic subsample of public-source training records (mix ablations)")
    ap.add_argument("--anchor", default="", help="JSON of frozen-base zero-shot distributions {record_id: {qid: {key: p}}} (kev.anchors); enables the anchoring loss")
    ap.add_argument("--anchor_w", type=float, default=0.0, help="weight of KL(base || model) toward the frozen base model's zero-shot distribution, per anchored question")
    ap.add_argument("--anchor_sources", default="", help="comma-separated sources to anchor (default: every record with a target)")
    ap.add_argument("--out", default="runs/kev")
    ap.add_argument("--data", default="", help="your own labelled requests, one JSON object per line (see kev.data.load_records); an alternative to --suite for fine-tuning, or combined with --suite and --replay")
    ap.add_argument("--replay", type=int, default=0, help="with --data and --suite: mix in this many records sampled (by --seed) from the suite's training partition, so a delta fine-tune does not forget the released recipe")
    ap.add_argument("--init_from", default="", help="delta mode: warm-start LoRA and the pointer head from an existing run "
                                                   "(local directory or hub id) instead of starting from the base model; keeps the "
                                                   "released model's in-domain skill while adapting to a new domain")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if min(a.epochs, a.accum, a.n_per_source, a.lora, a.batch, a.synthetic_repeat) < 1 or not 0 < a.public_frac <= 1:
        ap.error("epochs, accum, n_per_source, lora, batch and synthetic_repeat must be positive; 0 < public_frac <= 1")
    if a.dtype == "bf16" and a.device != "cuda":
        ap.error("--dtype bf16 requires --device cuda")
    if a.lr <= 0 or a.head_lr < 0 or a.weight_decay < 0 or min(a.ord_w, a.perm_kl, a.anchor_w) < 0 or not 0 <= a.perm_frac <= 1:
        ap.error("invalid learning rate or loss weights")
    if bool(a.anchor) != (a.anchor_w > 0):
        ap.error("--anchor and --anchor_w > 0 go together")
    distributed = DistributedContext.from_environment(a.device)
    try:
        train(a, ap, distributed)
    finally:
        distributed.close()


def train(a, ap, distributed):
    anchors = json.loads(Path(a.anchor).read_text()).get("targets", {}) if a.anchor else {}
    if a.anchor and distributed.is_main: print(f"anchor targets: {len(anchors)} records from {a.anchor}", flush=True)
    anchor_sources = set(a.anchor_sources.split(",")) if a.anchor_sources else None
    out_dir = Path(a.out)
    # Exactly one creator; an existing destination or mkdir failure reaches all ranks.
    distributed.run_main(lambda: out_dir.mkdir(parents=True, exist_ok=False))
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    dev = distributed.device
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if a.dtype == "bf16" else contextlib.nullcontext()
    manifest = json.loads((Path(a.suite) / "manifest.json").read_text()) if a.suite else None
    revision = manifest["base_revisions"].get(a.base) if manifest else None
    if a.base_revision:
        if revision and revision != a.base_revision: raise ValueError("--base_revision conflicts with the suite's pinned revision")
        revision = a.base_revision
    if manifest and not revision:
        raise ValueError("base not pinned by the suite; pass --base_revision")
    tok = load_tokenizer(a.base, revision=revision)
    model = DecisionModel(a.base, tok, dev, lora=a.lora, revision=revision, head_dim=a.head_dim, lora_targets=a.lora_targets,
                          option_isolation=bool(a.option_isolation), special_embeddings=bool(a.special_embeddings),
                          dtype=torch.bfloat16 if a.weights_dtype == "bf16" else torch.float32)
    if a.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    init_source = None
    if a.init_from:
        # delta mode (PR #9, Radexito): start from an already trained adapter + pointer head instead of the base model, so a
        # fine-tune on new data keeps what the released checkpoint knows. Compatibility is checked field by field BEFORE
        # loading, because peft loads matching keys silently and a half-loaded adapter still trains and still reports a loss.
        from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
        from .evaluate import resolve_run
        src = resolve_run(a.init_from)
        meta = torch.load(f"{src}/head.pt", map_location="cpu")
        checks = [("base", meta.get("base"), a.base), ("base_revision", meta.get("base_revision"), revision), ("lora", int(meta.get("lora", 0)), a.lora),
                  ("head_dim", int(meta.get("head_dim", 256)), a.head_dim), ("option_isolation", bool(meta.get("option_isolation", False)), bool(a.option_isolation)),
                  ("special_embeddings", bool(meta.get("special_embeddings", False)), bool(a.special_embeddings))]
        for field, theirs, ours in checks:
            if theirs != ours and not (field == "base_revision" and (theirs is None or ours is None)):
                raise ValueError(f"--init_from {src}: {field} is {theirs!r} there and {ours!r} here")
        weights = load_peft_weights(src, device="cpu")
        mine = set(get_peft_model_state_dict(model.lm))
        unexpected, missing = sorted(set(weights) - mine), sorted(mine - set(weights))
        if unexpected:
            raise ValueError(f"--init_from {src} carries {len(unexpected)} adapter tensors this model does not have (e.g. {unexpected[:2]}); check --lora_targets / --lora against its adapter_config.json")
        if missing:
            raise ValueError(f"--init_from {src} does not cover {len(missing)} of this model's adapter tensors (e.g. {missing[:2]}); check --lora_targets")
        set_peft_model_state_dict(model.lm, weights)
        model.head.load_state_dict(meta["head"])
        init_source = {"init_from": a.init_from, "resolved": str(src), "adapter_sha256": digest(Path(src) / "adapter_model.safetensors"), "head_sha256": digest(Path(src) / "head.pt")}
        print(f"delta: warm start from {src}: {len(weights)} adapter tensors and the pointer head loaded", flush=True)
    if distributed.is_main:
        print(f"device={dev} world_size={distributed.world_size} trainable params={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M", flush=True)

    holdout = manifest["holdout_sources"] if manifest else [s for s in a.holdout.split(",") if s]
    if a.replay and not (a.data and a.suite): ap.error("--replay needs both --data and --suite")
    if a.data:
        reqs = load_records(a.data)
        if a.replay:
            pool = load_split(a.suite, "train"); replay = random.Random(f"replay:{a.seed}").sample(pool, min(a.replay, len(pool)))
            print(f"replay: {len(replay)} of {len(pool)} suite training records mixed with {len(reqs)} from {a.data}", flush=True); reqs = reqs + replay
    else:
        reqs = load_split(a.suite, "train") if manifest else build(a.n_per_source, "train", a.seed, exclude=holdout)
    if not manifest or a.data:
        # frozen suites are filtered to the training context when they are frozen (kev.suite.select_unique); records built
        # on the fly here are not, so apply the same rule instead of letting the strict encoder abort the run (issue #5)
        kept = [r for r in reqs if fits_context(tok, r)]
        if len(kept) < len(reqs):
            print(f"dropped {len(reqs) - len(kept)} of {len(reqs)} records that exceed the training context "
                  f"({MAX_STATE} state / {MAX_BRANCH} branch / 2048 packed tokens)", flush=True)
        reqs = kept
    if not reqs:
        raise ValueError("empty training set")
    forbidden = {r["_meta"]["source"] for r in reqs} & set(EVAL_ONLY)
    if forbidden:
        raise ValueError(f"training partition contains eval-only sources: {sorted(forbidden)}")
    if a.train_sources:
        wanted = set(a.train_sources.split(","))
        unknown = wanted - {r["_meta"]["source"] for r in reqs}
        if unknown: raise ValueError(f"--train_sources not in the training partition: {sorted(unknown)}")
        reqs = [r for r in reqs if r["_meta"]["source"] in wanted]
        print(f"ablation: training on {sorted(wanted)} -> {len(reqs)} records", flush=True)
    if manifest:
        from .study_v3 import validate_training
        # --data records are the user's own (validated by load_records; never an eval-only source by construction of their
        # names); the suite's rules apply to the replay sample and to suite-only runs
        validate_training([r for r in reqs if not a.data or not r["_meta"]["source"].startswith(("custom", "night2_"))], manifest)
        if a.data and any(r["_meta"]["source"] in set(EVAL_ONLY) | set(manifest.get("eval_only_sources", [])) for r in reqs):
            raise ValueError("--data contains an eval-only source")
    SYNTHETIC = ("legacy_policy", "compositional", "contrastive")
    if a.public_frac < 1:
        mix_rng = random.Random(source_seed(a.seed, "public_frac"))
        public = [r for r in reqs if r["_meta"]["source"] not in SYNTHETIC]; synth = [r for r in reqs if r["_meta"]["source"] in SYNTHETIC]
        keep = sorted(mix_rng.sample(range(len(public)), int(round(a.public_frac * len(public)))))
        reqs = [public[i] for i in keep] + synth
        print(f"mix: public_frac {a.public_frac} -> {len(keep)} public + {len(synth)} synthetic records", flush=True)
    if a.synthetic_repeat > 1:
        extra = [r for r in reqs if r["_meta"]["source"] in SYNTHETIC] * (a.synthetic_repeat - 1)
        reqs = reqs + extra
        print(f"mix: synthetic_repeat {a.synthetic_repeat} -> +{len(extra)} records", flush=True)
    suite_hash = digest(Path(a.suite) / "manifest.json") if manifest else None
    distributed.run_main(lambda: write_json(out_dir / "training_config.json", {
        "args": vars(a), "suite_sha256": suite_hash, "base_revision": revision, "init_source": init_source,
        "ordinal_objective": "ranked_probability_score", "holdout": holdout,
        "world_size": distributed.world_size, "backend": distributed.backend,
        "global_batch": a.batch * a.accum * distributed.world_size,
    }))
    if distributed.is_main:
        print(f"{len(reqs)} training requests (holdout={holdout}), questions by type "
              f"{dict(Counter(q['qtype'] for r in reqs for q in materialize(r)['questions']))}")

    head_params = list(model.head.parameters()); head_ids = {id(p) for p in head_params}
    groups = [{"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
              {"params": head_params, "lr": a.head_lr or a.lr}]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)
    forward = distributed.wrap_model(model)
    micro_per_epoch = math.ceil(len(reqs) / (a.batch * distributed.world_size))
    steps = a.epochs * math.ceil(micro_per_epoch / a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr or a.lr], total_steps=max(steps, 1), pct_start=0.1)
    model.train(); t0 = time.time(); run = Counter(); step = 0; seen = 0
    tokens_seen = peak_mem = 0
    totals = Counter()
    for ep in range(a.epochs):
        # Every rank advances the same isolated RNG; per-record augmentation has
        # its own (epoch, record id) seed, independent of sharding.
        rng.shuffle(reqs)
        for shard in iter_global_batches(reqs, a.batch, a.accum, distributed.rank, distributed.world_size):
            chunk = shard.records
            recs, encs, perm_jobs, rec_ids, rec_sources, rec_weights = [], [], [], [], [], []
            # A fully empty rank still participates in DDP with a zero-weight
            # dummy forward/backward. This record is NOT sampled or accounted.
            for req in chunk or [shard.fallback]:
                item_rng = random.Random(source_seed(a.seed, f"{ep}:{req['_meta']['id']}"))
                variants = [augment(req, item_rng, p_none=a.p_none, p_none_distract=a.p_none_distract, p_distract=a.p_distract)]  # fresh permutation / distractors each epoch
                if a.p_none_pair > 0 and item_rng.random() < a.p_none_pair:
                    variants += none_pair(req, item_rng)
                for v in variants:
                    rec = materialize(v)
                    enc = model.encode(tok, rec, strict=True)
                    if len(enc["ids"]) > 2048:
                        raise ValueError("training request exceeds 2048 packed tokens")
                    recs.append(rec); encs.append(enc)
                    if chunk: tokens_seen += len(enc["ids"])
                    rec_ids.append(req["_meta"]["id"]); rec_sources.append(req["_meta"]["source"])
                    rec_weights.append(1.0 / len(variants))
                if a.perm_kl > 0 and item_rng.random() < a.perm_frac and any(q["qtype"] == "choice" and len(q["options"]) >= 3 for q in rec["questions"]):
                    rec2, perms = permuted_copy(rec, item_rng); perm_jobs.append((len(recs) - 1, model.encode(tok, rec2, strict=True), perms))
            with autocast:
                logits_b, logits2_b = forward(encs, [e for _, e, _ in perm_jobs])
            # Both forward paths share one DDP reducer invocation. Always keep a
            # zero-gradient connection to every returned tensor, including dummies.
            loss = zero_loss((logits_b, logits2_b))
            for logits, rec, rid, src, weight in zip(logits_b, recs, rec_ids, rec_sources, rec_weights) if chunk else []:
                ce = sum(question_loss(z.float(), q, dev, a.ord_w) for z, q in zip(logits, rec["questions"])) / len(logits)
                run["ce"] += weight * ce.item(); loss = loss + weight * ce
                if anchors and rid in anchors and (anchor_sources is None or src in anchor_sources):
                    terms = [t for t in (anchor_loss(z.float(), q, anchors[rid].get(q["qid"]), dev) for z, q in zip(logits, rec["questions"])) if t is not None]
                    if terms:
                        kl_a = sum(terms) / len(terms); loss = loss + weight * a.anchor_w * kl_a; run["anchor"] += weight * kl_a.item(); run["anchor_n"] += weight
            for (ri, enc2, perms), logits2 in zip(perm_jobs, logits2_b) if chunk else []:
                kl, n = 0.0, 0; tokens_seen += len(enc2["ids"])
                for z1, z2, perm in zip(logits_b[ri], logits2, perms):
                    if perm is None: continue
                    lp1 = F.log_softmax(z1.float(), -1); lp2 = F.log_softmax(z2.float(), -1)[torch.tensor([perm.index(j) for j in range(len(perm))], device=dev)]
                    kl = kl + 0.5 * (F.kl_div(lp2, lp1, log_target=True, reduction="sum") + F.kl_div(lp1, lp2, log_target=True, reduction="sum")); n += 1
                weight = rec_weights[ri]
                kl = kl / n; loss = loss + weight * a.perm_kl * kl; run["kl"] += weight * kl.item(); run["kl_n"] += weight
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            # Per-source-record variant averaging, then the exact GLOBAL group
            # denominator, compensating for DDP's mean across ranks.
            distributed.scale_loss(loss, shard.group_records).backward()
            run["n"] += len(chunk); seen += len(chunk)
            totals["loss_sum"] += loss.detach().item()
            totals["augmented_records"] += len(recs) if chunk else 0
            if dev == "mps": peak_mem = max(peak_mem, torch.mps.current_allocated_memory())
            elif dev == "cuda": peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())
            if shard.step:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if dev == "mps": torch.mps.empty_cache()
                if step % 10 == 0:
                    global_run = distributed.reduce_sums({key: run[key] for key in ("ce", "n", "kl", "kl_n", "anchor", "anchor_n")})
                    global_seen = distributed.reduce_sums({"seen": seen})["seen"]
                    if distributed.is_main:
                        print(f"ep{ep} step {step}/{steps} loss {global_run['ce']/global_run['n']:.3f} kl {global_run['kl']/max(global_run['kl_n'],1):.3f} anchor {global_run['anchor']/max(global_run['anchor_n'],1):.3f} {(time.time()-t0)/global_seen:.3f}s/rec", flush=True)
                    run = Counter()
    global_totals = distributed.reduce_sums({"records_seen": seen, "forward_tokens": tokens_seen,
        "loss_sum": totals["loss_sum"], "augmented_records": totals["augmented_records"]})
    peaks = distributed.reduce_max({"wall_seconds": time.time() - t0, "peak_device_bytes": peak_mem,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)})
    if int(global_totals["records_seen"]) != a.epochs * len(reqs):
        raise RuntimeError("distributed training did not consume every requested record")

    def save():
        model.lm.save_pretrained(a.out)
        torch.save({"head": model.head.state_dict(), "base": a.base, "base_revision": revision, "lora": a.lora, "head_dim": a.head_dim,
                    "option_isolation": bool(a.option_isolation), "special_embeddings": bool(a.special_embeddings), "weights_dtype": a.weights_dtype,
                    "holdout": holdout, "args": vars(a), "suite_sha256": suite_hash, "init_source": init_source}, f"{a.out}/head.pt")
        tok.save_pretrained(a.out)
        # This completion marker is last; the outer launcher publishes the task
        # manifest only after ALL torchrun workers return successfully.
        write_json(out_dir / "training_metrics.json", {
            **peaks, "records_seen": int(global_totals["records_seen"]),
            "requested_records": a.epochs * len(reqs), "truncated_records": 0, "rejected_records": 0,
            "optimizer_steps": step, "forward_tokens": int(global_totals["forward_tokens"]),
            "augmented_records": int(global_totals["augmented_records"]),
            "mean_training_loss": global_totals["loss_sum"] / global_totals["records_seen"],
            "device": dev, "dtype": a.dtype, "batch": a.batch, "accum": a.accum,
            "world_size": distributed.world_size, "backend": distributed.backend,
            "global_batch": a.batch * a.accum * distributed.world_size,
        })
        print("saved", a.out, flush=True)

    distributed.run_main(save)


if __name__ == "__main__":
    main()
