"""Distribution-only helpers for the pinned Kev reference trainer.

No model/data imports: the exact sharding and DDP math are testable on tiny CPU models.
"""
from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class BatchShard:
    records: list
    fallback: Any
    group_records: int
    index: int
    step: bool


def iter_global_batches(records, batch, accum, rank=0, world_size=1):
    """Shard an identically shuffled order, without sampling, duplication or padding.

    All ranks yield the same number of microsteps, including empty local tails.
    ``fallback`` is for a zero-weight collective-participation forward ONLY: it is
    never an observed training record or a counted token. Denominators count the
    actual global source records in the accumulation group, including short tails.
    """
    if min(batch, accum, world_size) < 1 or not 0 <= rank < world_size:
        raise ValueError("positive batch/accum/world_size and an in-range rank required")
    width = batch * world_size
    microsteps = (len(records) + width - 1) // width
    for index in range(microsteps):
        start = index * width
        local_start = start + rank * batch
        group_start = (index // accum) * accum * width
        yield BatchShard(
            records=list(records[local_start:min(local_start + batch, start + width)]),
            fallback=records[start],
            group_records=min(accum * width, len(records) - group_start),
            index=index,
            step=(index + 1) % accum == 0 or index + 1 == microsteps,
        )


class _ForwardPair(torch.nn.Module):
    """Enter the DDP reducer once even when permutation KL needs two backbone calls."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, encs, perm_encs):
        main = self.model.forward_batch(encs)
        permuted = self.model.forward_batch(perm_encs) if perm_encs else []
        return main, permuted


def zero_loss(outputs):
    """Keep the dummy graph connected to all returned logits, with zero gradient."""
    if isinstance(outputs, torch.Tensor):
        return outputs.sum() * 0.0
    terms = [zero_loss(value) for value in outputs]
    return sum(terms) if terms else 0.0


@dataclass
class DistributedContext:
    device: str
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "none"
    _owns_group: bool = False

    @classmethod
    def from_environment(cls, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        rank, world_size, local_rank = (int(os.environ.get(key, default))
                                        for key, default in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
        if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
            raise ValueError("invalid torchrun rank environment")
        if device == "cuda":
            # Must precede model construction AND NCCL initialization/object collectives.
            torch.cuda.set_device(local_rank)
        if world_size > 1 and device not in ("cuda", "cpu"):
            raise ValueError("distributed training requires CUDA/NCCL or CPU/Gloo")
        backend = ("nccl" if device == "cuda" else "gloo") if world_size > 1 else "none"
        context = cls(device, rank, world_size, local_rank, backend)
        if world_size > 1:
            if dist.is_initialized():
                raise RuntimeError("trainer must own its process group")
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timedelta(minutes=30))
            context._owns_group = True
        return context

    @property
    def is_main(self):
        return self.rank == 0

    def wrap_model(self, model):
        paired = _ForwardPair(model)
        if self.world_size == 1:
            return paired
        options = {"broadcast_buffers": False, "find_unused_parameters": False}
        if self.device == "cuda":
            options.update(device_ids=[self.local_rank], output_device=self.local_rank)
        return DistributedDataParallel(paired, **options)

    def scale_loss(self, local_loss_sum, global_group_records):
        if global_group_records < 1:
            raise ValueError("global accumulation group must contain records")
        # DDP averages gradients across ranks; undo that average before applying
        # the exact global-record denominator (also correct on empty-rank tails).
        return local_loss_sum * (self.world_size / global_group_records)

    def _reduce(self, values, operation):
        keys = sorted(values)
        tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=operation)
        return dict(zip(keys, tensor.cpu().tolist()))

    def reduce_sums(self, values):
        return self._reduce(values, dist.ReduceOp.SUM)

    def reduce_max(self, values):
        return self._reduce(values, dist.ReduceOp.MAX)

    def run_main(self, action):
        """Run an artifact action on rank zero; report its failure on every rank."""
        error = [None]
        if self.is_main:
            try:
                action()
            except Exception as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        if self.world_size > 1:
            dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(f"rank-zero action failed: {error[0]}")

    def close(self):
        # No success barrier here: an exception on one rank must not masquerade
        # as a successful checkpoint or strand peers in a cleanup-only barrier.
        if self._owns_group and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_group = False
