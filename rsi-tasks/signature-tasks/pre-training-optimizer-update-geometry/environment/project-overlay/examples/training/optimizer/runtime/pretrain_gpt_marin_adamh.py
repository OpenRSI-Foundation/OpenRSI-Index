# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run the locked Marin AdamH scaling recipes through Megatron."""

from __future__ import annotations

import math
import json
import os
import time
from pathlib import Path
from typing import Any

import pretrain_gpt as upstream
import torch
from megatron.core import parallel_state
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.training import initialize as megatron_initialize
from megatron.training import training
from optimizer_candidate import build_megatron_optimizer
from optimizer_candidate import install_runtime as install_optimizer_runtime


Z_LOSS_WEIGHT = 1.0e-7


def _install_zero_seed_support() -> None:
    """Permit Marin's locked seed 0 without changing its RNG streams."""
    original = megatron_initialize._set_random_seed

    def set_random_seed_allowing_zero(
        seed_: int,
        data_parallel_random_init: bool = False,
        te_rng_tracker: bool = False,
        inference_rng_tracker: bool = False,
        use_cudagraphable_rng: bool = False,
        pp_group=None,
        dp_group=None,
        tp_group=None,
        ep_group=None,
        etp_group=None,
    ) -> None:
        if seed_ != 0:
            return original(
                seed_,
                data_parallel_random_init,
                te_rng_tracker,
                inference_rng_tracker,
                use_cudagraphable_rng,
                pp_group,
                dp_group,
                tp_group,
                ep_group,
                etp_group,
            )

        # This is Megatron's normal seed setup with only the > 0 guard relaxed
        # to >= 0. Keeping the value at zero is required for Marin parity.
        pp_rank = (
            megatron_initialize.get_pg_rank(pp_group)
            if pp_group is not None
            else megatron_initialize.mpu.get_pipeline_model_parallel_rank()
        )
        seed = seed_ + (100 * pp_rank)
        if data_parallel_random_init:
            dp_rank = (
                megatron_initialize.get_pg_rank(dp_group)
                if dp_group is not None
                else megatron_initialize.mpu.get_data_parallel_rank()
            )
            seed += 10 * dp_rank
        megatron_initialize.random.seed(seed)
        megatron_initialize.np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tp_rank = megatron_initialize.get_pg_rank(tp_group) if tp_group is not None else None
            ep_rank = megatron_initialize.get_pg_rank(ep_group) if ep_group is not None else None
            etp_rank = megatron_initialize.get_pg_rank(etp_group) if etp_group is not None else None
            megatron_initialize.tensor_parallel.model_parallel_cuda_manual_seed(
                seed,
                te_rng_tracker,
                inference_rng_tracker,
                use_cudagraphable_rng,
                tp_rank=tp_rank,
                ep_rank=ep_rank,
                etp_rank=etp_rank,
            )

    megatron_initialize._set_random_seed = set_random_seed_allowing_zero


def _fan_in_truncated_normal(tensor: torch.Tensor) -> None:
    """Match Haliax Linear.init: truncated N(0, 1) / sqrt(fan_in)."""
    fan_in = tensor.shape[-1]
    std = 1.0 / math.sqrt(fan_in)
    torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)


def _embedding_truncated_normal(tensor: torch.Tensor) -> None:
    """Match Haliax EmbeddingStandardParam (scale = 1 / hidden size)."""
    std = 1.0 / tensor.shape[-1]
    torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)


def _install_z_loss() -> None:
    original = LanguageModule.compute_language_model_loss

    def with_marin_z_loss(self, labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # Megatron keeps vocabulary logits sharded under TP. Compute the global
        # log normalizer without gathering the full vocabulary. The detached
        # maximum is only a numerical shift; the autograd-aware sum reduction
        # supplies the exact local softmax gradient on every TP rank.
        tp_group = parallel_state.get_tensor_model_parallel_group()
        local_max = logits.detach().float().amax(dim=-1, keepdim=True)
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        local_exp_sum = torch.exp(logits.float() - local_max).sum(dim=-1)
        global_exp_sum = reduce_from_tensor_model_parallel_region(local_exp_sum, group=tp_group)
        log_normalizers = torch.log(global_exp_sum) + local_max.squeeze(-1)
        losses = original(self, labels, logits)
        if self.training and torch.is_grad_enabled():
            losses = losses + Z_LOSS_WEIGHT * log_normalizers.transpose(0, 1).square()
        return losses

    LanguageModule.compute_language_model_loss = with_marin_z_loss


def _install_evaluation_hook() -> None:
    lm_eval_enabled = os.environ.get("LM_EVAL_ENABLED", "0") == "1"
    longppl_enabled = os.environ.get("LONGPPL_ENABLED", "0") == "1"
    eval_only = os.environ.get("EVAL_ONLY", "0") == "1"
    if not lm_eval_enabled and not longppl_enabled:
        return

    harness_config = None
    harness_interval = None
    final_only_tasks: tuple[str, ...] = ()
    if lm_eval_enabled:
        from lm_eval_runner import config_from_environment, final_only_tasks_from_environment

        harness_config, harness_interval = config_from_environment()
        final_only_tasks = final_only_tasks_from_environment()

    longppl_config = None
    longppl_interval = None
    if longppl_enabled:
        from longppl_runner import config_from_environment as longppl_config_from_environment

        longppl_config = longppl_config_from_environment()
        longppl_interval = int(os.environ.get("LONGPPL_INTERVAL", "10000"))
        if longppl_interval <= 0:
            raise ValueError(f"LONGPPL_INTERVAL must be positive, got {longppl_interval}")

    original_save_checkpoint_and_time = training.save_checkpoint_and_time
    harness_iterations: set[int] = set()
    longppl_iterations: set[int] = set()

    if eval_only:
        original_setup_model_and_optimizer = training.setup_model_and_optimizer

        def setup_then_evaluate(*args: Any, **kwargs: Any) -> Any:
            result = original_setup_model_and_optimizer(*args, **kwargs)
            model = result[0]
            training_args = training.get_args()
            iteration = int(getattr(training_args, "iteration", 0) or 0)
            train_iters = int(getattr(training_args, "train_iters", 0) or 0)
            if iteration <= 0:
                raise ValueError("LongPPL eval-only mode requires a checkpoint with a positive iteration")
            if iteration != train_iters:
                raise ValueError(
                    "LongPPL eval-only mode requires the final checkpoint: "
                    f"loaded iteration {iteration}, expected {train_iters}"
                )
            if lm_eval_enabled and iteration not in harness_iterations:
                from lm_eval_runner import config_for_iteration, evaluate_with_harness

                if harness_config is None:
                    raise RuntimeError("LM Evaluation Harness configuration is missing")
                iteration_config = config_for_iteration(
                    harness_config,
                    iteration=iteration,
                    train_iters=train_iters,
                    final_only_tasks=final_only_tasks,
                )
                evaluate_with_harness(model, iteration_config, iteration)
                harness_iterations.add(iteration)
            if longppl_enabled and iteration not in longppl_iterations:
                from longppl_runner import evaluate_longppl

                if longppl_config is None:
                    raise RuntimeError("LongPPL configuration is missing")
                evaluate_longppl(model, longppl_config, iteration)
                longppl_iterations.add(iteration)
            return result

        training.setup_model_and_optimizer = setup_then_evaluate

    def save_then_evaluate(
        iteration: int,
        model: Any,
        optimizer: Any,
        opt_param_scheduler: Any,
        num_floating_point_operations_so_far: Any,
        checkpointing_context: Any,
        non_persistent_ckpt: bool = False,
        train_data_iterator: Any = None,
    ) -> Any:
        # Hook the actual save path rather than the per-step checkpoint check.
        # This also covers a final step that is not divisible by save_interval.
        result = original_save_checkpoint_and_time(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
            non_persistent_ckpt=non_persistent_ckpt,
            train_data_iterator=train_data_iterator,
        )
        train_iters = int(getattr(training.get_args(), "train_iters", 0) or 0)
        run_harness = bool(
            lm_eval_enabled
            and harness_interval is not None
            and not non_persistent_ckpt
            and iteration > 0
            and (iteration % harness_interval == 0 or iteration == train_iters)
            and iteration not in harness_iterations
        )
        run_longppl = bool(
            longppl_enabled
            and longppl_interval is not None
            and not non_persistent_ckpt
            and iteration > 0
            and (iteration % longppl_interval == 0 or iteration == train_iters)
            and iteration not in longppl_iterations
        )
        if run_harness or run_longppl:
            # An async checkpoint can return before its files are durable. Wait
            # for it here so an evaluation failure cannot lose this resume point.
            training.maybe_finalize_async_save(blocking=True)
            # The AdamH data split is training-only, so Megatron's native
            # validation callback never runs. Drive external evaluation only after the
            # matching checkpoint is durable and exclude its wall time from
            # train throughput accounting, just as native validation does.
            timers = training.get_timers()
            timers("interval-time").stop()
            try:
                if run_harness:
                    from lm_eval_runner import config_for_iteration, evaluate_with_harness

                    if harness_config is None:
                        raise RuntimeError("LM Evaluation Harness configuration is missing")
                    iteration_config = config_for_iteration(
                        harness_config,
                        iteration=iteration,
                        train_iters=train_iters,
                        final_only_tasks=final_only_tasks,
                    )
                    evaluate_with_harness(model, iteration_config, iteration)
                    harness_iterations.add(iteration)
                if run_longppl:
                    from longppl_runner import evaluate_longppl

                    if longppl_config is None:
                        raise RuntimeError("LongPPL configuration is missing")
                    evaluate_longppl(model, longppl_config, iteration)
                    longppl_iterations.add(iteration)
            finally:
                timers("interval-time", log_level=0).start(barrier=True)
        return result

    training.save_checkpoint_and_time = save_then_evaluate


def _install_cost_trace_hook() -> None:
    """Record charged per-step CUDA time and fixed-trajectory training loss.

    The timer brackets Megatron's complete ``train_step``: forward, backward,
    gradient communication, and optimizer update.  Every rank contributes and
    a world MAX is charged, matching the task's slowest-rank cost coordinate.
    Only global rank zero appends the durable trace.
    """

    original_train_step = training.train_step
    cumulative_gpu_seconds = 0.0
    run_dir = Path(os.environ.get("RUN_DIR", "."))
    trace_path = run_dir / "optimizer_cost_trace.jsonl"

    def measured_train_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal cumulative_gpu_seconds
        if not torch.cuda.is_available():
            return original_train_step(*args, **kwargs)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_train_step(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        cuda_seconds = start.elapsed_time(end) / 1000.0
        active_seconds = time.perf_counter() - wall_start
        elapsed = torch.tensor(active_seconds, device="cuda", dtype=torch.float64)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0
        step_gpu_seconds = float(elapsed.item()) * world_size
        cumulative_gpu_seconds += step_gpu_seconds
        if rank == 0:
            iteration = kwargs.get("iteration")
            if iteration is None and len(args) > 7:
                iteration = args[7]
            iteration = int(iteration or 0) + 1
            loss_value = None
            skipped = None
            if isinstance(result, tuple) and result:
                loss_dict = result[0]
                if isinstance(loss_dict, dict) and "lm loss" in loss_dict:
                    raw_loss = loss_dict["lm loss"]
                    loss_value = float(raw_loss.detach().float().item()) if torch.is_tensor(raw_loss) else float(raw_loss)
                if len(result) > 1:
                    skipped = int(result[1])
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "step_active_max_seconds": float(elapsed.item()),
                            "step_cuda_seconds_rank0": cuda_seconds,
                            "step_gpu_seconds": step_gpu_seconds,
                            "cumulative_gpu_seconds": cumulative_gpu_seconds,
                            "world_size": world_size,
                            "lm_loss": loss_value,
                            "skipped": skipped,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        return result

    training.train_step = measured_train_step


def main() -> None:
    """Install Marin-specific semantics and execute upstream pretraining."""
    install_optimizer_runtime()
    _install_zero_seed_support()
    _install_evaluation_hook()
    _install_cost_trace_hook()
    _install_z_loss()
    training.get_megatron_optimizer = build_megatron_optimizer

    upstream.print_rank_0(f"> PyTorch version ................ {upstream.get_torch_version()}")
    upstream.print_rank_0(f"> Megatron-Core version .......... {upstream.mcore_version}")
    upstream.print_rank_0(f"> Transformer Engine version ... {upstream.get_te_version()}")
    upstream.set_startup_timestamps(
        program_start=upstream._PROGRAM_START_TIME,
        main_entry=time.time(),
    )
    setattr(upstream.train_valid_test_datasets_provider, "is_distributed", True)
    pretrain, store = upstream.inprocess_restart.maybe_wrap_for_inprocess_restart(upstream.pretrain)

    args = upstream.parse_and_validate_args(
        extra_args_provider=upstream.add_modelopt_args if upstream.has_nvidia_modelopt else None,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    if upstream.has_nvidia_modelopt:
        upstream.maybe_enable_modelopt(args)
    if upstream.has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):
        model_cfg = upstream.gpt_config_from_args(
            args,
            model_config_cls=upstream.ModelOptModelConfig,
        )
    else:
        model_cfg = upstream.gpt_config_from_args(args)

    # These are the initializers actually called by the source Haliax model.
    # The recorded initializer_range=0.02 is export metadata and is not used by
    # LlamaEmbedding/Linear.init in the pinned Marin code.
    model_cfg.init_method = _fan_in_truncated_normal
    model_cfg.output_layer_init_method = _fan_in_truncated_normal
    model_cfg.embedding_init_method = _embedding_truncated_normal

    full_config = upstream.pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        upstream.train_valid_test_datasets_provider,
        upstream.ModelType.encoder_or_decoder,
        upstream.forward_step,
        store=store,
        get_embedding_ranks=upstream.get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
