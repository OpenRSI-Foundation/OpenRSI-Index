"""Public matched adapter. Formal selection, trace comparison and reward live in Judge.

Uses pinned AgentLoopManager, GenericAgentLoop, GymEnvironmentInteraction,
ALFWorldGame and DataProto. Overrides only scheduling, exact reset, bounds and packing.
"""
import asyncio
import ipaddress
import math
import os
import socket
import time
import uuid
from pathlib import Path


def configure_address():
    addresses = set()
    for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
        address = ipaddress.IPv4Address(item[4][0])
        if not (address.is_unspecified or address.is_loopback or address.is_link_local
                or address.is_multicast or address.is_reserved):
            addresses.add(str(address))
    if len(addresses) != 1:
        raise RuntimeError("address_discovery")
    address = addresses.pop()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((address, 0))
    os.environ["VLLM_HOST_IP"] = address
    return address


# Address discovery precedes the first Ray/vLLM import, including in actor imports.
HOST_IP = configure_address()
# Fixed backend mode must be established before Ray or any vLLM import, also
# when the module is imported in a worker with a minimal runtime environment.
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoTokenizer
from verl.protocol import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopMetrics, AgentLoopOutput
from opentinker.server.generic_agent_loop import GenericAgentLoop, GenericAgentData
from opentinker.environment.gym_environment_interaction import GymEnvironmentInteraction
from policy_client import ContractError, PolicyClient

MODEL = "/opt/models/qwen2.5-3b-instruct"
DATA = Path("/opt/data/alfworld")


@ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0)
class EnvironmentShard:
    """A single serial actor owns each TextWorld parser and all its episodes."""
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from opentinker.environment.alfworld.alfworld_game import ALFWorldGame
        self.game_type = ALFWorldGame
        self.games = {}

    def reset(self, key, record):
        if key in self.games:
            raise RuntimeError("duplicate_environment")
        game = self.game_type(max_steps=20, split=("train" if "/train/" in record["path"]
                                                else "eval_out_of_distribution"))
        # Upstream reset, wrappers, prompts and parser, with a singleton exact selection.
        selected = str((DATA / record["path"]).parent)
        game._get_cached_game_paths = lambda: [selected]
        observation = game.reset(seed=0)
        self.games[key] = game
        initial = game.get_initial_user_message()
        return {"system": game.get_system_prompt(),
                "observation": f"{initial}\n\n{observation}" if observation else initial}

    def step(self, key, action):
        result = self.games[key].step(action)
        return result.observation, result.reward, result.done, result.info

    def close(self, key):
        game = self.games.pop(key)
        if game._tw_env is not None:
            game._tw_env.close()


@ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0, max_concurrency=128)
class Broker:
    def __init__(self, servers):
        self.servers = servers
        self.client = None
        self.states = {}
        self.error = None
        self.events = []
        self.outstanding = [0, 0]
        self.changed = asyncio.Event()
        self.stop = False
        self.pump = None
        self.waits = []
        self.model_waits = []

    async def initialize(self, source, batch_size, shards):
        self.client = PolicyClient(source)
        await self.client.start()
        await self.client.call("reset", {"workers": 8, "replicas": 2,
                                        "batch_size": batch_size, "environment_shards": shards})
        self.pump = asyncio.create_task(self._pump())

    async def begin(self, ids, workers, batch_index):
        if self.states and any(x["state"] != "done" for x in self.states.values()):
            raise RuntimeError("incomplete_batch")
        self.batch_index = batch_index
        self.states = {tid: {"id": tid, "worker": worker, "state": "pending",
                            "prompt_tokens": None, "generated_tokens": 0,
                            "completed_turns": 0, "past_service_seconds": [],
                            "routing": [], "admitted": asyncio.Event(), "future": None,
                            "queued_at": None}
                       for tid, worker in zip(ids, workers, strict=True)}
        self.changed.set()

    async def admitted(self, tid):
        while not self.states[tid]["admitted"].is_set():
            self._raise()
            try:
                await asyncio.wait_for(self.states[tid]["admitted"].wait(), 1)
            except asyncio.TimeoutError:
                continue
        self._raise()

    def _raise(self):
        if self.error:
            raise self.error

    async def generate(self, tid, prompt):
        self._raise()
        state = self.states[tid]
        if state["state"] != "environment":
            raise RuntimeError("invalid_adapter_state")
        state["state"] = "ready"
        state["prompt_tokens"] = len(prompt)
        state["prompt"] = prompt
        state["queued_at"] = time.perf_counter()
        future = asyncio.get_running_loop().create_future()
        state["future"] = future
        self.changed.set()
        return await future

    async def completed_turn(self, tid):
        self._raise()
        self.states[tid]["completed_turns"] += 1

    async def finished(self, tid):
        self._raise()
        self.states[tid]["state"] = "done"
        self.changed.set()

    async def _dispatch(self, tid, replica):
        state = self.states[tid]
        start = time.perf_counter()
        try:
            output = await self.servers[replica].generate.remote(
                prompt_ids=state.pop("prompt"), request_id=tid,
                sampling_params={"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                                 "repetition_penalty": 1.0, "logprobs": True}, image_data=None)
            if (not output.token_ids or len(output.token_ids) > 128 or output.log_probs is None
                    or len(output.log_probs) != len(output.token_ids)
                    or not all(math.isfinite(x) for x in output.log_probs)):
                raise RuntimeError("backend_output")
            duration = time.perf_counter() - start
            self.model_waits.append(duration)
            state["generated_tokens"] += len(output.token_ids)
            state["past_service_seconds"].append(duration)
            self.events.append({"id": tid, "replica": replica, "service_seconds": duration,
                                "generated_tokens": len(output.token_ids)})
            state["state"] = "environment"
            state["future"].set_result(output)
        except BaseException as exc:
            self.error = exc
            if not state["future"].done():
                state["future"].set_exception(exc)
        finally:
            self.outstanding[replica] -= 1
            self.changed.set()

    def _decision(self, result):
        if not isinstance(result, dict) or set(result) != {"admit", "dispatch"}:
            raise ContractError("decision_shape", "schedule", "return exactly admit and dispatch lists")
        admit, dispatch = result["admit"], result["dispatch"]
        if (not isinstance(admit, list) or not isinstance(dispatch, list)
                or len(admit) > len(self.states) or len(dispatch) > len(self.states)):
            raise ContractError("decision_size", "schedule", "lists must be bounded by the current batch")
        seen = set()
        for tid in admit:
            if not isinstance(tid, str) or tid not in self.states or tid in seen:
                raise ContractError("admit_id", "admit", "use each pending trajectory ID at most once")
            seen.add(tid)
            if self.states[tid]["state"] != "pending":
                raise ContractError("admit_state", "admit", "admit only pending trajectories")
        seen = set()
        for item in dispatch:
            if not isinstance(item, dict) or set(item) != {"id", "replica"}:
                raise ContractError("dispatch_shape", "dispatch", "each record must contain id and replica")
            tid, replica = item["id"], item["replica"]
            if not isinstance(tid, str) or tid not in self.states or tid in seen:
                raise ContractError("dispatch_id", "dispatch.id", "dispatch each ready ID at most once")
            seen.add(tid)
            if self.states[tid]["state"] != "ready":
                raise ContractError("dispatch_state", "dispatch.id", "only ready turns can be dispatched")
            if type(replica) is not int or replica not in (0, 1):
                raise ContractError("replica", "dispatch.replica", "replica must be integer 0 or 1")
        return admit, dispatch

    async def _pump(self):
        try:
            while not self.stop:
                await self.changed.wait()
                self.changed.clear()
                if not self.states or all(x["state"] == "done" for x in self.states.values()):
                    continue
                self._raise()
                fields = ("id", "worker", "state", "prompt_tokens", "generated_tokens",
                          "completed_turns", "past_service_seconds", "routing")
                view = {"batch_index": self.batch_index,
                        "trajectories": [{k: x[k] for k in fields} for x in self.states.values()],
                        "replica_outstanding": self.outstanding[:], "events": self.events[:]}
                self.events.clear()
                admit, dispatch = self._decision(await self.client.call("schedule", view))
                for tid in admit:
                    self.states[tid]["state"] = "environment"
                    self.states[tid]["admitted"].set()
                for item in dispatch:
                    tid, replica = item["id"], item["replica"]
                    state = self.states[tid]
                    state["state"] = "running"
                    state["routing"].append(replica)
                    self.outstanding[replica] += 1
                    self.waits.append(time.perf_counter() - state["queued_at"])
                    asyncio.create_task(self._dispatch(tid, replica))
                if not admit and not dispatch and not any(
                        x["state"] in {"running", "environment"} for x in self.states.values()):
                    raise ContractError("schedule_stalled", "schedule", "dispatch or admit work when nothing is in flight")
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError) and self.stop:
                # shutdown deliberately cancels the idle pump after work drains.
                return
            self.error = exc
            for state in self.states.values():
                state["admitted"].set()
                future = state.get("future")
                if future is not None and not future.done():
                    future.set_exception(exc)

    async def shutdown(self):
        self.stop = True
        if self.pump:
            self.pump.cancel()
            await asyncio.gather(self.pump, return_exceptions=True)
        if self.client:
            await self.client.close()
        self._raise()
        return {"ready_wait_mean_sec": float(np.mean(self.waits)) if self.waits else 0.0,
                "ready_wait_max_sec": max(self.waits, default=0.0),
                "model_wait_count": len(self.model_waits),
                "model_wait_total_sec": sum(self.model_waits),
                "model_wait_mean_sec": float(np.mean(self.model_waits)) if self.model_waits else 0.0,
                "model_wait_max_sec": max(self.model_waits, default=0.0)}


class Interaction(GymEnvironmentInteraction):
    def __init__(self, shard, record):
        super().__init__({"max_steps": 20})
        self.shard, self.record = shard, record
        self.initial = None
        self.waits = {"reset": [], "step": []}

    async def _call_env_reset(self, instance_id, **kwargs):
        started = time.perf_counter()
        self.initial = await self.shard.reset.remote(instance_id, self.record)
        self.waits["reset"].append(time.perf_counter() - started)
        return self.initial["observation"]

    async def _call_env_step(self, instance_id, action):
        started = time.perf_counter()
        result = await self.shard.step.remote(instance_id, action)
        self.waits["step"].append(time.perf_counter() - started)
        return result

    async def finalize_interaction(self, instance_id, **kwargs):
        await self.shard.close.remote(instance_id)
        await super().finalize_interaction(instance_id, **kwargs)


class MatchedLoop(GenericAgentLoop):
    def __init__(self, tokenizer, broker, shard, record, tid):
        self.tokenizer, self.broker = tokenizer, broker
        self.interaction = Interaction(shard, record)
        self.tid = tid
        self.processor = None
        self.apply_chat_template_kwargs = {}
        self.loop = asyncio.get_running_loop()
        self.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)

    async def run(self, sampling_params=None, **kwargs):
        await self.broker.admitted.remote(self.tid)
        await self.interaction.start_interaction(self.tid)
        initial = self.interaction.initial
        messages = [{"role": "system", "content": initial["system"]},
                    {"role": "user", "content": initial["observation"]}]
        data = GenericAgentData(messages, {}, self.tid, self.interaction)
        trace = {"initial_observation": initial["observation"], "turns": [], "termination": None}
        start = time.perf_counter()
        try:
            # Inherit the official initial chat-template/tokenizer path.
            await self._handle_pending_state(data, {})
            initial_ids = data.prompt_ids[:]
            while data.assistant_turns < 20:
                if len(data.prompt_ids) >= 8192:
                    trace["termination"] = "context_limit"
                    break
                output = await self.broker.generate.remote(self.tid, data.prompt_ids[:])
                if len(data.prompt_ids) + len(output.token_ids) > 8192:
                    raise RuntimeError("context_bound")
                data.assistant_turns += 1
                data.prompt_ids.extend(output.token_ids)
                data.response_mask.extend([1] * len(output.token_ids))
                data.response_logprobs.extend(output.log_probs)
                assistant = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                data.messages.append({"role": "assistant", "content": assistant})
                done, observation, reward, info = await self.interaction.generate_response(self.tid, data.messages)
                data.user_turns += 1
                data.turn_scores.append(reward)
                trace["turns"].append({"token_ids": output.token_ids, "log_probs": output.log_probs,
                                       "assistant": assistant, "action": info["action_taken"],
                                       "observation": observation, "reward": reward,
                                       "environment_done": done})
                await self.broker.completed_turn.remote(self.tid)
                observation_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": observation}], add_generation_prompt=True,
                    tokenize=True)[len(self.system_prompt):]
                fits = len(data.prompt_ids) + len(observation_ids) <= 8192
                if fits:
                    data.prompt_ids.extend(observation_ids)
                    data.response_mask.extend([0] * len(observation_ids))
                    data.response_logprobs.extend([0.0] * len(observation_ids))
                    data.messages.append({"role": "user", "content": observation})
                if done:
                    trace["termination"] = "step_limit" if data.user_turns == 20 and not info.get("success") else "environment_done"
                    break
                if data.assistant_turns == 20:
                    trace["termination"] = "turn_limit"
                    break
                if not fits or len(data.prompt_ids) == 8192:
                    trace["termination"] = "context_limit"
                    break
            trace["prompt_ids"] = initial_ids
            trace["response_ids"] = data.prompt_ids[len(initial_ids):]
            trace["response_mask"] = data.response_mask
            trace["response_logprobs"] = data.response_logprobs
            if len(trace["response_ids"]) != len(data.response_mask):
                raise RuntimeError("mask_length")
            result = AgentLoopOutput(prompt_ids=initial_ids, response_ids=trace["response_ids"],
                                     response_mask=data.response_mask, response_logprobs=data.response_logprobs,
                                     reward_score=sum(data.turn_scores), num_turns=data.assistant_turns + data.user_turns + 1,
                                     metrics=AgentLoopMetrics(generate_sequences=time.perf_counter() - start),
                                     extra_fields={"trace": trace, "environment_waits": self.interaction.waits})
        finally:
            await self.interaction.finalize_interaction(self.tid)
        await self.broker.finished.remote(self.tid)
        return result


def pack(outputs, pad):
    """Original-order DataProto with full outputs; padding never truncates accepted tokens."""
    n = len(outputs)
    p = max(len(o.prompt_ids) for o in outputs)
    r = max(1, max(len(o.response_ids) for o in outputs))
    prompts = torch.full((n, p), pad, dtype=torch.int64)
    responses = torch.full((n, r), pad, dtype=torch.int64)
    mask = torch.zeros((n, r), dtype=torch.int64)
    attention = torch.zeros((n, p + r), dtype=torch.int64)
    logprobs = torch.zeros((n, r), dtype=torch.float32)
    traces = np.empty(n, dtype=object)
    environment_waits = np.empty(n, dtype=object)
    for i, output in enumerate(outputs):
        a, b = len(output.prompt_ids), len(output.response_ids)
        prompts[i, p-a:] = torch.tensor(output.prompt_ids, dtype=torch.int64)
        if b:
            responses[i, :b] = torch.tensor(output.response_ids, dtype=torch.int64)
            mask[i, :b] = torch.tensor(output.response_mask, dtype=torch.int64)
            logprobs[i, :b] = torch.tensor(output.response_logprobs, dtype=torch.float32)
        attention[i, p-a:p+b] = 1
        traces[i] = output.extra_fields["trace"]
        environment_waits[i] = output.extra_fields["environment_waits"]
    positions = torch.clamp(attention.cumsum(-1) - 1, min=0)
    tensors = TensorDict({"prompts": prompts, "responses": responses, "response_mask": mask,
                          "input_ids": torch.cat((prompts, responses), dim=-1),
                          "attention_mask": attention, "position_ids": positions,
                          "rollout_log_probs": logprobs}, batch_size=n)
    return DataProto(batch=tensors, non_tensor_batch={"trace": traces, "environment_waits": environment_waits})


@ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0)
class Worker:
    def __init__(self, config, servers, reward_router_address=None):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)

    async def generate(self, records, ids, shards, broker, indices):
        # Eager coroutine creation within each of eight static equal chunks.
        return await asyncio.gather(*[MatchedLoop(self.tokenizer, broker, shards[index % len(shards)], record, tid).run()
                                      for record, tid, index in zip(records, ids, indices, strict=True)])


class Manager(AgentLoopManager):
    def __init__(self, config):
        self.agent_loop_workers_class = Worker
        super().__init__(config, worker_group=None, rm_wg=None)
        self.pad = AutoTokenizer.from_pretrained(MODEL, local_files_only=True).pad_token_id

    def batch(self, records, broker, shards, batch_index):
        size = len(records)
        if size % 8:
            raise ValueError("batch_not_divisible_by_workers")
        chunk = size // 8
        ids = [uuid.uuid4().hex for _ in records]
        workers = [i // chunk for i in range(size)]
        ray.get(broker.begin.remote(ids, workers, batch_index))
        futures = [worker.generate.remote(records[i*chunk:(i+1)*chunk], ids[i*chunk:(i+1)*chunk],
                                           shards, broker, list(range(i*chunk, (i+1)*chunk)))
                   for i, worker in enumerate(self.agent_loop_workers)]
        outputs = [item for group in ray.get(futures) for item in group]
        return pack(outputs, self.pad)

    def drain_sync(self, clear=False):
        ray.get([s.wait_for_requests_to_drain.remote() for s in self.server_handles])
        if clear:
            ray.get([s.task_reset_prefix_cache.remote() for s in self.server_handles])
        def synchronize(worker):
            import torch
            torch.cuda.synchronize()
        ray.get([w.__ray_call__.remote(synchronize) for replica in self.rollout_replicas for w in replica.workers])


def fixed_config():
    from verl.workers.config import RolloutConfig
    from dataclasses import fields, is_dataclass

    def hydra_config(value):
        # BaseConfig defaults _target_ to an empty string. Hydra recursively
        # instantiates nested configs, so every dataclass needs its real target.
        if is_dataclass(value):
            result = {field.name: hydra_config(getattr(value, field.name)) for field in fields(value)}
            result["_target_"] = f"{type(value).__module__}.{type(value).__qualname__}"
            return result
        if isinstance(value, dict):
            return {key: hydra_config(item) for key, item in value.items()}
        if isinstance(value, list):
            return [hydra_config(item) for item in value]
        return value

    rollout = hydra_config(RolloutConfig(name="vllm", mode="async", temperature=0, top_p=1, top_k=-1,
                                  do_sample=False, n=1, prompt_length=4096, response_length=4096,
                                  dtype="bfloat16", gpu_memory_utilization=0.8, enforce_eager=True,
                                  free_cache_engine=False, tensor_model_parallel_size=1,
                                  data_parallel_size=1, pipeline_model_parallel_size=1,
                                  max_model_len=8192, max_num_seqs=32, max_num_batched_tokens=8192,
                                  calculate_log_probs=True, enable_chunked_prefill=True,
                                  enable_prefix_caching=True, load_format="safetensors"))
    rollout["_target_"] = "verl.workers.config.RolloutConfig"
    return OmegaConf.create({"actor_rollout_ref": {"model": {"path": MODEL, "trust_remote_code": False,
                                                               "override_config": {"attn_implementation": "sdpa"}},
                                                 "rollout": rollout},
                             "trainer": {"nnodes": 1, "n_gpus_per_node": 2},
                             "reward_model": {"enable": False, "enable_resource_pool": False}})


def run_pass(records, warmup, source, baseline, batch_size, shard_count, scratch):
    """One fresh process calls this once; no model/process state crosses passes."""
    if len(records) % batch_size or batch_size not in (8, 32) or shard_count not in (2, 8):
        raise ValueError("workload_shape")
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_properties(i).total_memory < 48_000_000_000 for i in range(2)):
        raise RuntimeError("gpu_contract")
    module_root = str(Path(__file__).parent)
    actor_env = {key: value for key, value in os.environ.items() if key in {
        "VLLM_HOST_IP", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
        "HF_HOME", "XDG_CACHE_HOME", "ALFWORLD_DATA", "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK",
        "TOKENIZERS_PARALLELISM", "PYTHONDONTWRITEBYTECODE", "WANDB_MODE", "RAY_USAGE_STATS_ENABLED"}}
    actor_env["VLLM_BATCH_INVARIANT"] = "1"
    actor_env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    actor_env["PYTHONPATH"] = module_root + ":/opt/src/opentinker:/opt/src/verl:/opt/src/alfworld"
    ray.init(_node_ip_address=HOST_IP, num_cpus=32, num_gpus=2, include_dashboard=False,
             object_store_memory=8 * 1024**3,
             _temp_dir=str(Path(scratch) / "ray"), log_to_driver=False,
             runtime_env={"env_vars": actor_env})
    broker = None
    try:
        manager = Manager(fixed_config())
        shards = [EnvironmentShard.remote() for _ in range(shard_count)]
        warm_broker = Broker.remote(manager.server_handles)
        try:
            ray.get(warm_broker.initialize.remote(baseline, 8, shard_count))
            manager.batch(warmup, warm_broker, shards, -1)
            ray.get(warm_broker.shutdown.remote())
        except BaseException:
            raise RuntimeError("warmup_failure") from None
        ray.kill(warm_broker)
        manager.drain_sync(clear=True)
        broker = Broker.remote(manager.server_handles)
        manager.drain_sync()
        started = time.perf_counter()
        ray.get(broker.initialize.remote(source, batch_size, shard_count))
        durations, traces, packed, environment_waits = [], [], [], []
        for i in range(0, len(records), batch_size):
            batch_start = time.perf_counter()
            result = manager.batch(records[i:i+batch_size], broker, shards, i // batch_size)
            traces.extend(result.non_tensor_batch["trace"].tolist())
            environment_waits.extend(result.non_tensor_batch["environment_waits"].tolist())
            # Full DataProto tensors, including all accepted masks, are correctness inputs.
            packed.append({key: value.tolist() for key, value in result.batch.items()})
            durations.append(time.perf_counter() - batch_start)
        wait_stats = ray.get(broker.shutdown.remote())
        broker = None
        manager.drain_sync()
        seconds = time.perf_counter() - started
        if not math.isfinite(seconds) or seconds <= 0 or len(traces) != len(records):
            raise RuntimeError("incomplete_timing")
        calls = sum(len(t["turns"]) for t in traces)
        tokens = sum(len(turn["token_ids"]) for t in traces for turn in t["turns"])
        environment_stats = {}
        for operation in ("reset", "step"):
            waits = [duration for episode in environment_waits for duration in episode[operation]]
            prefix = "environment_" + operation + "_wait"
            environment_stats.update({prefix + "_count": len(waits), prefix + "_total_sec": sum(waits),
                                      prefix + "_mean_sec": float(np.mean(waits)) if waits else 0.0,
                                      prefix + "_max_sec": max(waits, default=0.0)})
        return {"seconds": seconds, "traces": traces, "packed": packed,
                "episodes": len(traces), "model_calls": calls, "generated_tokens": tokens,
                "completed_trajectories_per_sec": len(traces) / seconds,
                "batch_mean_sec": float(np.mean(durations)), "batch_tail_sec": max(durations),
                **wait_stats, **environment_stats}
    finally:
        if broker is not None:
            try:
                ray.get(broker.shutdown.remote(), timeout=10)
            except BaseException:
                pass
        ray.shutdown()
