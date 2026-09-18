from __future__ import annotations

import contextlib
import hmac
import importlib.util
import json
import re
import sys
import threading
import types
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict


TASK = Path(__file__).resolve().parents[1]
ENVIRONMENT = TASK / "environment"


def _load_runner(monkeypatch: pytest.MonkeyPatch):
    core_spec = importlib.util.spec_from_file_location("replay_core", ENVIRONMENT / "replay_core.py")
    core = importlib.util.module_from_spec(core_spec)
    monkeypatch.setitem(sys.modules, "replay_core", core)
    assert core_spec.loader is not None
    core_spec.loader.exec_module(core)

    runner_spec = importlib.util.spec_from_file_location("dual_judge_eval_runner", ENVIRONMENT / "judge_eval.py")
    runner = importlib.util.module_from_spec(runner_spec)
    assert runner_spec.loader is not None
    runner_spec.loader.exec_module(runner)
    return runner


class _Image:
    size = (100, 100)

    def convert(self, mode: str):
        assert mode == "RGB"
        return self

    def resize(self, size: tuple[int, int]):
        self.size = size
        return self


class _Tensor:
    def __init__(self, length: int):
        self.shape = (1, length)

    def is_floating_point(self) -> bool:
        return False

    def to(self, device, **kwargs):
        return self


class _Tokens:
    def __init__(self, text: str, count: int):
        self.text = text
        self._count = count

    def numel(self) -> int:
        return self._count


class _Generated:
    def __init__(self, text: str, count: int):
        self.text = text
        self.count = count

    def __getitem__(self, key):
        assert isinstance(key, tuple) and key[0] == 0
        return _Tokens(self.text, self.count)


class _Tokenizer:
    def decode(self, tokens: _Tokens, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return tokens.text


class _Processor:
    def __init__(self, controller, processor_id: int):
        self.controller = controller
        self.processor_id = processor_id
        self.tokenizer = _Tokenizer()
        self.image_processor = types.SimpleNamespace(max_crops=99)

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False and add_generation_prompt is True
        return messages[0]["content"][0]["text"]

    def __call__(self, *, text: str, images, return_tensors: str, padding: bool,
                 max_crops: int, return_mm_token_type_ids: bool):
        match = re.search(r"case-(\d+)", text)
        assert match is not None
        case = int(match.group(1))
        assert images and return_tensors == "pt" and padding is True
        assert max_crops == 8 and return_mm_token_type_ids is False
        return {
            "input_ids": _Tensor(10 + case),
            "case": case,
            "processor_id": self.processor_id,
        }


class _Model:
    dtype = "float32"

    def __init__(self, controller, device_index: int):
        self.controller = controller
        self.device_index = device_index
        self.device = f"cuda:{device_index}"

    def eval(self):
        return self

    def generate(self, **batch):
        assert self.controller.active_device() == self.device_index
        assert batch["processor_id"] == self.device_index
        case = batch["case"]
        text = self.controller.generate(self.device_index, case)
        return _Generated(text, case + 1)


class _Controller:
    def __init__(self, *, device_count: int, predictions: dict[int, dict], synchronize_first: bool = False,
                 failure_device: int | None = None, failure: BaseException | None = None):
        self.device_count = device_count
        self.predictions = predictions
        self.synchronize_first = synchronize_first
        self.failure_device = failure_device
        self.failure = failure
        self.local = threading.local()
        self.lock = threading.Lock()
        self.first_devices: set[int] = set()
        self.barrier = threading.Barrier(2)
        self.release = threading.Event()
        self.healthy_finished = threading.Event()
        self.load_events: list[str] = []
        self.generate_events: list[tuple[int, int, str]] = []
        self.context_events: list[tuple[int, str]] = []

    def active_device(self):
        return getattr(self.local, "device", None)

    @contextlib.contextmanager
    def device(self, index: int):
        previous = self.active_device()
        self.local.device = index
        with self.lock:
            self.context_events.append((index, threading.current_thread().name))
        try:
            yield
        finally:
            self.local.device = previous

    def generate(self, device: int, case: int) -> str:
        with self.lock:
            self.generate_events.append((device, case, threading.current_thread().name))
            synchronize = self.synchronize_first and device not in self.first_devices
            self.first_devices.add(device)
        if synchronize:
            self.barrier.wait(timeout=2)
        if device == self.failure_device:
            self.release.set()
            assert self.failure is not None
            raise self.failure
        if self.failure_device is not None:
            assert self.release.wait(timeout=2)
            self.healthy_finished.set()
        return json.dumps({"thought": "", "action": self.predictions[case]})


def _install_expensive_doubles(monkeypatch: pytest.MonkeyPatch, controller: _Controller):
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.cuda = types.SimpleNamespace(device_count=lambda: controller.device_count, device=controller.device)
    fake_torch.inference_mode = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            processor_id = sum(event.startswith("processor:") for event in controller.load_events)
            controller.load_events.append(f"processor:{processor_id}")
            return _Processor(controller, processor_id)

    class AutoModelForImageTextToText:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            device = int(kwargs["device_map"][""].split(":")[1])
            controller.load_events.append(f"model:{device}")
            return _Model(controller, device)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = AutoProcessor
    fake_transformers.AutoModelForImageTextToText = AutoModelForImageTextToText
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    agent = types.ModuleType("agent")
    multimodal_agent = types.ModuleType("agent.multimodal_agent")

    class Template:
        def render(self, **context):
            return context["state"]["title"]

    class MultimodalAgent:
        def __init__(self, system_message: str = "system"):
            pass

    class KeyboardPress(BaseModel):
        model_config = ConfigDict(extra="ignore")
        key: str

    class SendMsgToUser(BaseModel):
        model_config = ConfigDict(extra="ignore")
        message: str

    actions = types.ModuleType("agent.actions")
    actions.KeyboardPress = KeyboardPress
    actions.SendMsgToUser = SendMsgToUser
    multimodal_agent.MultimodalAgent = MultimodalAgent
    multimodal_agent.USER_MSG_TEMPLATE = Template()
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.actions", actions)
    monkeypatch.setitem(sys.modules, "agent.multimodal_agent", multimodal_agent)


def _strategy() -> dict:
    return {
        "history_select_mode": "none",
        "history_budget_steps": 0,
        "history_stride": 1,
        "history_fields": set(),
        "thought_render": "omit",
        "action_format": "json",
        "url_truncate_len": 0,
        "page_state_fields": {"title"},
        "past_image_mode": "none",
        "past_image_count": 0,
        "past_image_scale": 1.0,
        "current_image_max_crops": 8,
        "max_new_tokens": 64,
    }


def _rows(actions: list[dict]) -> list[dict]:
    steps = []
    for case, action in enumerate(actions):
        steps.append({
            "case": case,
            "screenshot": _Image(),
            "action": action,
            "other_obs": {"title": f"case-{case}"},
        })
    return [{"instruction": {"high_level": "test"}, "trajectory": steps}]


def test_two_workers_infer_concurrently_and_reduce_in_original_task_order(monkeypatch):
    references = [
        {"name": "keyboard_press", "key": "A"},
        {"name": "send_msg_to_user", "message": "[exit]"},
        {"name": "keyboard_press", "key": "B"},
        {"name": "keyboard_press", "key": "C"},
        {"name": "keyboard_press", "key": "D"},
    ]
    predictions = {
        0: {"name": "keyboard_press", "key": "A"},
        1: {"name": "send_msg_to_user", "message": "[exit]"},
        2: {"name": "keyboard_press", "key": "wrong"},
        3: {"name": "keyboard_press", "key": "wrong"},
        4: {"name": "keyboard_press", "key": "D"},
    }
    controller = _Controller(device_count=2, predictions=predictions, synchronize_first=True)
    _install_expensive_doubles(monkeypatch, controller)
    runner = _load_runner(monkeypatch)
    score_order = []
    real_score_action = runner.score_action

    def observe_order(prediction, reference, step):
        score_order.append(step["case"])
        return real_score_action(prediction, reference, step)

    monkeypatch.setattr(runner, "score_action", observe_order)
    summary, reward = runner.run_replay(_rows(references), _strategy(), workers=2)

    assert controller.load_events == ["processor:0", "model:0", "processor:1", "model:1"]
    assert {device for device, _, _ in controller.generate_events} == {0, 1}
    assert len({thread for _, _, thread in controller.generate_events}) == 2
    assert score_order == [0, 1, 2, 3, 4]
    assert sorted(case for _, case, _ in controller.generate_events) == [0, 1, 2, 3, 4]
    assert summary == {
        "per_action": {
            "keyboard_press": {"mean": 0.5, "count": 4},
            "send_msg_to_user": {"mean": 1.0, "count": 1},
        },
        "parse_validity_rate": 1.0,
        "mean_prompt_tokens": 12.0,
        "mean_generated_tokens": 3.0,
        "total_steps": 5,
        "answer_token_f1_unscored": None,
    }
    assert reward == 0.75


def test_worker_failure_is_reraised_only_after_other_worker_finishes(monkeypatch):
    failure = RuntimeError("device failed")
    actions = [
        {"name": "keyboard_press", "key": "A"},
        {"name": "keyboard_press", "key": "B"},
    ]
    controller = _Controller(
        device_count=2,
        predictions={0: actions[0], 1: actions[1]},
        failure_device=0,
        failure=failure,
    )
    _install_expensive_doubles(monkeypatch, controller)
    runner = _load_runner(monkeypatch)

    with pytest.raises(RuntimeError) as caught:
        runner.run_replay(_rows(actions), _strategy(), workers=2)

    assert caught.value is failure
    assert controller.healthy_finished.is_set()


def test_default_mode_uses_one_worker_on_cuda_zero(monkeypatch):
    action = {"name": "keyboard_press", "key": "A"}
    controller = _Controller(device_count=1, predictions={0: action})
    _install_expensive_doubles(monkeypatch, controller)
    runner = _load_runner(monkeypatch)

    summary, reward = runner.run_replay(_rows([action]), _strategy(), public=True)

    assert controller.load_events == ["processor:0", "model:0"]
    assert [(device, case) for device, case, _ in controller.generate_events] == [(0, 0)]
    assert summary["total_steps"] == 1
    assert reward == 1.0


def test_two_worker_mode_requires_exactly_two_cuda_visible_devices(monkeypatch):
    controller = _Controller(device_count=1, predictions={})
    _install_expensive_doubles(monkeypatch, controller)
    runner = _load_runner(monkeypatch)

    with pytest.raises(RuntimeError, match="exactly two CUDA-visible GPUs"):
        runner.run_replay([], _strategy(), workers=2)

    assert controller.load_events == []


def test_formal_judge_facade_explicitly_requests_two_workers(monkeypatch):
    fixed_source = b"def run_replay(*args, **kwargs):\n    return kwargs\n"
    real_read_bytes = Path.read_bytes

    def read_bytes(path: Path):
        if str(path) == "/opt/molmoweb-task/judge_eval.py":
            return fixed_source
        return real_read_bytes(path)

    authority = types.ModuleType("asset_authority")
    authority.MAC_KEY = b"test"
    authority.RESERVED_KEY = "test"
    monkeypatch.setitem(sys.modules, "asset_authority", authority)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(hmac, "compare_digest", lambda left, right: True)
    spec = importlib.util.spec_from_file_location("formal_judge_facade", TASK / "tests" / "judge_eval.py")
    facade = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(facade)

    assert facade.run_replay([], {}, public=False) == {"public": False, "workers": 2}
