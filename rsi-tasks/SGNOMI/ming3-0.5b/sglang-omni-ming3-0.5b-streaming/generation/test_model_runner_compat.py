"""Exercise the archived overrides against the latest core's method contracts."""
import ast
import dataclasses
from pathlib import Path
import tarfile
import unittest

@dataclasses.dataclass
class BaseConfig:
    capacity: int = 64
    future_upstream_field: str = 'preserved'

@dataclasses.dataclass
class OmniConfig(BaseConfig):
    total_gpu_memory_fraction: float = .9
    kv_cache_bytes: int | None = None

class Base:
    def __init__(self):
        self.total_gpu_memory_fraction = .75
        self.kv_cache_bytes = None
        self.max_total_num_tokens = 1024
        self.device = 'cuda'
        self.gpu_id = 0
        self.shrink = False

    def init_kv_cache_configurator(self, draft_parallel_layout=None):
        self.observed_layout = draft_parallel_layout
        self.kv_cache_configurator = BaseConfig()

    def post_capture_resize_kv_pool(self, *, draft_runners=()):
        self.observed_drafts = draft_runners
        if self.shrink:
            self.max_total_num_tokens -= 1
        return 'resized'

def runner_class():
    archive = Path(__file__).resolve().parents[1] / 'environment/source.tar.gz'
    with tarfile.open(archive) as source:
        member = next(m for m in source if m.name.endswith('/sglang_omni/model_runner/sglang_model_runner.py'))
        tree = ast.parse(source.extractfile(member).read().decode())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SGLModelRunner')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
               ('init_kv_cache_configurator', 'post_capture_resize_kv_pool')]
    assert len(methods) == 2
    cls = ast.ClassDef(name='Runner', bases=[ast.Name(id='Base', ctx=ast.Load())],
                       keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    values = dict(Base=Base, dataclasses=dataclasses, OmniKVCacheConfigurator=OmniConfig,
                  free_gpu_memory_bytes=lambda *_: 1 << 30, format_bytes_gib=str)
    exec(compile(module, str(archive), 'exec'), values)
    return values['Runner']

class LatestCoreOverrideTest(unittest.TestCase):
    def test_layout_forwarding_preserves_all_config_fields_and_omni_budget(self):
        runner = runner_class()()
        runner.kv_cache_bytes = 1 << 20
        layout = object()
        runner.init_kv_cache_configurator(layout)
        self.assertIs(runner.observed_layout, layout)
        self.assertIsInstance(runner.kv_cache_configurator, OmniConfig)
        self.assertEqual(runner.kv_cache_configurator.future_upstream_field, 'preserved')
        self.assertEqual(runner.kv_cache_configurator.kv_cache_bytes, 1 << 20)
        self.assertEqual(runner.kv_cache_configurator.total_gpu_memory_fraction, .75)

    def test_draft_runner_forwarding_in_both_budget_paths(self):
        for budget in (None, 1 << 20):
            with self.subTest(budget=budget):
                runner = runner_class()()
                runner.kv_cache_bytes = budget
                drafts = (object(), object())
                self.assertEqual(runner.post_capture_resize_kv_pool(draft_runners=drafts), 'resized')
                self.assertIs(runner.observed_drafts, drafts)

    def test_explicit_budget_still_rejects_post_capture_shrink(self):
        runner = runner_class()()
        runner.kv_cache_bytes = 1 << 20
        runner.shrink = True
        with self.assertRaisesRegex(RuntimeError, 'cannot honor the declared byte budget'):
            runner.post_capture_resize_kv_pool(draft_runners=(object(),))

if __name__ == '__main__':
    unittest.main()
