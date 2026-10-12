"""CPU-only contract tests; fixtures below are synthetic, never calibration data."""
import ast
import contextlib
import copy
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / 'tests/protocol.json').read_text())
_read_text = Path.read_text


def read_protocol(path, *args, **kwargs):
    if path == Path('/tests/protocol.json'):
        return json.dumps(PROTOCOL)
    return _read_text(path, *args, **kwargs)


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with patch.object(Path, 'read_text', read_protocol):
        spec.loader.exec_module(module)
    return module


common = load_module('common')
lane = load_module('lane')


def pinned_nodes(relative, names):
    with tarfile.open(ROOT / 'tests/source.tar.gz') as archive:
        member = next(m for m in archive if m.name.partition('/')[2] == relative)
        tree = ast.parse(archive.extractfile(member).read(), filename=relative)
    nodes = []
    for node in tree.body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], 'id', None)
        elif isinstance(node, ast.AnnAssign):
            name = getattr(node.target, 'id', None)
        if name in names:
            nodes.append(node)
    return tree, nodes


def execute_nodes(nodes, namespace):
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    exec(compile(tree, '<pinned upstream functions>', 'exec'), namespace)


class ProtocolTests(unittest.TestCase):
    def test_cuda_uuid_display_formats_preserve_full_identity(self):
        raw = '6fc47d1e-023a-37f4-a4c6-1d5da8b3ae46'
        self.assertTrue(common.cuda_uuid_matches(raw, 'GPU-' + raw))
        self.assertTrue(common.cuda_uuid_matches('GPU-' + raw, 'GPU-' + raw))
        self.assertFalse(common.cuda_uuid_matches(raw[:-1] + '7', 'GPU-' + raw))
        self.assertFalse(common.cuda_uuid_matches(raw[:8], 'GPU-' + raw))
        self.assertFalse(common.cuda_uuid_matches('GPU-GPU-' + raw, 'GPU-' + raw))
        self.assertFalse(common.cuda_uuid_matches(raw, raw))

    def test_pinned_benchmark_seed_reaches_http_payload(self):
        _, build = pinned_nodes('benchmarks/eval/benchmark_tts_seedtts.py', {'_build_generation_kwargs'})
        _, payload = pinned_nodes('benchmarks/tasks/tts.py', {'_build_tts_payload', '_resolve_tts_generation_kwargs'})
        ns = {}
        execute_nodes(build + payload, ns)
        sample = SimpleNamespace(target_text='Contract test.', ref_audio='reference.wav', ref_text='Reference.')
        for model in PROTOCOL['lanes']:
            for seed in (42, 43, 44):
                config = SimpleNamespace(max_new_tokens=2048, token_count=None, temperature=None,
                                         top_p=None, top_k=None, repetition_penalty=None, seed=seed)
                kwargs = ns['_build_generation_kwargs'](config)
                request = ns['_build_tts_payload'](sample, model['model'], **kwargs)
                self.assertEqual(request['seed'], seed)
                self.assertEqual(request['max_new_tokens'], 2048)

    def test_pinned_benchmark_defaults_reach_managed_server_flags(self):
        tree, profiles = pinned_nodes('benchmarks/eval/benchmark_tts_seedtts.py', {
            '_ModelBenchmarkProfile', '_AUK_BENCHMARK_PROFILE', '_MODEL_BENCHMARK_PROFILES',
            '_model_checkpoint_name', '_profile_for_model'})
        defaults = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument' and node.args
                    and isinstance(node.args[0], ast.Constant)):
                flag = node.args[0].value
                if flag in ('--max-running-requests', '--cuda-graph-max-bs', '--max-queued-requests', '--quantization'):
                    defaults[flag[2:].replace('-', '_')] = next(
                        ast.literal_eval(kw.value) for kw in node.keywords if kw.arg == 'default')
        overrides = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                         and getattr(node.targets[0], 'id', None) == 'engine_overrides')
        _, server = pinned_nodes('benchmarks/benchmarker/utils.py', {'managed_omni_server'})
        commands = []
        ns = dict(dataclass=dataclass, field=field, Path=Path, sys=sys,
                  contextmanager=contextlib.contextmanager, STARTUP_TIMEOUT=1200,
                  logger=SimpleNamespace(info=lambda *a: None),
                  _ensure_port_available=lambda *a: None,
                  _pinned_model_path=lambda model, config: model,
                  _resolve_managed_server_engine_stage=lambda **kw: 'test_stage',
                  start_server_from_cmd=lambda cmd, *a, **kw: commands.append(cmd),
                  stop_server=lambda *a: None, wait_for_gpu_memory_release=lambda: None)
        execute_nodes(profiles + server, ns)
        for model in PROTOCOL['lanes']:
            ns.update(profile=ns['_profile_for_model'](model['model']), config=SimpleNamespace(**defaults))
            self.assertTrue(ns['profile'].forward_sglang_engine)
            execute_nodes([overrides], ns)
            with ns['managed_omni_server'](model_path=model['model'], port=model['port'], host='127.0.0.1',
                                           log_file=None, **ns['engine_overrides']):
                pass
            for name, value in PROTOCOL['benchmark_engine_overrides'].items():
                flag = '--test_stage.engine.' + name
                self.assertEqual(defaults[name], value)
                self.assertEqual(commands[-1][commands[-1].index(flag) + 1], str(value))

    def test_all_model_commands_pair_seeds_without_mutating_protocol(self):
        original = copy.deepcopy(PROTOCOL)
        for model in PROTOCOL['lanes']:
            for index, seed in enumerate((42, 43, 44)):
                with self.subTest(model=model['name'], seed=seed):
                    expected = [common.PYTHON] + model['argv'] + ['--seed', str(seed)]
                    self.assertEqual(lane.benchmark_argv(model, index), expected)
                    self.assertEqual(lane.benchmark_argv(model, index, transcribe_only=True),
                                     expected + ['--transcribe-only'])
        self.assertEqual(PROTOCOL, original)

    def test_duplicate_or_arrival_seed_flags_are_rejected(self):
        for args in (['--seed', '7'], ['--seed=7'], ['--arrival-seed', '7'], ['--arrival-seed=7']):
            model = copy.deepcopy(PROTOCOL['lanes'][0])
            model['argv'] += args
            with self.subTest(args=args), self.assertRaises(common.Failure):
                lane.benchmark_argv(model, 0)

    def test_invalid_seed_schedules_and_indices_are_rejected(self):
        for seeds in (None, [42, 43], [42, 42, 44], [True, 43, 44], [-1, 43, 44], [2**31, 43, 44]):
            with self.subTest(seeds=seeds), patch.dict(common.PROTOCOL, repetition_seeds=seeds):
                with self.assertRaises(common.Failure):
                    common.repetition_seed(0)
        for index in (-1, 3, True, 0.0):
            with self.subTest(index=index), self.assertRaises(common.Failure):
                common.repetition_seed(index)

    def test_native_output_must_record_seed_and_benchmark_overrides(self):
        config = dict(stream=False, warmup=16, concurrency=16, max_new_tokens=2048,
                      max_samples=None, sample_offset=0, temperature=None, top_p=None,
                      top_k=None, repetition_penalty=None, seed=42, arrival_seed=None,
                      max_running_requests=64, cuda_graph_max_bs=64)
        lane.check_tts_config(config, 42, 'synthetic-output')
        for key, wrong in (('seed', None), ('seed', 43), ('arrival_seed', 42),
                           ('concurrency', 64), ('max_running_requests', 16), ('cuda_graph_max_bs', 16)):
            with self.subTest(key=key, wrong=wrong), self.assertRaises(common.Failure):
                lane.check_tts_config({**config, key: wrong}, 42, 'synthetic-output')
        for key in config:
            missing = dict(config)
            del missing[key]
            with self.subTest(missing=key), self.assertRaises(common.Failure):
                lane.check_tts_config(missing, 42, 'synthetic-output')

    def test_calibration_and_candidate_records_reject_unpaired_or_missing_seeds(self):
        runs = {model['name']: [dict(
            repetition=i, seed=seed, metrics={key: 1.0 for key in common.METRICS},
            coverage=dict(tts=1088, wavs=1088, asr=1088, failed=0, skipped=0, tts_warmup=16, asr_warmup=64),
            contention=dict(windows=1, errors=0, peak_foreign_cores=0.0))
            for i, seed in enumerate((42, 43, 44))] for model in PROTOCOL['lanes']}
        common.validate_runs(runs, 'synthetic-record')
        for name in runs:
            for index in range(3):
                for replacement in (None, 999, True):
                    wrong = copy.deepcopy(runs)
                    if replacement is None:
                        del wrong[name][index]['seed']
                    else:
                        wrong[name][index]['seed'] = replacement
                    with self.subTest(name=name, index=index, seed=replacement), self.assertRaises(common.Failure):
                        common.validate_runs(wrong, 'synthetic-record')

    def test_schema_requires_seed_for_every_model(self):
        from jsonschema import Draft202012Validator
        schema = json.loads((ROOT / 'tests/calibration.schema.json').read_text())
        Draft202012Validator.check_schema(schema)
        for model in PROTOCOL['lanes']:
            item = schema['properties']['runs']['properties'][model['name']]['items']
            self.assertIn('seed', item['required'])
            seed_validator = Draft202012Validator(item['properties']['seed'])
            for seed in (42, 43, 44):
                seed_validator.validate(seed)
            for seed in (None, 999, True):
                self.assertTrue(list(seed_validator.iter_errors(seed)))


if __name__ == '__main__':
    unittest.main()
