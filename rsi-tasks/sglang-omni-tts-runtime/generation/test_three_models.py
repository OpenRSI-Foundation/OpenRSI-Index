"""Synthetic contract tests for the future three-model task; no GPU execution."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from test_protocol import ROOT, PROTOCOL, common, load_module

noise = load_module('noise')
engine = load_module('engine')
NAMES = {'base', 'moss', 'cosy'}


def runs():
    return {name: [dict(
        repetition=i, seed=seed,
        metrics={key: 0.01 if key == 'wer_corpus' else 10.0 for key in common.METRICS},
        coverage=dict(tts=1088, wavs=1088, asr=1088, failed=0, skipped=0,
                      tts_warmup=16, asr_warmup=64),
        contention=dict(windows=1, errors=0, peak_foreign_cores=0, mean_foreign_cores=0),
        resources=dict(rss_peak_bytes=1, uss_peak_bytes=1, gpu_peak_mib=1, samples=1),
        timings=dict(native_tts_and_asr_seconds=1, authoritative_asr_seconds=1,
                     repetition_seconds=2, startup_and_warmup_events=[]),
        audio=dict(duration_min_s=1, duration_max_s=1, duration_mean_s=1,
                   sample_rates=[24000], formats=[[1, 'PCM_16']]),
        diagnostic_stock_wer=0.01, native_performance_summary={})
        for i, seed in enumerate((42, 43, 44))] for name in NAMES}


def calibration():
    here = ROOT / 'tests'
    values = runs()
    identity = dict(
        source=PROTOCOL['source'], protocol_sha256=common.sha(here / 'protocol.json'),
        assets_sha256=common.sha(here / 'assets.json'),
        software=dict(versions={}, files_sha256={}, interpreter_sha256='0' * 64),
        hardware=dict(gpus=[{'uuid': f'GPU-{i}', 'index': str(i)} for i in range(3)],
                      cpus=[list(range(16*i, 16*(i+1))) for i in range(3)],
                      numa=[[0]] * 3, topology=[[]] * 3),
        evaluator_sha256={p.relative_to(here).as_posix(): common.sha(p)
                          for p in here.rglob('*') if p.is_file() and p.suffix != '.gz'})
    return dict(schema=1, status='calibrated', identity=identity, run_id='synthetic',
                created_unix=time.time() - 10, total_wall_seconds=1,
                source_sha256='1' * 64, runs=values, means=common.means(values))


class ThreeModelTests(unittest.TestCase):
    def test_only_retained_models_are_required_and_assets_omit_custom(self):
        self.assertEqual({lane['name'] for lane in PROTOCOL['lanes']}, NAMES)
        common.validate_runs(runs(), 'synthetic')
        for name in NAMES:
            incomplete = runs()
            del incomplete[name]
            with self.subTest(missing=name), self.assertRaises(common.Failure):
                common.validate_runs(incomplete, 'synthetic')
        old = runs()
        old['custom'] = copy.deepcopy(old['base'])
        with self.assertRaises(common.Failure):
            common.validate_runs(old, 'synthetic')
        environment = json.loads((ROOT / 'environment/assets.json').read_text())
        self.assertEqual(environment, json.loads((ROOT / 'tests/assets.json').read_text()))
        repos = {asset['repo'] for asset in environment}
        self.assertTrue({lane['model'] for lane in PROTOCOL['lanes']} <= repos)
        self.assertIn('Qwen/Qwen3-ASR-1.7B', repos)
        self.assertIn('OpenMOSS-Team/MOSS-Audio-Tokenizer', repos)
        self.assertFalse(any('CustomVoice' in repo for repo in repos))

    def test_calibration_freezes_three_models_and_rejects_legacy_records(self):
        record = calibration()
        with patch.object(noise, 'HERE', ROOT / 'tests'):
            frozen = noise.summarize([record])
            self.assertEqual(set(frozen['means']), NAMES)
            self.assertEqual(frozen['generation'], 0)
            for section in ('runs', 'means'):
                old = copy.deepcopy(record)
                old[section]['custom'] = copy.deepcopy(old[section]['base'])
                with self.subTest(section=section), self.assertRaises(common.Failure):
                    noise.validate_single(old, old['identity'])
            old = copy.deepcopy(record)
            old['identity']['hardware']['gpus'].append({'uuid': 'GPU-3'})
            with self.assertRaises(common.Failure):
                noise.validate_single(old, old['identity'])
            old = copy.deepcopy(record)
            old['identity']['protocol_sha256'] = 'legacy-protocol-v4'
            with self.assertRaises(common.Failure):
                noise.summarize([old])
            with self.assertRaises(common.Failure):
                noise.validate_frozen(frozen, old['identity'])

    def test_scoring_retains_every_three_model_guard(self):
        baseline = dict(means=common.means(runs()))
        candidate = copy.deepcopy(baseline['means'])
        candidate['cosy']['throughput_qps'] = 11.0
        comparisons, guards, _ = noise.compare(candidate, baseline)
        self.assertEqual(set(comparisons), NAMES)
        self.assertEqual(sum(map(len, guards.values())), 33)
        self.assertTrue(all(all(g.values()) for g in guards.values()))
        for name in NAMES:
            for metric in common.METRICS:
                broken = copy.deepcopy(candidate)
                broken[name][metric] = (0.021 if metric == 'wer_corpus'
                                       else 11.0 if metric in PROTOCOL['lower_metrics']
                                       else 9.0)
                with self.subTest(name=name, metric=metric):
                    self.assertFalse(noise.compare(broken, baseline)[1][name][metric])
        candidate['cosy']['throughput_qps'] = 10.0
        self.assertFalse(noise.compare(candidate, baseline)[1]['cosy']['throughput_qps'])

    def test_eligible_promotion_preserves_b0_reward_and_rejects_regression(self):
        with patch.object(noise, 'HERE', ROOT / 'tests'):
            initial = noise.summarize([calibration()])
            candidate = calibration()
            candidate.update(status='eligible', baseline=initial, reward=1.1)
            for run in candidate['runs']['cosy']:
                run['metrics']['throughput_qps'] = 11.0
            candidate['means'] = common.means(candidate['runs'])
            promoted = noise.freeze_accepted(candidate, initial)
            self.assertEqual(promoted['generation'], 1)
            self.assertEqual(promoted['initial_means'], initial['means'])
            self.assertEqual(set(promoted['audio_formats']), NAMES)
            candidate['baseline'] = promoted
            with self.assertRaises(common.Failure):
                noise.freeze_accepted(candidate, promoted)

    def test_hardware_accepts_three_disjoint_lanes_and_rejects_four(self):
        cpus = [list(range(i * 16, (i + 1) * 16)) for i in range(3)]
        env = dict(RSI_HARNESS_EXPECTED_GPU_UUIDS='GPU-0,GPU-1,GPU-2',
                   RSI_TTS_CPU_LANES=json.dumps(cpus), RSI_TTS_NUMA_LANES='[[0],[0],[0]]')
        raw = '\n'.join(f'{i},GPU-{i},synthetic,1,0,1,1,1,1,1,1,Default,Disabled' for i in range(4))

        def topology(path, *args, **kwargs):
            if path.name == 'physical_package_id':
                return '0'
            if path.name == 'core_id':
                return path.parts[-3].removeprefix('cpu')
            raise AssertionError(path)

        with (patch.dict(os.environ, env), patch.object(engine.subprocess, 'check_output', return_value=raw),
              patch.object(Path, 'read_text', topology),
              patch.object(Path, 'glob', return_value=[Path('/sys/node0')]),
              patch.object(os, 'sched_getaffinity', return_value=set(range(64)))):
            hw = engine.hardware()
            self.assertEqual(len(hw['gpus']), 3)
            self.assertEqual(hw['cpus'], cpus)
            for override in (
                {'RSI_HARNESS_EXPECTED_GPU_UUIDS': 'GPU-0,GPU-1,GPU-2,GPU-3'},
                {'RSI_HARNESS_EXPECTED_GPU_UUIDS': 'GPU-0,GPU-1,GPU-1'},
                {'RSI_TTS_CPU_LANES': json.dumps(cpus + [list(range(48, 64))])},
                {'RSI_TTS_NUMA_LANES': '[[0],[0],[0],[0]]'},
                {'RSI_TTS_CPU_LANES': json.dumps([cpus[0], cpus[0], cpus[2]])},
            ):
                with self.subTest(override=override), patch.dict(os.environ, override):
                    with self.assertRaises(common.Failure):
                        engine.hardware()

    def test_calibration_orchestrates_only_three_lanes_without_gpu_calls(self):
        fixture = calibration()
        launches = []

        def fake_lane(config, path, env):
            name = config['lane']['name']
            launches.append((name, config['gpu']['uuid']))
            return fixture['runs'][name]

        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            for name, value in dict(
                original=lambda p: None, clean_env=lambda *args: {},
                hardware=lambda: fixture['identity']['hardware'], assets=lambda: 'synthetic-assets',
                software=lambda: fixture['identity']['software'], scope=lambda **kw: [],
                cleanup_gpus=lambda *args: None, run_lane=fake_lane,
                tree_inventory=lambda p: {}, sha=lambda p: '0' * 64,
                HERE=ROOT / 'tests').items():
                stack.enter_context(patch.object(engine, name, value))
            probe = stack.enter_context(patch.object(engine.subprocess, 'run'))
            stack.enter_context(contextlib.redirect_stdout(output))
            # main sets a private umask; restore it for the test runner.
            previous = os.umask(0o077)
            try:
                self.assertEqual(engine.main(calibrate=True), 0)
            finally:
                os.umask(previous)
        record = json.loads(output.getvalue().removeprefix('RSI_TTS_CALIBRATION='))
        self.assertEqual(set(record['runs']), NAMES)
        self.assertEqual(sum(map(len, record['runs'].values())), 9)
        self.assertEqual(set(launches), {('base', 'GPU-0'), ('moss', 'GPU-1'), ('cosy', 'GPU-2')})
        self.assertEqual(probe.call_count, 1)
        self.assertNotIn('reward', record)


if __name__ == '__main__':
    unittest.main()
